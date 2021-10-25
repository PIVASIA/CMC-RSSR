# Based on code from 
# https://github.com/mkisantal/backboned-unet/blob/master/backboned_unet/unet.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class UpsampleBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out=None,
                 skip_in=0,
                 use_bn=True,
                 parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv,
        # or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in,
                                         out_channels=ch_out,
                                         kernel_size=(4, 4),
                                         stride=2,
                                         padding=1,
                                         output_padding=0,
                                         bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in,
                                   out_channels=ch_out,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=1,
                                   bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in,
                               out_channels=ch_out,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1,
                               bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric \
            else F.interpolate(x,
                               size=None,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):
    def __init__(self,
                 backbone,
                 n_channels,
                 classes=-1,
                 encoder_freeze=True,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True) -> None:
        """[summary]

        Args:
            backbone ([type]): [description]
            n_channels (int): number of input channels.
            classess (int): number of output classes.
                    If -1, return feature instead. Defaults to -1.
            encoder_freeze (bool, optional): freeze encoder part.
            decoder_filters (tuple, optional): [description].
                    Defaults to (256, 128, 64, 32, 16).
            parametric_upsampling (bool, optional): [description].
                    Defaults to True.
            shortcut_features (str, optional): [description].
                    Defaults to 'default'.
            decoder_use_batchnorm (bool, optional): [description].
                    Defaults to True.
        """
        super().__init__()

        self.backbone = backbone
        self.n_channels = n_channels

        # specifying skip feature and output names for backbone
        self.shortcut_features = [None, 'relu', 'layer1', 'layer2', 'layer3']
        self.bb_out_name = 'layer4'

        shortcut_chs, bb_out_chs = self._infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        # avoiding having more blocks than skip connections
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in \
                enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(
                UpsampleBlock(filters_in,
                              filters_out,
                              skip_in=shortcut_chs[num_blocks-i-1],
                              parametric=parametric_upsampling,
                              use_bn=decoder_use_batchnorm))

        self.final_conv = None
        if classes > 0:
            self.final_conv = nn.Conv2d(decoder_filters[-1],
                                        self.classes,
                                        kernel_size=(1, 1))

        if encoder_freeze:
            self._freeze_encoder()

    def forward(self, x):
        """ Forward propagation

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, features = self._forward_backbone(x)

        for skip_name, upsample_block in \
                zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        if self.final_conv is not None:
            x = self.final_conv(x)

        return x

    def _forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def _freeze_encoder(self):
        """ Freezing encoder parameters, the newly initialized
        decoder parameters are remaining trainable. """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _infer_skip_channels(self):
        """Getting the number of channels at skip connections
        and at the output of the encoder.

        Returns:
            channels: list of number_of_channels at every skip
                    connections of encoder
            out_channels: number_of_channels at output of encoder
        """
        x = torch.zeros(1, self.n_channels, 224, 224)
        channels = []

        # forward run in encoder to count channels
        # (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])

            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break

        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


if __name__ == "__main__":
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)

    from torchsummary import summary
    model = Unet(resnet18, 3)
    summary(model, (3, 256, 256))