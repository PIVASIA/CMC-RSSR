from __future__ import print_function

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from util import parse_option
from train_CMC import CMCModel
from models.unet import Unet


class DoubleUnetModel(pl.LightningModule):
    def __init__(self,
                 backbone_l,
                 backbone_ab,
                 channels_l,
                 channels_ab,
                 n_classes,
                 learning_rate,
                 momentum,
                 weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.channels_l = channels_l
        self.channels_ab = channels_ab

        self.encoder_l = Unet(backbone_l, len(self.channels_l))
        self.encoder_ab = Unet(backbone_ab, len(self.channels_ab))
        self.final_conv = nn.Conv2d(32,
                                    self.n_classes,
                                    kernel_size=(1, 1))

    def forward(self, x):
        l, ab = x[:, self.channels_l, ...], x[:, self.channels_ab, ...]
        feature_l = self.encoder_l(l)
        feature_ab = self.encoder_ab(ab)
        feature = torch.cat((feature_l, feature_ab), 1)
        out = self.final_conv(feature)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass


def _test_model(args):
    # load pre-trained CMC model as encoders
    encoder = CMCModel.load_from_checkpoint(
                        checkpoint_path=args.model_path)
    backbone_l = encoder.l_to_ab
    backbone_ab = encoder.ab_to_l

    # import torchvision.models as models
    # backbone_l = models.resnet18(pretrained=False)
    # backbone_ab = models.resnet18(pretrained=False)

    model = DoubleUnetModel(backbone_l,
                            backbone_ab,
                            args.channels_l,
                            args.channels_ab,
                            n_classes=10,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    from torchsummary import summary
    summary(model, (11, 256, 256), device="cpu")


if __name__ == "__main__":
    # parse the args
    args = parse_option(False)

    _test_model(args)
