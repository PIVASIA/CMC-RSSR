from __future__ import print_function

import os
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from piq import MultiScaleSSIMLoss

from dataset import MultispectralImageDataModule
import models.resnet as resnet
from models.unet import Unet
from util import parse_option

warnings.filterwarnings("ignore")


class CMCSemSegModel(pl.LightningModule):
    def __init__(self,
                 args):
        super().__init__()
        self.save_hyperparameters()

        self.model = args.model
        self.channels_l = args.channels_l
        self.channels_ab = args.channels_ab

        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay

        self._build_model()
        self._set_criterion()

    def _build_model(self):
        # set the model
        if self.model.startswith('resnet'):
            backbone = getattr(resnet, self.model, lambda: None)

            self.l_to_ab = Unet(backbone(in_channel=len(self.channels_l)), 
                                len(self.channels_l), 
                                n_classes=len(self.channels_ab))

            self.ab_to_l = Unet(backbone(in_channel=len(self.channels_ab)), 
                                len(self.channels_ab), 
                                n_classes=len(self.channels_l))
        else:
            raise ValueError(
                'model not supported yet {}'.format(self.model)
            )

    def _set_criterion(self):
        # setup criterion
        self.criterion = nn.MSELoss()
        # self.criterion_msssim = MultiScaleSSIMLoss()

    def _forward_l(self, x):
        x = x.float()
        output, feat = self.l_to_ab(x)
        return output, feat

    def _forward_ab(self, x):
        x = x.float()
        output, feat = self.ab_to_l(x)
        return output, feat

    def forward(self, inputs_l, inputs_ab):
        output_l, feat_l = self._forward_l(inputs_l)
        output_ab, feat_ab = self._forward_ab(inputs_ab)

        return [output_l, feat_l, output_ab, feat_ab]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, _, index = train_batch
        inputs = inputs.float()
        inputs_l, inputs_ab = inputs[:, self.channels_l, ...], inputs[:, self.channels_ab, ...]

        # forward
        out_l, _, out_ab, _ = self(inputs_l, inputs_ab)

        # calculating loss
        loss_l = self.criterion(out_l, inputs_ab)
        # msssim_loss_l = self.criterion_msssim(out_l, inputs_ab)

        loss_ab = self.criterion(out_ab, inputs_l)
        # msssim_loss_ab = self.criterion_msssim(out_ab, inputs_l)

        loss = loss_l + loss_ab
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return {
            "loss": loss
        }

    def validation_step(self, val_batch, batch_idx):
        inputs, _, index = val_batch
        inputs_l, inputs_ab = inputs[:, self.channels_l, ...], inputs[:, self.channels_ab, ...]

        # forward
        out_l, _, out_ab, _ = self(inputs_l, inputs_ab)

        # calculating loss
        loss_l = self.criterion(out_l, inputs_ab)
        # msssim_loss_l = self.criterion_msssim(out_l, inputs_ab)

        loss_ab = self.criterion(out_ab, inputs_l)
        # msssim_loss_ab = self.criterion_msssim(out_ab, inputs_l)

        loss = loss_l + loss_ab
        self.log('val_loss', loss, on_step=False, on_epoch=True)


def main(args):
    # set the datamodule
    dm = MultispectralImageDataModule(args.dataset_name,
                                      args.image_folder,
                                      args.train_image_list,
                                      args.test_image_list,
                                      args.label_folder,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size,
                                      augment=args.augment,
                                      num_workers=args.num_workers)
    dm.setup(stage="fit", validation_size=0.1)

    # set the model
    if os.path.isfile(args.resume):
        model = CMCSemSegModel.load_from_checkpoint(args.resume)
    else:
        model = CMCSemSegModel(args)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.model_folder,
        filename="{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         gpus=args.gpu,
                         max_epochs=args.epochs,
                         val_check_interval=5)
    trainer.fit(model, dm)


def _test_model(args):
    # set the datamodule
    dm = MultispectralImageDataModule(args.dataset_name,
                                      args.image_folder,
                                      args.train_image_list,
                                      args.test_image_list,
                                      args.label_folder,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size,
                                      augment=args.augment,
                                      num_workers=args.num_workers)
    dm.setup(stage="fit", validation_size=0.1)

    # set the model
    # if os.path.isfile(args.resume):
    #     model = CMCSemSegModel.load_from_checkpoint(args.resume)
    # else:
    #     model = CMCSemSegModel(args)
    
    inputs, _, _ = next(iter(dm.train_dataloader()))
    inputs_l, inputs_ab = inputs[:, args.channels_l, ...], inputs[:, args.channels_ab, ...]
    # feat_l, feat_ab = model(inputs_l, inputs_ab)

    # print(inputs_l.shape, inputs_ab.shape)
    # print(feat_l.shape, feat_ab.shape)

    # loss_fnc = MultiScaleSSIMLoss()

    # feat_ab = torch.relu(feat_ab)

    # print(feat_ab.min(), inputs_l.min())
    # loss_fnc(inputs_l, feat_ab)

    print(inputs_l.min(), inputs_l.max())
    print(inputs_ab.min(), inputs_ab.max())


if __name__ == '__main__':
    # parse the args
    args = parse_option(True)

    main(args)
    # _test_model(args)
