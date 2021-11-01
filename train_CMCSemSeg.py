from __future__ import print_function

import os
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
        self.l1 = nn.L1Loss()

    def _forward_l(self, x):
        feat = self.l_to_ab(x)
        return feat

    def _forward_ab(self, x):
        feat = self.ab_to_l(x)
        return feat

    def forward(self, x):
        x = x.float()

        l, ab = x[:, self.channels_l, ...], x[:, self.channels_ab, ...]
        feat_l = self._forward_l(l)
        feat_ab = self._forward_ab(ab)
        return feat_l, feat_ab

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, _, index = train_batch

        # forward
        out_l, out_ab = self(inputs)

        # calculating loss
        l1_loss = self.l1(out_l, out_ab)
        loss = l1_loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "l1_loss": l1_loss
        }

    def validation_step(self, val_batch, batch_idx):
        inputs, _, index = val_batch

        # forward
        out_l, out_ab = self(inputs)

        # calculating loss
        l1_loss = self.l1(out_l, out_ab)
        loss = l1_loss

        self.log('val_loss', loss, on_step=False, on_epoch=True)


def main():
    # parse the args
    args = parse_option(True)

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
    dm.setup(stage="fit")

    # set the model
    if os.path.isfile(args.resume):
        model = CMCModel.load_from_checkpoint(args.resume)
    else:
        model = CMCModel(dm.n_data, args)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.model_folder,
        filename="{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         gpus=args.gpu,
                         max_epochs=args.epochs)
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
