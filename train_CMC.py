from __future__ import print_function

import os
import sys
import time
import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import get_train_loader

from models.alexnet import alexnet, multispectral_alexnet
from models.resnet import ResNetV2, multispectral_ResNet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from util import AverageMeter, adjust_learning_rate, parse_option

warnings.filterwarnings("ignore")


class CMCModel(pl.LightningModule):
    def __init__(self,
                 n_data,
                 args):
        super().__init__()
        self.save_hyperparameters()

        self.n_data = n_data
        self.args = args
        self._set_model()

    def _set_model(self):
        # set the model
        if self.args.model == 'alexnet':
            if self.args.multispectral:
                self.model = multispectral_alexnet(self.args.feat_dim)
            else:
                self.model = alexnet(self.args.feat_dim)
        elif self.args.model.startswith('resnet'):
            if self.args.multispectral:
                self.model = multispectral_ResNet(channels_l=self.args.channels_l,
                                                  channels_ab=self.args.channels_ab,
                                                  name=self.args.model)
            else:
                self.model = ResNetV2(self.args.model)
        else:
            raise ValueError(
                'model not supported yet {}'.format(self.args.model)
            )

        # setup criterion
        self.contrast = NCEAverage(self.args.feat_dim,
                                   self.n_data,
                                   self.args.nce_k,
                                   self.args.nce_t,
                                   self.args.nce_m)
        self.criterion_l = NCECriterion(self.n_data)
        self.criterion_ab = NCECriterion(self.n_data)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.args.learning_rate,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, _, index = train_batch

        # bsz = inputs.size(0)
        inputs = inputs.float()

        # forward
        feat_l, feat_ab = self.model(inputs)
        out_l, out_ab = self.contrast(feat_l, feat_ab, index)

        l_loss = self.criterion_l(out_l)
        ab_loss = self.criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        loss = l_loss + ab_loss
        self.log(
            'performance',
            {
                "loss": loss,
                "l_loss": l_loss,
                "l_prob": l_prob,
                "ab_loss": ab_loss,
                "ab_prob": ab_prob
            },
            prog_bar=True,
            logger=True
        )

        return {
            "loss": loss,
            "l_loss": l_loss,
            "l_prob": l_prob,
            "ab_loss": ab_loss,
            "ab_prob": ab_prob
        }

    def validation_step(self, val_batch, batch_idx):
        pass


def main():
    # parse the args
    args = parse_option(True)

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    if os.path.isfile(args.resume):
        model = CMCModel.load_from_checkpoint(args.resume)
    else:
        model = CMCModel(n_data, args)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.model_folder,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=args.gpu)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
