from __future__ import print_function

import os
import sys
import time
import warnings

import torch
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import (ImageDataset, MultispectralImageDataset,
                     MultispectralRandomHorizontalFlip,
                     MultispectralRandomResizedCrop, RGB2Lab, ScalerPCA)
from models.alexnet import alexnet, multispectral_alexnet
from models.resnet import ResNetV2, multispectral_ResNetV2
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from util import AverageMeter, adjust_learning_rate, parse_option

warnings.filterwarnings("ignore")

DATASET_MEAN = [395.2578, 548.2824, 659.7652, 541.0414, 722.1775, 1016.3951, 1166.1165, 1232.2306, 1313.5893, 1347.3470, 900.3346]
DATASET_STD = [290.2383, 378.0133, 395.4832, 384.9488, 505.8480, 787.9814, 909.9235, 982.9118, 1055.6485, 1146.3106, 797.1838]


def get_train_loader(args):
    """get the train loader"""
    data_folder = args.data_folder
    image_list = args.image_list

    if not args.multispectral:
        normalize = transforms.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                         std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])

        transformations = [transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
                           transforms.RandomHorizontalFlip()]

        if args.resize_image_aug:
            transformations.insert(0, transforms.Resize((256, 256)))

        transformations += [RGB2Lab(), transforms.ToTensor(), normalize]
        train_transform = transforms.Compose(transformations)
        train_dataset = ImageDataset(data_folder, image_list, transform=train_transform)
        train_sampler = None
    else:
        transformations = [
            MultispectralRandomResizedCrop(224, scale=(args.crop_low, 1.)),
            MultispectralRandomHorizontalFlip()
        ]

        transformations += [
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
            transforms.ToTensor()
        ]
        train_transform = transforms.Compose(transformations)
        train_dataset = MultispectralImageDataset(data_folder,
                                                  image_list,
                                                  transform=train_transform)
        train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


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
        elif self.args.args.model.startswith('resnet'):
            if self.args.multispectral:
                self.model = multispectral_ResNetV2(self.args.model)
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

    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
