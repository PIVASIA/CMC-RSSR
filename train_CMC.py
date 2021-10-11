from __future__ import print_function

import os
import warnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from models.alexnet import alexnet, multispectral_alexnet
from models.resnet import ResNetV2, multispectral_ResNet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from util import parse_option
from multispectral_transform import (
                MultispectralRandomHorizontalFlip,
                MultispectralRandomResizedCrop,
                StandardScaler)
from dataset import MultispectralImageDataset
from constants import DATASET_MEAN, DATASET_STD

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
                self.model = multispectral_ResNet(
                                channels_l=self.args.channels_l,
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
        x = x.float()
        feat_l, feat_ab = self.model(x)
        return feat_l, feat_ab

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.args.learning_rate,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, _, index = train_batch

        # forward
        feat_l, feat_ab = self(inputs)

        # calculating loss
        out_l, out_ab = self.contrast(feat_l, feat_ab, index)

        l_loss = self.criterion_l(out_l)
        ab_loss = self.criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        loss = l_loss + ab_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "l_loss": l_loss,
            "l_prob": l_prob,
            "ab_loss": ab_loss,
            "ab_prob": ab_prob
        }

    def validation_step(self, val_batch, batch_idx):
        pass


class CMCDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        transformations = [
            MultispectralRandomResizedCrop(224, scale=(args.crop_low, 1.)),
            MultispectralRandomHorizontalFlip()
        ]

        transformations += [
            StandardScaler(DATASET_MEAN[self.args.dataset_name],
                           DATASET_STD[self.args.dataset_name]),
            transforms.ToTensor()
        ]
        self.transform = transforms.Compose(transformations)

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage=None):
        # called on every GPU
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = \
                MultispectralImageDataset(self.args.data_folder,
                                          self.args.image_list,
                                          transform=self.transform)
            self.n_data = len(self.train_dataset)

    def train_dataloader(self):
        train_sampler = None
        return torch.utils.data.DataLoader(
                        self.train_dataset,
                        batch_size=self.args.batch_size,
                        shuffle=(train_sampler is None),
                        num_workers=self.args.num_workers,
                        pin_memory=True,
                        sampler=train_sampler
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


def main():
    # parse the args
    args = parse_option(True)

    # set the datamodule
    dm = CMCDataModule(args)

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
