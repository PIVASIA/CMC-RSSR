from __future__ import print_function

import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import MultispectralImageDataModule
from util import parse_option
from train_CMC import CMCModel
from models.unet import Unet
from models.losses import DiceLoss


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

        # setup model
        self.encoder_l = Unet(backbone_l, len(self.channels_l))
        self.encoder_ab = Unet(backbone_ab, len(self.channels_ab))
        self.final_conv = nn.Conv2d(32,
                                    self.n_classes,
                                    kernel_size=(1, 1))
        
        # setup criterion
        # ignore class 0 which equipvalent to no label
        # self.criterion = DiceLoss(self.n_classes, ignore_index=0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x):
        x = x.float()
        
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
        inputs, label, index = train_batch

        # forward
        out = self(inputs)

        # calculating loss
        loss = self.criterion(out, label)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, label, index = val_batch

        # forward
        out = self(inputs)

        # calculating loss
        loss = self.criterion(out, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True)


def _test_model():
     # parse the args
    args = parse_option(False)

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
                            n_classes=args.n_classes,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion = DiceLoss(args.n_classes, ignore_index=0)

    dm = MultispectralImageDataModule(args.dataset_name,
                                      args.image_folder,
                                      args.train_image_list,
                                      args.test_image_list,
                                      args.label_folder,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size,
                                      augment=args.augment)
    dm.setup(stage="fit", train_val_split=True)

    inputs, targets, _ = next(iter(dm.train_dataloader()))
    out = model(inputs)
    loss = criterion(out, targets)

    # from torchsummary import summary
    # summary(model, (11, 256, 256), device="cpu")


def main():
    # parse the args
    args = parse_option(False)

    # setup datamodule
    print("--setup datamodule ...")
    dm = MultispectralImageDataModule(args.dataset_name,
                                      args.image_folder,
                                      args.train_image_list,
                                      args.test_image_list,
                                      args.label_folder,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size,
                                      augment=args.augment)
    dm.setup(stage="fit", train_val_split=True)

    print("--setup model ...")
    # load pre-trained CMC model as encoders
    encoder = CMCModel.load_from_checkpoint(
                        checkpoint_path=args.model_path)
    backbone_l = encoder.l_to_ab
    backbone_ab = encoder.ab_to_l

    # set the model
    if os.path.isfile(args.resume):
        model = DoubleUnetModel.load_from_checkpoint(args.resume)
    else:
        model = DoubleUnetModel(backbone_l,
                                backbone_ab,
                                args.channels_l,
                                args.channels_ab,
                                n_classes=args.n_classes, 
                                learning_rate=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_path,
        filename="{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                        min_delta=0.00, 
                                        patience=5, 
                                        verbose=False, 
                                        mode="min")

    print("--training ...")
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback],
                         gpus=args.gpu,
                         max_epochs=args.epochs)
    lr_finder = trainer.tuner.lr_find(model, dm)

    model.hparams.learning_rate = lr_finder.suggestion()
    trainer.fit(model, dm)


if __name__ == "__main__":
    # _test_model()
    main()
