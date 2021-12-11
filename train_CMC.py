from __future__ import print_function

import os
import warnings

import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/hunglv@piv.asia/cmc-species")
import mlflow.pytorch

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import MultispectralImageDataModule
from models.cmc import CMCModel
from util import parse_option

warnings.filterwarnings("ignore")


def main():
    mlflow.pytorch.autolog()
    
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
    model = CMCModel(dm.n_data, args)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.model_folder,
        filename="{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    early_stop_callback = EarlyStopping(monitor="train_loss", 
                                        min_delta=0.005, 
                                        patience=3, 
                                        verbose=False, 
                                        mode="min")

    callbacks=[checkpoint_callback, early_stop_callback]

    # use float (e.g 1.0) to set val frequency in epoch
    # if val_check_interval is integer, val frequency is in batch step
    training_params = {
        "callbacks": callbacks,
        "gpus": args.train_loss,
        "val_check_interval": 1.0,
        "max_epochs": args.epochs
    }

    if args.resume is not None:
        training_params["resume_from_checkpoint"] = args.resume
    
    trainer = pl.Trainer(**training_params)
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
