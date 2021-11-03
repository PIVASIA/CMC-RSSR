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
from train_CMCSemSeg import CMCSemSegModel

warnings.filterwarnings("ignore")


class FeatureExtractorTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        _, feat_l, _, feat_ab = self(batch)



def main(args):
    # load pre-trained CMC model as encoders
    model = CMCSemSegModel.load_from_checkpoint(
                        checkpoint_path=args.model_path)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.ab_to_l.upsample_blocks[4].register_forward_hook(get_activation('ab2l'))
    model.l_to_ab.upsample_blocks[4].register_forward_hook(get_activation('l2ab'))
    
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
    dm.setup(stage="predict")

    inputs, _, _ = next(iter(dm.test_dataloader()))
    inputs = inputs.float()
    inputs_l, inputs_ab = inputs[:, args.channels_l, ...], inputs[:, args.channels_ab, ...]
    output = model(inputs_l, inputs_ab)
        
    print(activation.keys())


def _test_model(args):
    # load pre-trained CMC model as encoders
    model = CMCSemSegModel.load_from_checkpoint(
                        checkpoint_path=args.model_path)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.ab_to_l.upsample_blocks[4].register_forward_hook(get_activation('ab2l'))
    model.l_to_ab.upsample_blocks[4].register_forward_hook(get_activation('l2ab'))
    
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
    dm.setup(stage="test")

    inputs, _, _ = next(iter(dm.test_dataloader()))
    inputs = inputs.float()
    inputs_l, inputs_ab = inputs[:, args.channels_l, ...], inputs[:, args.channels_ab, ...]
    output = model(inputs_l, inputs_ab)
        
    print(activation.keys())
    

if __name__ == '__main__':
    # parse the args
    args = parse_option(False)

    main(args)
    # _test_model(args)