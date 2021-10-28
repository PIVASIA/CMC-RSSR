from typing import Dict, List, Optional

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from PIL import Image

from multispectral import load_multispectral
from dataset import MultispectralImageDataset
from transform import (TransformParameters, random_transform,
                       apply_transform, adjust_transform_for_image,
                       StandardScaler)
from constants import DATASET_MEAN, DATASET_STD


class MultispectralImageDataset(Dataset):
    def __init__(self, 
                 images_to_use: List[str],
                 image_folder: str,
                 label_folder: Optional[str] = None,
                 label_mapping: Optional[dict] = None,
                 augment: bool = False,
                 augment_params: TransformParameters = None,
                 torch_transform=None):
        super(MultispectralImageDataset, self).__init__()

        self.images_to_use = images_to_use
        
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.label_mapping = label_mapping

        # Spatial transform
        self.augment = augment
        self.augment_params = augment_params or TransformParameters()

        # Tensor transform
        self.torch_transform = torch_transform or transforms.ToTensor()

    def __len__(self):
        return len(self.images_to_use)

    def __getitem__(self, idx):
        # read image
        img_path = os.path.join(self.image_folder, self.images_to_use[idx])

        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image's not existed: {0}".format(img_path))
        img = load_multispectral(img_path)

        # read label
        label = None
        if self.label_folder:
            basename = self.filenames[idx].split(".")[0]
            label_path = os.path.join(self.label_folder, "%s.png" % basename)
            if not os.path.isfile(label_path):
                raise FileNotFoundError("Label's not existed: {0}".format(label_path))
            label = Image.open(label_path).convert('L')
            label = np.array(label)
        
        # data augmentation
        if self.augment:
            transform = adjust_transform_for_image(random_transform(), 
                                                   img.shape, 
                                                   self.augment_params.relative_translation)
            img = apply_transform(transform, img, self.augment_params)
            if label:
                label = apply_transform(transform, label, self.augment_params)

        # transform to Tensor
        img = self.torch_transform(img)

        if label:
            label = torch.from_numpy(label).float()
        else:
            label = 0

        return img, label, idx


class MultispectralImageDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 image_folder: str,
                 train_image_list: str,
                 test_image_list: Optional[str] = None,
                 label_folder: Optional[str] = None,
                 label_mapping: Optional[dict] = None,
                 train_batch_size: int = 32,
                 test_batch_size: int = 16,
                 augment: bool = False
                 ):
        super().__init__()

        self.label_mapping = label_mapping
        self.train_image_list = train_image_list
        self.test_image_list = test_image_list
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size # use as test/val batch size
        self.augment = augment

        torch_transformations = [
            StandardScaler(DATASET_MEAN[dataset_name],
                           DATASET_STD[dataset_name]),
            transforms.ToTensor()
        ]
        self.torch_transform = transforms.Compose(torch_transformations)

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, 
              stage: str = None, 
              train_val_split: bool = False):
        # called on every GPU
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            with open(self.train_image_list, 'r', encoding="utf-8") as f:
                names = f.readlines()        
            images_to_use = [name.strip() for name in names]

            # train/val split
            if train_val_split:
                train_set_size = int(len(images_to_use) * 0.8)
                valid_set_size = len(images_to_use) - train_set_size
                train_image_list, val_image_list = random_split(images_to_use, [train_set_size, valid_set_size])
            else:
                train_image_list = images_to_use

            self.train_dataset = \
                MultispectralImageDataset(train_image_list,
                                          image_folder=self.image_folder,
                                          label_folder=self.label_folder,
                                          augment=self.augment,
                                          augment_params=None, # use default augment param
                                          torch_transform=self.torch_transform)
            self.n_data = len(self.train_dataset)

            if train_val_split:
                self.val_dataset = \
                    MultispectralImageDataset(val_image_list,
                                              image_folder=self.image_folder,
                                              label_folder=self.label_folder,
                                              augment=False,
                                              torch_transform=self.torch_transform)
        
        if stage == "test" or stage is None:
            with open(self.test_image_list, 'r', encoding="utf-8") as f:
                names = f.readlines()        
            images_to_use = [name.strip() for name in names]

            self.test_dataset = \
                    MultispectralImageDataset(images_to_use,
                                              image_folder=self.image_folder,
                                              label_folder=self.label_folder,
                                              augment=False,
                                              torch_transform=self.torch_transform)

    def train_dataloader(self):
        train_sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )

        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
        return None


if __name__ == '__main__':
    from util import parse_option
    args = parse_option(True)

    dm = MultispectralImageDataModule(args.dataset_name,
                                      args.image_folder,
                                      args.train_image_list,
                                      args.test_image_list,
                                      args.label_folder)
    dm.setup(stage="fit")