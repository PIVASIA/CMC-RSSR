import os

import torch
import torch.utils.data as datautils
from torchvision import transforms

from multispectral import load_multispectral
from multispectral_transform import (
                MultispectralRandomHorizontalFlip,
                MultispectralRandomResizedCrop,
                StandardScaler)

DATASET_MEAN = [395.2578, 548.2824, 659.7652, 541.0414, 722.1775, 1016.3951, 1166.1165, 1232.2306, 1313.5893, 1347.3470, 900.3346]
DATASET_STD = [290.2383, 378.0133, 395.4832, 384.9488, 505.8480, 787.9814, 909.9235, 982.9118, 1055.6485, 1146.3106, 797.1838]


class MultispectralImageDataset(datautils.Dataset):
    def __init__(self, folder_path, images_to_use, transform=None):
        super(MultispectralImageDataset, self).__init__()

        with open(images_to_use, 'r', encoding="utf-8") as f:
            names = f.readlines()
        self.filepaths = [os.path.join(folder_path, name.strip()) for name in names]

        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = load_multispectral(self.filepaths[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample, 0, idx


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
            StandardScaler(DATASET_MEAN, DATASET_STD),
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


if __name__ == '__main__':
    import sys
    dataset = MultispectralImageDataset(sys.argv[1], sys.argv[2])
    transform = MultispectralRandomResizedCrop((256, 256))
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        img, _, _ = dataset.__getitem__(i)
        transform(img)