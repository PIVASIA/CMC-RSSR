import os

import torch
import torch.utils.data as datautils

from multispectral import load_multispectral


class MultispectralImageDataset(datautils.Dataset):
    def __init__(self, folder_path, images_to_use, transform=None):
        super(MultispectralImageDataset, self).__init__()

        with open(images_to_use, 'r', encoding="utf-8") as f:
            names = f.readlines()
        self.filepaths = \
            [os.path.join(folder_path, name.strip()) for name in names]

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


if __name__ == '__main__':
    import sys
    dataset = MultispectralImageDataset(sys.argv[1], sys.argv[2])