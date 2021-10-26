import os

import torch.utils.data as datautils
from torchvision import transforms

from PIL import Image

from multispectral import load_multispectral
from transform import (TransformParameters, random_transform,
                       apply_transform, adjust_transform_for_image)


class MultispectralImageDataset(datautils.Dataset):
    def __init__(self, 
                 images_to_use,
                 img_folder_path,
                 label_folder_path=None,
                 augment=False,
                 augment_params=None,
                 torch_transform=None):
        super(MultispectralImageDataset, self).__init__()

        with open(images_to_use, 'r', encoding="utf-8") as f:
            names = f.readlines()        
        self.filenames = \
            [name.strip() for name in names]
        
        self.img_folder_path = img_folder_path
        self.label_folder_path = label_folder_path

        # Spatial transform
        self.augment = augment
        self.augment_params = augment_params or TransformParameters()

        # Tensor transform
        self.torch_transform = torch_transform or transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read image
        img_path = os.path.join(self.img_folder_path, self.filenames[idx])

        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image's not existed: {0}".format(img_path))
        img = load_multispectral(img_path)

        # read label
        label = None
        if self.label_folder_path:
            basename = self.filenames[idx].split(".")[0]
            label_path = os.path.join(self.label_folder_path, "%s.png" % basename)
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


if __name__ == '__main__':
    import sys
    dataset = MultispectralImageDataset(sys.argv[1], sys.argv[2], augment_image=True)

    import torch
    data_loader = torch.utils.data.DataLoader(
                                        dataset,
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True,
                                        sampler=None)
    print(len(data_loader))
    batch = next(iter(data_loader))
    print(batch[0].shape)
    print(batch[1])
    print(batch[2])