import os

import torch
import torch.utils.data as datautils
from torchvision import transforms

from PIL import Image

from multispectral import load_multispectral
from transform import TransformParameters, apply_transform, adjust_transform_for_image


class MultispectralImageDataset(datautils.Dataset):
    def __init__(self, 
                 images_to_use,
                 img_folder_path,
                 label_folder_path=None,
                 augment_generator=None,
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
        self.augment_generator = augment_generator
        self.augment_params = augment_params or TransformParameters()

        # Tensor transform
        self.torch_transform = torch_transform or transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read image
        img_path = os.path.join(self.img_folder_path, "%s.tif" % self.filenames[idx])

        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image's not existed: {0}".format(img_path))
        img = load_multispectral(img_path)

        # read label
        label = None
        if self.label_folder_path:
            label_path = os.path.join(self.label_folder_path, "%s.png" % self.filenames[idx])
            if not os.path.isfile(label_path):
                raise FileNotFoundError("Label's not existed: {0}".format(label_path))
            label = Image.open(label_path).convert('L')
            label = np.array(label)
        
        # spatial transform for data augmentation
        if self.augment_generator:
            transform = adjust_transform_for_image(next(self.augment_generator), 
                                                   image, 
                                                   self.augment_params.relative_translation)
            image = apply_transform(transform, image, self.augment_params)
            if label:
                label = apply_transform(transform, label, self.augment_params)

        # transform to Tensor
        img = self.torch_transform(img)
        if label:
            label = torch.from_numpy(label).float()

        return img, label, idx


if __name__ == '__main__':
    import sys
    dataset = MultispectralImageDataset(sys.argv[1], sys.argv[2])