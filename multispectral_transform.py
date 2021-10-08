import numpy as np
import math
import cv2
import random


def multispectral_resize(img, shape):
    out = np.zeros((shape[0], shape[1], img.shape[2]))
    for c in range(0, img.shape[2], 3):
        if c + 3 >= img.shape[2]:
            break
        
        out[..., c:c+3] = cv2.resize(img[..., c:c+3], shape)
    
    for l in range(c, img.shape[2]):
        out[..., l] = cv2.resize(img[..., l], shape)

    return out


def crop(img, top, left, height, width):
    return img[top:top+height, left:left+width, :]


def resized_crop(img, top, left, height, width, size):
    img = crop(img, top, left, height, width)
    img = multispectral_resize(img, size)
    return img


class MultispectralResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return multispectral_resize(img, self.size)


class MultispectralRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self. p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[:, ::-1, :]
        return img


class MultispectralRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.shape[0], img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)

                return i, j, h, w

        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        out = resized_crop(img, i, j, h, w, self.size)

        return out


class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img):
        img_shape = img.shape

        img = img.reshape((img_shape[0] * img_shape[1], img_shape[2]))
        img = (img - self.mean) / self.std
        img = img.reshape((img_shape[0], img_shape[1], img_shape[2]))

        return img


def unit_test_resize():
    for c in range(1, 20):
        img = np.ones((6, 6, c))
        r_img = multispectral_resize(img, (3, 3))
        print(r_img.shape)

if __name__ == '__main__':
    unit_test_resize()
    