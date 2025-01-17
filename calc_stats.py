# coding=utf-8
'''
created by Hung Luu, October,2021
This code is used to create remote sensing images tiles for self-supervised learning.
'''

import os
import argparse
from tqdm import tqdm
from glob import glob

import numpy as np
import torch

from multispectral import load_multispectral


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Learning dataset stats")
    # basic
    parser.add_argument('-i', '--input-path', type=str, required=True)
    args = parser.parse_args()

    return args


def fit_scaler(input_path):
    filepaths = glob(os.path.join(input_path, "*.TIF"))

    tmpImg = load_multispectral(filepaths[0])
    height, width, channels = tmpImg.shape

    # placeholders
    psum = torch.from_numpy(np.zeros(channels)).to(device)
    psum_sq = torch.from_numpy(np.zeros(channels)).to(device)

    # loop through images
    for filepath in tqdm(filepaths):
        tmpImg = load_multispectral(filepath)

        # verify images of same size
        if height != tmpImg.shape[0] or width != tmpImg.shape[1]:
            continue

        tmpImg = tmpImg.reshape(
            (tmpImg.shape[0] * tmpImg.shape[1], tmpImg.shape[2]))
        # convert to tensor
        tmpImg = torch.from_numpy(tmpImg).to(device)

        psum += tmpImg.sum(axis=0)
        psum_sq += (tmpImg ** 2).sum(axis=0)

    # pixel count
    count = len(filepaths) * height * width

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean.cpu(), total_std.cpu()


if __name__ == "__main__":
    args = parse_args()
    total_mean, total_std = fit_scaler(args.input_path)
    print("Mean", total_mean)
    print("Std", total_std)