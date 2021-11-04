import numpy as np

import argparse
import os
from glob import glob
from osgeo import gdal
from tqdm import tqdm

from multispectral import load_multispectral

def parse_args():
    parser = argparse.ArgumentParser(description="Making Remote Sensing Tiles for supervised learning")
    # basic
    parser.add_argument('-i', '--input-image', type=str, required=True, help='input image')
    parser.add_argument('-l', '--input-label', type=str, required=True, help='input label image')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='output')
    parser.add_argument('--target-width', type=int, default=256)
    parser.add_argument('--target-height', type=int, default=256)
    parser.add_argument('-t', '--threshold', type=float, default=0.3)

    args = parser.parse_args()

    return args


def main(args):
    tiles = []

    image = load_multispectral(args.input_image, nodata=0)

    label = load_multispectral(args.input_label, nodata=0)
    # extract points with label
    ys, xs = np.nonzero(label)
    
    valids = []
    for y, x in zip(ys, xs):
        is_all_zero = np.all((image[y, x, :] == 0))
        if not is_all_zero:
            valids.append([y, x, label[y, x]])


    with open(args.output_path, mode='w') as fou:
        writer = csv.writer(fou, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in valids:
            writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()
    main(args)
