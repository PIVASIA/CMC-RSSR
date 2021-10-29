import numpy as np

import argparse
import os
from glob import glob
from tqdm import tqdm
from PIL import Image

from constants import LABEL_MAPPING


def parse_args():
    parser = argparse.ArgumentParser(description="Mapping raw label images to zeros-based classes")
    # basic
    parser.add_argument('-i', '--input-dir', type=str, required=True, help='input dir with raw label images')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='output dir with mapped label images')

    args = parser.parse_args()

    return args


def main(args):
    for filepath in tqdm(glob(os.path.join(args.input_dir, "*.png"))):
        basename = os.path.basename(filepath)

        # Open the file
        label = Image.open(filepath).convert('L')
        label = np.array(label)

        for k, v in LABEL_MAPPING.items():
            label[label == k] = v
        
        label.save(os.path.join(args.output_dir, basename))


if __name__ == "__main__":
    args = parse_args()
    main(args)
