import numpy as np

import argparse
import os
from glob import glob
from osgeo import gdal
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Making Remote Sensing Tiles for self-supervised learning")
    # basic
    parser.add_argument('-i', '--input-path', type=str, required=True, help='input dir with tiles image')
    parser.add_argument('--valid-width', type=int, default=256)
    parser.add_argument('--valid-height', type=int, default=256)
    parser.add_argument('-t', '--threshold', type=float, default=0.3)

    args = parser.parse_args()

    return args


def main(args):
    valids = []
    for filepath in tqdm(glob(os.path.join(args.input_path, "*.TIF"))):
        basename = os.path.basename(filepath)

        # Open the file
        raster = gdal.Open(filepath)
        
        # validate size
        if raster.RasterYSize != args.valid_height or \
           raster.RasterXSize != args.valid_width:
           continue
        
        # validate no-data value
        sample = raster.GetRasterBand(1).ReadAsArray()
        nonzero = np.count_nonzero(sample)
        if (nonzero / (args.valid_height * args.valid_width)) < args.threshold:
            continue

        valids.append(basename)
    
    with open(os.path.join(args.input_path, "filenames.txt"), 'w') as fou:
        for filename in valids:
            fou.write(filename + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
