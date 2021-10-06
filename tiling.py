# coding=utf-8
'''
created by Hung Luu, October,2021
This code is used to create remote sensing images tiles for self-supervised learning.
'''

import os
import shutil
import sys
import argparse
from tqdm import tqdm
from glob import glob
import arcpy


def parse_args():
    parser = argparse.ArgumentParser(description="Making Remote Sensing Tiles for self-supervised learning")
    # basic
    parser.add_argument('-i', '--input-path', type=str, required=True)
    parser.add_argument('-o', '--output-path', type=str, required=True, help='output folder')
    parser.add_argument('-t', '--tile-size', type=int, default=256, help='tile size')
    # optional
    parser.add_argument('--overlap', type=int, default=16,
                        help='tile overlap size', required=False)

    args = parser.parse_args()

    return args


def tiling(filepath, output_path, tile_size=256, overlap=16):
    arcpy.AddMessage("Tiling for {0}".format(filepath))
    # temp workspace
    tempWorkSpace = os.path.join(output_path, 'tempworkspace')
    if os.path.exists(tempWorkSpace):
        shutil.rmtree(tempWorkSpace)
    if not os.path.exists(tempWorkSpace):
        os.mkdir(tempWorkSpace)

    arcpy.env.workspace = tempWorkSpace

    basename = os.path.basename(filepath).split(".")[0]
    prefix = basename + "_"
    # Equally split a large TIFF image by size of images
    arcpy.SplitRaster_management(filepath, output_path, prefix, 
                                 split_method="SIZE_OF_TILE",
                                 format="TIFF", 
                                 resampling_type="NEAREST", 
                                 tile_size="{0} {0}".format(tile_size), 
                                 overlap="{0}".format(overlap), 
                                 units="PIXELS")



def batch_tiling(input_path, output_path, tile_size=256, overlap=16):
    filepaths = glob(os.path.join(input_path, "*.TIF"))
    for filepath in filepaths:
        tiling(filepath, output_path, tile_size=tile_size, overlap=overlap)



if __name__ == "__main__":
    args = parse_args()
    batch_tiling(args.input_path, args.output_path, args.tile_size, args.overlap)