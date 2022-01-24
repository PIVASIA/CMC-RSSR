import numpy as np
from osgeo import gdal
import csv
import os
import sys
from datetime import datetime, timedelta
import glob
import argparse
from shutil import copyfile

options_list = [
    '-outsize 25% 25%'
]
options_string = " ".join(options_list)


parser 	= argparse.ArgumentParser(description='')
parser.add_argument('--datetime', default="16070800", type=str, help='')
args 	= parser.parse_args()

def generate_amv_datetime(datetime_str):
    img_dir = "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/img_ppm"
    img_dir2 = "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/img"
    amv_dir = "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/amv_ppm"
    amv_dir2 = "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/amv"
    yyyy 			= 2000 + int(datetime_str[0:2])
    mm 				= (int)(datetime_str[2:4])
    dd 				= (int)(datetime_str[4:6])
    hh 				= (int)(datetime_str[6:8])
    current_time 	= datetime(yyyy, mm, dd, hh, 0, 0)
    for prev in range(0, 24*90+1, 1):
        anchor_time 	= current_time 	+ timedelta(hours=prev)
        anchor_name 	= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld".format(anchor_time.year,
																	anchor_time.month,
																	anchor_time.day, 
																	anchor_time.hour, 
																	anchor_time.minute)
        print("processing %s" %anchor_name)
        filepath1 = os.path.join(img_dir,"%s.ppm" % anchor_name)
        filepath2 = os.path.join(img_dir2,"%s.ppm" % anchor_name)
        if (os.path.isfile(filepath1)):
            gdal.Translate(filepath2,
                    filepath1,
                    options=options_string)
        filepath3 = os.path.join(amv_dir,"%s.ppm" % anchor_name)
        filepath4 = os.path.join(amv_dir2,"%s.ppm" % anchor_name)
        if (os.path.isfile(filepath3)):
            copyfile(filepath3, filepath4)


if __name__ == '__main__':
	if args.datetime is not None:
		generate_amv_datetime(args.datetime)