import numpy as np
from osgeo import gdal
from PIL import Image

def load_multispectral(filepath1, filepath2):
    # Open the amv file
    raster = gdal.Open(filepath2)
    # populate data as numpy array
    sample = \
        np.zeros((raster.RasterYSize, raster.RasterXSize, 2*raster.RasterCount))
    for i in range(raster.RasterCount):
        sample[..., i] = raster.GetRasterBand(i + 1).ReadAsArray()
    
    # Open the img file    
    
    
    raster1 = gdal.Open(filepath1)
    for i in range(raster1.RasterCount):
        sample[..., i+3] = raster1.GetRasterBand(i + 1).ReadAsArray()
    return sample
