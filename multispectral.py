import numpy as np
from osgeo import gdal


def load_multispectral(filepath):
    # Open the file
    raster = gdal.Open()
    # populate data as numpy array
    sample = \
        np.zeros((raster.RasterYSize, raster.RasterXSize, raster.RasterCount))
    for i in range(raster.RasterCount):
        sample[..., i] = raster.GetRasterBand(i).astype(np.float32)

    return sample
