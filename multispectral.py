import numpy as np
from osgeo import gdal


def load_multispectral(filepath, nodata=0):
    # Open the file
    raster = gdal.Open(filepath)
    # populate data as numpy array
    sample = \
        np.zeros((raster.RasterYSize, raster.RasterXSize, raster.RasterCount), dtype=np.uint16)
    for i in range(raster.RasterCount):
        sample[..., i] = raster.GetRasterBand(i + 1).SetNoDataValue(nodata).ReadAsArray()

    return sample
