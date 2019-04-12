import cv2 as cv
import numpy as np
import csv
import scipy.linalg as SL
import scipy.special as SP
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from myfuncs import *
from osgeo import gdal

img1 = gdal.Open('HEBMS1.tif')
img2 = gdal.Open('HEBMS4.tif')
N = img1.RasterCount-1
rasters1 = np.array([np.uint8(img1.GetRasterBand(i).ReadAsArray() / 4) for i in range(2, N + 2)])
rasters2 = np.array([np.uint8(img2.GetRasterBand(i).ReadAsArray() / 4) for i in range(2, N + 2)])
rasters1, rasters2 = reg4(rasters1, rasters2, N)
imgg1 = cv.merge((rasters1[2], rasters1[1], rasters1[0]))
imgg2 = np.uint8(cv.merge((rasters2[2], rasters2[1], rasters2[0])))
res2 = CDetect4(rasters1, rasters2, 10, N, 3)

plt.figure()
plt.subplot(131)
plt.imshow(imgg1, 'gray')
plt.subplot(132)
plt.imshow(imgg2, 'gray')
plt.subplot(133)
plt.imshow(res2, 'gray')
plt.show()

cv.waitKey(0)
