import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

img1h = cv.merge((rasters1[0], rasters1[1], rasters1[2]))
img2h = np.uint8(cv.merge((rasters2[0], rasters2[1], rasters2[2])))

sp = img1h.shape
img1h = img1h[sp[0] % 32:, sp[1] % 32:, :]
img2h = img2h[sp[0] % 32:, sp[1] % 32:, :]

for i in range(5):
    img1d = cv.pyrDown(img1h)
    img2d = cv.pyrDown(img2h)
    img1l = cv.subtract(img1h, cv.pyrUp(img1d))
    img2l = cv.subtract(img2h, cv.pyrUp(img2d))
    img1h = img1d.copy()
    img2h = img2d.copy()
    plt.subplot(3, 5, i * 3 + 1), plt.imshow(img1l, 'gray')
    plt.subplot(3, 5, i * 3 + 2), plt.imshow(img2l, 'gray')
    img1l = np.array([img1l[:, :, j] for j in range(N)])
    img2l = np.array([img2l[:, :, j] for j in range(N)])
    change = CDetect4(img1l, img2l, 5, N, 0)

    plt.subplot(3, 5, i * 3 + 3), plt.imshow(change, 'gray')

plt.show()
