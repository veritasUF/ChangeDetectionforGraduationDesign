import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from myfuncs import *
from osgeo import gdal

img1 = gdal.Open('HEBMS1.tif')
img2 = gdal.Open('HEBMS4.tif')
N = img1.RasterCount-1
T = 5
rasters1 = np.array([np.uint8(img1.GetRasterBand(i).ReadAsArray() / 4) for i in range(2, N + 2)])
rasters2 = np.array([np.uint8(img2.GetRasterBand(i).ReadAsArray() / 4) for i in range(2, N + 2)])
rasters1, rasters2 = reg4(rasters1, rasters2, N)
imgg1 = cv.merge((rasters1[2], rasters1[1], rasters1[0]))
imgg2 = np.uint8(cv.merge((rasters2[2], rasters2[1], rasters2[0])))

img1h = cv.merge((rasters1[0], rasters1[1], rasters1[2]))
img2h = np.uint8(cv.merge((rasters2[0], rasters2[1], rasters2[2])))

sp = img1h.shape
img1h = img1h[sp[0] % (1 << T):, sp[1] % (1 << T):, :]
img2h = img2h[sp[0] % (1 << T):, sp[1] % (1 << T):, :]
change = []
for i in range(T):
    img1d = cv.pyrDown(img1h)
    img2d = cv.pyrDown(img2h)
    img1l = cv.subtract(img1h, cv.pyrUp(img1d))
    img2l = cv.subtract(img2h, cv.pyrUp(img2d))
    # plt.subplot(1, 3, 1), plt.imshow(cv.equalizeHist(cv.cvtColor(img1l, cv.COLOR_BGR2GRAY)), 'gray')
    # plt.subplot(1, 3, 2), plt.imshow(cv.equalizeHist(cv.cvtColor(img2l, cv.COLOR_BGR2GRAY)), 'gray')
    img1l = np.array([img1h[:, :, j] for j in range(N)])
    img2l = np.array([img2h[:, :, j] for j in range(N)])

    change.append(CDetect4(img1l, img2l, 10, N, 0))

    img1h = img1d.copy()
    img2h = img2d.copy()

    # plt.subplot(1, 3, 3), plt.imshow(change[-1], 'gray')
    # plt.show()

img1h = np.array([img1h[:, :, j] for j in range(N)])
img2h = np.array([img2h[:, :, j] for j in range(N)])
change.append(CDetect4(img1h, img2h, 10, N, 0))
ls_ = change[T]
for i in range(T-1, -1, -1):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, change[i])

plt.subplot(121), plt.imshow(ls_, 'gray')
ret, th1 = cv.threshold(ls_, 3, 255, cv.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
th1 = cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel)
plt.subplot(122), plt.imshow(th1, 'gray')
plt.figure()
for i in range(T+1):
    plt.subplot(2, 3, i+1), plt.imshow(change[i], 'gray')
plt.show()
