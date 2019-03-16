import cv2 as cv
import numpy as np
import scipy.linalg as SL
import scipy.special as SP
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.cluster import KMeans
from sklearn import metrics
from myfuncs import *
from osgeo import gdal

img1 = gdal.Open('E:/Downloads/HEBMS1.tif')
img2 = gdal.Open('E:/Downloads/HEBMS3.tif')

rasters1 = np.array([np.uint8(img1.GetRasterBand(i).ReadAsArray()/4) for i in range(1, img1.RasterCount+1)])
rasters2 = np.array([np.uint8(img2.GetRasterBand(i).ReadAsArray()/4) for i in range(1, img2.RasterCount+1)])
rasters1, rasters2 = reg4(rasters1, rasters2)
sp = rasters1[0].shape
rasters1 = np.array([rasters1[i].reshape(-1) for i in range(img1.RasterCount)])
rasters2 = np.array([rasters2[i].reshape(-1) for i in range(img2.RasterCount)])
# img3 = cv.merge((rasters1[2], rasters1[1], rasters1[0]))
# img4 = cv.merge((rasters1[2], rasters1[1], rasters1[0]))
# plt.subplot(121)
# plt.imshow(img3, 'gray')
# plt.subplot(122)
# plt.imshow(img4, 'gray'), plt.show()
av1 = np.average(rasters1, axis=1)
av2 = np.average(rasters2, axis=1)
rasters1 = np.array([rasters1[i] - av1[i] for i in range(img1.RasterCount)])
rasters2 = np.array([rasters2[i] - av2[i] for i in range(img2.RasterCount)])

weight = np.array([1 for i in rasters1[0]])

for __iter__ in range(1):
    covxy = np.cov(rasters1, rasters2, aweights=weight)
    cov11 = covxy[:4, :4]
    cov22 = covxy[4:, 4:]
    cov12 = covxy[:4, 4:]
    cov21 = cov12.T

    invcov22 = np.linalg.inv(cov22)
    d1, v = SL.eigh(cov12@invcov22@cov21, cov11)
    v1 = v
    # aux1 = v.T@cov11@v
    # aux2 = 1/np.sqrt(np.diag(aux1))
    # aux3 = np.array([aux2 for i in range(4)])
    # v1 = v*aux3
    v2 = invcov22@cov21@v1
    aux1 = v2.T@cov22@v2
    aux2 = 1/np.sqrt(np.diag(aux1))
    aux3 = np.array([aux2 for i in range(4)])
    v2 = v2*aux3

    # invcov11 = np.linalg.inv(cov11)
    # d2, v = SL.eigh(cov21 @ invcov11 @ cov12, cov22)
    # v2 = v

    delta = v1.T@rasters1-v2.T@rasters2

    sigma2 = 2*(1-d1**0.5)
    Tj = np.array([np.sum(delta[:, i]**2 / sigma2) for i in range(delta.shape[1])])

    weight = 1 - SP.gammainc(2, Tj/2)

    # Z = weight.reshape(-1, 1)
    # res = KMeans(n_clusters=3, random_state=16).fit_predict(Z)
    # res2 = res.reshape(sp)
    # res2 = np.array([[1 if i == 1 else 0 for i in j] for j in res2])
    # plt.figure(), plt.imshow(res2, 'gray'), plt.show()
    print(__iter__)

res2 = np.array([1 if i > 0.0001 else 0 for i in weight])
res2 = res2.reshape(sp)
# Z = weight.reshape(-1, 1)
# res = KMeans(n_clusters=16, random_state=16).fit_predict(Z)
# res2 = res.reshape(sp)
#res2 = np.array([[1 if i == 1 else 0 for i in j] for j in res2])
plt.imshow(res2, 'gray'), plt.show()

cv.waitKey(0)
