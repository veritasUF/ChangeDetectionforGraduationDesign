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
img2 = gdal.Open('E:/Downloads/HEBMS2.tif')

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
av1=np.average(rasters1)
av2=np.average(rasters1)
rasters1 = rasters1 - av1
rasters2 = rasters2 - av2

weight = [1 for i in rasters1[0]]

for __iter__ in range(10):
    covxy = np.cov(rasters1, rasters2, aweights=weight)
    cov11 = covxy[:4, :4]
    cov22 = covxy[4:, 4:]
    cov12 = covxy[:4, 4:]
    cov21 = cov12.T

    invcov22 = np.linalg.inv(cov22)
    d, v1 = SL.eig(cov12@invcov22@cov21, cov11)
    d = d.real
    indices = np.argsort(d)
    d = d[indices]
    v1 = v1[:, indices]

    # aux1 = v.T@cov11@v
    # aux2 = 1/np.sqrt(np.diag(aux1))
    # aux3 = np.array([aux2 for i in range(4)])
    # v1 = v*aux3
    #
    # aux1 = v.T@cov22@v
    # aux2 = 1/np.sqrt(np.diag(aux1))
    # aux3 = np.array([aux2 for i in range(4)])
    # v2 = v*aux3

    invcov11 = np.linalg.inv(cov11)
    d, v2 = SL.eig(cov21 @ invcov11 @ cov12, cov22)
    d = d.real
    indices = np.argsort(d)
    d = d[indices]
    v2 = v2[:, indices]

    delta = v1.T@rasters1-v2.T@rasters2
    sigma2 = 2*(1-np.sqrt(d))
    Tj = np.array([np.sum(delta[:, i]**2 / sigma2) for i in range(delta.shape[1])])

    weight = SP.gammainc(2, Tj/2)

    # Z = weight.reshape(-1, 1)
    # res = KMeans(n_clusters=2, random_state=16).fit_predict(Z)
    # res2 = res.reshape(sp)
    # plt.figure(), plt.imshow(res2, 'gray'), plt.show()
    print(__iter__)

Z = weight.reshape(-1, 1)
res = KMeans(n_clusters=2, random_state=16).fit_predict(Z)
res2 = res.reshape(sp)

plt.imshow(res2, 'gray'), plt.show()

cv.waitKey(0)
