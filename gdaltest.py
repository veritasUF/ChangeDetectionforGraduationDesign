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
N = img1.RasterCount
rasters1 = np.array([np.uint8(img1.GetRasterBand(i).ReadAsArray() / 4) for i in range(1, N + 1)])
rasters2 = np.array([np.uint8(img2.GetRasterBand(i).ReadAsArray() / 4) for i in range(1, N + 1)])
rasters1, rasters2 = reg4(rasters1, rasters2, N)
imgg1 = cv.merge((rasters1[2], rasters1[1], rasters1[0]))
imgg2 = np.uint8(cv.merge((rasters2[2], rasters2[1], rasters2[0])))
sp = rasters1[0].shape
rasters1 = np.array([rasters1[i].reshape(-1) for i in range(N)], np.float64)
rasters2 = np.array([rasters2[i].reshape(-1) for i in range(N)], np.float64)

weight = np.array([1 for i in rasters1[0]])
delta = rasters1

for __iter__ in range(2):
    covxy = np.cov(rasters1, rasters2, aweights=weight)
    av1 = np.average(rasters1, 1, weight)
    av2 = np.average(rasters2, 1, weight)
    X = np.array([rasters1[i] - av1[i] for i in range(N)])
    Y = np.array([rasters2[i] - av2[i] for i in range(N)])
    cov11 = covxy[:N, :N]
    cov22 = covxy[N:, N:]
    cov12 = covxy[:N, N:]
    cov21 = cov12.T

    invcov22 = np.linalg.inv(cov22)
    d, v1 = SL.eigh(cov12 @ invcov22 @ cov21, cov11)
    if d[N-1] > 0.999:
        break
    v2 = invcov22 @ cov21 @ v1
    aux1 = v2.T @ cov22 @ v2
    aux2 = 1 / np.sqrt(np.diag(aux1))
    aux3 = np.array([aux2 for i in range(N)])
    v2 = v2 * aux3

    delta = v1.T @ X - v2.T @ Y

    sigma2 = 2 * (1 - d ** 0.5)
    Tj = np.array([np.sum(delta[:, i] ** 2 / sigma2) for i in range(delta.shape[1])])

    weight = 1 - SP.gammainc(N/2, Tj/2)

    # res2 = np.array([1 if i > 0 else 0 for i in weight])
    # res2 = res2.reshape(sp)
    #
    # plt.subplot(4, 5, __iter__+1), plt.imshow(res2, 'gray'), plt.show()
    print(__iter__)
    print(weight.sum())

# with open('weight.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(weight)
# with open('delta.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     for row in delta:
#         csv_writer.writerow(row)
# res2 = np.array([255 if i > 0 else 0 for i in weight], np.uint8)
# res2 = res2.reshape(sp)
# kernel = np.ones((3, 3), np.uint8)
# res2 = cv.morphologyEx(res2, cv.MORPH_CLOSE, kernel)
# plt.imshow(res2, 'gray'), plt.show()

Z = (np.sum(delta*delta, axis=0)**0.5).reshape(-1, 1)
res = KMeans(n_clusters=4, init='k-means++', n_jobs=-1).fit_predict(Z)
res2 = res.reshape(sp)

plt.figure()
plt.subplot(131)
plt.imshow(imgg1, 'gray')
plt.subplot(132)
plt.imshow(imgg2, 'gray')
plt.subplot(133)
plt.imshow(res2, 'gray')
plt.show()

cv.waitKey(0)
