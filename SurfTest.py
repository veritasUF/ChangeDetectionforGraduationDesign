import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from myfuncs import *
from osgeo import gdal

img1 = cv.imread('HEB4.tif', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('HEB2.tif', cv.IMREAD_GRAYSCALE)

# img3, img4 = reg(img1, img2)
# img1 = gdal.Open('HEBMS1.tif')
# img2 = gdal.Open('HEBMS3.tif')
#
# img1 = np.array(img1.GetRasterBand(2).ReadAsArray()/4, np.uint8)
# img2 = np.array(img2.GetRasterBand(2).ReadAsArray()/4, np.uint8)

img1, img2 = reg(img1, img2)
# img1 = img1[1787:2893, 1800:2318]
# img2 = img2[1787:2893, 1800:2318]
# img1, img2 = reg(img1, img2)

ans = CDetectpca(img1, img2, 7, 15, 0)
a = CDetect(img1, img2, 4, 3)

'''
plt.subplot(221)
plt.imshow(img1, 'gray')
plt.subplot(222)
plt.imshow(img2, 'gray')
plt.subplot(223)
plt.imshow(img3, 'gray')
plt.subplot(224)
plt.imshow(img4, 'gray'), plt.show()
'''
# a = CDetect(img1, img2, 20, 3)

plt.figure()
plt.subplot(141)
plt.imshow(img1, 'gray')
plt.subplot(142)
plt.imshow(img2, 'gray')
plt.axis('off')
plt.subplot(143)
plt.imshow(a, 'gray')
plt.axis('off')
plt.subplot(144)
plt.imshow(ans, 'gray')
plt.axis('off')


plt.show()
