import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.cluster import KMeans
from sklearn import metrics
from myfuncs import *
from osgeo import gdal

img1 = cv.imread('E:/Downloads/HEB1.tif', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/Downloads/HEB3.tif', cv.IMREAD_GRAYSCALE)
#imgg = gdal.Open('E:/Downloads/GF2_PMS1_E126.2_N45.6_20171201_L1A0002813059/GF2_PMS1_E126.2_N45.6_20171201_L1A0002813059-PAN1.tiff')

img3, img4 = reg(img1, img2)
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

a = Img_PCA(diff(img3, img4))

kernel = np.ones((7, 7), np.uint8)
if a.sum() > a.size/2:
    a = np.float32(127 * a)
    a = cv.morphologyEx(a, cv.MORPH_CLOSE, kernel)
else:
    a = np.float32(127 * a)
    a = cv.morphologyEx(a, cv.MORPH_OPEN, kernel)

plt.figure()
plt.subplot(131)
plt.imshow(img1, 'gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(img2, 'gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(a, 'gray')
plt.axis('off')

plt.show()
