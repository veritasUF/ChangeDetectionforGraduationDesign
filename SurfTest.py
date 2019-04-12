import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from myfuncs import *
from osgeo import gdal

img1 = cv.imread('E:/Downloads/HEB1.tif', cv.IMREAD_UNCHANGED)
img2 = cv.imread('E:/Downloads/HEB3.tif', cv.IMREAD_UNCHANGED)

# img3, img4 = reg(img1, img2)
# img1 = gdal.Open('HEBMS1.tif')
# img2 = gdal.Open('HEBMS3.tif')
#
# img1 = np.array(img1.GetRasterBand(2).ReadAsArray()/4, np.uint8)
# img2 = np.array(img2.GetRasterBand(2).ReadAsArray()/4, np.uint8)

img1, img2 = reg(img1, img2)
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
a = CDetect(img1, img2, 0)

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
