import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.cluster import KMeans
from sklearn import metrics
from myfuncs import *
from osgeo import gdal

img1 = gdal.Open('E:/Downloads/HEBMS1.tif')
img2 = gdal.Open('E:/Downloads/HEBMS3.tif')

img1 = np.uint8(img1.GetRasterBand(3).ReadAsArray()/4)
img2 = np.uint8(img2.GetRasterBand(3).ReadAsArray()/4)
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
a = CDetect(img3, img4, 3)


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
