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
img1, img2 = reg(img1, img2)

a = ChangeDetection(img1, img2, 7)

kernel = np.ones((7, 7), np.uint8)  # 阈值到底是怎么确定的
opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
opening2 = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel)
th2 = cv.adaptiveThreshold(opening, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 151, 5)
th3 = cv.adaptiveThreshold(opening2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 151, 5)

# 确保二值化中包含跑道的部分为亮部，去噪点

th2 = 255 - th2 if th2.sum() > th2.size / 2 else th2
th3 = 255 - th3 if th3.sum() > th3.size / 2 else th3
th2 = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel)
th3 = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)

# 直线检测，以检测出的直线跑道作为掩模
mask = np.zeros(th2.shape, np.uint8)
lines = cv.HoughLinesP(th2, 1, np.pi/720, 2000, minLineLength=2000, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2-y1) < 20 or abs((x1-x2)/(y1-y2)) > 50:  # 去除配准边区
        continue
    cv.line(mask, (x1, y1), (x2, y2), 255, 2)

# 膨胀掩模约28像素，增加容错
mask = cv.dilate(mask, kernel, iterations=4)

# 掩住！再直线检测
th3 = np.multiply(th3, mask)
th3 = np.multiply(th3, a)
mask2 = np.zeros(th2.shape, np.uint8)
lines = cv.HoughLinesP(th3, 1, np.pi/720, 100, minLineLength=200, maxLineGap=20)
length = np.array([((line[0][0]-line[0][2])**2+(line[0][1]-line[0][3])**2)**0.5 for line in lines])
mx = length.max()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(mask2, (x1, y1), (x2, y2), 255, 2)

plt.subplot(231)
plt.imshow(img1, 'gray'), plt.axis('off')
plt.subplot(232)
plt.imshow(mask, 'gray'), plt.axis('off')
plt.subplot(233)
plt.imshow(th2, 'gray'), plt.axis('off')
plt.subplot(234)
plt.imshow(img2, 'gray'), plt.axis('off')
plt.subplot(235)
plt.imshow(mask2, 'gray'), plt.axis('off')
plt.subplot(236)
plt.imshow(th3, 'gray'), plt.axis('off'), plt.show()