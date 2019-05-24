import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.cluster import KMeans
from sklearn import metrics
from myfuncs import *
from osgeo import gdal
from time import time

t0 = time()
img1 = cv.imread('HEB4.tif', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('HEB2.tif', cv.IMREAD_GRAYSCALE)
img1, img2 = reg(img1, img2)

# a = CDetect(img1, img2, 4, 3)

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
    cv.line(mask, (x1, y1), (x2, y2), 1, 2)

# 膨胀掩模约28像素，增加容错
mask = cv.dilate(mask, kernel, iterations=4)

# 掩住
th3 = np.multiply(th3, mask)

img1 = np.multiply(img1, mask)
img2 = np.multiply(img1, mask)
plt.imshow(img1, 'gray'), plt.show()
print(time()-t0)
a = CDetectpca(img1, img2, h=3, S=5, n=3)
print(time()-t0)

th3 = np.multiply(th3, a)

# 腐蚀：将有宽度直线检测化为普通直线检测，腐蚀3像素，即限制最小宽度2.4米
th3 = cv.erode(th3, (3, 3))

# 再直线检测
mask2 = np.zeros(th2.shape, np.uint8)
lines = cv.HoughLinesP(th3, 1, np.pi/720, 100, minLineLength=200, maxLineGap=5)
length = np.array([((line[0][0]-line[0][2])**2+(line[0][1]-line[0][3])**2)**0.5 for line in lines])
mx = length.max()
for line in lines:
    x1, y1, x2, y2 = line[0]
    if ((x2-x1)**2+(y2-y1)**2)**0.5 == mx:
        cv.line(mask2, (x1, y1), (x2, y2), 255, 2)
    # else:
    #     cv.line(mask2, (x1, y1), (x2, y2), 128, 2)

print(mx)
# plt.subplot(231)
# plt.imshow(img1, 'gray'), plt.axis('off')
# plt.subplot(232)
# plt.imshow(mask, 'gray'), plt.axis('off')
# plt.subplot(233)
# plt.imshow(th2, 'gray'), plt.axis('off')
# plt.subplot(234)
# plt.imshow(a, 'gray'), plt.axis('off')
# plt.subplot(235)
plt.imshow(mask2, 'gray'), plt.axis('off')
# plt.subplot(236)
# plt.imshow(th3, 'gray'), plt.axis('off')
plt.show()