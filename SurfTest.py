import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.cluster import KMeans
from sklearn import metrics
from myfuncs import *

img1 = cv.imread('E:/Downloads/forest_1992.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/Downloads/forest_1986.png', cv.IMREAD_GRAYSCALE)

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

delta = abs(img3-img4)
Z = delta.reshape(-1, 1)
y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(Z)
score= metrics.calinski_harabaz_score(Z, y_pred)


a = Img_PCA(CD_diff(img3,img4))
ret,a = cv.threshold(a,1,255,cv.THRESH_BINARY)
'''

a = Img_PCA(diff(img3, img4))

# kernel = np.ones((3, 3), np.uint8)
# if a.sum() > a.size/2:
#     a = np.float32(127 * a)
#     a = cv.morphologyEx(a, cv.MORPH_CLOSE, kernel)
# else:
#     a = np.float32(127 * a)
#     a = cv.morphologyEx(a, cv.MORPH_OPEN, kernel)

plt.figure()
plt.subplot(221)
plt.imshow(img1, 'gray')
plt.subplot(223)
plt.imshow(img2, 'gray')
plt.subplot(222)
plt.imshow(img3, 'gray')
plt.subplot(224)
plt.imshow(img4, 'gray')
plt.figure()
plt.imshow(a, 'gray')

plt.show()
