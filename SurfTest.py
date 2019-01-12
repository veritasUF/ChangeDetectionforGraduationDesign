import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from myfuncs import *

img1 = cv.imread('E:/Downloads/forest_1986.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/Downloads/forest_1992.png',cv.IMREAD_GRAYSCALE)

img3, img4 = reg(img1, img2)

'''
a = Img_PCA(CD_diff(img3,img4))
print(a)
ret,a = cv.threshold(a,127,255,cv.THRESH_BINARY)
print(a)
'''

a = Img_PCA(diff(img3, img4))
b = diff(Img_PCA(img3), Img_PCA(img4))

plt.figure()
plt.subplot(221)
plt.imshow(img3, 'gray')
plt.subplot(222)
plt.imshow(img4, 'gray')
plt.subplot(223)
plt.imshow(a, 'gray')
plt.subplot(224)
plt.imshow(b, 'gray')

plt.show()