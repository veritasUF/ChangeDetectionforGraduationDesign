import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from myfuncs import *
from osgeo import gdal

img1 = cv.imread('E:/downloads/forest_1986.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/downloads/forest_1992.png', cv.IMREAD_GRAYSCALE)

# img3, img4 = reg(img1, img2)
# img1 = gdal.Open('HEBMS1.tif')
# img2 = gdal.Open('HEBMS3.tif')
#
# img1 = np.array(img1.GetRasterBand(2).ReadAsArray()/4, np.uint8)
# img2 = np.array(img2.GetRasterBand(2).ReadAsArray()/4, np.uint8)

img1, img2 = reg(img1, img2)
h = 5
sp = img1.shape
img1 = img1[sp[0]%h:, sp[1]%h:]
img2 = img2[sp[0]%h:, sp[1]%h:]
dif = np.array(abs(-1*img1 + img2), np.uint8)
sp = dif.shape


def getblock(xd, x, y, h):
    x = x - h // 2
    y = y - h // 2
    vec = []
    for i in range(h):
        for j in range(h):
            vec.append(xd[(x + i)%sp[0], (y + j)%sp[1]])

    return np.array(vec)

blocks = []
for i in range(sp[0]):
    for j in range(sp[1]):
        blocks.append(getblock(dif, i, j, h))
blocks = np.array(blocks).reshape(sp[0], sp[1], h*h)

H,W = np.array(sp)//h
samples = np.array([np.array([blocks[h//2+h*i] for i in range(H)])[:,h//2+h*j,:] for j in range(W)]).reshape(-1,h*h)
pca = PCA(n_components=3, whiten=True)
pca.fit(samples)
res = np.zeros((sp[0],sp[1],3))
for i in range(sp[0]):
    for j in range(sp[1]):
        res[i,j] = pca.transform(blocks[i,j].reshape(1,-1))

res = res.reshape(-1,3)
ans = KMeans(n_clusters=2, init='k-means++', n_jobs=-1).fit_predict(res)
ans = ans.reshape(sp)
plt.imshow(ans, 'gray'), plt.show()

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
a = CDetect(img1, img2, 20, 3)

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
