import cv2 as cv
import numpy as np
import csv
import scipy.linalg as SL
import scipy.special as SP
import matplotlib.pyplot as plt
from myfuncs import *
from osgeo import gdal
reader = csv.reader(open("weight.csv"))
weight = np.array([np.float64(i) for i in reader])[0]
reader = csv.reader(open("delta.csv"))
delta = np.array([[np.float64(i) for i in row] for row in reader])
sp = (1264, 976)
# res2 = np.array([255 if i == 0 else 0 for i in weight], np.uint8)
# res2 = res2.reshape(sp)
# kernel = np.ones((5, 5), np.uint8)
# res2 = cv.morphologyEx(res2, cv.MORPH_CLOSE, kernel)
# plt.imshow(res2, 'gray'), plt.show()

Z = (np.sum(delta*delta, axis=0)**0.5).reshape(-1, 1)
res = KMeans(n_clusters=8, random_state=16).fit_predict(Z)
res2 = np.uint8(res.reshape(sp))
# for i in range(res2.shape[0]):
#     for j in range(res2.shape[1]):
#         res2[i][j] = 1 if res2[i][j] == 1 else 0

# res2 = 1 - res2 if res2.sum() < res2.size / 2 else res2
# kernel = np.ones((7, 7), np.uint8)
# res2 = cv.morphologyEx(res2, cv.MORPH_CLOSE, kernel)

plt.figure()
plt.imshow(res2, 'gray'), plt.show()

cv.waitKey(0)
