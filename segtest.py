import cv2 as cv
import numpy as np
import csv
import scipy.linalg as SL
import scipy.special as SP
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from myfuncs import *
from osgeo import gdal
import random as rng
rng.seed(12345)

img1 = gdal.Open('wv3.tif')
N = img1.RasterCount
rasters1 = np.array([np.uint8(img1.GetRasterBand(i).ReadAsArray() / 4) for i in range(1, N+1)])

src = cv.merge((rasters1[1], rasters1[2], rasters1[4]))
src = src[600:, :800]
src = cv.resize(src, None, fx=4, fy=4, interpolation = cv.INTER_CUBIC)
# src = cv.imread('HEB1.tif',cv.IMREAD_GRAYSCALE)
# src = src[2000:2700, 800:1500]
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# src = clahe.apply(src)
# src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

cv.imshow('Source Image', src)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated

imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian
# imgResult = src

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
# imgLaplacian = np.clip(imgLaplacian, 0, 255)
# imgLaplacian = np.uint8(imgLaplacian)
#cv.imshow('Laplace Filtered Image', imgLaplacian)
cv.imshow('New Sharped Image', imgResult)
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
# _, bw = cv.threshold(bw, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
bw = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 95, 2)

# kernel2 = np.ones((3,3), dtype=np.uint8)
# bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel2)

cv.imshow('Binary Image', bw)
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)
_, dist = cv.threshold(dist, 0.15, 1.0, cv.THRESH_BINARY)
# Dilate a bit the dist image
kernel1 = np.ones((5,5), dtype=np.uint8)
dist = cv.erode(dist,kernel1)
# dist = cv.medianBlur(dist, 5)
# kernel2 = np.ones((5,5), dtype=np.uint8)
# dist = cv.morphologyEx(dist, cv.MORPH_OPEN, kernel2)
cv.imshow('Peaks', dist)
dist_8u = dist.astype('uint8')
# Find total markers
_, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
    cv.drawContours(markers, contours, i, (i+1), -1)
# Draw the background marker
cv.circle(markers, (5,5), 3, (255,255,255), -1)
cv.imshow('Markers', markers*10000)
cv.watershed(imgResult, markers)
#mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
cv.imshow('Markers_v2', mark)
# Generate random colors
count = [markers[markers == i].size for i in range(5000000)]
colors = []
for contour in contours:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours):
            dst[i,j,:] = colors[index-1] if count[index] < 7000 else [0,0,0]
# Visualize the final image
cv.imshow('Final Result', dst)
cv.waitKey()
