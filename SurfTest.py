import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('E:/Downloads/StereoMP_1m_BW_8bit/po_97258_pan_0000000.tif')
img2 = cv.imread('E:/Downloads/StereoMP_1m_BW_8bit/po_97258_pan_0010000.tif')

#surf = cv.xfeatures2d.SURF_create(400)
surf = cv.xfeatures2d.SIFT_create()

kp1,des1 = surf.detectAndCompute(img1,None)
kp2,des2 = surf.detectAndCompute(img2,None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])


#plt.imshow(cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2), 'gray'),plt.show()

src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ])
#print(src_pts)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,1.0)

img3 = cv.warpPerspective(img1,M,img2.shape[:2])
plt.figure()
plt.imshow(img1, 'gray')
plt.figure()
plt.imshow(img3, 'gray')
plt.figure()
plt.imshow(img2, 'gray'),plt.show()