import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import myfuncs

img1 = cv.imread('E:/Downloads/burn_1992.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/Downloads/burn_1986.png',cv.IMREAD_GRAYSCALE)

img3,img4 = mySurf.reg(img1, img2)
'''
#surf = cv.xfeatures2d.SURF_create(2)
#surf = cv.xfeatures2d.SIFT_create()

kp1,des1 = surf.detectAndCompute(img1,None)
kp2,des2 = surf.detectAndCompute(img2,None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.6*n.distance: #阈值是怎么确定的
        good.append([m])

#print(kp1[good[0][0].queryIdx].pt[0])
#print(img1.item(int(kp1[good[0][0].queryIdx].pt[0]), int(kp1[good[0][0].queryIdx].pt[1])))
plt.imshow(cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2), 'gray'),plt.show()

src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ])
#print(src_pts)
src_val = [ img1.item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good ]
dst_val = [ img2.item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good ]
#print(src_val)
#print(dst_val)
plt.scatter(src_val,dst_val)
z1 = np.polyfit(dst_val, src_val, 1)
p1 = np.poly1d(z1)
print(p1)
plt.plot(p1(dst_val),dst_val,'r'),plt.show()

M, mask = cv.findHomography(dst_pts, src_pts)

img3 = cv.warpPerspective(img1,M,img2.shape)

img2 = np.round(p1(img2)) 
'''
plt.figure()
plt.imshow(img3,'gray')
plt.figure()
plt.imshow(img4,'gray'),plt.show()
'''
plt.figure()
plt.subplot(131)
plt.imshow(img1, 'gray')
plt.subplot(132)
plt.imshow(img2, 'gray')
plt.subplot(133)
plt.imshow(img2-img3, 'gray'),plt.show()
'''