#配准算法
#依赖于opencv库
#暂为面向过程形式，未来有机会（并没有）可能改写成OOP形式

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

surf = cv.xfeatures2d.SIFT_create()
#surf = cv.xfeatures2d.SURF_create(2)

def reg(img1, img2):
    kp1,des1 = surf.detectAndCompute(img1,None)
    kp2,des2 = surf.detectAndCompute(img2,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance: #阈值是怎么确定的
            good.append([m])

    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ])

    src_val = [ img1.item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good ]
    dst_val = [ img2.item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good ]

    z1 = np.polyfit(dst_val, src_val, 1)
    p1 = np.poly1d(z1)
    #print(p1)
    M, mask = cv.findHomography(dst_pts, src_pts)

    img3 = cv.warpPerspective(img1,M,img2.shape)
    img4 = np.round(p1(img2))
    return img3,img4
