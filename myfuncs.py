import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.linalg as SL
import scipy.special as SP
from time import time

# 配准算法
# 依赖于opencv库
# 暂为面向过程形式，未来有机会（并没有）可能改写成OOP形式

surf = cv.xfeatures2d.SIFT_create()

# surf = cv.xfeatures2d.SURF_create(400)


def reg(img1, img2):
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.55 * n.distance:  # 阈值是怎么确定的
            good.append([m])

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good])

    src_val = [img1.item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good]
    dst_val = [img2.item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good]

    # img9 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # plt.imshow(img9), plt.axis('off'), plt.show()

    # 辐射配准
    z1 = np.polyfit(dst_val, src_val, 1)
    p1 = np.poly1d(z1)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    img3 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    img4 = np.uint8(np.round(p1(img2)))

    return img3, img4


def reg4(img1, img2, rastercount):
    p = []
    M, mask = [], []
    for i in range(rastercount):
        kp1, des1 = surf.detectAndCompute(img1[0], None)
        kp2, des2 = surf.detectAndCompute(img2[0], None)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:  # 阈值是怎么确定的
                good.append([m])

        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good])

        src_val = [img1[i].item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good]
        dst_val = [img2[i].item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good]

        # plt.scatter(src_val, dst_val), plt.show()
        # img9 = cv.drawMatchesKnn(img1[i], kp1, img2[i], kp2, good, None, flags=2)
        # plt.imshow(img9), plt.axis('off'), plt.show()

        # 辐射配准
        z1 = np.polyfit(dst_val, src_val, 1)
        p.append(np.poly1d(z1))

        if i == 0:
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    img3 = np.array([cv.warpPerspective(img1[i], M, (img2[i].shape[1], img2[i].shape[0])) for i in range(rastercount)])
    img4 = np.array([p[i](img2[i]) for i in range(rastercount)])

    return img3, img4


# PCA与作差算法，作差算法中集成了Kmeans聚类

def diff(img1, img2):
    delta = abs(img2 - img1)

    Z = delta.reshape(-1, 1)
    res = KMeans(n_clusters=2, init='k-means++', n_jobs=-1).fit_predict(Z)
    res2 = res.reshape(delta.shape)

    return res2


def Img_PCA(img1):
    pca = PCA(n_components=0.9, svd_solver='full')
    sig = pca.fit_transform(img1)
    img = pca.inverse_transform(sig)
    return img


def CDetect(img1, img2, max_iter=1, n=0):
    # 变化检测方法
    # n为要丢弃的检测结果的最大直径，默认为0，即保留所有变化的点
    # 未集成配准
    sp=img1.shape
    rasters1 = np.float64(img1.reshape(-1))
    rasters2 = np.float64(img2.reshape(-1))

    weight = np.ones(sp).reshape(-1)

    for __iter__ in range(max_iter):
        av1 = np.average(rasters1, weights=weight)
        av2 = np.average(rasters2, weights=weight)
        X = rasters1 - av1
        Y = rasters2 - av2
        covxy = np.cov(X, Y, aweights=weight)
        cov11 = covxy[0, 0]
        cov22 = covxy[1, 1]
        cov12 = covxy[1, 0]
        cov21 = cov12

        invcov22 = 1/cov22
        d = cov12 * invcov22 * cov21 / cov11
        v1 = 1 / cov11**0.5
        if d > 0.999:
            break
        v2 = 1 / cov22**0.5

        delta = v1 * X - v2 * Y

        sigma2 = 2 * (1 - d ** 0.5)
        Tj = delta ** 2 / sigma2

        weight = 1 - SP.gammainc(0.5, Tj / 2)

    res2 = np.array([1 if i > 0 else 0 for i in weight], np.uint8)
    res2 = res2.reshape(sp)

    if n != 0:
        kernel = np.ones((n, n), np.uint8)
        res2 = cv.morphologyEx(res2, cv.MORPH_CLOSE, kernel)
    return res2

def CDetect4(rasters1, rasters2, max_iter, N=1, n=0):
    #输入为已配准的图像，N为通道数，n为丢弃阈值，max_iter为最大迭代数
    # if N == 1:
    #     return CDetect(rasters1, rasters2, n)

    sp = rasters1[0].shape
    rasters1 = np.array([rasters1[i].reshape(-1) for i in range(N)], np.float64)
    rasters2 = np.array([rasters2[i].reshape(-1) for i in range(N)], np.float64)

    weight = np.array([1 for i in rasters1[0]])

    for __iter__ in range(max_iter):
        av1 = np.average(rasters1, 1, weight)
        av2 = np.average(rasters2, 1, weight)
        X = np.array([rasters1[i] - av1[i] for i in range(N)])
        Y = np.array([rasters2[i] - av2[i] for i in range(N)])
        covxy = np.cov(X, Y, aweights=weight)
        cov11 = covxy[:N, :N]
        cov22 = covxy[N:, N:]
        cov12 = covxy[:N, N:]
        cov21 = cov12.T

        invcov22 = np.linalg.inv(cov22)
        d, v1 = SL.eigh(cov12 @ invcov22 @ cov21, cov11)
        if d[N - 1] > 0.999:
            break
        v2 = invcov22 @ cov21 @ v1
        aux1 = v2.T @ cov22 @ v2
        aux2 = 1 / np.sqrt(np.diag(aux1))
        aux3 = np.array([aux2 for i in range(N)])
        v2 = v2 * aux3

        delta = v1.T @ X - v2.T @ Y

        sigma2 = 2 * (1 - d ** 0.5)
        Tj = np.array([np.sum(delta[:, i] ** 2 / sigma2) for i in range(delta.shape[1])])

        weight = 1 - SP.gammainc(N / 2, Tj / 2)

    res2 = np.array([1 if i > 0 else 0 for i in weight], np.uint8)
    res2 = res2.reshape(sp)

    if n != 0:
        kernel = np.ones((n, n), np.uint8)
        res2 = cv.morphologyEx(res2, cv.MORPH_CLOSE, kernel)
    return res2















