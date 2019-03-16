import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from time import time

# 配准算法
# 依赖于opencv库
# 暂为面向过程形式，未来有机会（并没有）可能改写成OOP形式

surf = cv.xfeatures2d.SIFT_create()

# surf = cv.xfeatures2d.SURF_create(20)


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


def reg4(img1, img2):
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

    src_val = [img1[0].item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good]
    dst_val = [img2[0].item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good]

    # img9 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # plt.imshow(img9), plt.axis('off'), plt.show()

    # 辐射配准
    # z1 = np.polyfit(dst_val, src_val, 1)
    # p1 = np.poly1d(z1)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    img3 = np.array([cv.warpPerspective(img1[i], M, (img2[i].shape[1], img2[i].shape[0])) for i in range(4)])
    img4 = np.array([np.uint8(np.round(img2[i])) for i in range(4)])

    return img3, img4



# PCA与作差算法，作差算法中集成了Kmeans聚类

def diff(img1, img2):
    delta = abs(img2 - img1)

    Z = delta.reshape(-1, 1)
    res = KMeans(n_clusters=2, random_state=16).fit_predict(Z)
    res2 = res.reshape(delta.shape)

    return res2


def Img_PCA(img1):
    t0 = time()

    pca = PCA(n_components=0.8, svd_solver='full', whiten=True)
    sig = pca.fit_transform(img1)
    img = pca.inverse_transform(sig)
    print(time()-t0)
    return img


# 变化检测方法
# n为要丢弃的检测结果的最大直径，默认为0，即保留所有变化的点
# 未集成配准
def CDetect(img1, img2, n=0):
    change = np.uint8(diff(Img_PCA(img1), Img_PCA(img2)))
    if change.sum() < change.size / 2:
        change = 1 - change

    if n == 0:
        return change

    kernel = np.ones((n, n), np.uint8)
    change = cv.morphologyEx(change, cv.MORPH_CLOSE, kernel)
    return change


def covw(A, B, w):

    def meanw(X, W):
        return (X*W).sum() / W.sum()

    C = np.vstack((A, B))
    N = C[0].__len__()
    mn = [meanw(i, w) for i in C]
    for i in range(C.__len__()):
        C[i] -= mn[i]
    swn = (N-1)*sum(w/N)
    ans = [[(sum(w*i*j))/swn for i in C] for j in C]
    return np.array(ans)














