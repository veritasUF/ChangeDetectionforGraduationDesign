import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
        if m.distance < 0.6 * n.distance:  # 阈值是怎么确定的
            good.append([m])

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good])

    src_val = [img1.item(int(kp1[m[0].queryIdx].pt[1]), int(kp1[m[0].queryIdx].pt[0])) for m in good]
    dst_val = [img2.item(int(kp2[m[0].trainIdx].pt[1]), int(kp2[m[0].trainIdx].pt[0])) for m in good]

    img9 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img9), plt.show()

    # 辐射配准
    z1 = np.polyfit(dst_val, src_val, 1)
    p1 = np.poly1d(z1)
    # print(p1)
    M, mask = cv.findHomography(src_pts, dst_pts)

    img3 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    img4 = np.round(p1(img2))

    return img3, img4


# PCA与作差算法，作差算法中集成了Kmeans聚类

def diff(img1, img2):
    delta = abs(img2 - img1)

    Z = delta.reshape(-1, 1)
    res = KMeans(n_clusters=2, random_state=16).fit_predict(Z)
    res2 = res.reshape(delta.shape)

    return res2


def Img_PCA(delta):
    U, S, V = np.linalg.svd(delta)
    SS = np.zeros(U.shape)
    for i in range(S.shape[0]):
        SS[i][i] = S[i]

    def Pick_k(s):
        sval = np.sum(s)
        for i in range(s.shape[0]):
            if np.sum(s[:i]) >= 0.6 * sval:
                break
        return i + 1

    k = Pick_k(S)
    Uk = U[:, 0:k]
    Sk = SS[0:k, 0:k]
    Vk = V[0:k, :]
    im = np.dot(np.dot(Uk, Sk), Vk)
    return np.int8(np.round(im))
