"""
此模块存放匹配与矫正算法

@author: lfc
"""

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def sift_match(image, temp_img, index_params=None, search_params=None, min_match_count=10):
    """
    该函数用于计算两图像的特征点，并进行匹配得到两图像间得单应性矩阵H
    :param image: 图片
    :param temp_img: 模板图片
    :param index_params: 匹配算法
    :param search_params:
    :param min_match_count: 两图像至少有多少个好点，不足此参数，则返回None
    :return: 单应性矩阵H或None， 其中 image * H = temp_img
    """
    if temp_img is None or image is None:
        return None
    if search_params is None:
        search_params = dict(checks=50)
    if index_params is None:
        index_params = dict(algorithm=0, trees=5)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(temp_img, None)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matches2 = flann.knnMatch(des2, des1, k=2)
    good = []
    good1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:

            good.append(m)
    for m, n in matches2:
        if m.distance < 0.7 * n.distance:
            good1.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        x1, y1 = cvt_pos(0, 0, H1)
        x2, y2 = cvt_pos(0, image.shape[1], H1)
        x3, y3 = cvt_pos(image.shape[0], 0, H1)
        x4, y4 = cvt_pos(image.shape[0], image.shape[1], H1)
        max_x = max(x1, x2, x3, x4)
        min_x = min(x1, x2, x3, x4)
        max_y = max(y1, y2, y3, y4)
        min_y = min(y1, y2, y3, y4)
        # print(x1, x2, x3, x4)
        # print(y1, y2, y3, y4)
        # print(max_x, min_x, max_y, min_y)
        cv2.rectangle(image, (0, min_y), (max_x- min_x, max_y), 2)
        cv2.imshow('rect', image)
        cv2.waitKey(0)

        # print(src_pts[4][0], dst_pts[4], image.shape, temp_img.shape)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        """srcArr = np.float32([[src_pts[0][0][0], src_pts[0][0][1]], [src_pts[1][0][0], src_pts[1][0][1]], [src_pts[2][0][0], src_pts[2][0][1]], [src_pts[3][0][0], src_pts[3][0][1]]])
        dstArr = np.float32([[dst_pts[0][0][0], dst_pts[0][0][1]], [dst_pts[1][0][0], dst_pts[1][0][1]], [dst_pts[2][0][0], dst_pts[2][0][1]], [dst_pts[3][0][0], dst_pts[3][0][1]]])
        MM = cv2.getPerspectiveTransform(dstArr, srcArr)"""

        # x1, y1 = cvt_pos(44, 631, H)
        # print(x1, y1)
        # print(src_pts[4][0], dst_pts[4], image.shape[0], temp_img.shape)

        return H

    else:
        return None


def harris_match(image, temp_img, index_params=None, search_params=None, min_match_count=10):
    pass


def match_template(image, temp_img):
    match_method = cv2.TM_SQDIFF
    result = cv2.matchTemplate(image, temp_img, method=match_method)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    match_loc = None
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result, None)
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        match_loc = min_loc
    else:
        match_loc = max_loc
    # cv2.imshow("rate", result)
    # cv2.waitKey(0)
    return match_loc


def rectification(img1, img2, H=None):
    """
    矫正算法
    :param img1: 待矫正图像
    :param img2: 模板图
    :param H: 单应性矩阵
    :return: None 或 矫正后的图像
    """
    if img1 is None or img2 is None:
        return None
    if H is None:
        H = sift_match(img1, img2)
        if H is None:
            print("两图像相似性不足，无法矫正")
            return None
        img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        return img
    elif H.shape == (3, 3):
        img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        return img
    else:
        return None


def cvt_pos(u, v, mat):
    x = (mat[0][0]*u+mat[0][1]*v+mat[0][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    y = (mat[1][0]*u+mat[1][1]*v+mat[1][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    return (int(x), int(y))


if __name__ == "__main__":
    img1 = cv2.imdecode(np.fromfile(r"/home/wisnton/code/zjn_img/template_img/zhandian_template/37_2_0.png",
                                    np.uint8), 1)
    img2 = cv2.imdecode(np.fromfile(r"/home/wisnton/code/data_image/visible_pictures/2023_03_13/1/38_2_0.png",
                                    np.uint8), 1)
    img = rectification(img2, img1)
    match_template(img2, img)
    cv2.imshow('sifted_img', img)
    cv2.waitKey(0)
