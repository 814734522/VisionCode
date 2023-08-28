"""
此模块用于存放图像增强算法

@author: lfc
"""

import numpy as np
import cv2
from skimage import img_as_float64


class Retinex:
    def __init__(self):
        pass

    @staticmethod
    def singleScale(img, sigma):

        """
        Single-scale Retinex

        Parameters :

        img : input image
        sigma : the standard deviation in the X and Y directions, for Gaussian filter
        """

        ssr = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return ssr

    @staticmethod
    def multiScale(img, sigmas: list):

        """
        Multi-scale Retinex

        Parameters :

        img : input image
        sigma : list of all standard deviations in the X and Y directions, for Gaussian filter
        """

        retinex = np.zeros_like(img)
        for s in sigmas:
            retinex += Retinex.singleScale(img, s)

        msr = retinex / len(sigmas)
        return msr

    @staticmethod
    def crf(img, alpha, beta):
        """
        CRF (Color restoration function)

        Parameters :

        img : input image
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        """
        img_sum = np.sum(img, axis=2, keepdims=True)

        color_rest = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_rest

    @staticmethod
    def MSRCR(img, sigmas=None, alpha=125, beta=46, G=5, b=25.0):
        """
        MSRCR (Multi-scale retinex with color restoration)

        Parameters :

        img : input image
        sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        G : final gain
        b : offset
        """
        if sigmas is None:
            sigmas = [15, 80, 250]
        img = img_as_float64(img) + 1

        img_msr = Retinex.multiScale(img, sigmas)
        img_color = Retinex.crf(img, alpha, beta)
        img_msrcr = G * (img_msr * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255

        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        # cv2.imshow("image", img_msrcr)
        # cv2.waitKey(0)
        return img_msrcr


if __name__ == '__main__':
    SIGMA_LIST = [15, 80, 250]
    ALPHA = 125.0
    BETA = 46.0
    G = 5.0
    OFFSET = 25.0
    image = cv2.imdecode(
        np.fromfile(r'/home/wisnton/catkin_ws_HN/src/vision_code/src/HN_process/images_test/4_3-1.png',
                    np.uint8), 1)
    Retinex.MSRCR(image, SIGMA_LIST, ALPHA, BETA, G, OFFSET)
    print(111)
