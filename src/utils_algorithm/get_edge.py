import cv2
import numpy as np


def get_edge(img, kernel_size, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    kernel = np.ones((size, size), np.float32) / (size ** 2)
    img_thresh = cv2.filter2D(img_thresh, -1, kernel)
    edges = cv2.Canny(img_thresh, 50, 150, apertureSize=3)
    Matrix = np.ones((2, 2), np.uint8)
    img_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, Matrix)
    return img_edge