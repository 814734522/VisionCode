"""
此模块存放ocr算法

@author: lfc
"""
import numpy as np
import cv2
from paddleocr import PaddleOCR, draw_ocr
from utils_algorithm.image_enhancement import Retinex


class OcrImage:
    def __init__(self, use_angle_cls=True, use_gpu=False, cls=True, det=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, use_gpu=use_gpu)
        self.cls = cls
        self.det = det

    def identify_image(self, image):
        texts = self.ocr.ocr(image, cls=self.cls, det=self.det)
        if len(texts[0]) == 0:
            return None
        info = []
        for text in texts[0]:
            arr = np.array(text[0], dtype=np.int32)
            line = (text[1][0], arr, text[1][1])   # (文本信息， 文本所在坐标(x,y)， 文本置信度)
            info.append(line)
        return info


if __name__ == '__main__':
    ocr_img = OcrImage(det=True)
    img = cv2.imdecode(np.fromfile(r"../HN_process/test/1.JPG", np.uint8), 1)
    # rect = [[786, 278], [1043, 419]]
    # im = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    image = Retinex.MSRCR(img)
    text_info = ocr_img.identify_image(image)
    if text_info is not None:
        for info in text_info:
            # print(info)
            cv2.rectangle(image, tuple(info[1][0]), tuple(info[1][2]), (0, 0, 255), 1)

    # ocr = PaddleOCR.ocr(img, det=False)
    cv2.imwrite("./ocr.jpg", image)
    cv2.waitKey(0)
