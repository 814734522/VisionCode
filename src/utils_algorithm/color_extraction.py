import cv2
import warnings
import numpy as np


class HsvExtraction:
    def __init__(self):
        self.black = ((0, 0, 0), (180, 255, 46))
        self.grey = ((0, 0, 46), (180, 43, 220))
        self.white = ((0, 0, 221), (180, 30, 255))
        self.red = ([[0, 156], 43, 46], [[10, 180], 255, 255])
        self.orange = ((11, 43, 46), (25, 255, 255))
        self.yellow = ((26, 43, 46), (34, 255, 255))
        self.green = ((35, 43, 46), (77, 255, 255))
        self.cyan = ((78, 43, 46), (99, 255, 255))
        self.blue = ((100, 43, 46), (124, 255, 255))
        self.purple = ((125, 43, 46), (155, 255, 255))

    def extraction_color(self, image, color):
        warnings.simplefilter('always')
        if not isinstance(color, str):
            warnings.warn("颜色参数错误")
            return None
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == 'black':
            image = cv2.inRange(img_hsv, self.black[0], self.black[1])
            return image
        elif color == 'grey':
            image = cv2.inRange(img_hsv, self.grey[0], self.grey[1])
            return image
        elif color == 'white':
            image = cv2.inRange(img_hsv, self.white[0], self.white[1])
            return image
        elif color == 'red':
            lower1 = (self.red[0][0][0], self.red[0][1], self.red[0][2])
            lower2 = (self.red[0][0][1], self.red[0][1], self.red[0][2])
            upper1 = (self.red[1][0][0], self.red[1][1], self.red[1][2])
            upper2 = (self.red[1][0][1], self.red[1][1], self.red[1][2])
            mask1 = cv2.inRange(img_hsv, lower1, upper1)
            mask2 = cv2.inRange(img_hsv, lower2, upper2)
            image = mask1 + mask2
            return image
        elif color == 'orange':
            image = cv2.inRange(img_hsv, self.orange[0], self.orange[1])
            return image
        elif color == 'yellow':
            image = cv2.inRange(img_hsv, self.yellow[0], self.yellow[1])
            return image
        elif color == 'green':
            image = cv2.inRange(img_hsv, self.green[0], self.green[1])
            return image
        elif color == 'cyan':
            image = cv2.inRange(img_hsv, self.cyan[0], self.cyan[1])
            return image
        elif color == 'blue':
            image = cv2.inRange(img_hsv, self.blue[0], self.blue[1])
            return image
        elif color == 'purple':
            image = cv2.inRange(img_hsv, self.purple[0], self.purple[1])
            return image
        else:
            warnings.warn("颜色参数不正确")
            return None


if __name__ == '__main__':
    detector = HsvExtraction()
    img = cv2.imdecode(np.fromfile(r"D:\华能项目\code\vision_code\images\img.png",
                                   np.uint8), 1)
    detector.extraction_color(img, 'red')
