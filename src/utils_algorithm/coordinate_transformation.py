"""
此模块用于存放坐标转换与图像切割算法

@author: lfc
"""

import cv2
import warnings


def cart2polar(img, opt):
    """
    该函数用于将图像从笛卡尔坐标映射到极坐标，并裁剪极坐标图像
    :param img: 笛卡尔坐标图像
    :param opt: 存放极点的笛卡尔坐标，变换后图片大小
    裁剪角度范围（开始极角，终止极角），裁剪长度范围（开始极径，终止极径）
    :return: 已裁剪的极坐标图像
    """
    warnings.simplefilter('always')
    if img is None:
        warnings.warn("No image init")
        return None
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS  # INTER_LINEAR 双线性插值，WARP_FILL_OUTLIERS填充所有目标图像像素
    pole = opt['pole']
    start_angle, end_angle = opt['angle_range']
    min_diameter, max_diameter = opt['polar_diameter_range']
    polar_img_size = opt['polar_size']
    transformer_max_diameter = opt['t_max_d']
    if transformer_max_diameter > min(int(img.shape[0] / 2), int(img.shape[1] / 2)):
        return None

    polar_img = cv2.warpPolar(img, tuple(polar_img_size), tuple(pole), transformer_max_diameter, flags)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    angle_map_factor = polar_img_size[1] / 360
    diameter_map_factor = polar_img_size[0] / transformer_max_diameter

    clip_polar = polar_img[int(min_diameter * diameter_map_factor):int(max_diameter * diameter_map_factor),
                 int(start_angle * angle_map_factor):int(end_angle * angle_map_factor)]

    return clip_polar


if __name__ == '__main__':
    # hsv_extraction = HsvExtraction()
    rect = [[826, 610], [1387, 929]]
    im = cv2.imread("/home/wisnton/catkin_ws_HN/src/vision_code/src/HN_process/images/4_3-1.png")
    im1 = im[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    opt = {'t_max_d': 47,
           'pole': (199, 260),
           'angle_range': [160, 250],
           'polar_diameter_range': [0, 45],
           'polar_size': [200, 600],
           'span': [0, 1],
           'unet_weight': '/home/wisnton/catkin_ws_HN/vision_code/src/weights/pytorch/pointer_scale_1.pt'}
    # image = hsv_extraction.extraction_color(im, 'red')
    cv2.imshow("image", im)
    cv2.waitKey(0)
    image_polar = cart2polar(im, opt)
    cv2.imshow("image_polar", image_polar)
    cv2.waitKey(0)
