# -*- coding: utf-8 -*-

"""

pip install opencv-contrib-python

"""

import os
import re
import json
import time


def get_file_name(path, suffix):
    """
    获取指定目录下所有指定后缀的文件名
    比如 suffix = ".png"
    file_list = getFileName("./", ".png")
    """
    file_list = []
    f_list = os.listdir(path)
    # print f_list
    file_data = []
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == suffix:
            # print i
            nn = re.findall(r'^\d+\_\d+\_\d+\_?[a-z]*$', os.path.splitext(i)[0])
            if nn:
                file_list.append(i)
    return file_list


def read_result(name):
    resul = {}

    return resul


def save_file(resul):
    # rul_dict = json.dumps(resul,  ensure_ascii=False)
    path = "../result_file/" + str(resul['task_id']) + "-" + str(resul['room']) \
           + "-" + str(resul['station_id']) + "_" + str(resul['room_num']) + '.json'
    f2 = open(path, 'w')
    f2.write(resul)
    f2.close()
    listDir("../result_file")       # 定期删除结果文件


def listDir(fileDir):
    for eachFile in os.listdir(fileDir):
        if os.path.isfile(fileDir + "/" + eachFile):  # 如果是文件，判断最后修改时间，符合条件进行删除
            ft = fileDir + "/" + str(eachFile)
            ltime = int(ft.st_mtime)  # 获取文件最后修改时间
            ntime = int(time.time()) - 3600 * 24 * 30  # 现在时间减30天
            if ltime <= ntime:
                os.remove(fileDir + "/" + eachFile)  # 删除3小时前的文件
        elif os.path.isdir(fileDir + "/" + eachFile):  # 如果是文件夹，继续递归
            listDir(fileDir + "/" + eachFile)


def find_all_pic(image_path):
    images = []
    for eachFile in os.listdir(image_path):
        if os.path.isfile(image_path + "/" + eachFile):
            file = str(image_path + "/" + eachFile)
            images.append(file)
        elif os.path.isdir(image_path + "/" + eachFile):  # 如果是文件夹，继续递归
            listDir(image_path + "/" + eachFile)
    return images


# 读取保存的json结果文件
def read_resul(info):
    path = "../result_file/" + str(info) + ".json"
    with open(path, 'r', encoding='utf-8') as f:
        resul_dict = json.load(f)
    return resul_dict


# 图像裁剪
def clip_images(image, rect):
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    if len(image.shape) == 2:
        return image[y1:y2, x1:x2]
    return image[y1:y2, x1:x2, :]

def get_hsv_mean(image, image_hsv):
    src = image.copy()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(src, 3)
    gaussian = cv2.GaussianBlur(median, (3, 3), sigmaX=1, sigmaY=1)
    _, thresh = cv2.threshold(gaussian, 210, 255, cv2.THRESH_BINARY)
    mask = []
    for i in range((image.shape[0])):
        for j in range((image.shape[1])):
            if thresh[i][j] >= 0:
                mask.append(image_hsv[i][j][:])
    hsv_mean = [int(np.mean(mask[:][0])), int(np.mean(mask[:][1])), int(np.mean(mask[:][2]))]
    return hsv_mean