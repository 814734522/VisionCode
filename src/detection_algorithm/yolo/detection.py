import time
import torch
import cv2 as cv
import numpy as np
import random
import json
from detection_algorithm.yolo.models.experimental import attempt_load  # 加载模型
from detection_algorithm.yolo.utils.general import non_max_suppression, scale_coords, xyxy2xywh  # nms 坐标缩放
from detection_algorithm.yolo.utils.torch_utils import select_device  # 加载设别
from detection_algorithm.yolo.utils.augmentations import letterbox
from pathlib import Path


broker = 'broker.emqx.io'
port = 1883
topic = "/rule/video_collapse_box"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'

classes = {0:' person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
  12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
  26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
  39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',  47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
  54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
  69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddybear', 78: 'hair drier', 79: 'toothbrush',}


def publish(client,pubmsg):
    data = json.dumps(pubmsg)
    msg = f"{data}"
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


class Detect(object):
    def __init__(self, weights, image_w, image_h, class_code, cfg, iou, max_bbox, dev):
        """
                构造函数 - 初始化参数
            inputs
            ------
            weights         模型权重  str
            image_w         图像宽 int
            image_h         图像高  int
            class_code      类别编码 dict
            cfg_            置信度阈值 float
            iou_            交并比阈值 float
            max_bbox_       最大框维度 int
            dev             计算设别ID  str
        """
        self.W = weights
        self.img_w = image_w
        self.img_h = image_h
        self.class_code = class_code
        self.cfg_ = cfg
        self.iou_ = iou
        self.max_bbox_ = max_bbox
        # 读取权重 加载模型
        self.w = str(self.W[0] if isinstance(self.W, list) else self.W)
        if dev == 'cpu':
            self.dev_ = dev
        else:
            self.dev_ = select_device(dev)
        self.model = torch.jit.load(self.w) if 'torchscript' in self.w else attempt_load(self.W)

    def resize_zoom(self, image):
        """
            修改图像维度并归一化
        """
        image_ = letterbox(image, (640, 640))[0]  # 自适应图像缩放
        image_ = image_ / 255.0
        return image_

    @classmethod
    def read_data(cls, image_path):
        """
            读取图像
        """
        image = cv.imread(image_path)
        return image, image.shape

    def predict_(self, data, dev, IMG, target):
        """
                开始预测
            inputs
            ------
            data    修改维度并归一化过后的图像 array
            dev     设备ID  str
            IMG     原始图像  array
            outputs
            -------
            IMG     检测结果 array
        """

        img = data
        # 转换通道位置 HWC转CHW
        img = img[:, :, ::-1].transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        if dev == 'cpu':
            # 矩阵转化tensor
            img = torch.from_numpy(img.copy())
        else:
            # 矩阵转化tensor 并加入cuda设备
            img = torch.from_numpy(img.copy()).cuda()
        # float64转换float32
        img = img.to(torch.float32)
        # 开始预测
        predict = self.model(img, augment='store_true', visualize='store_true')[0]
        # NMS
        predict = non_max_suppression(predict, self.cfg_, self.iou_, None, False, max_det=self.max_bbox_)
        targets = []
        classes = []
        # 绘制bbox信息
        for i, det in enumerate(predict):
            if len(det):
                # 坐标缩放
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], IMG.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if int(cls) in target:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        x1 = int(xywh[0] - xywh[2] / 2)
                        x1 = x1 if x1 < int(img.shape[0]) else int(img.shape[0]) - 1
                        x2 = int(xywh[0] + xywh[2] / 2)
                        x2 = x2 if x2 < int(img.shape[0]) else int(img.shape[0]) - 1
                        y1 = int(xywh[1] - xywh[3] / 2)
                        y1 = y1 if y1 < int(img.shape[1]) else int(img.shape[1]) - 1
                        y2 = int(xywh[1] + xywh[3] / 2)
                        y2 = y2 if y2 < int(img.shape[1]) else int(img.shape[1]) - 1
                        gn = torch.tensor(IMG.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xywh = xywh / [img.shape[0], img.shape[0], img.shape[0], img.shape[0]]
                        # print(xywh)
                        color_dict = {'1': [0, 0, 250], '2': [190, 90, 92], '3': [142, 154, 78], '4': [2, 76, 82],
                                      '5': [119, 80, 5], '6': [189, 163, 234]}  # Rgb3个值：B G R
                        color_single = list
                        if int(cls) in [2, 3, 5]:  # 根据训练的文件.yaml中的name类中的名称修改
                            color_single = color_dict['3']
                        elif self.class_code[int(cls)] == 'person':
                            color_single = color_dict['2']
                        # 类别筛选
                        class_and_cfg = self.class_code[(int(cls))]  # + ' ' + str(conf)
                        # xyxy = int(xyxy)
                        # 绘制bbox
                        IMG = cv.rectangle(IMG, ((x1, y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        # 写入类别置信度
                        cv.putText(IMG, class_and_cfg, (int(xyxy[0]), int(xyxy[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                                   color_single, 2)

                        classes.append(cls)
                        targets.append([[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])]])

        # cv.putText(IMG,"this detection using:" + str(round(run_time_ms, 2)) + "ms", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75,
        #            (0, 255, 255), 2)
        return IMG, targets, classes


if __name__ == '__main__':

    current_path = Path(__file__).resolve().parents[0]          # 当前文件路径文件路径
    boot_path = Path(__file__).resolve().parents[2]
    im_path = str(Path.joinpath(current_path, 'bus.jpg'))
    weight_path = str(Path.joinpath(boot_path, 'weights/pytorch/yolov5s.pt'))
    print(weight_path)
    frame = cv.imread(im_path)
    detects = Detect(weight_path, 640, 640, classes, 0.3, 0.5, 10, 'cpu')
    data = detects.resize_zoom(frame)
    frame, _, _ = detects.predict_(data, 'cpu', frame, (0, 2, 3, 5))
    cv.imshow("1", frame)
    cv.waitKey(0)