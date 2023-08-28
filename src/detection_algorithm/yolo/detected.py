# 导入需要的库
import os
import sys
from pathlib import Path
import numpy as np
# import cv2
import torch
import torch.backends.cudnn as cudnn

# 初始化目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from detection_algorithm.yolo.utils.general import (LOGGER, check_img_size, cv2,
                                                    non_max_suppression, scale_boxes, xyxy2xywh)
from detection_algorithm.yolo.utils.torch_utils import select_device, time_sync

# 导入letterbox
from detection_algorithm.yolo.utils.augmentations import letterbox

current_path = Path(__file__).resolve().parents[0]  # 当前文件路径文件路径
boot_path = Path(__file__).resolve().parents[2]
im_path = str(Path.joinpath(current_path, 'bus.jpg'))
weight_path = (Path.joinpath(boot_path, 'weights/pytorch/yolov5n.pt'))
# weights = weight_path  # 权重文件地址   .pt文件
source = ROOT / 'data/images'  # 测试数据文件(图片或视频)的保存路径
data = ROOT / 'data/coco128.yaml'  # 标签文件地址   .yaml文件


class Detect():
    def __init__(self, weights, imgz, conf_thres, iou_thres, max_det, device, targets, classes_path, agnostic_nms, augment, visualize, half, dnn, is_save):
        self.weights = weights
        self.imgsz = imgz  # 输入图片的大小 默认640(pixels)
        self.conf_thres = conf_thres  # object置信度阈值 默认0.25  用在nms中
        self.iou_thres = iou_thres  # 做nms的iou阈值 默认0.45   用在nms中
        self.max_det = max_det  # 每张图片最多的目标数量  用在nms中
        self.device = select_device(device)  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None
        self.data = ROOT / classes_path  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
        self.agnostic_nms = agnostic_nms  # 进行nms是否也除去不同类别之间的框 默认False
        self.augment = augment  # 预测是否也要采用数据增强 TTA 默认False
        self.visualize = visualize  # 特征图可视化 默认FALSE
        self.half = half  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        self.dnn = dnn  # 使用OpenCV DNN进行ONNX推理
        self.targets = targets
        self.is_save_label = is_save      #是否保存为label文件
        # 获取设备
        # self.device0 = select_device(self.device)

        # 载入模型
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # 检查图片尺寸

        # Half
        # 使用半精度 Float16 推理
        self.half &= (
                                 self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()

    def resize_zoom(self, image):
        """
            修改图像维度并归一化
        """
        image_ = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        return image_

    def detect(self, img):
        """
            目标检测部分
        """
        # Dataloader
        # 载入数据

        # Run inference
        # 开始预测
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        # 对图片进行处理
        im0 = img
        # Padded resize
        im = letterbox(im0, self.imgsz, self.stride, auto=self.pt)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 预测
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        dt[2] += time_sync() - t3

        # 用于存放结果
        detections = []
        classes = []
        # Process predictions
        for i, det in enumerate(pred):  # per image 每张图片
            seen += 1
            # im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    if self.targets is None or int(cls) in self.targets:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        x1 = int(xywh[0] - xywh[2] / 2)
                        x1 = x1 if x1 < int(img.shape[1]) else int(img.shape[1]) - 1
                        x2 = int(xywh[0] + xywh[2] / 2)
                        x2 = x2 if x2 < int(img.shape[1]) else int(img.shape[1]) - 1
                        y1 = int(xywh[1] - xywh[3] / 2)
                        y1 = y1 if y1 < int(img.shape[0]) else int(img.shape[0]) - 1
                        y2 = int(xywh[1] + xywh[3] / 2)
                        y2 = y2 if y2 < int(img.shape[0]) else int(img.shape[0]) - 1
                        color_dict = {'1': (0, 100, 250), '2': (190, 90, 92), '3': (142, 154, 78), '4': (2, 76, 82),
                                      '5': (119, 80, 5), '6': (189, 163, 234)}  # Rgb3个值：B G R
                        color_single = list
                        if int(cls) in [2, 7, 5]:  # 根据训练的文件.yaml中的name类中的名称修改
                            color_single = color_dict['2']
                        elif self.names[int(cls)] == 'person':
                            color_single = color_dict['1']
                        # 类别筛选
                        class_and_cfg = self.names[(int(cls))]  + ' ' + str(round(np.float32(conf), 2))
                        # 绘制bbox
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), color_single,
                                           3)
                        
                        # 写入类别置信度
                        cv2.putText(img, class_and_cfg, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                   color_single, 2)

                        classes.append(int(cls))

                    # conf = float(conf)
                    # detections.append({'class': cls, 'conf': conf, 'position': xywh})
        # print(img.shape)
        # cv2.imshow("1", img)
        # cv2.waitKey(0)
        # 输出结果
        # for i in detections:
        # print(classes)
        # 推测的时间
        LOGGER.info(f'({t3 - t2:.3f}s)')
        return img, classes


if __name__ == '__main__':
    detect1 = Detect(weight_path, (640, 640), 0.5, 0.5, 100, 'cuda:0', None, 'data/coco128.yaml', False, False, False, False, False, False)
    path = './bus.jpg'
    img = cv2.imread(path)
    # 传入一张图片
    # image = detect1.resize_zoom(img)
    # im, _ = detect1.detect(image)
    # print(image.shape, im.shape)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        if frame is not None:
            frame, _ = detect1.detect(frame)
            cv2.imshow("Detected Window", frame)
            if cv2.waitKey(50) == 27:
                break
