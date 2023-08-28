import json
import queue
import rospy
import numpy as np
# import cv2
from queue import Queue
from threading import Thread
import time
from camera_msgs.msg import cameras_cap, camera_image
import os
from pathlib import Path
from utils_algorithm.load_config import read_yaml
from detection_algorithm.yolo.detected import Detect
import cv2 as cv

current_path = Path(__file__).resolve().parents[0]  # 当前文件路径文件路径
cfg_path = str(Path.joinpath(current_path, 'config.yaml'))
cfg = read_yaml(cfg_path)
yolo_yaml = cfg['yolo']['classes']


class ImageProcessMain:

    def __init__(self):
        rospy.init_node('image_pub', anonymous=True)
        rospy.Subscriber("/cameras/image", cameras_cap, self.cameras_Callback, queue_size=1)
        # rospy.Subscriber("/vision_process/synchronous_data", synchronous_data, self.synchronous_data_Callback)
        self.process_result_pub = rospy.Publisher('/vision_process/result', cameras_cap, queue_size=1)
        self.back_meter_index = False
        self.result_path = str(Path.joinpath(Path(__file__).resolve().parents[0], "result_file/"))
        self.im_queue = Queue(1)
        self.cfg = read_yaml(cfg_path)  # 导入本地配置文件  提取部分配置信息  填充消息发布数据结构体
        self.weight_path = (Path.joinpath(current_path, 'weights/pytorch/yolov5n.pt'))
        self.detects = Detect(self.weight_path, (320, 320), 0.5, 0.5, 100, 'cuda:0', False, False, False, False, False)
        # print(1)


    def cameras_Callback(self, msg):
        rospy.loginfo("camera_dh_data_Callback......")
        try:
            self.im_queue.put_nowait(msg)  # 队列满后不阻塞 直接报错  在回调函数中不要阻塞
            # print('put')
        except queue.Full:
            pass

            # print('im_queue is full error')

    # 图像处理和结果发布
    def data_process_pub(self):
        rate = rospy.Rate(0.5)  # ROS Rate at 20Hz
        while True:
            try:
                self.process_im_msg = self.im_queue.get_nowait()
            except queue.Empty:
                self.process_im_msg = []

            if self.process_im_msg:  # False、0、''、[]、{}、()都为假
                time_start = time.perf_counter()
                try:
                    # 接收到图像
                    # 处理图像算法接口   处理图像算法接口
                    self.pub_msg = cameras_cap()
                    self.pub_msg.angle = self.process_im_msg.angle
                    self.pub_msg.pos_x = self.process_im_msg.pos_x
                    self.pub_msg.pos_y = self.process_im_msg.pos_y
                    # name = 0
                    for index, _ in enumerate(self.process_im_msg.im_arr):
                        # name +=1
                        im = np.frombuffer(self.process_im_msg.im_arr[index].data, dtype=np.uint8)
                        # im = cv2.imdecode(im, cv2.IMREAD_COLOR)  # 单机测试需注释

                        # im = cv2.imdecode(im_encode, cv2.IMREAD_COLOR)  #无需编码
                        # im = np.frombuffer(self.process_im_msg.image, dtype=np.uint8)  #从内存获取数据
                        h = self.process_im_msg.im_arr[index].height
                        w = self.process_im_msg.im_arr[index].width
                        im = im.reshape(h, w, 3)
                        # cv.imwrite(str(name) + ".jpg", im)
                        # data = self.detects.resize_zoom(im)
                        frame, classes = self.detects.detect(im)
                        # frame = im
                        resul = camera_image()
                        # print(1)
                        imencode = cv.imencode('.jpg', frame)[1]
                        resul.data = imencode.tobytes()
                        resul.len = imencode.size
                        resul.width = frame.shape[0]
                        resul.height = frame.shape[1]
                        resul.stamp = self.process_im_msg.im_arr[index].stamp
                        if classes is not None:
                            if 0 in classes and 2 not in classes and 3 not in classes and 7 not in classes:
                                resul.detect_type = 1
                                resul.describe = ""
                            elif 0 not in classes and (2 in classes or 3 in classes or 7 in classes):
                                resul.detect_type = 2
                                resul.describe = ""
                            else:
                                resul.detect_type = 3
                                resul.describe = ""
                        else:
                            resul.detect_type = 0
                            resul.describe = ""
                        resul.encoding = str(0)
                        self.pub_msg.im_arr.append(resul)
                        # print("algorith result:", self.pub_msg.pos_x, self.pub_msg.pos_y)
                    # cal_result  # 该张图片所有仪表的处理结果
                    self.process_result_pub.publish(self.pub_msg)
                    # 获取结束时间
                    time_end = time.perf_counter()
                    # 计算运行时间
                    run_time = time_end - time_start
                    run_time_ms = run_time * 1000
                    print(str(run_time_ms))
                    # 以任务号，站点信息作为文件名保存在./result文件夹下
                    # print(0)
                except Exception as e:
                    print("process picture error: ")
                    print(str(e))
                    pass
            # 是否需要查询数据
            rate.sleep()

    def process_run(self):
        self.process_thread = Thread(target=self.data_process_pub)
        self.process_thread.start()
        rate = rospy.Rate(4)  # ROS Rate at 20Hz
        while not rospy.is_shutdown():
            rate.sleep()
            # print('ros main loop')


if __name__ == '__main__':
    im_process = ImageProcessMain()
    im_process.process_run()