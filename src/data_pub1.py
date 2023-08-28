
import rospy
import numpy as np
from camera_hk_msgs.msg import camera_hk_data
from camera_msgs.msg import cameras_cap, camera_image, camera_status
import cv2
from pathlib import Path
import os
from utils_algorithm.load_config import read_yaml
import re
import time
cfg = read_yaml('config.yaml')


#读本地文件夹的图片内容 发布出去
def read_image_publish():

    current_path = Path(__file__).resolve().parents[0]  #当前文件路径文件路径
    image_path = Path.joinpath(current_path, 'Images/person')
    images_name = os.listdir(str(image_path))   #文件夹下所有图片名列表
    list.sort(images_name)
    picture_num = len(images_name)   #文件夹下所有图片数目

    rospy.init_node("image_pub", anonymous=True)
    image_pub = rospy.Publisher('/cameras/image', cameras_cap, queue_size=1)  # rospy发布话题
    # ir_image_pub = rospy.Publisher('/camera_dh/data', camera_dh_data, queue_siz=1) # rospy发布红外图像数据
    rate = rospy.Rate(30)    # rospy 主程序循环频率
    num = 0
    idx = 0
    ratio = 60

    while not rospy.is_shutdown():
        rate.sleep()
        num = num+1
        if num % ratio == 0 and idx < picture_num:    # 每隔固定时间读取一张图片
            name = images_name[idx]
            full_name = str(Path.joinpath(image_path, images_name[idx]))
            im = cv2.imread(full_name)   # 读图像
            im_encode = cv2.imencode('.jpg', im)[1]
            pub_data = cameras_cap()  # 单次多相机抓拍
            pub_data.pos_x = np.float64(100)  # 小车在建图中x坐标
            pub_data.pos_y = np.float64(100)  # 小车在建图中y坐标
            pub_data.angle = np.float64(100)
            pic = 0
            while pic != 4:
                pic += 1
                images = camera_image()
                images.data = im_encode.tobytes()
                images.width = np.uint32(im.shape[0])  # 图像宽
                images.height = np.uint32(im.shape[1])  # 图像高
                images.stamp = np.int32(1)
                images.encoding = str(1)
                images.len = np.uint32(im_encode.size)
                pub_data.im_arr.append(images)
            # image = cv.imdecode(np.fromfile(full_name, dtype=np.uint8), 1)  # 读图像
            image_pub.publish(pub_data)
            print("publish image", pic)
            num = 0
            idx += 1
            if idx == picture_num:
                idx = 0

    # for name in images_name:


if __name__ == '__main__':
    read_image_publish()
