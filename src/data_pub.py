from camera_msgs.msg import camera_image, cameras_cap
import cv2 as cv
from pathlib import Path
import rospy
import os
import numpy as np

current_path = Path(__file__).resolve().parents[0]



def data_pub():

    image_path = Path.joinpath(current_path, 'Images/person')
    images_name = os.listdir(str(image_path))  # 文件夹下所有图片名列表
    list.sort(images_name)
    rospy.init_node("Image", anonymous=True)
    image_pub = rospy.Publisher('/camera/cap', cameras_cap, queue_size=1)  # rospy发布话题
    rate = rospy.Rate(10)  # rospy 主程序循环频率
    num = 0

    ratio = 60

    while not rospy.is_shutdown():
        rate.sleep()
        pub_data = cameras_cap()  # 单次多相机抓拍
        pub_data.pos_x = np.float64(100)            # 小车在建图中x坐标
        pub_data.pos_y = np.float64(100)            # 小车在建图中y坐标
        pub_data.angle = np.float64(100)
        pic = 0
        num += 1
        if num % ratio == 0:
            idx = 0
            while pic != 4:
                """ int32       stamp
                    uint32      height
                    uint32      width
                    string      encoding
                    uint32      len
                    uint8[]     data
                """
                """images = camera_image()             # 照片数据类型
                full_name = str(Path.joinpath(image_path, images_name[idx]))
                im = cv.imread(full_name)           # 读图像
                im_encode = cv.imencode('.jpg', im)[1]
                images.data = im_encode.tobytes()
                images.width = np.uint32(im.shape[0])        # 图像宽
                images.height = np.uint32(im.shape[1])        # 图像高
                images.stamp = np.int32(1)
                images.encoding = str(1)
                images.len = np.uint32(im_encode.size)
                # image = cv.imdecode(np.fromfile(full_name, dtype=np.uint8), 1)  # 读图像
                pub_data.im_arr.append(images)            # 加入单张相片数据"""
                idx += 1
                pic = pic + 1
                print("add image")
            image_pub.publish(cameras_cap)              # 发布一次数据
            print("pub success!")


if __name__ == '__main__':
    data_pub()