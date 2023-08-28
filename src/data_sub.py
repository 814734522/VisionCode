from camera_msgs.msg import camera_image, cameras_cap
import rospy
import cv2 as cv


def data_callback(msg):
    rospy.loginfo("camera_cap_Callback......")
    cap_image = msg.im_arr
    for images in cap_image:
        image = images.data
        cv.imshow("data", image)
        cv.waitKey(0)


if __name__ == '__main__':
    rospy.init_node('ImageProcessNode', anonymous=True)
    # process_result_sub = rospy.Subscriber("", cameras_cap, queue_size=1)
    rospy.Subscriber("/camera/cap", cameras_cap, data_callback, queue_size=1)
