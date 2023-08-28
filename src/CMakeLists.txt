cmake_minimum_required(VERSION 2.8.3)
project(vision_code)

find_package(catkin REQUIRED COMPONENTS
        rospy
        std_msgs
        cv_bridge
        sensor_msgs
#        camera_hk_msg
        )

#add_message_files(
#        FILES
#        Image.msg
#)

#generate_messages(
#        DEPENDENCIES
#        test_py
#        sensor_msgs
#)

catkin_package(
        #INCLUDE_DIRS include
        #LIBRARIES test_py
        CATKIN_DEPENDS rospy cv_bridge
#        DEPENDS system_lib
)

include_directories(
        ${catkin_INCLUDE_DIRS}
)
