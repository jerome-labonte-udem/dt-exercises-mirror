cmake_minimum_required(VERSION 2.8.3)
project(led_interpreter)


find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  roscpp
  rospy
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES led_interpreter
  CATKIN_DEPENDS duckietown_msgs roscpp rospy
  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
