#!/usr/bin/env python
import sys
import rospy
import cv2
import os
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError


from duckietown import DTROS


import exercise as mooc_exercise

class image_subscriber(DTROS):

  def __init__(self,node_name):
    super(image_subscriber, self).__init__(node_name=node_name)
    self.veh=os.environ['VEHICLE_NAME'] 
    camera_topic="/"+self.veh+"/camera_node/image/compressed"
    out_topic="/"+self.veh+"/mooc/canny/image/compressed"

    self.bridge = CvBridge()

    self.image_sub = rospy.Subscriber(camera_topic,CompressedImage,self.callback)
    self.image_pub = rospy.Publisher(out_topic,CompressedImage, queue_size=10)

  def callback(self,data):
    try:
      cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
    except CvBridgeError as e:
      print(e)
    edges = mooc_exercise.CannyF(cv_image)
    self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(edges))

if __name__ == '__main__':
    image_subscriber_node = image_subscriber("image_subscriber")
    rospy.spin()