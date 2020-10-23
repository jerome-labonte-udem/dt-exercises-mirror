#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import BoolStamped


class DecoderNode(object):

    def __init__(self):
        rospy.init_node('image_decoder')
        self.active = True
        self.bridge = CvBridge()

        self.publish_freq = self.setupParam("~publish_freq", 30.0)
        self.publish_duration = rospy.Duration.from_sec(1.0 / self.publish_freq)
        self.pub_raw = rospy.Publisher("~image/raw", Image, queue_size=1)
        self.pub_compressed = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        self.last_stamp = rospy.Time.now()
        self.sub_compressed_img = rospy.Subscriber("~compressed_image", CompressedImage,
                                                   self.cbImg, queue_size=1)
        self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)  # Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " % (rospy.get_name(), param_name, value))
        return value

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def cbImg(self, msg):
        if not self.active:
            return
        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        else:
            self.last_stamp = now
        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = msg.header.frame_id
        self.pub_raw.publish(img_msg)
        self.pub_compressed.publish(msg)


if __name__ == '__main__':
    node = DecoderNode()
    rospy.spin()
