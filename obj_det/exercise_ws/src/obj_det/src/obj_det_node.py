#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading
from obj_det_msg.msg import ObjDetList, ObjDet
from image_processing.anti_instagram import AntiInstagram
import cv2
from model.model_sol import Wrapper

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )


        # Construct publishers
        self.pub_obj_dets = rospy.Publisher("~obj_dets",
                                           SegmentList,
                                           queue_size=1,
                                           dt_topic_type=TopicType.PERCEPTION)

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )
        
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()
        
        self.model_wrapper = Wrapper()

        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )
            
        image = cv2.resize(image, (224,224))    # TODO change BGR to RGB
        bboxes = self.model_wrapper.predict(image)[0]
        
        msg = SegmentList()
        msg.header = self.image_msg.header
        msg.segments.extend(bboxes)
        
        
        self.pub_obj_dets.publish(msg)
    
    def det2seg(self, bboxes):
        obj_det_list = []
        for x1, y1, x2, y2 in bboxes:
            det = ObjDet()
            det.x_min = x1
            det.y_min = y1
            det.x_max = x2
            det.y_max = y2
        return obj_det_list



if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = ObjectDetectionNode(node_name='obj_det_node')
    # Keep it spinning
    rospy.spin()
