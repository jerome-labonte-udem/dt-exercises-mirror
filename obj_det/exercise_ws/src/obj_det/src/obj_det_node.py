#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds, SegmentList, Segment
from obj_det_msg.msg import ObjDetList, ObjDet
from image_processing.anti_instagram import AntiInstagram
import cv2
from model.model_sol import Wrapper
from cv_bridge import CvBridge

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )


        # Construct publishers
        self.pub_obj_dets = rospy.Publisher("/agent/obj_det_node/obj_dets",
                                           SegmentList,
                                           queue_size=1,
                                           dt_topic_type=TopicType.PERCEPTION)

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "/agent/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "/agent/anti_instagram_node/thresholds",
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
        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images
        
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
        
        orig_y, orig_x = image.shape[0], image.shape[1]
        scale_y, scale_x = orig_y/224, orig_x/224
        
        image = cv2.resize(image, (224,224))
        bboxes, classes, scores = self.model_wrapper.predict(image)
        
        msg = SegmentList()
        msg.header = image_msg.header
        msg.segments.extend(self.det2seg(bboxes[0], classes[0], scale_x, scale_y))
        
        
        self.pub_obj_dets.publish(msg)
    
    def det2seg(self, bboxes, classes, scale_x, scale_y):
        print(bboxes)
        print(classes)
        # TODO filter the predictions: this environment here is a bit different, and your model might output a bit
        # of noise. For example, you might see a bunch of predictions with x1=223.4 and x2=224, which obviously makes
        # no sense. You should remove these. 
        
        obj_det_list = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            
            x_min = int(np.round(x1 * scale_x))
            y_min = int(np.round(y1 * scale_y))
            x_max = x_min + int(np.round(x2 * scale_x))
            y_max = y_min + int(np.round(y2 * scale_y))
            
            
            det = Segment() # novnc doesn't see our custom messages, so we're hacking our way through by using segments.
            det.pixels_normalized[0].x = x_min
            det.pixels_normalized[0].y = y_min
            det.pixels_normalized[1].x = x_max
            det.pixels_normalized[1].y = y_max
            det.color = classes[i]
        return obj_det_list



if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = ObjectDetectionNode(node_name='obj_det_node')
    # Keep it spinning
    rospy.spin()
