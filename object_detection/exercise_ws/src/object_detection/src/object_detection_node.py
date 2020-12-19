#!/usr/bin/env python3
import cv2
import rospkg
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, DTParam, NodeType, ParamType, TopicType
from duckietown_msgs.msg import (AntiInstagramThresholds, BoolStamped,
                                 FSMState, LanePose, StopLineReading,
                                 Twist2DStamped, WheelsCmdStamped)
from image_processing.anti_instagram import AntiInstagram
from object_detection.model import Wrapper
from sensor_msgs.msg import CompressedImage, Image


class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        self.initialized = False
        self.image_count = 0
        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        self.pub_image = rospy.Publisher(
            "~detection/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

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
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file', '.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute)
        self.initialized = True
        self.log("Initialized!")

    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        if not self.initialized:
            return
        self.image_count += 1
        if self.image_count % 5 != 0:
            return

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

        image = cv2.resize(image, (224, 224))
        rgb_image = image[:, :, ::-1].copy()
        bboxes, classes, scores = self.model_wrapper.predict(rgb_image)

        msg = BoolStamped()
        msg.header = image_msg.header
        found_duckie = self.det2bool(bboxes[0], classes[0], scores[0])
        msg.data = found_duckie

        self.pub_obj_dets.publish(msg)

        # If there are any subscribers to the debug topics, generate a debug image and publish it
        if self.pub_image.get_num_connections() > 0:
            for i, box in enumerate(bboxes[0]):
                if int(classes[0][i]) == 1:
                    x1, y1, x2, y2 = box
                    center_x = x1 + (x2 - x1) / 2
                    if (scores[0][i] > 0.85) and (56 < center_x < 168) and ((x2 - x1) * (y2 - y1)) > 300:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            msg = self.bridge.cv2_to_compressed_imgmsg(image)
            msg.header = image_msg.header
            self.pub_image.publish(msg)

    def det2bool(self, bboxes, classes, scores):
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            label = classes[i]
            center_x = x1 + (x2 - x1) / 2

            if label != 1:
                continue
            else:
                if (scores[i] > 0.85) and (56 < center_x < 168) and ((x2 - x1) * (y2 - y1)) > 300:
                    return True
        return False


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
