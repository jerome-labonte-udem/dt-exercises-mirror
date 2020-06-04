import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import WheelsCmdStamped
import numpy as np
import os
import cv2
from zuper_nodes_python2 import logger


class ROSAgent(object):
    def __init__(self):
        logger.info('ROSAgent::__init__ starts')
        # Get the vehicle name, which comes in as HOSTNAME
        self.vehicle = os.getenv('HOSTNAME')


        logger.info('Creating subscriber')
        # Subscribes to the output of the lane_controller_node
        self.ik_action_sub = rospy.Subscriber('/{}/wheels_driver_node/wheels_cmd'.format(
            self.vehicle), WheelsCmdStamped, self._ik_action_cb)

        # Place holder for the action
        self.action = np.array([0.0, 0.0])
        self.updated = True

        # Publishes onto the corrected image topic
        # since image out of simulator is currently rectified
        logger.info('creating publishers')
        self.cam_pub = rospy.Publisher('/{}/camera_node/image/compressed'.format(
            self.vehicle), CompressedImage, queue_size=10)

        # Publisher for camera info - needed for the ground_projection
        self.cam_info_pub = rospy.Publisher('/{}/camera_node/camera_info'.format(
            self.vehicle), CameraInfo, queue_size=1)

    def init_node(self):
        # Initializes the node
        logger.info("In init_node")
        print("PIn init_node")
        try:
            # rospy.init_node('ROSAgent', disable_signals=True)
            rospy.init_node('ROSAgent',log_level=rospy.INFO)
            logger.info('node initialized')
            print("PNode initialized")
        except BaseException as e:
            logger.info('exception in init_node: %s' % e)
            print("PException in init_node")
            raise

        logger.info("AIDO_DATA_OUT: "+os.getenv("AIDONODE_DATA_OUT"))

        # 15Hz ROS Cycle - TODO: What is this number?
        self.r = rospy.Rate(15)

        # This gets run (looged to /root/.ros/log)
        logger.info('ROSAgent::__init__ complete.')
        print("PROSAGENT::__INIT__COMPLETE")

    def _ik_action_cb(self, msg):
        """
        Callback to listen to last outputted action from lane_controller_node
        Stores it and sustains same action until new message published on topic
        """
        vl = msg.vel_left
        vr = msg.vel_right
        self.action = np.array([vl, vr])
        self.updated = True

    def _publish_info(self):
        """
        Publishes a default CameraInfo - TODO: Fix after distortion applied in simulator
        """
        self.cam_info_pub.publish(CameraInfo())

    def _publish_img(self, obs):
        """
        Publishes the image to the compressed_image topic, which triggers the lane following loop
        """
        logger.info("got image")
        img_msg = CompressedImage()

        time = rospy.get_rostime()
        img_msg.header.stamp.secs = time.secs
        img_msg.header.stamp.nsecs = time.nsecs

        img_msg.format = "jpeg"
        contig = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)
        img_msg.data = np.array(cv2.imencode('.jpg', contig)[1]).tostring()

        self.cam_pub.publish(img_msg)

    def print_test(self):
        logger.info("This prints")
        print("This also prints")
