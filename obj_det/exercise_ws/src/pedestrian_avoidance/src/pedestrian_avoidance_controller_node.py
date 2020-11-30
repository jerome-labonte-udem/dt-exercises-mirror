#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds

class PedestrianAvoidanceControllerNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(PedestrianAvoidanceControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )


        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        rospy.Timer(rospy.Duration(0.1), self.cbPubCarCmd)

    def cbPubCarCmd(self,event):
        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        # Add commands to car message
        car_control_msg.v = 0
        car_control_msg.omega = 0
        self.pub_car_cmd.publish(car_control_msg)

if __name__ == "__main__":
    # Initialize the node
    pedestrian_avoidance_controller_node = PedestrianAvoidanceControllerNode(node_name='obj_det_node')
    # Keep it spinning
    rospy.spin()
