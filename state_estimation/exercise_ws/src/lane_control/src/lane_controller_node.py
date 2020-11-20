#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, Segment, SegmentList, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # TODO erase about up to initialize
        # Add the node parameters to the parameters dictionary
        self.params = {"look_ahead": rospy.get_param("~look_ahead"),
                       "v_min": rospy.get_param("~v_min"),
                       "v_max": rospy.get_param("~v_max"),
                           "k": rospy.get_param("~k"),
                      "use_k": rospy.get_param("~use_k")}
        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        # Construct subscribers
        self.sub_seglist_reading = rospy.Subscriber("~segment_list",
                                                    SegmentList,
                                                    self.cb_seglist,
                                                    queue_size=1)

        self.log("Initialized!")

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header
        parameters = {"look_ahead": rospy.get_param("~look_ahead"),
                       "v_min": rospy.get_param("~v_min"),
                       "v_max": rospy.get_param("~v_max"),
                           "k": rospy.get_param("~k"),
                      "use_k": rospy.get_param("~use_k")}
        self.pp_controller.update_parameters(parameters)
        v, omega = self.pp_controller.pure_pursuit(self.pose_msg.d, self.pose_msg.phi)
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)


    def cb_seglist(self, seglist_msg):
        print(seglist_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""
        print("PARAM CHANGED !")
        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
