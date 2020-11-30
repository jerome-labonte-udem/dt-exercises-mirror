#!/usr/bin/env python3

import rospy

from duckietown_msgs.msg import Twist2DStamped, FSMState

from duckietown.dtros import DTROS, NodeType, TopicType


class CarCmdSwitchNode(DTROS):
    """

    Warnings:
        This node will probably be redesigned soon.

    Subscriber:
        ~mode (:obj:`duckietown_msgs/FSMState`): Current control mode of the duckiebot

    Publisher:
        ~cmd (:obj`duckietown_msgs/Twist2DStamped`): The car command messages corresponding
            to the selected mode
    """
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(CarCmdSwitchNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Read parameters
        self._mappings = rospy.get_param("~mappings")
        self._source_topic = rospy.get_param("~source_topics")
        self._mode_topic = rospy.get_param("~mode_topic")
        self.current_src_name = "joystick"

        # Construct publisher
        self.pub_cmd = rospy.Publisher(
            "~cmd",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        # Construct subscribers
        self.sub_fsm_state = rospy.Subscriber(
            self._mode_topic,
            FSMState,
            self.fsm_state_cb
        )

        self.sub_dict = dict()
        for src_name, topic_name in self._source_topic.items():
            self.sub_dict[src_name] = rospy.Subscriber(
                topic_name,
                Twist2DStamped,
                self.wheels_cmd_cb,
                callback_args=src_name
            )
        # ---
        self.log("Initialized.")
        self.log(
            "Sources:" +
            '\n\t- '.join([''] + [
                "'%s': '%s'" % (k, v) for k, v in self._source_topic.items()
            ])
        )

    def fsm_state_cb(self,fsm_state_msg):
        self.current_src_name = self._mappings.get(fsm_state_msg.state)
        if self.current_src_name == "stop":
            self.publish_stop()
            self.log("Car cmd switched to STOP in state %s." % fsm_state_msg.state)
        elif self.current_src_name is None:
            self.logwarn(
                "FSMState %s not handled. No msg pass through the switch." % fsm_state_msg.state
            )
        else:
            self.log("Car cmd switched to %s in state %s." % (
                self.current_src_name, fsm_state_msg.state
            ))

    def wheels_cmd_cb(self,msg,src_name):
        if src_name == self.current_src_name:
            self.pub_cmd.publish(msg)

    def publish_stop(self):
        msg = Twist2DStamped()
        msg.v = 0
        msg.omega = 0
        self.pub_cmd.publish(msg)


if __name__ == '__main__':
    # Create the DaguCar object
    car_cmd_switch_node = CarCmdSwitchNode(node_name='car_cmd_switch_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
