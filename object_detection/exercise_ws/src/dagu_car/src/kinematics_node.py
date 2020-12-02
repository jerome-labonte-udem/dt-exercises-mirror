#!/usr/bin/env python3

import rospy
import yaml
import time
import os.path

from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from std_srvs.srv import EmptyResponse, Empty

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType


class KinematicsNode(DTROS):
    """
    The `KinematicsNode` maps car commands send from various nodes to wheel commands
    that the robot can execute.

    The `KinematicsNode` performs both the inverse and forward kinematics calculations. Before
    these were implemented in separate nodes, but due to their similarity and parameter sharing,
    they are now combined.

    `KinematicsNode` utilises the car geometry as well as a number of tunining and limiting
    parameters to calculate the wheel commands that the wheels should execute in order for the
    robot to perform the desired car commands (inverse kinematics). Then it uses these wheel
    commands in order to do an open-loop velocity estimation (the forward kinematics part).

    TODO: Add link/explanation/illustration of the car geometry and the math used

    All the configuration parameters of this node can be changed dynamically while the node
    is running via `rosparam set` commands.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~gain (:obj:`float`): scaling factor applied to the desired velocity, default is 1.0
        ~trim (:obj:`float`): trimming factor that is typically used to offset differences in the
            behaviour of the left and right motors, it is recommended to use a value that results in
            the robot moving in a straight line when forward command is given, default is 0.0
        ~baseline (:obj:`float`): the distance between the two wheels of the robot, default is 0.1
        ~radius (:obj:`float`): radius of the wheel, default is 0.0318
        ~k (:obj:`float`): motor constant, assumed equal for both motors, default is 27.0
        ~limit (:obj:`float`): limits the final commands sent to the motors, default is 1.0
        ~v_max (:obj:`float`): limits the input velocity, default is 1.0
        ~omega_max (:obj:`float`): limits the input steering angle, default is 8.0

    Subscriber:
        ~car_cmd (:obj:`Twist2DStamped`): The requested car command

    Publisher:
        ~wheels_cmd (:obj:`WheelsCmdStamped`): The corresponding resulting wheel commands
        ~velocity (:obj:`Twist2DStamped`): The open-loop estimation of the robot velocity

    Service:
        ~save_calibration:
            Saves the current set of kinematics parameters (the ones in the Configuration section)
                to `/data/config/calibrations/kinematics/HOSTNAME.yaml`.

    """
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(KinematicsNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Get the vehicle name
        self.veh_name = rospy.get_namespace().strip("/")

        # Read parameters from a robot-specific yaml file if such exists
        self.read_params_from_calibration_file()

        # Get static parameters
        self._baseline = rospy.get_param('~baseline')
        self._radius = rospy.get_param('~radius')
        self._k = rospy.get_param('~k')
        self._v_max = rospy.get_param('~v_max')
        self._omega_max = rospy.get_param('~omega_max')
        # Get editable parameters
        self._gain = DTParam(
            '~gain',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )
        self._trim = DTParam(
            '~trim',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )
        self._limit = DTParam(
            '~limit',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        # Prepare the save calibration service
        self.srv_save = rospy.Service("~save_calibration", Empty, self.srv_save_calibration)

        # Setup publishers
        self.pub_wheels_cmd = rospy.Publisher(
            "~wheels_cmd",
            WheelsCmdStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        self.pub_velocity = rospy.Publisher(
            "~velocity",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        # Setup subscribers
        self.sub_car_cmd = rospy.Subscriber(
            "~car_cmd",
            Twist2DStamped,
            self.car_cmd_callback
        )
        # ---
        self.log("Initialized with: %s" % self._get_details_str())

    def _get_details_str(self):
        return "[" \
            "gain: %s; " % self._gain.value + \
            "trim: %s; " % self._trim.value + \
            "baseline: %s; " % self._baseline + \
            "radius: %s; " % self._radius + \
            "k: %s; " % self._k + \
            "limit: %s; " % self._limit.value + \
            "omega_max: %s; " % self._omega_max + \
            "v_max: %s;" % self._v_max + \
            "]"

    def read_params_from_calibration_file(self):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjsuts the ROS paramaters for the
        node with the new values.

        """
        # Check file existence
        fname = self.get_calibration_filepath(self.veh_name)
        # Use the default values from the config folder if a robot-specific file does not exist.
        if not os.path.isfile(fname):
            self.logwarn("Kinematics calibration %s not found! Using default instead." % fname)
        else:
            with open(fname, 'r') as in_file:
                try:
                    yaml_dict = yaml.load(in_file)
                except yaml.YAMLError as exc:
                    self.logfatal("YAML syntax error. File: %s fname. Exc: %s" % (fname, exc))
                    rospy.signal_shutdown()
                    return

            # Set parameters using value in yaml file
            if yaml_dict is None:
                # Empty yaml file
                return
            for param_name in ["gain", "trim", "baseline", "k", "radius", "limit"]:
                param_value = yaml_dict.get(param_name)
                if param_name is not None:
                    rospy.set_param("~"+param_name, param_value)
                else:
                    # Skip if not defined, use default value instead.
                    pass

    def srv_save_calibration(self, req=None):
        """
        Saves the current kinematics paramaters to a robot-specific file at
        `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`.

        Args:
            req: Not used.

        Returns:
            EmptyResponse

        """

        # Write to a yaml file
        data = {
            "calibration_time": time.strftime("%Y-%m-%d-%H-%M-%S"),
            "gain": self._gain.value,
            "trim": self._trim.value,
            "baseline": self._baseline,
            "radius": self._radius,
            "k": self._k,
            "limit": self._limit.value,
            "v_max": self._v_max,
            "omega_max": self._omega_max
        }

        # Write to file
        file_name = self.get_calibration_filepath(self.veh_name)
        with open(file_name, 'w') as outfile:
            outfile.write(yaml.dump(data, default_flow_style=False))

        # ---
        self.log("Saved kinematic calibration to %s with values: %s" % (
            file_name, self._get_details_str()
        ))

        return EmptyResponse()

    def car_cmd_callback(self, msg_car_cmd):
        """
        A callback that reposponds to received `car_cmd` messages by calculating the
        corresponding wheel commands, taking into account the robot geometry, gain and trim
        factors, and the set limits. These wheel commands are then published for the motors to use.
        The resulting linear and angular velocities are also calculated and published.

        Args:
            msg_car_cmd (:obj:`Twist2DStamped`): desired car command

        """

        # INVERSE KINEMATICS PART

        # trim the desired commands such that they are within the limits:
        msg_car_cmd.v = self.trim(
            msg_car_cmd.v,
            low=-self._v_max,
            high=self._v_max
        )
        msg_car_cmd.omega = self.trim(
            msg_car_cmd.omega,
            low=-self._omega_max,
            high=self._omega_max
        )

        # assuming same motor constants k for both motors
        k_r = self._k
        k_l = self._k

        # adjusting k by gain and trim
        k_r_inv = (self._gain.value + self._trim.value) / k_r
        k_l_inv = (self._gain.value - self._trim.value) / k_l

        omega_r = (msg_car_cmd.v + 0.5 * msg_car_cmd.omega * self._baseline) / self._radius
        omega_l = (msg_car_cmd.v - 0.5 * msg_car_cmd.omega * self._baseline) / self._radius

        # conversion from motor rotation rate to duty cycle
        # u_r = (gain + trim) (v + 0.5 * omega * b) / (r * k_r)
        u_r = omega_r * k_r_inv
        # u_l = (gain - trim) (v - 0.5 * omega * b) / (r * k_l)
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = self.trim(u_r, -self._limit.value, self._limit.value)
        u_l_limited = self.trim(u_l, -self._limit.value, self._limit.value)

        # Put the wheel commands in a message and publish
        msg_wheels_cmd = WheelsCmdStamped()
        msg_wheels_cmd.header.stamp = msg_car_cmd.header.stamp
        msg_wheels_cmd.vel_right = u_r_limited
        msg_wheels_cmd.vel_left = u_l_limited
        self.pub_wheels_cmd.publish(msg_wheels_cmd)

        # FORWARD KINEMATICS PART

        # Conversion from motor duty to motor rotation rate
        omega_r = msg_wheels_cmd.vel_right / k_r_inv
        omega_l = msg_wheels_cmd.vel_left / k_l_inv

        # Compute linear and angular velocity of the platform
        v = (self._radius * omega_r + self._radius * omega_l) / 2.0
        omega = (self._radius * omega_r - self._radius * omega_l) / self._baseline

        # Put the v and omega into a velocity message and publish
        msg_velocity = Twist2DStamped()
        msg_velocity.header = msg_wheels_cmd.header
        msg_velocity.v = v
        msg_velocity.omega = omega
        self.pub_velocity.publish(msg_velocity)

    @staticmethod
    def trim(value, low, high):
        """
        Trims a value to be between some bounds.

        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound

        Returns:
            the trimmed value
        """
        return max(min(value, high), low)

    @staticmethod
    def get_calibration_filepath(name):
        """
        Returns the path to the robot-specific configuration file,
        i.e. `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`.

        Args:
            name (:obj:`str`): the Duckiebot name

        Returns:
            :obj:`str`: the full path to the robot-specific calibration file

        """
        cali_file_folder = '/data/config/calibrations/kinematics/'
        cali_file = cali_file_folder + name + ".yaml"
        return cali_file


if __name__ == '__main__':
    # Initialize the node
    kinematics_node = KinematicsNode(node_name='kinematics_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
