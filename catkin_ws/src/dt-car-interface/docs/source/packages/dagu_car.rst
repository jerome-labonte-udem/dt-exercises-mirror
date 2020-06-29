dagu\_car package
=================

.. contents::

The `dagu_car` package contains three core nodes that facilitate a Duckiebot's movement. `KinematicsNode` implements the forward and reverse kinematics models of a Duckiebot, that is it maps movement commands in terms of translational and rotational velocities (`car_cmd` messages) to the corresponding wheel rotation speeds (`wheel_cmd` messages). `CarCmdSwitchNode` is used to select one among various sources of movement commands (`car_cmd` messages). This is necessary when, for example, switching between joystick control and lane following, or between the lane following and the intersection navigation controllers. Finally, `VelocityToPoseNode` integrates the executed wheel commands and provides the current pose of the bot relative to its position and orientation when the node was started.

CarCmdSwitchNode
----------------

.. automodule:: nodes.car_cmd_switch_node


KinematicsNode
--------------

.. automodule:: nodes.kinematics_node


VelocityToPoseNode
------------------

.. automodule:: nodes.velocity_to_pose_node

