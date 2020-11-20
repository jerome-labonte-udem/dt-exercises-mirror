import numpy as np
from typing import Tuple
import rospy

class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def pure_pursuit(self, d: float, phi: float) -> Tuple[float, float]:
        """
        Takes distance to and angle to lane and returns the velocity and omega
        :param d: distance from d_ref to lane
        :param phi: angular distance from phi_ref to lane
        """
        v_max = self.parameters["v_max"]
        v_min = self.parameters["v_min"]
        x = np.arcsin(d/self.parameters["look_ahead"])
        alpha = -phi
        # if abs(d) > L we get nan. So we correct by trying to go back straight to the lane
        if not np.isnan(x):
            alpha -= x
        elif d < 0:
            alpha -= np.arcsin(-1)
        else:
            alpha -= np.arcsin(1)
        v = max(v_min, v_max - np.abs(np.sin(alpha)) * v_max)
        if self.parameters["use_k"]:
            omega = self.parameters["k"] * np.sin(alpha)
        else:
            omega = 2 * v * np.sin(alpha) / self.parameters["look_ahead"]
        return v, omega
