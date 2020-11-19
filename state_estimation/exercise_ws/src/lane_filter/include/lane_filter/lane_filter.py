
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
import rospy # TODO remove once optimal noise is found
import os

class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
            'encoder_variance_d',
            'encoder_variance_phi'
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        veh = os.getenv("VEHICLE_NAME")

        self.encoder_variance_d = rospy.get_param(f"/{veh}/lane_filter_node/lane_filter_histogram_kf_configuration/encoder_variance_d")
        self.encoder_variance_phi = rospy.get_param(f"/{veh}/lane_filter_node/lane_filter_histogram_kf_configuration/encoder_variance_phi")

        # TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return
        Q = np.array([[self.encoder_variance_d, 0],
                      [0, self.encoder_variance_phi]])
        # Get distance traveled by both wheels since last prediction
        dist_right = right_encoder_delta / self.encoder_resolution * self.wheel_radius * 2 * np.pi
        dist_left = left_encoder_delta / self.encoder_resolution * self.wheel_radius * 2 * np.pi

        # Get change in phi from traveled distance and wheels distance
        # This assumes that the angular velocity was constant, so but wheels traced arcs around a same point.
        delta_phi = (dist_right - dist_left) / self.baseline

        # This approximates the change in d using the previous phi. It assumes a straight line in the direction
        # phi and a traveled distance equal to the average traveled by each wheels. This is close to reality
        # for short distance
        mu_phi = self.belief["mean"][1]
        delta_d = np.sin(mu_phi) * (dist_left + dist_right) / 2  # TODO review this approximation if needed

        mu_prev = self.belief["mean"]  # [mu_d, mu_phi]
        cov_prev = self.belief["covariance"]  # [[cov(mu, mu), cov(mu, phi)], [cov(phi, mu), cov(mu, mu)])
        delta_mu = np.array([delta_d, delta_phi])
        #print("mean before predict", mu_prev)
        #print("cov before predict", cov_prev)
        # Update the previous belief by adding the computed deltas
        predicted_mu = mu_prev + delta_mu
        predicted_cov = cov_prev + Q
        #print("predicted mean", predicted_mu)
        #print("predicted cov", predicted_cov)
        self.belief["mean"] = predicted_mu
        self.belief["covariance"] = predicted_cov

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)
        if measurement_likelihood is not None:
            # TODO: Parameterize the measurement likelihood as a Gaussian
            cov_prev = self.belief["covariance"]

            H = np.array([[1, 0],
                          [0, 1]])
            # We keep the cell with the most vote as the mean
            argmax_idx = np.argmax(measurement_likelihood)
            argmax_row = argmax_idx // measurement_likelihood.shape[1]
            argmax_col = argmax_idx % measurement_likelihood.shape[1]
            max_d = self.d_min + argmax_row * self.delta_d
            max_phi = self.phi_min + argmax_col * self.delta_phi

            # We assume that if many segment agree on d or phi, then there's less variance

            conf_d = np.sum(measurement_likelihood[argmax_row, :])
            conf_phi = np.sum(measurement_likelihood[:, argmax_col])
            noise_d = max(0.02, 0.5 * (1 - conf_d))  # The higher the confidence, the lower the noise
            noise_phi = max(0.05, (1 - conf_phi))
            z = np.array([max_d, max_phi])
            #print("measurement estimate", z)
            R = np.array([[noise_d, 0],
                          [0, noise_phi]])
            #print("measurement noise", R)
            residual_mu = z - H @ self.belief["mean"]
            residual_cov = H @ cov_prev @ H.T + R

            # TODO: Apply the update equations for the Kalman Filter to self.belief
            K = cov_prev @ H.T @ np.linalg.inv(residual_cov)

            self.belief["mean"] += K @ residual_mu
            self.belief["covariance"] = cov_prev - K @ H @ cov_prev
            #print("mean after update", self.belief["mean"])
            #print("cov after update", self.belief["covariance"])

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
               self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
                                 np.sum(measurement_likelihood)
        return measurement_likelihood





    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray