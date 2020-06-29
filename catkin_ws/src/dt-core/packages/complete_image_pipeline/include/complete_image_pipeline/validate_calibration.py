from collections import OrderedDict
import os

from duckietown_segmaps import FAMILY_SEGMAPS
import duckietown_utils as dtu
from duckietown_utils.cli import D8App
from easy_algo import get_easy_algo_db
from ground_projection import GroundProjection, NoHomographyInfoAvailable
import numpy as np
from pi_camera import NoCameraInfoAvailable

from .image_simulation import simulate_image

__all__ = [
    'ValidateCalibration',
]


class ValidateCalibration(D8App):
    """

        This program validates the intrinsic/extrinsics calibrations.

    """

    cmd = 'rosrun complete_image_pipeline validate_calibration'

    usage = """

Use as follows:

    $ %(prog)s [robot names]

Example:

    $ %(prog)s shamrock emma

"""

    def define_program_options(self, params):
        params.add_string('output', short='o', help='Output dir',
                          default='out-validate_calibration')
        params.accept_extra()

    def go(self):
        extra = self.options.get_extra()
        db = get_easy_algo_db()

        if len(extra) == 0:
            query = dtu.get_current_robot_name()
        else:
            query = extra

        robots = db.query('robot', query, raise_if_no_matches=True)
        self.debug('robots: %s' % sorted(robots))

        actual_map_name = 'DT17_scenario_four_way'

        out = self.options.output
        create_visuals(robots, actual_map_name, out)

        for robot_name in robots:
            _ok = try_simulated_localization(robot_name)


def try_simulated_localization(robot_name):
    actual_map_name = 'DT17_scenario_straight_straight'
    template = 'DT17_template_straight_straight'

    line_detector_name = 'baseline'
    lane_filter_name = 'baseline'
    lane_filter_name = 'moregeneric_straight'
    image_prep_name = 'baseline'
    d = 0.01
    phi = np.deg2rad(5)
    max_phi_err = np.deg2rad(5)
    max_d_err = 0.03
    outd = 'out-try_simulated_localization-%s' % robot_name

    from complete_image_pipeline_tests.synthetic import test_synthetic_phi
    # XXX: should not include the _tests module

    test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                       image_prep_name, lane_filter_name, d, phi, outd,
                       max_phi_err=max_phi_err,
                       max_d_err=max_d_err)


def create_visuals(robots, actual_map_name, out):
    db = get_easy_algo_db()
    actual_map = db.create_instance(FAMILY_SEGMAPS, actual_map_name)
    res = OrderedDict()
    res2 = OrderedDict()

    for i, robot_name in enumerate(sorted(robots)):
        dtu.logger.info('%d/%d: %s' % (i, len(robots), robot_name))
        try:
            gp = GroundProjection(robot_name)
        except (NoCameraInfoAvailable, NoHomographyInfoAvailable) as e:
            dtu.logger.warning('skipping %r: %s' % (robot_name, e))
            continue
        gpg = gp.get_ground_projection_geometry()
        pose = np.eye(3)
        simulated_data = \
            simulate_image(actual_map, pose, gpg, blur_sigma=1)
        res[robot_name] = simulated_data.rectified_synthetic_bgr
        res2[robot_name] = simulated_data.distorted_synthetic_bgr
    if not res:
        msg = 'No images to draw.'
        dtu.logger.error(msg)
    else:
        output = os.path.join(out, 'distorted')
        dtu.write_bgr_images_as_jpgs(res2, output)
        output = os.path.join(out, 'rectified')
        dtu.write_bgr_images_as_jpgs(res, output)
