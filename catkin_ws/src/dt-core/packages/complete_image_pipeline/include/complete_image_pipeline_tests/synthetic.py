import os

from complete_image_pipeline import run_pipeline, simulate_image
from duckietown_segmaps import FAMILY_SEGMAPS
import duckietown_utils as dtu
from easy_algo import get_easy_algo_db
from ground_projection.ground_projection_interface import get_ground_projection
from localization_templates import FAMILY_LOC_TEMPLATES, TemplateStraight
import numpy as np

actual_map_name = 'DT17_scenario_straight_straight'
template = 'DT17_template_straight_straight'
robot_name = dtu.DuckietownConstants.ROBOT_NAME_FOR_TESTS
line_detector_name = 'baseline'
image_prep_name = 'baseline'
# image_prep_name = 'bilinear'

lane_filter_names = []
lane_filter_names += ['moregeneric_straight']
# lane_filter_names += ['baseline']
raise_if_error_too_large = True

max_phi_err = np.deg2rad(5)
max_d_err = 0.021

ntries = 2


def dirn(misc):
    outd = dtu.get_output_dir_for_test()
    return os.path.join(outd, misc)


@dtu.unit_test
def test_synthetic_zero_zerophi():
    d = 0
    phi = np.deg2rad(0)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                       image_prep_name, lane_filter_name, d, phi , outd)


@dtu.unit_test
def test_synthetic_pos_zerophi():
    d = 0.054
    phi = np.deg2rad(0.5)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                           image_prep_name, lane_filter_name, d, phi , outd)


@dtu.unit_test
def test_synthetic_neg_posphi():
    d = -0.05
    phi = np.deg2rad(15)

    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)

        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                           image_prep_name, lane_filter_name, d, phi , outd)


@dtu.unit_test
def test_synthetic_zero_posphi():
    d = 0
    phi = np.deg2rad(15)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                           image_prep_name, lane_filter_name, d, phi , outd)


@dtu.unit_test
def test_synthetic_zero_negphi():
    d = 0
    phi = np.deg2rad(-20)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                           image_prep_name, lane_filter_name, d, phi , outd)


@dtu.unit_test
def test_synthetic_zero_bignegphi():
    d = 0
    phi = np.deg2rad(-50)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                               image_prep_name, lane_filter_name, d, phi, outd)


@dtu.unit_test
def test_synthetic_zero_bigposphi():
    d = 0
    phi = np.deg2rad(+50)
    for lane_filter_name in lane_filter_names:
        outd = dirn(lane_filter_name)
        for _ in range(ntries):
            test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                               image_prep_name, lane_filter_name, d, phi, outd)


def test_synthetic_phi(actual_map_name, template, robot_name, line_detector_name,
                   image_prep_name, lane_filter_name, d, phi, outd,
                   max_phi_err=max_phi_err,
                   max_d_err=max_d_err):

    # important to have deterministic results
    # The randomness is in the line extraction
    np.random.seed(42)
    location = np.zeros((), dtype=TemplateStraight.DATATYPE_COORDS)
    location['phi'] = phi
    location['d'] = d
    _res, stats = test_synthetic(actual_map_name, template, robot_name, line_detector_name,
                                image_prep_name, lane_filter_name, location, outd)
    error = stats['error']
    estimate = stats['estimate']
    fail = False
    msg = 'location: %s  estimate: %s error: %s ' % (location, estimate, error)
    dtu.logger.info(msg)

    if np.abs(error['phi']) > max_phi_err:
        msg += '\nError in phi too big (%s > %s) ' % (np.abs(error['phi']), max_phi_err)
        fail = True
    if np.abs(error['d']) > max_d_err:
        msg += '\nError in d too big (%s > %s)' % (np.abs(error['d']), max_d_err)
        fail = True
    if fail:
        dtu.logger.error(msg)

        if raise_if_error_too_large:
            if not dtu.on_duckiebot():
                raise Exception(msg)
    return not fail


@dtu.contract(actual_map_name='str')
def test_synthetic(actual_map_name, template, robot_name, line_detector_name,
                   image_prep_name, lane_filter_name, pose_or_location, outd):

    np.random.seed(42)
    db = get_easy_algo_db()
    actual_map = db.create_instance(FAMILY_SEGMAPS, actual_map_name)

    # first, load calibration for robot
    easy_algo_db = get_easy_algo_db()
    dtu.logger.debug('looking for localization template %r' % template)
    localization_template = easy_algo_db.create_instance(FAMILY_LOC_TEMPLATES, template)

    gp = get_ground_projection(robot_name)
    # GroundProjectionGeometry
    gpg = gp.get_ground_projection_geometry()

    if pose_or_location.shape == (3, 3):  # SE(2)
        pose = pose_or_location
        location = localization_template.coords_from_pose(pose)
    else:
        location = pose_or_location
        pose = localization_template.pose_from_coords(location)

    simulation_data = simulate_image(actual_map, pose, gpg, blur_sigma=0.3)

    image = simulation_data.distorted_synthetic_bgr

#     anti_instagram_name='identity' # skip
    anti_instagram_name = 'baseline'

    all_details = False
    res, stats = run_pipeline(image, gp,
                              line_detector_name=line_detector_name,
                              image_prep_name=image_prep_name,
                              lane_filter_name=lane_filter_name,
                              anti_instagram_name=anti_instagram_name,
                              all_details=all_details,
                              ground_truth=pose,
                              actual_map=actual_map)

    error = np.empty_like(location)
    for k in error.dtype.fields:
        error[k] = stats['estimate'][k] - location[k]
    stats['error'] = error

    res = dtu.resize_small_images(res)

    dtu.write_bgr_images_as_jpgs(res, outd, extra_string=outd.split('/')[-1])
    return res, stats


if __name__ == '__main__':
    dtu.run_tests_for_this_module()
