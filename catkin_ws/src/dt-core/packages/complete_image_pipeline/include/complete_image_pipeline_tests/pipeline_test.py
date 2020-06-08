from complete_image_pipeline.pipeline import run_pipeline
import duckietown_utils as dtu

from ground_projection import GroundProjection


@dtu.unit_test
def single_image1():
    p = dtu.require_resource('frame0002.jpg')
    image_cv = dtu.image_cv_from_jpg_fn(p)

    line_detector_name = 'baseline'
    image_prep_name = 'baseline'
    lane_filter_name = 'baseline'
    anti_instagram_name = 'baseline'
    robot_name = dtu.DuckietownConstants.ROBOT_NAME_FOR_TESTS
    gp = GroundProjection(robot_name)

    res, _stats = run_pipeline(image_cv, gp,
                         line_detector_name=line_detector_name,
                         image_prep_name=image_prep_name,
                         lane_filter_name=lane_filter_name,
                         anti_instagram_name=anti_instagram_name,
                         all_details=False, ground_truth=None)

    outd = dtu.get_output_dir_for_test()
    dtu.write_jpgs_to_dir(res, outd)

