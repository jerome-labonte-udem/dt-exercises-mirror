import os

import duckietown_utils as dtu
from easy_logs import get_local_bag_file, NotAvailableLocally
from easy_logs.app_with_logs import D8AppWithLogs
from ground_projection import GroundProjection
from quickapp import QuickApp
import rosbag

from .pipeline import run_pipeline

__all__ = [
    'SingleImagePipelineLog',
]


class SingleImagePipelineLog(D8AppWithLogs, QuickApp):
    """
        Runs the vision pipeline on the first image in a log.
    """

    cmd = 'rosrun complete_image_pipeline single_image_pipeline_log'

    def define_options(self, params):
        g = "Pipeline"
        params.add_string('anti_instagram', default='baseline',
                          help="Which anti_instagram to use", group=g)
        params.add_string('line_detector', default='baseline',
                          help="Which line detector to use", group=g)
        params.add_string('image_prep', default='baseline',
                          help="Which image prep to use", group=g)
        params.add_string('lane_filter', default='baseline',
                          help="Which lane filter to use", group=g)

        params.add_flag('details')

        params.accept_extra()

    def define_jobs_context(self, context):
        db = self.get_easy_logs_db()

        extra = self.options.get_extra()
        if len(extra) == 0:
            query = '*'
        else:
            query = extra
        logs = db.query(query)

        line_detector = self.options.line_detector
        image_prep = self.options.image_prep
        lane_filter = self.options.lane_filter
        anti_instagram = self.options.anti_instagram
        all_details = self.options.details

        print('anti_instagram: %s' % anti_instagram)
        print('image_prep: %s' % image_prep)
        print('line_detector: %s' % line_detector)
        print('lane_filter: %s' % lane_filter)

        for k, log in logs.items():
            d = os.path.join(self.options.output, k)
            context.comp(look_at, log, d,
                         anti_instagram, line_detector, image_prep, lane_filter, all_details)


def look_at(log, output, anti_instagram, line_detector, image_prep, lane_filter, all_details):
    filename = get_local_bag_file(log)

    bag = rosbag.Bag(filename)

    vehicle_name = dtu.which_robot(bag)

    dtu.logger.info('Vehicle name: %s' % vehicle_name)

    gp = GroundProjection(vehicle_name)

    topic = dtu.get_image_topic(bag)
    res = dtu.d8n_read_all_images_from_bag(bag, topic, max_images=1)

    image_cv = res[0]['rgb']

#     dtu.logger.debug(dtu.describe_value(image_cv))

    image_cv_bgr = dtu.bgr_from_rgb(image_cv)

    dtu.DuckietownConstants.show_timeit_benchmarks = True
    res, _stats = run_pipeline(image_cv_bgr, gp=gp,
                               anti_instagram_name=anti_instagram,
                               line_detector_name=line_detector,
                               image_prep_name=image_prep,
                               lane_filter_name=lane_filter,
                               all_details=all_details)

    res = dtu.resize_small_images(res)

    dtu.write_bgr_images_as_jpgs(res, output)
