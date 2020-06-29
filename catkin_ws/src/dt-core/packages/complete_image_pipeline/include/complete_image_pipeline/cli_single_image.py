import os

import duckietown_utils as dtu
from duckietown_utils.cli import D8App
from ground_projection import GroundProjection

from .pipeline import run_pipeline

__all__ = [
    'SingleImagePipeline',
]


class SingleImagePipeline(D8App):
    """
        Runs the vision pipeline on a single image.
    """

    cmd = 'rosrun complete_image_pipeline single_image_pipeline'

    def define_program_options(self, params):
        g = "Input/output"
        params.add_string('image', help="Image to use.", group=g, default=None)
        params.add_string('output', default=None, short='-o', help='Output directory', group=g)
        g = "Pipeline"
        params.add_string('line_detector', default='baseline', help="Which line detector to use", group=g)
        params.add_string('image_prep', default='baseline', help="Which image prep to use", group=g)
        params.add_string('anti_instagram', default='baseline', help="Which anti_instagram to use", group=g)
        params.add_string('lane_filter',
                          default='moregeneric_straight',
                          help="Which lane_filter to use", group=g)

    def go(self):
        vehicle_name = dtu.get_current_robot_name()

        output = self.options.output
        if output is None:
            output = 'out-pipeline'  #  + dtu.get_md5(self.options.image)[:6]
            self.info('No --output given, using %s' % output)

        if self.options.image is not None:
            image_filename = self.options.image
            if image_filename.startswith('http'):
                image_filename = dtu.get_file_from_url(image_filename)

            bgr = dtu.bgr_from_jpg_fn(image_filename)
        else:
            print("Validating using the ROS image stream...")
            import rospy
            from sensor_msgs.msg import CompressedImage

            topic_name = os.path.join('/', vehicle_name, 'camera_node/image/compressed')

            print('Let\'s wait for an image. Say cheese!')

            # Dummy to get ROS message
            rospy.init_node('single_image')
            img_msg = None
            try:
                img_msg = rospy.wait_for_message(topic_name, CompressedImage, timeout=10)
                print('Image captured!')
            except rospy.ROSException as e:
                print('\n\n\nDidn\'t get any message!: %s\n MAKE SURE YOU USE DT SHELL COMMANDS OF VERSION 4.1.9 OR HIGHER!\n\n\n' % (e,))

            bgr = dtu.bgr_from_rgb(dtu.rgb_from_ros(img_msg))
            self.info('Picture taken: %s ' % str(bgr.shape))

        gp = GroundProjection(vehicle_name)

        dtu.DuckietownConstants.show_timeit_benchmarks = True
        res, _stats = run_pipeline(bgr, gp,
                                   line_detector_name=self.options.line_detector,
                                   image_prep_name=self.options.image_prep,
                                   anti_instagram_name=self.options.anti_instagram,
                                   lane_filter_name=self.options.lane_filter)

        self.info('Resizing images..')
        res = dtu.resize_small_images(res)
        self.info('Writing images..')
        dtu.write_bgr_images_as_jpgs(res, output)

