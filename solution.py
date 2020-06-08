#!/usr/bin/env python2
from __future__ import unicode_literals
from PIL import Image
import io

import os
import time

import argparse
import numpy as np
import roslaunch
from rosagent import ROSAgent

from zuper_nodes_python2 import logger, wrap_direct


class ROSBaselineAgent(object):
    def __init__(self, in_sim, launch_file):
        # Now, initialize the ROS stuff here:

        vehicle_name = os.getenv('VEHICLE_NAME')

        if not in_sim:
            # logger.info('Configuring logging')
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            # roslaunch.configure_logging(uuid)
            # print('configured logging 2')
            roslaunch_path = os.path.join(os.getcwd(), launch_file)
            logger.info('Creating ROSLaunchParent')
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, [roslaunch_path])

            logger.info('about to call start()')

            self.launch.start()
            logger.info('returning from start()')

        # Start the ROSAgent, which handles publishing images and subscribing to action
        logger.info('starting ROSAgent()')
        self.agent = ROSAgent()
        logger.info('started ROSAgent()')

        logger.info('completed __init__()')

    def on_received_seed(self, context, data):
        logger.info('Received seed from pipes')
        np.random.seed(data)

    def on_received_episode_start(self, context, data):
        logger.info("Starting episode")
        context.info('Starting episode %s.' % data)

    def on_received_observations(self, context, data):
        logger.info("received observation")
        jpg_data = data['camera']['jpg_data']
        obs = jpg2rgb(jpg_data)
        self.agent._publish_img(obs)
        self.agent._publish_info()

    def on_received_get_commands(self, context, data):
        logger.info("Agent received GetCommand request")
        while not self.agent.updated:
            time.sleep(0.01)

        pwm_left, pwm_right = self.agent.action
        if self.agent.started:  # before starting, we send empty commnands to keep connection
            self.agent.updated = False

        rgb = {'r': 0.5, 'g': 0.5, 'b': 0.5}
        commands = {
            'wheels': {
                'motor_left': pwm_left,
                'motor_right': pwm_right
            },
            'LEDS': {
                'center': rgb,
                'front_left': rgb,
                'front_right': rgb,
                'back_left': rgb,
                'back_right': rgb

            }
        }
        context.write('commands', commands)

    def finish(self, context):
        context.info('finish()')


def jpg2rgb(image_data):
    """ Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


if __name__ == '__main__':

    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logger.setLevel(LOGLEVEL)
    logger.warn("Logger set to level: "+str(logger.level))
    if logger.level > 20:
        logger.warn("Logging is set to {}, info msg will no be shown".format(LOGLEVEL))
    logger.info("Started solution2.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--sim", action="store_true", help="Add this option to start the car interface")
    parser.add_argument("--launch_file",default="lf_slim.launch", help="launch file that should be used (default: lf_slim.launch")
    args = parser.parse_args()
    agent = ROSBaselineAgent(in_sim=args.sim, launch_file = args.launch_file)
    logger.info("Created agent in solution2 main")
    wrap_direct(agent)
