#!/usr/bin/env python
import gym
# noinspection PyUnresolvedReferences
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
import numpy as np

from utils.env import launch_env

# ROS imports
import os
import rospy
import roslaunch
from rosagent import ROSAgent
import subprocess
import time


def launch_local_experiment(gym_environment=None):
    # Use our environment launcher
    env = launch_env(gym_environment)

    observation = env.reset()
    # While there are no signal of completion (simulation done)
    # we run the predictions for a number of episodes, don't worry, we have the control on this part

    # We need to launch the ROS stuff in the background
    # ROSLaunch API doesn't play well with our environment setup, so we use subprocess
    import subprocess
    subprocess.Popen(["roslaunch lf_slim.launch"], shell=True)
    
    # Start the ROSAgent, which handles publishing images and subscribing to action 
    agent = ROSAgent()
    r = rospy.Rate(15)

    while not rospy.is_shutdown():
        # we passe the observation to our model, and we get an action in return
        # we tell the environment to perform this action and we get some info back in OpenAI Gym style
        
        # To trigger the lane following pipeline, we publish the image 
        # and camera_infos to the correct topics defined in rosagent
        agent._publish_img(observation)
        agent._publish_info()

        # The action is updated inside of agent by other nodes asynchronously
        action = agent.action
        # Edge case - if the nodes aren't ready yet
        if np.array_equal(action, np.array([0, 0])):
            continue
            
        observation, reward, done, info = env.step(action)
        # here you may want to compute some stats, like how much reward are you getting
        # notice, this reward may no be associated with the challenge score.

        # it is important to check for this flag, the Evalution Engine will let us know when should we finish
        # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
        # from this submission
        if 'simulation_done' in info:
            break
        if done:
            env.reset()

        # Run the main loop at 15Hz
        r.sleep()

if __name__ == '__main__':
    print('Starting local experiment')
    launch_local_experiment()
