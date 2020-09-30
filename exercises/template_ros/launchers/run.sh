#!/bin/bash

source /code/exercise_ws/devel/setup.bash
python3 solution.py &
roslaunch --wait duckietown_demos lane_following.launch
