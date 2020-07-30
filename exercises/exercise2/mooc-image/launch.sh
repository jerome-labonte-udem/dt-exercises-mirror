#!/bin/bash

set -e

roslaunch duckietown_demos lane_following.launch veh:=$VEHICLE_NAME
