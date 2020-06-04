#!/bin/bash
echo "Startup script"

rospack find duckietown_msgs
if [ $? -ne 0 ]; then 
  echo "Need to build environment first ..."
  catkin build --workspace catkin_ws
fi
source catkin_ws/devel/setup.bash

#echo "starting roscore"
#roscore &
#if [ $? -ne 0 ]; then
#  echo "Unable to start roscore, aborting ..."
#  exit 1
#fi


# echo "starting roscore..."
# stdbuf -o L roscore &
# rospid=$!
# 
# sleep 2
# 
# if ! pgrep "roscore" > /dev/null; then
#   echo "error while starting roscore"
#   exit 1
# fi
# echo "Started roscore."

echo "starting car interface"
./launch_car_interface.sh
if [ $? -ne 0 ]; then
  echo "error while starting car interface"
  exit 1
fi

echo "starting ros bridge"
python2 solution2.py --sim
errc=$?
if [ $errc -ne 0 ]; then
  echo "Error code $errc while running ros bridge, aborting ..."
  exit 1
fi

echo "Executed python2 solution2.py"
#jupyter-notebook --allow-root --ip="*" --no-browser
