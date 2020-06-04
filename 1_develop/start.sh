#!/bin/bash


function log_pid() {
  echo "[$BASHPID] $1"
}

function catkin_build() {
  rospack find duckietown_msgs
  if [ $? -ne 0 ]; then 
    log_pid "Need to build environment first, building ..."
    catkin build --workspace catkin_ws
    log_pid "Done building workspace"
  fi
  log_pid "Sourcing catkin_ws/devel/setup.bash"
  source catkin_ws/devel/setup.bash
}

function start_car_interface() {
  log_pid "Starting car interface..."
  ./launch_car_interface.sh
  if [ $? -ne 0 ]; then
    log_pid "error while starting car interface"
    exit 1
  fi
  log_pid "Car interface started"
}

function start_notebook() {
  log_pid "Starting notebook in second thread"
  jupyter-notebook --allow-root --ip="*" --no-browser
  if [ $? -ne 0 ]; then
    log_pid "Received error while running jupyter notebook"
    exit 1
  fi
  log_pid "Done running notebook" 
}

function start_ros_bridge() {
  log_pid "starting ros bridge"
  python2 solution2.py --sim
  errc=$?
  if [ $errc -ne 0 ]; then
    log_pid "[$BASHPID] Error code $errc while running ros bridge, aborting ..."
    exit 1
  fi
  log_pid "Done running ros bridge"
}



log_pid "Running startup script"
catkin_build
set -e # now catch errors and stop
start_car_interface
start_notebook &
start_ros_bridge

exit $?
