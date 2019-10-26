#!/bin/bash
rm -f /tmp/.X99-lock
rm -f /tmp/.X11-unix/X99 
./run_display.bash
source sim_ws/devel/setup.bash
roslaunch gymdt gymdt.launch
