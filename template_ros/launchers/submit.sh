#!/bin/bash
roscore &
./run.sh
sleep 5
rostopic pub /$VEHICLE_NAME/fsm_node/mode duckietown_msgs/FSMState '{header: {}, state: "LANE_FOLLOWING"}'
