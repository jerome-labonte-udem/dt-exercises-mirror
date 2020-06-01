#!/bin/bash

prefix="$0 > "
_kill_procs() {
   echo "$prefix _kill_procs()"
#  kill -TERM $gym
#  wait $gym
  kill -TERM $xvfb
}

## Setup a trap to catch SIGTERM and relay it to child processes
trap _kill_procs SIGTERM
trap _kill_procs INT

# make sure there is no xserver running (only neded for docker-compose)
#killall Xvfb || true
rm -f /tmp/.X99-lock || true

# Start Xvfbs
echo "$prefix Starting xvfb"
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
xvfb=$!
echo "$prefix Started xvfb with PID $xvfb"
export DISPLAY=:99


echo "$prefix Now running gym_bridge.py"
python3 gym_bridge.py
ret=$?
echo "$prefix gym_simulation_launcher terminated with return code $ret"


echo "$prefix Killing xvfb..."
kill $xvfb

echo "$prefix Killed."

echo "$prefix Graceful exit of launch.sh with return code $ret"
exit $ret
