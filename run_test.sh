#!/bin/bash

# Quick smoke test to see if the workflow is working.
# We start the workflow and check if all the 
# required container are started.

cd control
echo "Starting workflow, please wait"
dts exercises test --sim --staging >/dev/null &
PROCESSID=$!
sleep 20

C1=`docker ps | grep duckietown/challenge-aido_lf-baseline-duckietown | wc -l`
C2=`docker ps | grep duckietown/dt-gui-tools | wc -l`
C3=`docker ps | grep duckietown/dt-commons | wc -l`
C4=`docker ps | grep duckietown/mooc-fifos-connector | wc -l`
C5=`docker ps | grep duckietown/challenge-aido_lf-simulator-gym | wc -l`
OUTPUT=`docker ps -an5`

echo "Stopping workflow"
kill $PROCESSID

echo "Removing containers ..."
dts exercises test --sim --staging --stop >/dev/null

if [ $((C1+C2+C3+C4+C5)) -eq 5 ]; then
  echo "Test passed !"
else
  echo "Some containers have not started successfully:"
  echo "$OUTPUT"
  echo "Use docker logs for the failed container"
fi
