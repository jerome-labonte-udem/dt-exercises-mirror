#!/bin/bash
set -e

# check if ROS_MASTER_URI is set
ROS_MASTER_URI_IS_SET=0
if [ ! -z "${ROS_MASTER_URI}" ]; then
  ROS_MASTER_URI_IS_SET=1
fi

# constants
ROS_SETUP=(
  "/opt/ros/${ROS_DISTRO}/setup.bash"
  "${INSTALL_DIR}/setup.bash"
  "${SOURCE_DIR}/catkin_ws/devel/setup.bash"
  "${SOURCE_DIR}/setup.sh"
)

# check the mandatory arguments
VEHICLE_NAME_IS_SET=1
if [ ${#VEHICLE_NAME} -le 0 ]; then
  # vehicle name not set, use `hostname`
  VEHICLE_NAME_IS_SET=0
  VEHICLE_NAME=$(hostname)
  echo "The environment variable VEHICLE_NAME is not set. Using '${VEHICLE_NAME}'."
fi
export VEHICLE_NAME="${VEHICLE_NAME}"

# check optional arguments
VEHICLE_IP_IS_SET=0
if [ ${#VEHICLE_IP} -ne 0 ]; then
  VEHICLE_IP_IS_SET=1
  echo "The environment variable VEHICLE_IP is set to '${VEHICLE_IP}'. Adding to /etc/hosts."
  {
    echo "${VEHICLE_IP} ${VEHICLE_NAME} ${VEHICLE_NAME}.local" >> /etc/hosts
  } || {
    echo "Failed writing to /etc/hosts. Will continue anyway."
  }
fi

# if vehicle name is set, vehicle ip is then compulsory
if [ "${VEHICLE_NAME_IS_SET}" -eq "1" ] && [ "${VEHICLE_IP_IS_SET}" -eq "0" ]; then
  echo "If you set the variable VEHICLE_NAME, you must set the variable VEHICLE_IP as well. Aborting..."
  exit -1
fi

# configure hosts
if [ "${VEHICLE_NAME_IS_SET}" -eq "0" ]; then
  # vehicle name is not set (assume vehicle is localhost)
  {
    echo "127.0.0.1 ${VEHICLE_NAME} ${VEHICLE_NAME}.local" >> /etc/hosts
  } || {
    echo "Failed writing to /etc/hosts. Will continue anyway."
  }
fi

# setup ros environment
#TODO(andrea): check if necessary when we switch to ROS2
for ROS_SETUP_FILE in "${ROS_SETUP[@]}"; do
  if [ -f "${ROS_SETUP_FILE}" ]; then
    source "${ROS_SETUP_FILE}";
  fi
done

# configure ROS IP
#TODO(andrea): remove when we switch to ROS2
CONTAINER_IP=$(hostname -I 2>/dev/null | cut -d " " -f 1)
export ROS_IP=${CONTAINER_IP}

# configure ROS MASTER URI
#TODO(andrea): remove when we switch to ROS2
if [ "${ROS_MASTER_URI_IS_SET}" -eq "0" ]; then
  export ROS_MASTER_URI="http://${VEHICLE_NAME}.local:11311/"
fi

# robot_type - directory set in init_sd_card/command.py
ROBOT_TYPE_FILE=/data/stats/init_sd_card/parameters/robot_type
if [ -f "${ROBOT_TYPE_FILE}" ]; then
    export ROBOT_TYPE=`cat ${ROBOT_TYPE_FILE}`
else
    echo "Warning: robot_type file does not exist."
fi

# reuse LAUNCHFILE as CMD if the var is set and the first argument is `--`
if [ ${#LAUNCHFILE} -gt 0 ] && [ "$1" == "--" ]; then
  shift
  exec bash -c "$LAUNCHFILE $*"
else
  exec "$@"
fi
