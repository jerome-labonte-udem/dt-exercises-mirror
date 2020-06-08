# Install script for directory: /duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/duckietown/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE PROGRAM FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/_setup_util.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE PROGRAM FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/env.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/setup.bash;/duckietown/catkin_ws/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE FILE FILES
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/setup.bash"
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/local_setup.bash"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/setup.sh;/duckietown/catkin_ws/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE FILE FILES
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/setup.sh"
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/local_setup.sh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/setup.zsh;/duckietown/catkin_ws/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE FILE FILES
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/setup.zsh"
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/duckietown/catkin_ws/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/duckietown/catkin_ws/install" TYPE FILE FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/.rosinstall")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs/msg" TYPE FILE FILES
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AntiInstagramHealth.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AntiInstagramTransform.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AntiInstagramTransform_CB.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AprilTagDetection.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AprilTagDetectionArray.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AprilTagsWithInfos.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/AprilTagExtended.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/BoolStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/CarControl.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/CoordinationClearance.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/CoordinationSignal.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/DroneControl.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/DroneMode.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/DuckiebotLED.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/EncoderStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/FSMState.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/IntersectionPose.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/IntersectionPoseImg.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/IntersectionPoseImgDebug.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/KinematicsParameters.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/KinematicsWeights.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LanePose.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LEDDetection.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LEDDetectionArray.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LEDDetectionDebugInfo.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LEDInterpreter.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LEDPattern.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LightSensor.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/LineFollowerStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/MaintenanceState.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ObstacleImageDetection.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ObstacleImageDetectionList.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ObstacleProjectedDetection.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ObstacleProjectedDetectionList.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ObstacleType.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ParamTuner.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pose2DStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Rect.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Rects.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/SceneSegments.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Segment.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/SegmentList.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/SignalsDetection.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/SignalsDetectionETHZ17.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/SourceTargetNodes.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/StopLineReading.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/TagInfo.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ThetaDotSample.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/ToFStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Trajectory.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/TurnIDandType.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Twist2DStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/VehicleCorners.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/VehiclePose.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vsample.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/WheelsCmd.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/WheelsCmdStamped.msg"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/WheelsCmdDBV2Stamped.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs/srv" TYPE FILE FILES
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/ChangePattern.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/GetVariable.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/IMUstatus.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/LFstatus.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/SensorsStatus.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/SetCustomLEDPattern.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/SetFSMState.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/SetValue.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/SetVariable.srv"
    "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/srv/ToFstatus.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs/cmake" TYPE FILE FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/duckietown_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/duckietown/catkin_ws/devel/.private/duckietown_msgs/include/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/duckietown/catkin_ws/devel/.private/duckietown_msgs/share/roseus/ros/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/duckietown/catkin_ws/devel/.private/duckietown_msgs/share/common-lisp/ros/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/duckietown/catkin_ws/devel/.private/duckietown_msgs/share/gennodejs/ros/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/duckietown/catkin_ws/devel/.private/duckietown_msgs/lib/python2.7/dist-packages/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/duckietown/catkin_ws/devel/.private/duckietown_msgs/lib/python2.7/dist-packages/duckietown_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/duckietown_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs/cmake" TYPE FILE FILES "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/duckietown_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs/cmake" TYPE FILE FILES
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/duckietown_msgsConfig.cmake"
    "/duckietown/catkin_ws/build/duckietown_msgs/catkin_generated/installspace/duckietown_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/duckietown_msgs" TYPE FILE FILES "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/duckietown/catkin_ws/build/duckietown_msgs/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/duckietown/catkin_ws/build/duckietown_msgs/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
