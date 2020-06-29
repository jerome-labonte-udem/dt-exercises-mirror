execute_process(COMMAND "/duckietown/catkin_ws/build/pi_camera/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/duckietown/catkin_ws/build/pi_camera/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
