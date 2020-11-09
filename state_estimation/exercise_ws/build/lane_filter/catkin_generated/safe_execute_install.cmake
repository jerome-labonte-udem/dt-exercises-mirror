execute_process(COMMAND "/code/exercise_ws/build/lane_filter/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/code/exercise_ws/build/lane_filter/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
