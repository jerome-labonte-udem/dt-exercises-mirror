# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "ground_projection: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/melodic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg;-Iduckietown_msgs:/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ground_projection_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_custom_target(_ground_projection_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ground_projection" "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" "geometry_msgs/Point:duckietown_msgs/Pixel"
)

get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_custom_target(_ground_projection_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ground_projection" "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" "duckietown_msgs/Vector2D:geometry_msgs/Point"
)

get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_custom_target(_ground_projection_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ground_projection" "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg;/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection
)
_generate_srv_cpp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv"
  "${MSG_I_FLAGS}"
  "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection
)
_generate_srv_cpp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection
)

### Generating Module File
_generate_module_cpp(ground_projection
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ground_projection_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ground_projection_generate_messages ground_projection_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_cpp _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_cpp _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_cpp _ground_projection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ground_projection_gencpp)
add_dependencies(ground_projection_gencpp ground_projection_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ground_projection_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg;/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection
)
_generate_srv_eus(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv"
  "${MSG_I_FLAGS}"
  "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection
)
_generate_srv_eus(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection
)

### Generating Module File
_generate_module_eus(ground_projection
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ground_projection_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ground_projection_generate_messages ground_projection_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_eus _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_eus _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_eus _ground_projection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ground_projection_geneus)
add_dependencies(ground_projection_geneus ground_projection_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ground_projection_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg;/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection
)
_generate_srv_lisp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv"
  "${MSG_I_FLAGS}"
  "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection
)
_generate_srv_lisp(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection
)

### Generating Module File
_generate_module_lisp(ground_projection
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ground_projection_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ground_projection_generate_messages ground_projection_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_lisp _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_lisp _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_lisp _ground_projection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ground_projection_genlisp)
add_dependencies(ground_projection_genlisp ground_projection_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ground_projection_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg;/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection
)
_generate_srv_nodejs(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv"
  "${MSG_I_FLAGS}"
  "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection
)
_generate_srv_nodejs(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection
)

### Generating Module File
_generate_module_nodejs(ground_projection
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ground_projection_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ground_projection_generate_messages ground_projection_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_nodejs _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_nodejs _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_nodejs _ground_projection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ground_projection_gennodejs)
add_dependencies(ground_projection_gennodejs ground_projection_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ground_projection_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg;/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Pixel.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
)
_generate_srv_py(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv"
  "${MSG_I_FLAGS}"
  "/duckietown/catkin_ws/src/dt-ros-commons/packages/duckietown_msgs/msg/Vector2D.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
)
_generate_srv_py(ground_projection
  "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
)

### Generating Module File
_generate_module_py(ground_projection
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ground_projection_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ground_projection_generate_messages ground_projection_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetGroundCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_py _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/GetImageCoord.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_py _ground_projection_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/duckietown/catkin_ws/src/dt-core/packages/ground_projection/srv/EstimateHomography.srv" NAME_WE)
add_dependencies(ground_projection_generate_messages_py _ground_projection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ground_projection_genpy)
add_dependencies(ground_projection_genpy ground_projection_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ground_projection_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ground_projection
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ground_projection_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(ground_projection_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(ground_projection_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET duckietown_msgs_generate_messages_cpp)
  add_dependencies(ground_projection_generate_messages_cpp duckietown_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ground_projection
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ground_projection_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(ground_projection_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(ground_projection_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET duckietown_msgs_generate_messages_eus)
  add_dependencies(ground_projection_generate_messages_eus duckietown_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ground_projection
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ground_projection_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(ground_projection_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(ground_projection_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET duckietown_msgs_generate_messages_lisp)
  add_dependencies(ground_projection_generate_messages_lisp duckietown_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ground_projection
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ground_projection_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(ground_projection_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(ground_projection_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET duckietown_msgs_generate_messages_nodejs)
  add_dependencies(ground_projection_generate_messages_nodejs duckietown_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ground_projection/.+/__init__.pyc?$"
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ground_projection_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(ground_projection_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(ground_projection_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET duckietown_msgs_generate_messages_py)
  add_dependencies(ground_projection_generate_messages_py duckietown_msgs_generate_messages_py)
endif()
