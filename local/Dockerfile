FROM duckietown/dt-car-interface:daffy-amd64 AS dt-car-interface

FROM duckietown/dt-core:daffy-amd64

COPY --from=dt-car-interface ${CATKIN_WS_DIR}/src/dt-car-interface ./${CATKIN_WS_DIR}/src/dt-car-interface


# define repository path
ARG REPO_NAME
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
WORKDIR "${REPO_PATH}"



# create repo directory
RUN mkdir -p "${REPO_PATH}"

RUN git clone https://github.com/duckietown/duckiefleet.git /data/config

COPY dt_dependent_node "${REPO_PATH}/dt-dependent_node"
# Copy or init your packages in here
#COPY dt_dependent_node custom_ws/src/dt_dependent_node
#RUN chmod +x custom_ws/src/dt_dependent_node/dt_dependent_node.py

# Here's a copied package, slighly edited. We don't launch this, but you could easily 
# launch this by finding `custom_line_detector`# and putting that in the 
# lf_slim.launch file  
#COPY custom_line_detector custom_ws/src/custom_line_detector
# Don't forget to "chmod" any node you may need inside!



RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

#RUN /bin/bash -c " catkin build --workspace custom_ws/"
#RUN source ${}/devel/setup.bash


RUN /bin/bash -c "source ${CATKIN_WS_DIR}/devel/setup.bash"

# Copy your launch file over
COPY lf_slim.launch .
#### END CUSTOM CATKIN_WS ####

