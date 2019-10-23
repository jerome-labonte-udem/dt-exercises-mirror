# Definition of Submission container

# We start from the ROS Template
FROM duckietown/challenge-aido_lf-template-ros:daffy

RUN rm -r /workspace; mkdir /workspace
WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt 

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py ./

# For ROS Agent - Additional Files
COPY rosagent.py ./
COPY lf_slim.launch ./

## Uncomment these to build your own catkin_ws
#### START CUSTOM CATKIN_WS ####

# RUN /bin/bash -c "mkdir -p custom_ws/src/"

## Copy or init your packages in here
# COPY dt_dependent_node custom_ws/src/dt_dependent_node
# RUN chmod +x custom_ws/src/dt_dependent_node/dt_dependent_node.py

## Do not change the below line! This ensures that your workspace is overlayed on top of the Duckietown stack!  
## MAKE sure this line is present in the build: This workspace overlays: /home/software/catkin_ws/devel;/opt/ros/kinetic
# RUN /bin/bash -c "source /home/software/catkin_ws/devel/setup.bash && catkin_init_workspace && cd ../.."
# RUN /bin/bash -c "source /home/software/catkin_ws/devel/setup.bash && catkin_make -j -C custom_ws/"
# RUN echo "source custom_ws/devel/setup.bash" >> ~/.bashrc

#### END CUSTOM CATKIN_WS ####

# DO NOT MODIFY: your submission won't run if you do
ENV HOSTNAME=default

ENV DUCKIEFLEET_ROOT /data/config

RUN mkdir -p /data/config
RUN git clone https://github.com/duckietown/duckiefleet.git /data/config

# let's see what you've got there...
CMD ["python", "solution.py"]
