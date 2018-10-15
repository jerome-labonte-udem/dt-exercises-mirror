# Definition of Submission container

# We start from the AIDO1_LF1 ROS template
FROM duckietown/challenge-aido1_lf1-template-ros:v3

RUN ["cross-build-start"]

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py ./

# For ROS Agent - Additional Files
COPY rosagent.py ./
COPY lf_slim.launch ./
COPY catkin_ws ./catkin_ws/

# Uncomment these to build your own catkin_ws - warning: not the fastest!
# RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && catkin_make -j -C catkin_ws/"
# And adding it to the path
# RUN echo "source $(pwd)/catkin_ws/devel/setup.bash" >> ~/.bashrc

# DO NOT MODIFY: your submission won't run if you do
ENV DUCKIETOWN_SERVER=evaluator

# For ROS Agent - pulls the default configuration files for Duckietown stack
# Think of this as the vehicle name
ENV HOSTNAME=default

RUN ["cross-build-end"]
ENTRYPOINT ["qemu3-arm-static"]

# let's see what you've got there...
CMD ["/bin/bash", "-ci", "./solution.py"]