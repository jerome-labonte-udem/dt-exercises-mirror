# mooc-exercises

## Local development workflow with the simulator

### Quick start

in the root of this repo, run `make start`. The first time you run it, it will download the images, it might take some time. Once it starts, you will see logs from different docker containers that have started, the last one currently looks like this:

middleware_manager_1  | INFO:launcher:Sent episode start to sim
middleware_manager_1  | INFO:launcher:Sent episode start to an agent


You can access the VNC graphical user interface in a browser at http://localhost:8087. This should bring up a linux desktop (might also take a few seconds to show up). This is useful to see the simulation but it also gives you access to all of the `ros` tools. Open a terminal and type `rqt_image_view` (This application should also be available through the desktop). In the dropdown list, choose the correct `ros` topic to visualise: `/default/camera_node/image/compressed`. You can resize to have a better view. This is what the robot in the simulator sees. 

In a new terminal, you can also run `dt-launcher-joystick` to bring up a virtual joystick. This should allow you to control your virtual robot. By default, the whole lane_following is started and the nodes are running.

### Mounting local code

The repo contains a `overlay_ws`. This is a catkin workspace that will be mounted, build and sourced. Any package that you place in here will have precedence over the nodes that are running by default. You can place a package in here and modify the behavior of `dt-core`.

## Deprecated

Everything under this have been modified and will be removed. This documentation is left here during the migration procedure

### ~~How does it work~~ ?

when you run `make start`, the following container are started by docker-compose:

* a container for roscore. We could remove this at some point but I found it easier to use this way for development
* a container running lanefollow, which should be the agent from local development workflow. It is basically what we had a udem last year: https://github.com/duckietown-udem/udem-fall19-public/blob/master/DockerfileNB
* novnc container
* simulator: the goal was to use the same container that is used in the challenge server, so it is this image: https://github.com/duckietown/challenge-aido_LF-simulator-gym/tree/mooc-exercises
* middleware: In order to be able to use the challenge server's simulator image as-is, I re-used the same code to interface it. The middleware is used to handle the fifos connections and is really close to this: https://github.com/duckietown/challenge-aido_LF/tree/daffy-aido4/experiment_manager, though it was modified to remove the need for the scenario maker and use parameters from the param file.

## ~~Challenge submission~~

Since the older version of `1_develop` and `3_submit` have been merged, it should be easy to submit our agent to the AIDO challenge server. In order to submit your code, you can either modify the `lf_slim` launch file to be your launch file, or provide another launch file. To do so, modify the `Dockerfile` in the root by appending your launchfile name to the command:

`CMD ["python", "solution.py"]`
would become

`CMD ["python", "solution.py", "my_lf.launch"]`
(This will most likely be made easier through the param file, users should not modify the dockerfile)

## ~~Description of files~~

* `catkin_ws/`
    * Contains the user's packages. Has `dt-core`, `dt-car-interface` and `dt-ros-commons` as submodules. (This is why the repo should be clone using `git clone --recursive git@github.com:duckietown/mooc-exercises.git`). This folder is mounted into the notebook/agent docker image in the local development workflow.
* `exercises/`
    * (Experimental, not implemented) This was intended to host the different exercises
* `docker-compose`
    * This is the file that is used to start all the containers required for the local development workflow
* `Dockerfile`
    * Needed for `dts challenge submit` and `dts challenge evaluate`. It basically packages the submission into a docker image
* `Makefile`
    * Makes it easier to clean and start the workflow properly.
* `submission.yaml`
    * required for challenge submission
* `lf_slim.launs`
    * default launch file for challenge submission
* `setup/`
    * contains all the required configuration files for local development workflow

#### ~~Configuration files:~~

* `fifos-connector/`
    * This is the middleware image. Currently used locally, but is should be removed and an image from dockerhub will be used instead.
* `notebooks/`
    * Those are the exercises from last year, imported for convenience
* `utils/`
    * common code for the notebook exercises from last year
* `DockerfileNB`
    * Dockerfile for the Notebook/agent image.
* `launch_car_interface.sh`
    * convenience script to launch the car interface automatically (could be moved inside `start.sh`)
* `start.sh`
    * Entrypoint of the notebook/agent container. This will automate all the stuff required:
        * run catkin build if required (since catkin is mounted it is not always required)
        * start the car interface
        * start the notebook
        * start the ros bridge (`solution.py`)
* `requirement*`
    * pip requirement files
* `rosagent.py`
    * ros node to interface the agent. This node does not `spin()` or loop, the event handling is done in `solution.py`.
* `solution.py`
    * This file is used both by the notebook/agent image and when we submit our agent to the AIDO server. It import `rosagent` and forward all the comunication to the FIFOS. Everything that communicates with the simulator goes through this (Send commands, receive observation, etc). The line `wrap_direct(agent)` is blocking and will act as a event handler instead of `rospy.spin()`.

* `parameters.env`
    * Contains all the parameters for the containers. Can be used to tune the simulator.

### ~~TODOS~~

- [ ] One recurrent issue is the framerate. The number of renders was decreased to improve speed, but having more than 15 fps would be great.
- [ ] The notebook container has not been updated since last year, everything should be moved to python3 once ente is merge into daffy
- [ ] At some point we might want to move away from docker-compose and makefile. Everything will be migrated to python and `dts mooc ...`
- [ ] The workflow to make a submission should be made as easy as possible, i.e. we should not have to edit the dockerfile. It might be possible to use `parameters.env`
- [ ] Some of the knobs for the simulator are either not implemented, not working properly or not tested. We should have every param that we need working in the param file
- [ ] All the images that we need should be moved to the duckietown dockerhub and added to CI. They should be made as dt-projects
- [ ] It might be a better experience to have noVNC and notebook/agent container in the same container. We should also make use of the dashboard for everything that we can
- [ ] We should reuse as many docker layers as possible. base images have been changed recently and making sure we reuse the new base images when possible would save time and space.
- [ ] Move as much stuff as we can into setup and out of root of the repo to make it easier to see what is actually useful to student.
- [ ] Remove all the outputs from the containers (mostly simulator) such that the jupyter and novnc links stays visible at the bottom (alternatively, automatically open them in the browser)
