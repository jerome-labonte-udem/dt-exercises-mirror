# `dt-exercises`

This is the `dt-exercises` repo. It contains a suite of exercises that are designed to be used for learning. 

## Getting Started

Start by installing the requirements:

    $ pip install -r requirements.txt

## How to use this Repository

Enter into an exercise folder. From there you can do the following:

- Look at the description of the exercise.
- `dts exercises notebooks` will spin up the jupyter server to follow the notebook tutorials (if any) that are
 related to the exercise.
- `dts exercises build` will build the code (if any) in the exercise.
- `dts exercises test --sim` will run the code using the Duckietown simulator.
- `dts exercises test --duckiebot_name ![ROBOT_NAME] (--local)` will run the exercise on real Duckiebot hardware. If
 the `--local` option is specified then the agent will run on your local machine but communicate over the network with your Duckiebot.
 
After running `dts exercises test` you can point your browser to `http://localhost:8087` which will give you a desktop environment that you can use to view the status of your agent. For example if it is a ROS agent, you can use `rqt_image_view` or `rviz` to look at debugging outputs. 
 
Typically there will be code for you to edit. Once you are happy with your code, in many cases you can *evaluate* it locally or *submit* it to a challenges for official evaluation against our challenge infrastructure. These are done with:

- `dts challenges evaluate` for local evaluation and 
- `dts challenges submit` for official submission. 
 
You can also run your submission on your Duckiebot with:

- `dts duckiebot submit`
 

## The Anatomy of an Exercise



### ~~TODOS~~

- [ ] One recurrent issue is the framerate. The number of renders was decreased to improve speed, but having more than 15 fps would be great.
- [x] The notebook container has not been updated since last year, everything should be moved to python3 once ente is merge into daffy
- [x] At some point we might want to move away from docker-compose and makefile. Everything will be migrated to python and `dts mooc ...`
- [x] The workflow to make a submission should be made as easy as possible, i.e. we should not have to edit the dockerfile. It might be possible to use `parameters.env`
- [ ] Some of the knobs for the simulator are either not implemented, not working properly or not tested. We should have every param that we need working in the param file
- [ ] All the images that we need should be moved to the duckietown dockerhub and added to CI. They should be made as dt-projects
- [ ] It might be a better experience to have noVNC and notebook/agent container in the same container. We should also make use of the dashboard for everything that we can
- [x] We should reuse as many docker layers as possible. base images have been changed recently and making sure we reuse the new base images when possible would save time and space.
- [ ] Move as much stuff as we can into setup and out of root of the repo to make it easier to see what is actually useful to student.
- [x] Remove all the outputs from the containers (mostly simulator) such that the jupyter and novnc links stays visible at the bottom (alternatively, automatically open them in the browser)
