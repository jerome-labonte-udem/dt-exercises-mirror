FROM duckietown/challenge-aido_lf-template-ros:daffy-amd64

COPY requirements.* ./
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pip list

RUN mkdir /code/exercise_ws

COPY exercise_ws/src /code/exercise_ws/src
COPY launchers /code

RUN bash -c "source /code/catkin_ws/devel/setup.bash && catkin build --workspace /code/exercise_ws"
RUN bash -c "source /code/exercise_ws/devel/setup.bash"

ENTRYPOINT ["bash", "run.sh"]