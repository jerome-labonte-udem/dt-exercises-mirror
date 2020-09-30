FROM duckietown/challenge-aido_lf-template-ros:daffy-amd64

COPY requirements.* ./
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pip list

COPY exercise_ws /code/
COPY launchers /code

RUN bash -c "catkin clean --workspace exercise_ws --yes"
RUN bash -c "catkin build --workspace exercise_ws"
RUN bash -c "source /code/exercise_ws/devel/setup.bash"

ENTRYPOINT ["bash", "run.sh"]