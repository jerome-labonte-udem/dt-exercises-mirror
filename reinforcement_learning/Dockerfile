FROM duckietown/challenge-aido_lf-template-ros:daffy-amd64

COPY requirements.* ./
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pip list

RUN mkdir /code/exercise_ws

COPY exercise_ws/src /code/exercise_ws/src
COPY launchers /code

RUN /bin/bash -c "roscore &"

CMD ["bash", "/code/submit.sh"]