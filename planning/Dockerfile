FROM duckietown/challenge-aido_lf-template-ros:daffy-amd64

RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip list

RUN mkdir /code/exercise_ws

COPY exercise_ws/src /code/exercise_ws/src
COPY launchers /code

RUN /bin/bash -c "roscore &"

CMD ["bash", "/code/submit.sh"]
