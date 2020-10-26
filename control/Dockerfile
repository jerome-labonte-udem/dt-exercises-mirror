FROM duckietown/challenge-aido_lf-baseline-duckietown:daffy-amd64

COPY requirements.* ./
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pip list

RUN mkdir /code/exercise_ws

COPY exercise_ws /code/exercise_ws
COPY launchers /code

CMD ["bash", "/code/submit.sh"]
