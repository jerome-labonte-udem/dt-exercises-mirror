FROM duckietown/gym-duckietown-server-python3:daffy


WORKDIR /project

RUN apt-get update && apt-get install -y libgtk2.0-dev

RUN git --version
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get update && apt-get install -y git
RUN git --version

COPY requirements.txt requirements.txt
RUN python --version
RUN pip install -r requirements.txt
RUN pip list



COPY . .
ENV DISABLE_CONTRACTS=1

ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
