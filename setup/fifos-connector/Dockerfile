# FROM duckietown/aido3-base-python3:daffy
ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

COPY requirements.resolved .
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pip list

COPY . .
ENV DISABLE_CONTRACTS=1



ENTRYPOINT ["python3", "middleware.py"]
