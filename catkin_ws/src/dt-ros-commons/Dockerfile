# parameters
ARG REPO_NAME="dt-ros-commons"

ARG ARCH=arm32v7
ARG MAJOR=daffy
ARG ROS_DISTRO=kinetic
ARG BASE_TAG=${MAJOR}-${ARCH}
ARG BASE_IMAGE=dt-ros-${ROS_DISTRO}-base

# extend dt-commons
ARG SUPER_IMAGE=dt-commons
ARG SUPER_IMAGE_MAJOR=daffy
ARG SUPER_IMAGE_TAG=${SUPER_IMAGE_MAJOR}-${ARCH}
FROM duckietown/${SUPER_IMAGE}:${SUPER_IMAGE_TAG} as dt-commons

# define base image
FROM duckietown/${BASE_IMAGE}:${BASE_TAG}

# copy stuff from the super image
COPY --from=dt-commons /environment.sh /environment.sh

# configure environment
ENV SOURCE_DIR /code
ENV CATKIN_WS_DIR "${SOURCE_DIR}/catkin_ws"
ENV DUCKIEFLEET_ROOT "/data/config"
ENV ROS_LANG_DISABLE gennodejs:geneus:genlisp
ENV READTHEDOCS True
WORKDIR "${CATKIN_WS_DIR}"

# define repository path
ARG REPO_NAME
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
WORKDIR "${REPO_PATH}"

# create repo directory
RUN mkdir -p "${REPO_PATH}"

# copy dependencies files only
COPY ./dependencies-apt.txt "${REPO_PATH}/"
COPY ./dependencies-py.txt "${REPO_PATH}/"

# install apt dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' dependencies-apt.txt | uniq) \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install -r ${REPO_PATH}/dependencies-py.txt

# copy the source code
COPY . "${REPO_PATH}/"

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# configure entrypoint
COPY assets/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# store module name
LABEL org.duckietown.label.module.type "${REPO_NAME}"
ENV DT_MODULE_TYPE "${REPO_NAME}"

# store module metadata
ARG ARCH
ARG MAJOR
ARG BASE_TAG
ARG BASE_IMAGE
LABEL org.duckietown.label.architecture "${ARCH}"
LABEL org.duckietown.label.code.location "${REPO_PATH}"
LABEL org.duckietown.label.code.version.major "${MAJOR}"
LABEL org.duckietown.label.base.image "${BASE_IMAGE}:${BASE_TAG}"

# define maintainer
LABEL maintainer="Andrea F. Daniele (afdaniele@ttic.edu)"
