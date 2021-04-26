FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

COPY . /tmp/gfootball

RUN apt-get update && \
      apt-get --no-install-recommends install -yq \
      git cmake build-essential \
      libgl1-mesa-dev libsdl2-dev \
      libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
      libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
      libsdl-sge-dev python3-pip

# && \

RUN \
      pip3 install --upgrade pip setuptools && \
      pip3 install psutil dm-sonnet==1.* && \
      pip3 install tensorflow==1.15.*

# && \

RUN \
      pip3 install "git+https://github.com/openai/baselines.git@master"

# && \

RUN \
      pip3 install /tmp/gfootball && \
      pip3 install click pyyaml && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
