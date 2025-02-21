FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

LABEL maintainer="Developer"
LABEL description="TensorFlow GPU Development Container"
LABEL version="1.0"

RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN	apt update --fix-missing && apt-get --no-install-recommends install -y wget gpg apt-transport-https 

RUN	apt update --fix-missing && apt-get --no-install-recommends install -y \
        git \
        gdb \
        curl \
        ca-certificates \
        build-essential \
        host \
        vim-gtk3 \
        sudo \
        man \
        tmux \
        net-tools \
        cmake \
        openssh-server \
        iproute2 \
        inetutils-ping \
        python3-dev \
        python-is-python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
	&& rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${PYTHONPATH}:/root/.local/bin"
WORKDIR /
COPY ./requirements.txt /
RUN	python -m pip install --upgrade pip && \
        pip install --requirement /requirements.txt

#de-escalate
# RUN groupadd -g 1000 developer && \
#     useradd -m --uid 1000 --gid 1000 developer
# ARG DEV_PW
# RUN echo "developer:${DEV_PW}" | chpasswd
# RUN adduser developer sudo
# RUN mkdir -p /home/developer/.keras /home/developer/tensorflow_datasets /opt/project && \
#     chown -R 1000:1000 /home/developer /opt/project
# RUN mkdir -p /home/developer/.keras /home/developer/tensorflow_datasets /opt/project /workspace && \
#     chown -R developer:developer /home/developer /opt/project /workspace && \
#     chmod -R 755 /home/developer /opt/project /workspace
    
# USER developer
RUN mkdir -p /opt/project /workspace
WORKDIR /workspace

# Default CMD
CMD ["/bin/bash"]