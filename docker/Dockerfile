FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    wget \
    bash-completion \
    libgl1 \
    libgl1-mesa-glx \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# --------------------------- TD-MPC2 ----------------------------
COPY docker/requirements.txt /root
RUN pip install -r /root/requirements.txt && \
    pip cache purge && \
    rm /root/requirements.txt && \
    conda install -y numpy==1.26.3

# Fix moviepy issue in wandb
RUN sed -i 's/moviepy\.editor/moviepy/g' /opt/conda/lib/python3.11/site-packages/wandb/sdk/data_types/video.py

WORKDIR /root/tdmpc2

COPY ./tdmpc2 .

# success!
RUN echo "Successfully built TD-MPC2 Docker image!"
