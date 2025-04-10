# Define arguments for PyTorch and CUDA versions
ARG PYTORCH="2.5.1"
ARG CUDA="12.1"
# Use PyTorch CUDA development image with cuDNN 9 as base
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn9-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

# Avoid Public GPG key error (updated for Ubuntu 20.04 base in PyTorch 2.5.1 image)
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Install required system packages
RUN apt-get update \
    && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA to install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Ensure python3 points to python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Clone your GitHub repository
# REPLACE with your actual repo URL
RUN git clone https://github.com/datadrivenharshal/Apprely_Tractor_detection.git repo && \
    mv repo/* . && \
    rm -rf repo

# Install Ultralytics
RUN python3 -m pip install ultralytics

# Install Streamlit
RUN python3 -m pip install streamlit

# Install additional common dependencies
RUN python3 -m pip install \
    opencv-python \
    numpy \
    pillow \
    matplotlib \
    pandas \
    scipy \
    scikit-learn

# If your repo has a requirements.txt, install those dependencies
# Uncomment the following line if applicable
# RUN [ -f requirements.txt ] && python3 -m pip install -r requirements.txt || echo "No requirements.txt found"

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Open a bash shell
CMD ["/bin/bash"]