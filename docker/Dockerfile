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

# Avoid Public GPG key error
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 || true \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Install required system packages and Python 3.10
RUN apt-get update && apt-get install -y \
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
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

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

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Open a bash shell
CMD ["/bin/bash"]
