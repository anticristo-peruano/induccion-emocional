FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /repo

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

    RUN pip3 install --upgrade pip
    RUN pip3 install torch==2.0.0+cu121


COPY requirements.txt /repo/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /repo

CMD ["bash"]