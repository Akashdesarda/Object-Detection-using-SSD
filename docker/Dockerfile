FROM nvidia/cuda:10.0-runtime-ubuntu16.04
ENV CUDNN_VERSION 7.6.5.32
ENV PATH /usr/local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends wget ssh git unzip nano gcc software-properties-common python3 python3-pip python3-setuptools \
    apt-transport-https lsb-release openssh-client ffmpeg libsm6 libxrender1 libxext6 \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 ca-certificates netbase && \
	rm -rf /var/lib/apt/lists/*

RUN git clone git@github.com:Akashdesarda/Object-Detection-using-SSD.git

WORKDIR Object-Detection-using-SSD

RUN pip3 install -r requirements.txt

