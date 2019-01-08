FROM nvidia/cuda:8.0-cudnn6-devel
MAINTAINER Adam Van Etten

##############################################
# Build
# cd /raid/simrdwn/docker
# nvidia-docker build --no-cache -t simrdwn .
# NV_GPU=0 nvidia-docker run -it -v /raid:/raid --name simrdwn_gpu0 simrdwn
##############################################

##############################################
# Errors encountered in the past
# https://github.com/tensorflow/tensorflow/issues/16268
# error in export graph, edit 
# vi /opt/tensorflow-models/research/object_detection/exporter.py
#  change line 71 to say:
#    rewrite_options = rewriter_config_pb2.RewriterConfig()
##############################################


###################
# initial packages (some are likely unneccessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    	protobuf-compiler \
      	build-essential \
        curl \
        git \
		cmake \
        rsync \
        software-properties-common \
        unzip \
        zip \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        zlib1g-dev \
        libopencv-dev \
        python-opencv \
		python-tk \
        build-essential autoconf libtool libcunit1-dev \
        libproj-dev libgdal-dev libgeos-dev libjson0-dev vim python-gdal \
    	# requirements for keras
    	python-h5py \
    	python-yaml \
    	python-pydot \
		graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###################
# python packages
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip --no-cache-dir install \
		protobuf \
		ipykernel \
    	jupyter \
    	matplotlib \
    	numpy \
    	scipy \
    	sklearn \
		scikit-image \
		pandas \
		h5py \
		shapely \
    	&& \
    python -m ipykernel.kernelspec


###################
# Install TensorFlow GPU version.
#pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
#pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
ARG TENSORFLOW_VERSION=1.4.1
ARG TENSORFLOW_DEVICE=gpu
ARG TENSORFLOW_APPEND=_gpu
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl


###################
# keras
ARG KERAS_VERSION=2.0.6
ENV KERAS_BACKEND=tensorflow
RUN pip --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}


###################
# object detection api
# https://github.com/phipleg/tensorflow-object-detector/blob/master/Dockerfile
# git clone --branch my_abc http://git.abc.net/git/abc.git

RUN git clone --branch r1.5 https://github.com/tensorflow/models.git /opt/tensorflow-models
WORKDIR /opt/tensorflow-models/research
RUN pip install -e .
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH $PYTHONPATH:/opt/tensorflow-models:/opt/tensorflow-models/slim
ENV PYTHONPATH $PYTHONPATH:/opt/tensorflow-models:/opt/tensorflow-models/research/slim
# test (do this after container has been created)?
#RUN python /opt/tensorflow-models/research/object_detection/builders/model_builder_test.py


###################
# Set up our notebook config.
WORKDIR /raid/
# TensorBoard
RUN ["/bin/bash"]
