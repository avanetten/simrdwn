# inspired by sn4_baseline
FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER avanetten

# IF YOU WANT PROGRESS PRINTED TO TERMINAL
# Update model_main to log to screen...
# https://stackoverflow.com/questions/52016255/tensorflow-object-detection-api-not-displaying-global-steps
# #. Add tf.logging.set_verbosity(tf.logging.INFO) after the import section of the model_main.py script. It will display a summary after every 100th step. (Can change frequency by log_step_count)
# vi /tensorflow/models/research/object_detection/model_main.py
# insert in on line 27:
#  tf.logging.set_verbosity(tf.logging.INFO)
# change line 63 to: 
#  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, log_step_count_steps=10)

# once started run:
# export PYTHONPATH=$PYTHONPATH:/tensorflow/models/research/:/tensorflow/models/research/slim

# # check if it's using gpu
# python
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# resources:
#. https://github.com/jkjung-avt/hand-detection-tutorial

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  	&& apt-get install -y --no-install-recommends \
	    bc \
	    bzip2 \
	    ca-certificates \
	    curl \
	    git \
	    libgdal-dev \
	    libssl-dev \
	    libffi-dev \
	    libncurses-dev \
	    libgl1 \
	    jq \
	    nfs-common \
	    parallel \
	    python-dev \
	    python-pip \
	    python-wheel \
	    python-setuptools \
	    unzip \
	    vim \
		tmux \
	    wget \
	    build-essential \
        libopencv-dev \
        python-opencv \
	  && apt-get clean \
	  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
# https://repo.continuum.io/miniconda/
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
# ENV TINI_VERSION v0.16.1
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda config --add channels conda-forge 
	# && \
	# conda update conda
    # conda config --remove channels defaults && \

# set up conda environment and add to $PATH
RUN conda create -n simrdwn python=3.6 \
                    && echo "source activate simrdwn" > ~/.bashrc
ENV PATH /opt/conda/envs/simrdwn/bin:$PATH

# install GPU version of tensorflow
RUN source activate simrdwn && \
    conda install -n simrdwn -c defaults tensorflow-gpu=1.13.1

# install keras with tf backend
ENV KERAS_BACKEND=tensorflow
RUN source activate simrdwn \
  && conda install -n simrdwn keras=2.2.4

RUN source activate simrdwn \
	&& conda install -n simrdwn \
				  libgdal=2.4.1 \
    			  gdal=2.4.1 \
	              geopandas=0.5.0 \
				  opencv=4.1.0 \
	              affine=2.2.2 \
	              pyproj=1.9.6 \
	              pyhamcrest=1.9.0 \
	              cython=0.29.10 \
				  contextlib2=0.5.5 \
	              ncurses=6.1 \
	              jupyter=1.0.0 \
	              jupyterlab=0.35.6 \
	              ipykernel=5.1.1 \
	              matplotlib=3.1.0 \
	              numpy=1.16.4 \
	              pandas=0.24.2 \
	              pillow=6.0.0 \
	              pip=19.1.1 \
	              scipy=1.3.0 \
	              scikit-image=0.15.0 \
	              scikit-learn=0.21.2 \
	              shapely=1.6.4 \
	              rtree=0.8.3 \
	              testpath=0.4.2 \
	              tqdm=4.32.1 \
				  statsmodels=0.10.1 \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all 
#	%% conda build purge-all \

# Below is a hack to get around conda issues (2019-07-19)
RUN source activate simrdwn \
    && conda install -n simrdwn \
				-c conda-forge rasterio=1.0.24 \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all 
				  
# tf object detection api
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research/
RUN git checkout r1.13.0
#WORKDIR /tensorflow/models/research/
#RUN protoc object_detection/protos/*.proto --python_out=.
# WORKDIR /tensorflow/models/research/
# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/:/tensorflow/models/slim
# ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/:/tensorflow/models/research/slim

# also need coco api
# manually
# From tensorflow/models/research/
WORKDIR /tensorflow/models/research/
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
RUN pip install pycocotools

# From tensorflow/models/research/
WORKDIR /tensorflow/models/research/
RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/:/tensorflow/models/slim
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/:/tensorflow/models/research/slim

# # this always seems to fail, so do it manually (as above)
# RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
# WORKDIR /cocoapi/PythonAPI
# RUN make
# RUN cp -r pycocotools /tensorflow/models/research/
# # From tensorflow/models/research/
# WORKDIR /tensorflow/models/research/
# RUN protoc object_detection/protos/*.proto --python_out=.

# # test
# WORKDIR /tensorflow/models/research/
# RUN PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# RUN python /tensorflow/models/research/object_detection/builders/model_builder_test.py
#
# # RUN source activate simrdwn && \
# #      python /tensorflow/models/research/object_detection/builders/model_builder_test.py
# # # ENV MODELS_DIR /tensorflow/models/research/
# # # PYTHONPATH=${MODELS_DIR}/research:$MODELS_DIR/research/slim \
# # #PYTHON $MODELS_DIR/research/object_detection/builders/model_builder_test.py

# # add a jupyter kernel for the conda environment in case it's wanted
RUN source activate simrdwn && python -m ipykernel.kernelspec
RUN python -m ipykernel.kernelspec


###################
# Set up our notebook config.
WORKDIR /
# TensorBoard
# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006
RUN ["/bin/bash"]
