# SIMRDWN #


![Alt text](/results/__examples/header.jpg?raw=true "")

The Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN) codebase combines some of the leading object detection algorithms into a unified framework designed to rapidly detect objects both large and small in overhead imagery.  By design, test images can be of arbitrarily large size.  This work seeks to extend the [YOLT](https://arxiv.org/abs/1805.09512) modification of [YOLO](https://pjreddie.com/darknet/yolo/) to include the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).  Therefore, one can train models and test on arbitrary image sizes with [YOLO](https://pjreddie.com/darknet/yolo/), [Faster R-CNN](https://arxiv.org/abs/1506.01497), [SSD](https://arxiv.org/abs/1512.02325), or [R-FCN](https://arxiv.org/abs/1605.06409).  

### For more information, see:

1. Our arXiv paper: [Satellite Imagery Multiscale Rapid Detection with Windowed Networks](https://arxiv.org/abs/1809.09978)

2. Our original [YOLT paper](https://arxiv.org/abs/1805.09512)

3. The original [YOLT repository](https://github.com/CosmiQ/yolt) (now deprecated)


____
## Install SIMRDWN

SIMRDWN is designed to run within a docker container on a GPU-enabled system.

##### A. Install NVIDIA driver (384.X or greater)  

##### B. Intall [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

##### C. Download SIMRDWN codebase

    # assume local directory is /raid/simrdwn
	git clone https://github.com/cosmiq/simrdwn.git /raid/simrdwn 

##### D. Create and launch Docker file

All commands should be run in the docker file, create it with the following commands:

    cd /raid/simrdwn/docker
	# create a docker image named "simrdwn"
    nvidia-docker build --no-cache -t simrdwn .   
	# create a docker container named simrdwn_train
    nvidia-docker run -it -v /raid:/raid --name simrdwn_train simrdwn  

##### E. Compile the modified Darknet C code; download weights

    cd /raid/simrdwn/yolt
    make
	cd /raid/simrdwn/yolt/input_weights
	curl -O https://pjreddie.com/media/files/yolov2.weights 

##### F. Edit exporter for TensorFlow

There is an [error in the exporter function](https://github.com/tensorflow/tensorflow/issues/16268), so edit _exporter.py_

	vi /opt/tensorflow-models/research/object_detection/exporter.py
	# comment out line 72
	# change line 71 to say:
	#    rewrite_options = rewriter_config_pb2.RewriterConfig()


____
____

## 1. Prepare Training Data



####  1A. Create YOLT Format

Training data needs to be initially transformed to the YOLO format of training images in an "images" folder and bounding box labels in a "labels" folder.  For example, an image "images/ex0.png" has a corresponding label "labels/ex0.txt". We also need to define the object classes with a .pbtxt file, such as _/simrdwn/data/class\_labels\_car.pbtxt_. Labels are bounding boxes of the form 

    <object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height. Object-class is a zero-indexed integer. Running a script such as:

	python /raid/simrdwn/core/prep_data_cowc.py 

extracts training windows of reasonable size (usually 416 or 544 pixels in extent) from large labeleled images of the [COWC](https://gdo152.llnl.gov/cowc/) dataset.  The script then transforms the labels corresponding to these windows into the correct format and creates a list of all training input images in _/simdwn/data/cowc\_yolt\_train\_list.txt_.


####  1B. Create .tfrecord 
If the tensorflow object detection API models are being run, we must transform training data into the .tfrecord format.  The _simrdwn/core/prep\_data\_cowc.py_ script does this, but it can also be explicitly created via the _simrdwn/core/preprocess\_tfrecords.py_ script: 
	
	python /raid/simrdwn/core/preprocess_tfrecords.py \
	    --image_list_file /raid/simrdwn/data/cowc_yolt_train_list.txt \
	    --pbtxt_filename /raid/simrdwn/data/class_labels_car.pbtxt \
	    --outfile /raid/simrdwn/data/cowc_train.tfrecord \
	    --val_frac 0.0

____

## 2. Train

We can train either YOLT models or tensorflow object detection API models. Commonly altered variables (such as filenames or batch\_size) will be altered automatically in the config file when training is kicked off.  TensorFlow configs reside in the _/simrdwn/configs_ directory; further example of tensorflow config files reside [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)). YOLT configs can be found in _/simrdwn/yolt/cfg_.  Training can be run within the docker container with commands such as:

	# SSD COWC car search
	python /raid/simrdwn/core/simrdwn.py \
		--framework ssd \
		--mode train \
		--outname inception_v2_cowc \
		--label_map_path /raid/simrdwn/data/class_labels_car.pbtxt \
		--tf_cfg_train_file /raid/simrdwn/configs/_orig/ssd_inception_v2_simrdwn.config \
		--train_tf_record /raid/simrdwn/data/cowc_train.tfrecord \
		--max_batches 30000 \
		--batch_size 16 \
		--gpu 0
		
	# YOLT COWC car search
	python /raid/simrdwn/core/simrdwn.py \
		--framework yolt \
		--mode train \
		--outname dense_cowc \
		--yolt_object_labels_str car \
		--yolt_cfg_file yolt.cfg  \
		--weight_dir /simrdwn/yolt/input_weights \
		--weight_file yolov2.weights \
		--yolt_train_images_list_file cowc_yolt_train_list.txt \
		--label_map_path /raid/simrdwn/data/class_labels_car.pbtxt \
		--max_batches 30000 \
		--batch_size 64 \
		--subdivisions 16 \
		--gpu 0
		
		
	# Faster R-CNN 3-vehicle search (ResNet 101)
	python /raid/simrdwn/core/simrdwn.py \
		--framework faster_rcnn \
		--mode train \
		--outname resnet101_3class_vehicles \
		--label_map_path /simrdwn/data/class_labels_airplane_boat_car.pbtxt \
		--tf_cfg_train_file /raid/simrdwn/configs/faster_rcnn_resnet101_simrdwn.config \
		--train_tf_record /raid/simrdwn/data/labels_airplane_boat_car_train.tfrecord \
		--max_batches 30000 \
		--batch_size 16 \
		--gpu 0
		

The training script will create a results directory in _/simrdwn/results_ with the filename [framework] + [outname] + [date].  Since one cannot run TensorBoard with YOLT, we include scripts _/simrdwn/core/yolt_plot_loss.py_ and _/simrdwn/core/tf_plot_loss.py_ thatcan be called during training to inspect model convergence.  An example convergence plot is shown below.
![Alt text](/results/__examples/tf_loss_plot.png?raw=true "Figure 1")

____

## 3. Test

During the test phase, input images of arbitrary size are processed.  

1.	Slice test images into the window size used in training.
2.  Run inference on windows with the desired model
3.  Stitch windows back together to create original test image
4.  Run non-max suppression on overlapping predictions
5.  Make plots of predictions (optional)

<a/>

	
	# SSD vehicle search
	python /raid/simrdwn/core/simrdwn.py \
		--framework ssd \
		--mode valid \
		--outname inception_v2_cowc \
		--label_map_path /raid/simrdwn/data/class_labels_car.pbtxt \
		--train_model_path train_ssd_inception_v2_cowc_2019_01_07_09-24-04 \
		--valid_testims_dir cowc/Utah_AGRC  \
		--use_tfrecords 0 \
		--min_retain_prob=0.15 \
		--keep_valid_slices 0 \
		--slice_overlap 0.1 \
		--slice_sizes_str 544 \
		--valid_slice_sep __ \
		--plot_thresh_str 0.5 \
		--valid_make_legend_and_title 0 \
		--edge_buffer_valid 1 \
		--valid_box_rescale_frac 1 \
		--alpha_scaling 1 \
		--show_labels 0
		
	# YOLT vehicle search
	python /raid/simrdwn/core/simrdwn.py \
		--framework yolt \
		--mode valid \
		--outname dense_cowc \
		--yolt_object_labels_str car \
		--train_model_path train_yolt_dense_cowc_2019_01_07_09-13-22 \
		--weight_file yolt_20000_tmp.weights \
		--yolt_cfg_file yolt.cfg \
		--valid_testims_dir cowc/Utah_AGRC  \
		--use_tfrecords 0 \
		--min_retain_prob=0.15 \
		--keep_valid_slices 0 \
		--slice_overlap 0.1 \
		--slice_sizes_str 544 \
		--valid_slice_sep __ \
		--plot_thresh_str 0.2 \
		--valid_make_legend_and_title 0 \
		--edge_buffer_valid 1 \
		--valid_box_rescale_frac 1 \
		--alpha_scaling 1 \
		--show_labels 0

Outputs will be something akin to the images below.  The _alpha\_scaling_ flag makes the bounding box opacity proportional to prediction confidence, and the _show\_labels_ flag prints the object class at the top of the bounding box.
![Alt text](/results/__examples/ex0.png?raw=true "Figure 2")
![Alt text](/results/__examples/ex1.png?raw=true "Figure 3")
	
	
____	
_If you plan on using SIMRDWN in your work, please consider citing [YOLO](https://arxiv.org/abs/1612.08242), the [TensorFlow Object Detection API](https://arxiv.org/abs/1611.10012), [YOLT](https://arxiv.org/abs/1805.09512), and [SIMRDWN](https://arxiv.org/abs/1809.09978)._
