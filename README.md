# SIMRDWN #


![Alt text](/results/__examples/header.jpg?raw=true "")

The Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN) codebase combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery.  This work seeks to extend the [YOLT](https://arxiv.org/abs/1805.09512) modification of [YOLO](https://pjreddie.com/darknet/yolo/) to include the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).  Therefore, one can train models and test on arbitrary image sizes with [YOLO (versions 2 and 3)](https://pjreddie.com/darknet/yolo/), [Faster R-CNN](https://arxiv.org/abs/1506.01497), [SSD](https://arxiv.org/abs/1512.02325), or [R-FCN](https://arxiv.org/abs/1605.06409).  

### For more information, see:

1. Our arXiv paper: [Satellite Imagery Multiscale Rapid Detection with Windowed Networks](https://arxiv.org/abs/1809.09978)

2. Our [blog](https://medium.com/the-downlinq) (e.g. [1](https://medium.com/the-downlinq/simrdwn-adapting-multiple-object-detection-frameworks-for-satellite-imagery-applications-991dbf3d022b), [2](https://medium.com/the-downlinq/giving-simrdwn-a-spin-part-i-7032d7bf120a))

2. Our original [YOLT paper](https://arxiv.org/abs/1805.09512)

3. The original [YOLT repository](https://github.com/CosmiQ/yolt) (now deprecated)
 

____
## Running SIMRDWN

____

### 0. Installation

SIMRDWN is built to execute within a docker container on a GPU-enabled machine.  The docker command creates an Ubuntu 16.04 image with CUDA 9.0, python 3.6, and tensorflow-gpu version 1.13.1. 

1. Clone this repository (e.g. to _/simrdwn_)

2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
 
3. Build docker file.

		cd /simrdwn/docker
		nvidia-docker build --no-cache -t simrdwn .
	
4. Spin up the docker container (see the [docker docs](https://docs.docker.com/engine/reference/commandline/run/) for options) 

        nvidia-docker run -it -v /simrdwn:/simrdwn --name simrdwn_container0 simrdwn
	
5. Compile the Darknet C program for both YOLT2 and YOLT3.
      
	    cd /simrdwn/yolt2
	    make
	    cd /simrdwn/yolt3
	    make

6. Get help on SIMRDWN options
	
		python /simrdwn/simrdwn/core/simrdwn.py --help
	

____

### 1. Prepare Training Data



####  1A. Create YOLT Format

Training data needs to be transformed to the YOLO format of training images in an "images" folder and bounding box labels in a "labels" folder.  For example, an image "images/ex0.png" has a corresponding label "labels/ex0.txt". Labels are bounding boxes of the form 

    <object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height.  Running a script such as _/simrdwn/data\_prep_/parse\_cowc.py_ extracts training windows of reasonable size (usually 416 or 544 pixels in extent) from large labeleled images of the [COWC](https://gdo152.llnl.gov/cowc/) dataset.  The script then transforms the labels corresponding to these windows into the correct format and creates a list of all training input images in _/data/train\_data/training\_list.txt_.  We also need to define the object classes with a .pbtxt file, such as _/data/training\_data/class\_labels\_car.pbtxt_.  Class integers should be 1-indexed in the .pbtxt file.

####  1B. Create .tfrecord (optional)
If the tensorflow object detection API models are being run, we must transform the training data into the .tfrecord format.  This is accomplished via the _simrdwn/core/preprocess\_tfrecords.py_ script.
	
	python /simrdwn/core/preprocess_tfrecords.py \
	    --image_list_file /simrdwn/data/cowc_labels_car_list.txt \
	    --pbtxt_filename /simrdwn/data/class_labels_car.pbtxt \
	    --outfile /simrdwn/data/cowc_labels_car_train.tfrecord \
	    --outfile_val /simrdwn/data/cowc_labels_car_val.tfrecord \
	    --val_frac 0.1

____

### 2. Train

We can train either YOLT models or tensorflow object detection API models.  If we are using tensorflow, the config file may need to be updated in the _/simrdwn/configs_ directory (further example config files reside [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)).
Training can be run with commands such as:

	# SSD vehicle search
	python /simrdwn/core/simrdwn.py \
		--framework ssd \
		--mode train \
		--outname inception_v2_cowc \
		--label_map_path /simrdwn/data/class_labels_car.pbtxt \
		--tf_cfg_train_file _altered_v0/ssd_inception_v2_simrdwn.config \
		--train_tf_record cowc/cowc_train.tfrecord \
		--max_batches 30000 \
		--batch_size 16 
	
	# YOLT vechicle search
	python /simrdwn/core/simrdwn.py \
		--framework yolt2 \
		--mode train \
		--outname dense_cars \
		--yolt_cfg_file ave_dense.cfg  \
		--weight_file yolo.weights \
		--yolt_train_images_list_file cowc_yolt_train_list.txt \
		--label_map_path class_labels_car.pbtxt \
		--max_batches 30000 \
		--batch_size 64 \
		--subdivisions 16

____

### 3. Test

During the test phase, input images of arbitrary size are processed.  

1.	Slice test images into the window size used in training.
2.  Run inference on windows with the desired model
3.  Stitch windows back together to create original test image
4.  Run non-max suppression on overlapping predictions
5.  Make plots of predictions (optional)

	
	
		# SSD vehicle search
		python /raid/local/src/simrdwn/src/simrdwn.py \
			--framework ssd \
			--mode test \
			--outname inception_v2_cowc \
			--label_map_path class_labels_car.pbtxt \
			--train_model_path [ssd_train_path] \
			--tf_cfg_train_file ssd_inception_v2_simrdwn.config \
			--use_tfrecords=0 \
			--testims_dir cowc/Utah_AGRC  \
			--keep_test_slices 0 \
			--test_slice_sep __ \
			--test_make_legend_and_title 0 \
			--edge_buffer_test 1 \
			--test_box_rescale_frac 1 \
			--plot_thresh_str 0.2 \
			--slice_sizes_str 416 \
			--slice_overlap 0.2 \
			--alpha_scaling 1 \
			--show_labels 0
				
		# YOLT vehicle search
		python /raid/local/src/simrdwn/core/simrdwn.py \
			--framework yolt2 \
			--mode test \
			--outname dense_cowc \
			--label_map_path class_labels_car.pbtxt \
			--train_model_path [yolt2_train_path] \
			--weight_file ave_dense_final.weights \
			--yolt_cfg_file ave_dense.cfg \
			--testims_dir cowc/Utah_AGRC  \
			--keep_test_slices 0 \
			--test_slice_sep __ \
			--test_make_legend_and_title 0 \
			--edge_buffer_test 1 \
			--test_box_rescale_frac 1 \
			--plot_thresh_str 0.2 \
			--slice_sizes_str 416 \
			--slice_overlap 0.2 \
			--alpha_scaling 1 \
			--show_labels 1
	
	Outputs will be something akin to the images below.  The _alpha\_scaling_ flag makes the bounding box opacity proportional to prediction confidence, and the _show\_labels_ flag prints the object class at the top of the bounding box.
	![Alt text](/results/__examples/ex0.png?raw=true "Figure 1")
	![Alt text](/results/__examples/ex1.png?raw=true "Figure 2")
	
	
	
_If you plan on using SIMRDWN in your work, please consider citing [YOLO](https://arxiv.org/abs/1612.08242), the [TensorFlow Object Detection API](https://arxiv.org/abs/1611.10012), [YOLT](https://arxiv.org/abs/1805.09512), and [SIMRDWN](https://arxiv.org/abs/1809.09978)._


