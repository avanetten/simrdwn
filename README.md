# SIMRDWN (python 2.7) #


![Alt text](/results/__examples/header.jpg?raw=true "")

The Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN) codebase combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery.  This work seeks to extend the [YOLT](https://arxiv.org/abs/1805.09512) modification of [YOLO](https://pjreddie.com/darknet/yolo/) to include the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).  Therefore, one can train models and test on arbitrary image sizes with [YOLO](https://pjreddie.com/darknet/yolo/), [Faster R-CNN](https://arxiv.org/abs/1506.01497), [SSD](https://arxiv.org/abs/1512.02325), or [R-FCN](https://arxiv.org/abs/1605.06409).  

### For more information, see:

1. Our arXiv paper: [Satellite Imagery Multiscale Rapid Detection with Windowed Networks](https://arxiv.org/abs/1809.09978)

2. Our original [YOLT paper](https://arxiv.org/abs/1805.09512)

3. The original [YOLT repository](https://github.com/CosmiQ/yolt) (now deprecated)
 

____
## Running SIMRDWN

____

### 0. Create Docker file

All commands should be run in the docker file, create it via the following commands

	cd /simrdwn/docker
	nvidia-docker build --no-cache -t simrdwn .
	nvidia-docker run -it -v /raid:/raid --name simrdwn_train_gpu0 simrdwn
	

Compile the Darknet C program

	cd /simrdwn/yolt
	make


____

### 1. Prepare Training Data



####  1A. Create YOLT Format

Training data needs to be transformed to the YOLO format of training images in an "images" folder and bounding box labels in a "labels" folder.  For example, an image "images/ex0.png" has a corresponding label "labels/ex0.txt". Labels are bounding boxes of the form 

    <object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height.  Running a script such as _/simrdwn/core/parse\_cowc.py_ extracts training windows of reasonable size (usually 416 or 544 pixels in extent) from large labeleled images of the [COWC](https://gdo152.llnl.gov/cowc/) dataset.  The script then transforms the labels corresponding to these windows into the correct format and creates a list of all training input images in _/simdwn/data/training\_list.txt_.

We also need to define the object classes with a .pbtxt file, such as _/simrdwn/data/class\_labels\_car.pbtxt_

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
		--outname inception_v2_3class_vehicles \
		--label_map_path /simrdwn/data/class_labels_airplane_boat_car.pbtxt \
		--tf_cfg_train_file /simrdwn/configs/_altered_v0/ssd_inception_v2_simrdwn.config \
		--train_tf_record /simrdwn/data/labels_airplane_boat_car_train.tfrecord \
		--max_batches 30000 \
		--batch_size 16 \
		--gpu 0
		
	# YOLT vechicle search
	python /simrdwn/core/simrdwn.py \
		--framework yolt \
		--mode train \
		--outname dense_3class_vehicles \
		--yolt_object_labels_str tmp0,airplane,boat,car \
		--yolt_cfg_file ave_dense.cfg  \
		--weight_dir /simrdwn/yolt/input_weights \
		--weight_file yolo.weights \
		--yolt_train_images_list_file labels_airplane_boat_car_list.txt \
		--label_map_path /simrdwn/data/class_labels_airplane_boat_car.pbtxt \
		--nbands 3 \
		--max_batches 30000 \
		--batch_size 64 \
		--subdivisions 16 \
		--gpu 0


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
			--mode valid \
			--outname inception_v2_3class_vehicles \
			--pbtxt_filename '/simrdwn/data/class_labels_airplane_boat_car.pbtxt' \
			--inference_graph_path '/simrdwn/results/ssd_output_inference_graph/frozen_inference_graph.pb' \
			--valid_testims_dir 'vechicle_test'  \
			--keep_valid_slices 0 \
			--valid_slice_sep __ \
			--valid_make_legend_and_title 0 \
			--edge_buffer_valid 1 \
			--valid_box_rescale_frac 1 \
			--plot_thresh_str 0.2 \
			--slice_sizes_str 416 \
			--slice_overlap 0.2 \
			--alpha_scaling 0 \
			--show_labels 0
	
			
		# YOLT vehicle search
		python /raid/local/src/simrdwn/core/simrdwn.py \
			--framework yolt \
			--mode valid \
			--outname dense_3class_vehicles \
			--yolt_object_labels_str tmp0,airplane,tmp2,boat,tmp4,car \
			--train_model_path train_yolt_3class \
			--weight_file ave_dense_final.weights \
			--yolt_cfg_file ave_dense.cfg \
			--valid_testims_dir 'vechicle_test'  \
			--keep_valid_slices 0 \
			--valid_slice_sep __ \
			--valid_make_legend_and_title 0 \
			--edge_buffer_valid 1 \
			--valid_box_rescale_frac 1 \
			--plot_thresh_str 0.2 \
			--slice_sizes_str 416 \
			--slice_overlap 0.2 \
			--alpha_scaling 1 \
			--show_labels 1
	
	Outputs will be something akin to the images below.  The _alpha\_scaling_ flag makes the bounding box opacity proportional to prediction confidence, and the _show\_labels_ flag prints the object class at the top of the bounding box.
	![Alt text](/results/__examples/ex0.png?raw=true "Figure 1")
	![Alt text](/results/__examples/ex1.png?raw=true "Figure 2")
	
	
	
_If you plan on using SIMRDWN in your work, please consider citing [YOLO](https://arxiv.org/abs/1612.08242), the [TensorFlow Object Detection API](https://arxiv.org/abs/1611.10012), [YOLT](https://arxiv.org/abs/1805.09512), and [SIMRDWN](https://arxiv.org/abs/1809.09978)._
