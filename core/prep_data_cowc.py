#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 06:14:20 2019

@author: avanetten

Parse COWC dataset for SIMRDWN training

Data located at:
    https://gdo152.llnl.gov/cowc/
    cd /raid/data/
    wget -r -np  ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets

"""

import os
import sys
import shutil
import numpy as np

path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_simrdwn_core)
import preprocess_tfrecords
import yolt_data_prep_funcs
import parse_cowc


###############################################################################
# path variables (may need to be edited! )
cowc_data_dir = '/raid/data/gdo152.ucllnl.org/cowc/datasets/ground_truth_sets/'
simrdwn_data_dir = '/raid/simrdwn/data/'
train_out_dir = '/raid/simrdwn/training_datasets/cowc/'
test_out_dir = '/raid/simrdwn/test_images/cowc/'
label_map_file = 'class_labels_car.pbtxt'
verbose = True

label_map_path = os.path.join(simrdwn_data_dir, label_map_file)

################################################################################
#### Construct argument parser
#import argparse
#parser = argparse.ArgumentParser()
#
## general settings
#parser.add_argument('--cowc_data_dir', type=str, default='/raid/data/cowc/datasets/ground_truth_sets',
#                    help="Location of  ground truth labels")
#parser.add_argument('--simrdwn_data_dir', type=str, default='/raid/local/src/simrdwn/data/',
#                    help="location to create training data lists and records")
#parser.add_argument('--train_out_dir', type=str, default='/raid/local/src/simrdwn/training_datasets/cowc/',
#                    help="Location of output for sliced images")
#parser.add_argument('--label_map_file', type=str, default='class_labels_car.pbtxt',
#                    help="File (in simrdwn_data_dir) with class labels")
##parser.add_argument('--annotation_suffix', type=str, default='_Annotated_Cars.png',
##                    help="Suffix of annoation files")
##parser.add_argument('--input_box_size', type=int, default=10,
##                    help="Default input car size, in pixels")
#parser.add_argument('--verbose', type=int, default=0,
#                    help="verbose switch")
#args = parser.parse_args()
#verbose = bool(args.verbose)
#
## set total label_map_path
#label_map_path = os.path.join(args.simrdwn_data_dir, args.label_map_file)
#simrdwn_data_dir = args.simrdwn_data_dir
#cowc_data_dir = args.cowc_data_dir
#train_out_dir = args.train_out_dir

##############################
# list of train and test directories
# for now skip Columbus and Vahingen since they are grayscale
ground_truth_dir = cowc_data_dir #os.path.join(args.cowc_data_dir, 'datasets/ground_truth_sets/')
train_dirs = ['Potsdam_ISPRS', 'Selwyn_LINZ', 'Toronto_ISPRS']
test_dirs = ['Utah_AGRC']
annotation_suffix = '_Annotated_Cars.png'
##############################

##############################
# infer training output paths
labels_dir = os.path.join(train_out_dir, 'labels/')
images_dir = os.path.join(train_out_dir, 'images/')
im_list_name = os.path.join(train_out_dir, 'cowc_yolt_train_list.txt')
tfrecord_train = os.path.join(train_out_dir, 'cowc_train.tfrecord')
sample_label_vis_dir = os.path.join(train_out_dir, 'sample_label_vis/')
#im_locs_for_list = output_loc + train_name + '/' + 'training_data/images/'
#train_images_list_file_loc = yolt_dir + 'data/'
# create output dirs
for d in [train_out_dir, test_out_dir, labels_dir, images_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
##############################

##############################
# set yolt training box size
car_size = 3      # meters
GSD = 0.15        # meters
yolt_box_size = np.rint(car_size/GSD) # size in pixels
print ("yolt_box_size (pixels):", yolt_box_size)
##############################

##############################
# slicing variables
slice_overlap = 0.1
zero_frac_thresh = 0.2
sliceHeight, sliceWidth = 544, 544 # for for 82m windows
##############################

##############################
# set yolt category params from pbtxt
label_map_dict = preprocess_tfrecords.load_pbtxt(label_map_path, verbose=False)
print ("label_map_dict:", label_map_dict)
# get ordered keys
key_list = sorted(label_map_dict.keys())
category_num = len(key_list)
# category list for yolt
cat_list = [label_map_dict[k] for k in key_list]
print ("cat list:", cat_list)
yolt_cat_str = ','.join(cat_list)
print ("yolt cat str:", yolt_cat_str)
# create yolt_category dictionary (should start at 0, not 1!)
yolt_cat_dict = {x:i for i,x in enumerate(cat_list)}
print ("yolt_cat_dict:", yolt_cat_dict)
# conversion between yolt and pbtxt numbers (just increase number by 1)
convert_dict = {x:x+1 for x in range(100)}
print ("convert_dict:", convert_dict)
##############################


##############################
# Slice large images into smaller chunks 
##############################
if os.path.exists(im_list_name):
    run_slice = False
else:
    run_slice = True

for i,d in enumerate(train_dirs):
    dtot = os.path.join(ground_truth_dir, d)
    print ("dtot:", dtot)

    # get label files
    files = os.listdir(dtot)
    annotate_files = [f for f in files if f.endswith(annotation_suffix)]
    #print ("annotate_files:", annotate_files
    
    for annotate_file in annotate_files:
        annotate_file_tot = os.path.join(dtot, annotate_file)
        name_root = annotate_file.split(annotation_suffix)[0]
        imfile = name_root + '.png'
        imfile_tot = os.path.join(dtot, imfile)
        outroot = d + '_' + imfile.split('.')[0]
        print ("\nName_root", name_root)
        #print ("annotate_file:", annotate_file
        #print ("  imfile:", imfile
        #print ("  imfile_tot:", imfile_tot
        print ("  outroot:", outroot)
            
        if run_slice:
            parse_cowc.slice_im_cowc(imfile_tot, annotate_file_tot, outroot, 
                     images_dir, labels_dir, yolt_cat_dict, cat_list[0], yolt_box_size, 
                     sliceHeight=sliceHeight, sliceWidth=sliceWidth, 
                     zero_frac_thresh=zero_frac_thresh, overlap=slice_overlap, 
                     pad=0, verbose=verbose)
##############################

##############################
# Get list for simrdwn/data/, copy to data dir
##############################
train_ims = [os.path.join(images_dir, f) for f in  os.listdir(images_dir)]
f = open(im_list_name, 'wb')
for item in train_ims:
    f.write("%s\n" % item)
f.close()
# copy to data dir
shutil.copy(im_list_name, simrdwn_data_dir)
##############################

##############################
# Ensure labels were created correctly by plotting a few
##############################
max_plots=50
thickness=2
yolt_data_prep_funcs.plot_training_bboxes(labels_dir, images_dir, ignore_augment=False,
                     sample_label_vis_dir=sample_label_vis_dir, 
                     max_plots=max_plots, thickness=thickness, ext='.png')





##############################
# Make a .tfrecords file
##############################
preprocess_tfrecords.yolt_imlist_to_tf(im_list_name, label_map_dict, 
                                       TF_RecordPath=tfrecord_train,
                                       TF_PathVal='', val_frac=0.0, 
                                       convert_dict=convert_dict, verbose=True)
# copy train file to data dir
shutil.copy(tfrecord_train, simrdwn_data_dir)


##############################
# Copy test images to test dir
print ("Copying test images to:", test_out_dir)
for td in test_dirs:
    td_tot_in = os.path.join(ground_truth_dir, td)
    td_tot_out = os.path.join(test_out_dir, td)
    if not os.path.exists(td_tot_out):
        os.makedirs(td_tot_out)
    # copy non-label files
    for f in os.listdir(td_tot_in):
        if f.endswith('.png') and not f.endswith(('_Cars.png', '_Negatives.png', '.xcf')):
            shutil.copy2(os.path.join(td_tot_in, f), td_tot_out)
    # copy everything?
    #os.system('cp -r ' + td_tot + ' ' + test_out_dir)
    ##shutil.copytree(td_tot, test_out_dir)
##############################
