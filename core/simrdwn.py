#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

# run nvidia-docker nteractive shell 
nvidia-docker run -it -v /raid:/raid —name yolt_name darknet 

"""

from __future__ import print_function
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import argparse
import shutil
import copy
#import cv2
#import csv
#import pickle
#from osgeo import ogr
#from subprocess import Popen, PIPE, STDOUT
#import math
#import random
#from collections import OrderedDict
#import matplotlib.pyplot as plt

##########################
# import slice_im, post_process scripts
path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_simrdwn_core)
import slice_im
import preprocess_tfrecords
import parse_tfrecord
import post_process
import utils
#import export_model
#import parse_tfrecord
#import yolt_post_process
sys.stdout.flush()
##########################

###############################################################################
def update_args(args):
    '''Construct inferred values'''
    
    ###########################################################################
    # CONSTRUCT INFERRED VALUES
    ###########################################################################
    
    ##########################
    # GLOBAL VALUES
    # set directory structure

    # append '/' to end of simrdwn_dir
    #if not args.yolt_dir.endswith('/'): args.yolt_dir += '/'
    #if not args.simrdwn_dir.endswith('/'): args.simrdwn_dir += '/'
    #args.src_dir = os.path.join(args.simrdwn_dir, 'core')
    args.src_dir = os.path.dirname(os.path.realpath(__file__)) 
    args.simrdwn_dir = os.path.dirname(args.src_dir) 
    args.this_file = os.path.join(args.src_dir, 'simrdwn.py')
    args.results_topdir = os.path.join(args.simrdwn_dir, 'results')
    args.test_images_dir = os.path.join(args.simrdwn_dir, 'test_images')
    args.data_dir = os.path.join(args.simrdwn_dir, 'data')
    args.yolt_dir = os.path.join(args.simrdwn_dir, 'yolt')
    args.yolt_weight_dir = os.path.join(args.yolt_dir, 'input_weights')
    args.yolt_cfg_dir = os.path.join(args.yolt_dir, 'cfg')
    args.yolt_plot_file = os.path.join(args.src_dir, 'yolt_plot_loss.py')
    args.tf_plot_file = os.path.join(args.src_dir, 'tf_plot_loss.py')

    ##########################################
    # Get datetime and set outlog file
    args.now = datetime.datetime.now()
    args.date_string = args.now.strftime('%Y_%m_%d_%H-%M-%S')
    #print "Date string:", date_string
    args.res_name = args.mode + '_' + args.framework + '_' + args.outname \
                    + '_' + args.date_string
    args.results_dir = os.path.join(args.results_topdir, args.res_name) 
    args.log_dir = os.path.join(args.results_dir, 'logs')
    args.log_file = os.path.join(args.log_dir, args.res_name + '.log')
    args.yolt_loss_file = os.path.join(args.log_dir, 'yolt_loss.txt')
    args.labels_log_file = os.path.join(args.log_dir, 'labels_list.txt') 
    
    #args.valid_make_pngs = bool(args.valid_make_pngs)
    args.valid_make_legend_and_title = bool(args.valid_make_legend_and_title)
    args.keep_valid_slices = bool(args.keep_valid_slices)

    ##########################
    # set possible extensions for image files
    args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG', 
                           '.jpg', '.JPEG', '.jpeg']    
    # set cuda values
    if args.gpu >= 0:
        args.use_GPU, args.use_CUDNN = 1, 1
    else:
        args.use_GPU, args.use_CUDNN = 0, 0
     
    # make label_map_dic (key=int, value=str), and reverse
    if len(args.label_map_path) > 0:
        args.label_map_dict = preprocess_tfrecords.load_pbtxt(args.label_map_path, verbose=False)
    else:
        args.label_map_dict = {}
    args.label_map_dict_rev = {v: k for k,v in args.label_map_dict.iteritems()}
    
    # infer lists from args
    if len(args.yolt_object_labels_str) == 0:
        args.yolt_object_labels = [args.label_map_dict[ktmp] for ktmp in 
                              sorted(args.label_map_dict.keys())]
        args.yolt_object_labels_str = ','.join(args.yolt_object_labels)
    else:
        args.yolt_object_labels = args.yolt_object_labels_str.split(',')
        # also set label_map_dict, if it's empty
        if len(args.label_map_path) == 0:
            for itmp,val in enumerate(args.yolt_object_labels):
                args.label_map_dict[itmp] = val
            args.label_map_dict_rev = {v: k for k,v in args.label_map_dict.iteritems()}

    # set total dict
    args.label_map_dict_tot = copy.deepcopy(args.label_map_dict)
    args.label_map_dict_rev_tot = copy.deepcopy(args.label_map_dict_rev)

    
    args.yolt_classnum = len(args.yolt_object_labels)
    args.yolt_final_output = 1 * 1 * args.boxes_per_grid * (args.yolt_classnum + 4 + 1)

    # plot thresh and slice sizes
    args.plot_thresh = np.array(args.plot_thresh_str.split(args.str_delim)).astype(float)
    args.slice_sizes = np.array(args.slice_sizes_str.split(args.str_delim)).astype(int)

    # weight file and yolt cfg file
    #args.weight_file_tot = os.path.join(args.weight_dir, args.weight_file)
    if args.mode.upper() == 'TRAIN':
        args.weight_file_tot = os.path.join(args.yolt_weight_dir, args.weight_file)
    else:    
        args.weight_file_tot = os.path.join(args.results_topdir, args.train_model_path + '/' + args.weight_file)
    args.yolt_cfg_file_tot = os.path.join(args.log_dir, args.yolt_cfg_file)
    if args.mode == 'valid':
        # assume weights and cfg are in the training dir
        args.yolt_cfg_file_in = os.path.join(os.path.dirname(args.weight_file_tot), 'logs/', args.yolt_cfg_file)
    else:
        # assume weights are in weight_dir, and cfg in cfg_dir
        args.yolt_cfg_file_in = os.path.join(args.yolt_cfg_dir, args.yolt_cfg_file)


    ##########################   
    # set training files
    args.yolt_train_images_list_file_tot = os.path.join(args.data_dir, args.yolt_train_images_list_file)
    # set tf cfg file out
    tf_cfg_base = os.path.basename(args.tf_cfg_train_file)
    #tf_cfg_root = tf_cfg_base.split('.')[0]
    args.tf_cfg_train_file_out = os.path.join(args.results_dir, tf_cfg_base)
    args.tf_model_output_directory = os.path.join(args.results_dir, 'frozen_model')
    #args.tf_model_output_directory = os.path.join(args.results_dir, tf_cfg_root)

    ##########################
    # set validation files
    # first prepend paths to directories
    #args.valid_weight_dir_tot = os.path.join(args.results_topdir, args.valid_weight_dir)
    #args.valid_weight_file_tot = args.valid_weight_dir + args.valid_weight_file
    
    # keep raw testims dir if it starts with a '/'
    if args.valid_testims_dir.startswith('/'):
        args.valid_testims_dir_tot = args.valid_testims_dir
    else:
        args.valid_testims_dir_tot = os.path.join(args.test_images_dir, args.valid_testims_dir)
        
    print ("os.listdir(args.valid_testims_dir_tot:", os.listdir(args.valid_testims_dir_tot))

    # set test list 
    try:
        if args.nbands == 3:
            print ("os.listdir(args.valid_testims_dir_tot:", os.listdir(args.valid_testims_dir_tot))
            args.valid_ims_list = [f for f in os.listdir(args.valid_testims_dir_tot) \
                                       if f.endswith(tuple(args.extension_list))]
            print ("args.valid_ims_list:", args.valid_ims_list)
        else:
            args.valid_ims_list = [f for f in os.listdir(args.valid_testims_dir_tot) \
                                       if f.endswith('#1.png')]
    except:
        args.valid_ims_list = []
    # more validation files
    args.rotate_boxes = bool(args.rotate_boxes)
    args.yolt_valid_classes_files = [os.path.join(args.results_dir, l + '.txt') \
                                           for l in args.yolt_object_labels]
    
    # set total location of validation image file list
    args.valid_presliced_list_tot = os.path.join(args.simrdwn_dir, args.valid_presliced_list)
    if len(args.valid_presliced_tfrecord_part) > 0:
        args.valid_presliced_tfrecord_tot = os.path.join(args.simrdwn_dir, 
                                                 args.valid_presliced_tfrecord_part 
                                                 + '/valid_splitims.tfrecord')
        args.valid_tfrecord_out = os.path.join(args.results_dir, 'predictions.tfrecord')
    else:
        args.valid_presliced_tfrecord_tot, args.valid_tfrecord_out  = '', ''
    
    if len(args.valid_presliced_list) > 0:
        args.valid_splitims_locs_file = args.valid_presliced_list_tot
    else:
        args.valid_splitims_locs_file = os.path.join(args.results_dir, args.valid_splitims_locs_file_root)      
    #args.valid_tfrecord_file = os.path.join(args.results_dir, args.valid_tfrecord_root)
    #args.val_prediction_pkl = os.path.join(args.results_dir, args.valid_prediction_pkl_root)
    #args.val_df_tfrecords_out = os.path.join(args.results_dir, 'predictions.tfrecord')
    args.val_df_path_init = os.path.join(args.results_dir, args.val_df_root_init)
    args.val_df_path_aug = os.path.join(args.results_dir, args.val_df_root_aug)

    args.inference_graph_path_tot = os.path.join(args.results_topdir, 
                                                args.train_model_path \
                                                + '/frozen_model/frozen_inference_graph.pb')
    ##########################
    # get second validation classifier values
    args.slice_sizes2 = []
    if len(args.label_map_path2) > 0:

        # label dict
        args.label_map_dict2 = preprocess_tfrecords.load_pbtxt(args.label_map_path2, verbose=False)
        args.label_map_dict_rev2 = {v: k for k,v in args.label_map_dict2.iteritems()}
        
        # to update label_map_dict just adds second classifier to first
        nmax_tmp = max(args.label_map_dict.keys())
        for ktmp, vtmp in args.label_map_dict2.iteritems():
            args.label_map_dict_tot[ktmp+nmax_tmp] = vtmp
        args.label_map_dict_rev_tot = {v: k for k,v in args.label_map_dict_tot.iteritems()}
        
        # infer lists from args
        args.yolt_object_labels2 = [args.label_map_dict2[ktmp] for ktmp in sorted(args.label_map_dict2.keys())]
        args.yolt_object_labels_str2 = ','.join(args.yolt_object_labels2)

        # set classnum and final output
        args.classnum2 = len(args.yolt_object_labels2)
        args.yolt_final_output2 = 1 * 1 * args.boxes_per_grid * (args.classnum2 + 4 + 1)

        # plot thresh and slice sizes
        args.plot_thresh2 = np.array(args.plot_thresh_str2.split(args.str_delim)).astype(float)
        args.slice_sizes2 = np.array(args.slice_sizes_str2.split(args.str_delim)).astype(int)

        # validation files2
        args.yolt_valid_classes_files2 = [os.path.join(args.results_dir, l + '.txt') \
                                               for l in args.yolt_object_labels2]
        if len(args.valid_presliced_list2) > 0:
            args.valid_presliced_list_tot2 = os.path.join(args.simrdwn_dir, args.valid_presliced_list2)
        else:
            args.valid_splitims_locs_file2 = os.path.join(args.results_dir, args.valid_splitims_locs_file_root2)      
        #args.val_prediction_pkl2 = os.path.join(args.results_dir, args.valid_prediction_pkl_root2)
        #args.val_df_tfrecords_out2 = os.path.join(args.results_dir, 'predictions2.tfrecord')
        args.valid_tfrecord_out2 = os.path.join(args.results_dir, 'predictions2.tfrecord')
        args.val_df_path_init2 = os.path.join(args.results_dir, args.val_df_root_init2)
        args.val_df_path_aug2 = os.path.join(args.results_dir, args.val_df_root_aug2)
        args.weight_file_tot2 = os.path.join(args.results_topdir, args.train_model_path + '/' + args.weight_file2)
        #args.weight_file_tot2 = os.path.join(args.train_model_path2, args.weight_file2)
        #args.weight_file_tot2 = os.path.join(args.weight_dir2, args.weight_file2)
        args.yolt_cfg_file_tot2 = os.path.join(args.log_dir, args.yolt_cfg_file2)

        if args.mode == 'valid':
            args.yolt_cfg_file_in2 = os.path.join(os.path.dirname(args.weight_file_tot2), 'logs/', args.yolt_cfg_file2)
        else:
            args.yolt_cfg_file_in2 = os.path.join(args.yolt_cfg_dir, args.yolt_cfg_file2)

        args.inference_graph_path_tot2 = os.path.join(args.results_topdir, 
                                                args.train_model_path2 \
                                                + '/frozen_model/frozen_inference_graph.pb')

    
    # total validation
    args.val_df_path_tot = os.path.join(args.results_dir, args.val_df_root_tot)
    #args.val_prediction_pkl_tot = os.path.join(args.results_dir, args.valid_prediction_pkl_root_tot)
    args.val_prediction_df_refine_tot = os.path.join(args.results_dir, 
                                                     args.val_prediction_df_refine_tot_root_part +
                                                     '_thresh=' + str(args.plot_thresh[0])) 

    ##########################
    # Plotting params
    args.figsize = (12,12)
    args.dpi = 300

    ##########################
    # yolt test settings, assume test images are in test_images
    args.yolt_test_im_tot = os.path.join(args.test_images_dir, args.yolt_test_im)
    #args.testweight_file = args.weight_dir + args.testweight_file
    # populate test_labels
    test_labels_list = []
    if args.mode == 'test':
        # populate labels
        with open(os.path.join(args.data_dir, args.test_labels), 'rb') as fin:
            for l in fin.readlines():
                # spaces in names screws up the argc in yolt.c
                test_labels_list.append(l[:-1].replace(' ', '_'))
        # overwrite yolt_object_labels, and yolt_object_labels_str
        args.yolt_object_labels = test_labels_list
        args.yolt_object_labels_str = ','.join([str(ltmp) for ltmp in \
                                           test_labels_list])
        
    # set side length from cfg root, classnum, and final output
    # default to length of 13
    #try:
    #    args.side = int(args.yolt_cfg_file.split('.')[0].split('x')[-1])  # Grid size (e.g.: side=20 gives a 20x20 grid)
    #except:
    #    args.side = 13
           
    # set yolt_cfg_file, assume raw cfgs are in cfg directory, and the cfg file
    #   will be copied to log_dir.  Use the cfg in logs as input to yolt.c
    #   with valid, cfg will be in results_dir/logs/
    # if using valid, assume cfg file is in valid_weight_dir, else, assume
    #   it's in yolt_dir/cfg/
    # also set weight file
    

    ## set batch size based on network size?
    #if args.side >= 30:
    #    args.batch_size = 16                 # batch size (64 for 14x14, 32 for 28x28)
    #    args.subdivisions = 8               # subdivisions per batch (8 for 14x14 [yields 8 images per batch], 32 for 28x28)
    #elif args.side >= 16:
    #    args.batch_size = 64                 # batch size (64 for 14x14, 32 for 28x28)
    #    args.subdivisions = 16               # subdivisions per batch (8 for 14x14 [yields 8 images per batch], 32 for 28x28)
    #else:
    #    args.batch_size = 64
    #    args.subdivisions = 8

        
    ## initialize val files to empty
    #figname_val = ''#results_dir + valid_image + '_valid_thresh=' + str(plot_thresh) + '.png'
    #pkl_val = ''#results_dir + valid_image + '_boxes_thresh=' + str(plot_thresh) + '.pkl'
    #valid_image = ''
    #valid_im,  valid_files, yolt_valid_classes_files = [],[],[]
    #valid_dir, valid_splitims_locs_file = '', ''
    ###########################
    
    return args


###############################################################################
def update_tf_train_config(config_file_in, config_file_out,
                     label_map_path='', train_tf_record='', 
                     train_val_tf_record='', num_steps=10000,
                     batch_size=32,
                     verbose=False):

    '''
    edit tf trainig config file to reflect proper paths
	For details on how to set up the pipeline, see:
		https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md 
	For example .config files:
		https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
		also located at: /cosmiq/simrdwn/configs

    to test:
        config_file_in = '/cosmiq/simrdwn/tf/configs/faster_rcnn_inception_v2_simrdwn.config'
        config_file_out = '/cosmiq/simrdwn/tf/configs/faster_rcnn_inception_v2_simrdwn_tmp.config'
    '''
        
    fin = open(config_file_in, 'r')
    fout = open(config_file_out, 'w')   
    line_minus_two = ''
    line_list = []
    for i,line in enumerate(fin):
        if verbose:
            print (i, line)
        line_list.append(line)
        
        # set line_minus_two
        if i > 1:
            line_minus_two = line_list[i-2].strip()
        
        # assume train_path is first, val_path is second
        if (line.strip().startswith('input_path:')) and (line_minus_two.startswith('train_input_reader:')):
            line_out = '    input_path: "' + str(train_tf_record) + '"\n'
            
        elif (line.strip().startswith('input_path:')) and (line_minus_two.startswith('eval_input_reader:')):
            line_out = '    input_path: "' + str(train_val_tf_record) + '"\n'
        
        elif line.strip().startswith('label_map_path:'):
            line_out = '  label_map_path: "' + str(label_map_path) + '"\n'
            
        elif line.strip().startswith('batch_size:'):
            line_out = '  batch_size: ' + str(batch_size) + '\n'

        elif line.strip().startswith('num_steps:'):
            line_out = '  num_steps: ' + str(num_steps) + '\n'

        else:
            line_out = line
        fout.write(line_out)
        
    fin.close()
    fout.close() 

###############################################################################
def tf_train_cmd(tf_cfg_train_file, results_dir):#, log_file):
    '''Train a model with tensorflow object detection api
    nohup python /opt/tensorflow-models/research/object_detection/train.py \
			    --logtostderr \
			    --pipeline_config_path=/raid/local/src/simrdwn/tf/configs/ssd_inception_v2_simrdwn.config \
			    --train_dir=/raid/local/src/simrdwn/outputs/ssd >> \
				train_ssd_inception_v2_simrdwn.log & tail -f train_ssd_inception_v2_simrdwn.log
                
    nohup python /opt/tensorflow-models/research/object_detection/train.py 
    --logtostderr 
    --pipeline_config_path=/raid/local/src/simrdwn/results/train_ssd_ssd_train_3class_2018_01_24_05-28-37/ssd_inception_v2_simrdwn.config 
    --train_dir=/raid/local/src/simrdwn/results/train_ssd_ssd_train_3class_2018_01_24_05-28-37  >> 
    /raid/local/src/simrdwn/results/train_ssd_ssd_train_3class_2018_01_24_05-28-37/logs/train_ssd_ssd_train_3class_2018_01_24_05-28-37.log 
    & tail -f /raid/local/src/simrdwn/results/train_ssd_ssd_train_3class_2018_01_24_05-28-37/logs/train_ssd_ssd_train_3class_2018_01_24_05-28-37.log

    '''
    
    #suffix = ' >> ' + log_file + ' & tail -f ' + log_file
    #suffix =  >> ' + log_file
    suffix = ''

    cmd_arg_list = [
            'python',
            '/opt/tensorflow-models/research/object_detection/train.py',
            '--logtostderr',
            '--pipeline_config_path=' + tf_cfg_train_file,
            '--train_dir=' + results_dir,
            suffix
            ]
    
    cmd = ' '.join(cmd_arg_list)
      
    return cmd

###############################################################################
def tf_export_model_cmd(trained_dir='', tf_cfg_train_file='pipeline.config',  
                        model_output_root='frozen_model'):
                        #num_steps=100000):
    '''export trained model with tensorflow object detection api'''

    # get max training batches completed
    checkpoints_tmp = [ftmp for ftmp in os.listdir(trained_dir) 
            if ftmp.startswith('model.ckpt')]
    #print ("checkpoints tmp:", checkpoints_tmp)
    nums_tmp = [int(z.split('model.ckpt-')[-1].split('.')[0]) for z in checkpoints_tmp]
    #print ("nums_tmp:", nums_tmp)
    num_max_tmp = np.max(nums_tmp)

    cmd_arg_list = [
            'python',
            '/opt/tensorflow-models/research/object_detection/export_inference_graph.py',
            '--input_type image_tensor',
            '--pipeline_config_path=' + os.path.join(trained_dir, tf_cfg_train_file),
            '--trained_checkpoint_prefix=' + os.path.join(trained_dir, 'model.ckpt-' + str(num_max_tmp)),
            #'--trained_checkpoint_prefix=' + os.path.join(results_dir, 'model.ckpt-' + str(num_steps)),
            '--output_directory=' + os.path.join(trained_dir, model_output_root)
            ]
    
    cmd = ' '.join(cmd_arg_list)
      
    return cmd

###############################################################################    
def tf_infer_cmd_dual(inference_graph_path='',
                          input_file_list='', 
                          in_tfrecord_path='', 
                          out_tfrecord_path='',
                          use_tfrecords=1,
                          min_thresh=0.05,
                          GPU=0,
                          BGR2RGB=0,
                          output_csv_path='',
                          infer_src_path='/raid/local/src/simrdwn/core'):
                          #infer_src_path='/raid/local/src/simrdwn/src/infer_detections.py'):
    '''
    Run infer_detections.py with the given input tfrecord or input_file_list
    
    Infer output tfrecord
    		python /raid/local/src/simrdwn/src/infer_detections.py \
		--input_tfrecord_paths=/raid/local/src/simrdwn/data/qgis_labels_car_boat_plane_val.tfrecord \
		--inference_graph=/raid/local/src/simrdwn/outputs/ssd/output_inference_graph/frozen_inference_graph.pb \
		--output_tfrecord_path=/raid/local/src/simrdwn/outputs/ssd/val_detections_ssd.tfrecord 
     df.to_csv(outfile_df)
    '''
    
    cmd_arg_list = [
            'python' ,
            infer_src_path + '/' + 'infer_detections.py',
            '--inference_graph=' + inference_graph_path,
            '--GPU=' + str(GPU), 
            '--use_tfrecord=' + str(use_tfrecords),

            # first method, with tfrecords
            '--input_tfrecord_paths=' + in_tfrecord_path,
            '--output_tfrecord_path=' + out_tfrecord_path,
            
            # second method, with file list
            '--input_file_list=' + input_file_list,
            '--BGR2RGB=' + str(BGR2RGB),
            '--output_csv_path=' + output_csv_path,
            '--min_thresh=' + str(min_thresh)
            ]          
    cmd = ' '.join(cmd_arg_list)
      
    return cmd


###############################################################################
def yolt_command(yolt_cfg_file_tot='',
                 weight_file_tot='',
                 results_dir='',
                 log_file='',
                 yolt_loss_file='',
                 mode='train',
                 yolt_object_labels_str='',
                 yolt_classnum=1,
                 nbands=3,
                 gpu=0,
                 single_gpu_machine=0,
                 yolt_train_images_list_file_tot='',
                 valid_splitims_locs_file='',
                 test_im_tot='',
                 test_thresh=0.2,
                 yolt_nms_thresh=0,
                 min_retain_prob=0.025):
    
    '''
    Define YOLT commands
    yolt.c expects the following inputs:
    // arg 0 = GPU number
    // arg 1 'yolt'
    // arg 2 = mode
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *test_filename = (argc > 5) ? argv[5]: 0;
    float plot_thresh = (argc > 6) ? atof(argv[6]): 0.2;
    float nms_thresh = (argc > 7) ? atof(argv[7]): 0;
    char *train_images = (argc > 8) ? argv[8]: 0;
    char *results_dir = (argc > 9) ? argv[9]: 0;
    //char *valid_image = (argc >10) ? argv[10]: 0;
    char *valid_list_loc = (argc > 10) ? argv[10]: 0;
    char *names_str = (argc > 11) ? argv[11]: 0;
    int len_names = (argc > 12) ? atoi(argv[12]): 0;
    int nbands = (argc > 13) ? atoi(argv[13]): 0;
    char *loss_file = (argc > 14) ? argv[14]: 0;
    '''

    ##########################
    # set gpu command
    if single_gpu_machine == 1:#use_aware:
        gpu_cmd = ''
    else:
        gpu_cmd = '-i ' + str(gpu) 
        #gpu_cmd = '-i ' + str(3-args.gpu) # originally, numbers were reversed

    ##########################
    # SET VARIABLES ACCORDING TO MODE (SET UNNECCESSARY VALUES TO 0 OR NULL)      
    # set train prams (and prefix, and suffix)
    if mode == 'train':
        train_ims = yolt_train_images_list_file_tot
        prefix = 'nohup'
        suffix = ' >> ' + log_file + ' & tail -f ' + log_file
    else:
        train_ims = 'null'
        prefix = ''
        suffix =  ' 2>&1 | tee -a ' + log_file
        
    # set test params
    if mode == 'test':
        test_im = test_im_tot
        test_thresh = test_thresh
    else:
        test_im = 'null'
        test_thresh = 0
    
    # set valid params
    if mode == 'valid':
        #valid_image = args.valid_image_tmp
        valid_list_loc = valid_splitims_locs_file
    else:
        #valid_image = 'null'
        valid_list_loc = 'null'
            
    
    ##########################

    c_arg_list = [
            prefix,
            './yolt/darknet',
            gpu_cmd,
            'yolt2',
            mode,
            yolt_cfg_file_tot,
            weight_file_tot,
            test_im,
            str(test_thresh),
            str(yolt_nms_thresh),
            train_ims,
            results_dir,
            valid_list_loc,
            yolt_object_labels_str,
            str(yolt_classnum),
            str(nbands),
            yolt_loss_file,
            str(min_retain_prob),
            suffix
            ]
            
    cmd = ' '.join(c_arg_list)
      
    print ("Command:\n", cmd )       
      
    return cmd


###############################################################################
def recompile_darknet(yolt_dir):
    '''compile darknet'''
    os.chdir(yolt_dir)
    cmd_compile0 = 'make clean'
    cmd_compile1 = 'make'
    
    print (cmd_compile0)
    utils.run_cmd(cmd_compile0)
    
    print (cmd_compile1)
    utils.run_cmd(cmd_compile1)    
        
###############################################################################
def replace_yolt_vals_train_compile(yolt_dir='', mode='train', 
                              yolt_cfg_file_tot='',
                              yolt_final_output='',
                              yolt_classnum=2,
                              nbands=3,
                              max_batches=512,
                              batch_size=16,
                              subdivisions=4,
                              boxes_per_grid=5,
                              yolt_input_width=416,
                              yolt_input_height=416,
                              use_GPU=1,
                              use_opencv=1,
                              use_CUDNN=1):

    '''For either training or compiling, 
    edit cfg file in darknet to allow for custom models
    editing of network layers must be done in vi, this function just changes 
    parameters such as window size, number of trianing steps, etc'''
        
    #################
    # Makefile
    if mode == 'compile':
        yoltm = os.path.join(yolt_dir, 'Makefile')
        yoltm_tmp = yoltm + 'tmp'
        f1 = open(yoltm, 'r')
        f2 = open(yoltm_tmp, 'w')   
        for line in f1:
            if line.strip().startswith('GPU='):
                line_out = 'GPU=' + str(use_GPU) + '\n'
            elif line.strip().startswith('OPENCV='):
                line_out = 'OPENCV=' + str(use_opencv) + '\n'
            elif line.strip().startswith('CUDNN='):
                line_out = 'CUDNN=' + str(use_CUDNN) + '\n'            
            else:
                line_out = line
            f2.write(line_out)
        f1.close()
        f2.close()    
        # copy old yoltm
        utils.run_cmd('cp ' + yoltm + ' ' + yoltm + '_v0')
        # write new file over old
        utils.run_cmd('mv ' + yoltm_tmp + ' ' + yoltm)

    #################
    # cfg file
    elif mode == 'train':
        yoltcfg = yolt_cfg_file_tot
        yoltcfg_tmp = yoltcfg + 'tmp'
        f1 = open(yoltcfg, 'r')
        f2 = open(yoltcfg_tmp, 'w')    
        # read in reverse because we want to edit the last output length
        s = f1.readlines()
        s.reverse()
        sout = []
        
        fixed_output = False
        for line in s:
            #if line.strip().startswith('side='):
            #    line_out='side=' + str(side) + '\n'
            if line.strip().startswith('channels='):
                line_out = 'channels=' + str(nbands) + '\n'
            elif line.strip().startswith('classes='):
                line_out = 'classes=' + str(yolt_classnum) + '\n'
            elif line.strip().startswith('max_batches'):
                line_out = 'max_batches=' + str(max_batches) + '\n'
            elif line.strip().startswith('batch='):
                line_out = 'batch=' + str(batch_size) + '\n'  
            elif line.strip().startswith('subdivisions='):
                line_out = 'subdivisions=' + str(subdivisions) + '\n'  
            elif line.strip().startswith('num='):
                line_out = 'num=' + str(boxes_per_grid) + '\n'         
            elif line.strip().startswith('width='):
                line_out = 'width=' + str(yolt_input_width) + '\n'         
            elif line.strip().startswith('height='):
                line_out = 'height=' + str(yolt_input_height) + '\n'         
            # change final output, and set fixed to true
            #elif (line.strip().startswith('output=')) and (not fixed_output):
            #    line_out = 'output=' + str(final_output) + '\n'
            #    fixed_output=True
            elif (line.strip().startswith('filters=')) and (not fixed_output):
                line_out = 'filters=' + str(yolt_final_output) + '\n'
                fixed_output=True                
            else:
                line_out = line
            sout.append(line_out)
            
        sout.reverse()
        for line in sout:
            f2.write(line)
           
        f1.close()
        f2.close()
        
        # copy old yoltcfg?
        utils.run_cmd('cp ' + yoltcfg + ' ' + yoltcfg[:-4] + 'orig.cfg')
        # write new file over old
        utils.run_cmd('mv ' + yoltcfg_tmp + ' ' + yoltcfg)    
    #################
   
    else:
        return
    

###############################################################################
def split_valid_im(im_root_with_ext, valid_testims_dir_tot, results_dir,
                   log_file,
                   slice_sizes=[416],
                   slice_overlap=0.2,
                   valid_slice_sep='__',
                   zero_frac_thresh=0.5,
                   ):
    
    '''split files for valid step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')
    
    1. get image path (args.valid_image_tmp) from image root name 
            (args.valid_image_tmp)
    2. slice test image and move to results dir

    '''

    
    # get image root, make sure there is no extension
    im_root = im_root_with_ext.split('.')[0]
    im_path = os.path.join(valid_testims_dir_tot, im_root_with_ext)
    
    # slice validation plot into manageable chunks
    
    # slice (if needed)
    if slice_sizes[0] > 0:
    #if len(args.slice_sizes) > 0:
        # create valid_splitims_locs_file 
        # set valid_dir as in results_dir
        valid_split_dir = os.path.join(results_dir,  im_root + '_split' + '/')
        valid_dir_str = '"Valid_split_dir: ' +  valid_split_dir + '\n"'
        print ("Valid_dir:", valid_dir_str[1:-2])
        os.system('echo ' + valid_dir_str + ' >> ' + log_file)
        #print "valid_split_dir:", valid_split_dir
        
        # clean out dir, and make anew
        if os.path.exists(valid_split_dir):
            if (not valid_split_dir.startswith(results_dir)) \
                    or len(valid_split_dir) < len(results_dir) \
                    or len(valid_split_dir) < 10:
                print ("valid_split_dir too short!!!!:", valid_split_dir)
                return
            shutil.rmtree(valid_split_dir, ignore_errors=True)
        os.mkdir(valid_split_dir)

        # slice
        for s in slice_sizes:
            slice_im.slice_im(im_path, im_root, 
                              valid_split_dir, s, s, 
                              zero_frac_thresh=zero_frac_thresh, 
                              overlap=slice_overlap,
                              slice_sep=valid_slice_sep)
            valid_files = [os.path.join(valid_split_dir, f) for \
                                   f in os.listdir(valid_split_dir)]
        n_files_str = '"Num files: ' + str(len(valid_files)) + '\n"'
        print (n_files_str[1:-2])
        os.system('echo ' + n_files_str + ' >> ' + log_file)
        
    else:
        valid_files = [im_path]
        valid_split_dir = os.path.join(results_dir, 'nonsense')

    return valid_files, valid_split_dir


###############################################################################
def prep_valid_files(results_dir, log_file, valid_ims_list, 
              valid_testims_dir_tot, valid_splitims_locs_file,
              slice_sizes=[416],
              slice_overlap=0.2,
              valid_slice_sep='__',
              zero_frac_thresh=0.5,
              ):
    '''Split images and save split image locations to txt file'''
        
    # split validation images, store locations 
    t0 = time.time()
    valid_split_str = '"Splitting validation files...\n"'
    print (valid_split_str[1:-2])
    os.system('echo ' + valid_split_str + ' >> ' + log_file)
    print ("valid_ims_list:", valid_ims_list)

    valid_files_locs_list = []
    valid_split_dir_list = []
    # !! Should make a tfrecord when we split files, instead of doing it later 
    for i,valid_base_tmp in enumerate(valid_ims_list):
        iter_string = '"\n' + str(i+1) + ' / ' + \
            str(len(valid_ims_list)) + '\n"'
        print (iter_string[1:-2])
        os.system('echo ' + iter_string + ' >> ' + log_file)
        #print "\n", i+1, "/", len(args.valid_ims_list)
        
        # dirty hack: ignore file extensions for now
        #valid_base_tmp_noext = valid_base_tmp.split('.')[0]
        #valid_base_string = '"valid_base_tmp_noext:' \
        #                    + str(valid_base_tmp_noext) + '\n"'
        valid_base_string = '"valid_file: ' + str(valid_base_tmp) + '\n"'
        print (valid_base_string[1:-2])
        os.system('echo ' + valid_base_string + ' >> ' + log_file)
        
        # split data 
        #valid_files_list_tmp, valid_split_dir_tmp = split_valid_im(valid_base_tmp, args)
        valid_files_list_tmp, valid_split_dir_tmp = \
                split_valid_im(valid_base_tmp, valid_testims_dir_tot, 
                               results_dir, log_file,
                               slice_sizes=slice_sizes,
                               slice_overlap=slice_overlap,
                               valid_slice_sep=valid_slice_sep,
                               zero_frac_thresh=zero_frac_thresh)
        # add valid_files to list
        valid_files_locs_list.extend(valid_files_list_tmp)
        valid_split_dir_list.append(valid_split_dir_tmp)

    # swrite valid_files_locs_list to file (file = valid_splitims_locs_file)
    print ("Total len valid files:", len(valid_files_locs_list))
    print ("valid_splitims_locs_file:", valid_splitims_locs_file)
    # write list of files to valid_splitims_locs_file
    with open (valid_splitims_locs_file, "wb") as fp:
       for line in valid_files_locs_list:
           if not line.endswith('.DS_Store'):
               fp.write(line + "\n")

    t1 = time.time()
    cmd_time_str = '"\nLength of time to split valid files: ' \
                    + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str)
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)
               
    return valid_files_locs_list, valid_split_dir_list

###############################################################################
#def run_valid(framework, infer_cmd, results_dir, log_file, 
#              valid_files_locs_list, valid_split_dir_list,
#              slice_sizes=[416],
#              valid_testims_dir_tot='',
#              yolt_valid_classes_files='', 
#              val_df_path_init='',
#              val_df_path_aug='',
#              label_map_dict={},
#              # second classifier
#              infer_cmd2='',
#              label_map_dict2={},
#              slice_sizes2=[800],
#              val_df_path_init='',
#              val_df_path_aug='',
#              # slicing and plotting#              # slicing and plotting
#              valid_slice_sep='__',
#              edge_buffer_valid=1,
#              max_edge_aspect_ratio=4,
#              valid_box_rescale_frac=1.0,
#              rotate_boxes=False,
#              plot_thresh=0.33,
#              nms_overlap_thresh=0.5,
#              show_labels=True,
#              alpha_scaling=True,
#              plot_line_thickness=2,
#              keep_valid_slices='False',
#              ):
#    '''Evaluate multiple large images'''
    
###############################################################################
def run_valid(framework='YOLT', 
              infer_cmd='', 
              results_dir='', 
              log_file='',
              #valid_files_locs_list=[], #valid_split_dir_list,
              #valid_presliced_tfrecord_tot='',
              n_files=0,
              valid_tfrecord_out='',
              slice_sizes=[416],
              valid_testims_dir_tot='',
              yolt_valid_classes_files='', 
              label_map_dict={},
              val_df_path_init='',
              #val_df_path_aug='',
              valid_slice_sep='__',
              edge_buffer_valid=1,
              max_edge_aspect_ratio=4,
              valid_box_rescale_frac=1.0,
              rotate_boxes=False,
              min_retain_prob=0.05,
              #plot_thresh=0.33,
              #nms_overlap_thresh=0.5,
              #show_labels=True,
              #alpha_scaling=True,
              #plot_line_thickness=2,
              #keep_valid_slices='False',
              ):
    '''Evaluate multiple large images'''
    

    ## get colormap
    #colormap, color_dict = post_process.make_color_legend('', label_map_dict)
    #print ("colormap:", colormap)
    #print ("color_dict:", color_dict)

    # run for each image
    #t00 = time.time()    
    
    # determine object labels and number of files
    #yolt_object_labels = [label_map_dict[ktmp] for ktmp in sorted(label_map_dict.keys())]
    #n_files = len(valid_files_locs_list)  #file_len(valid_splitims_locs_file)

    t0 = time.time()
    # run command
    os.system(infer_cmd)       #run_cmd(outcmd)
    t1 = time.time()
    cmd_time_str = '"\nLength of time to run command: ' +  infer_cmd \
                    + ' for ' + str(n_files) + ' cutouts: ' \
                    + str(t1 - t0) + ' seconds\n"'
    print (cmd_time_str )
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)


    # run second classsifier?....

    if framework.upper() != 'YOLT':
        
        # if we ran inference with a tfrecord, we must now parse that into
        #   a dataframe
        if len(valid_tfrecord_out) > 0:
            df_init = parse_tfrecord.tf_to_df(valid_tfrecord_out, 
                max_iter=500000, 
                label_map_dict=label_map_dict, 
                tf_type='test',
                output_columns = ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category'],
                replace_paths=())
            # use numeric categories
            label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}
            df_init['Category'] = [label_map_dict_rev[vtmp] for vtmp in df_init['Category'].values]
            # save to file            
            df_init.to_csv(val_df_path_init)
        else:
            print ("Read in val_df_path_init:", val_df_path_init)
            df_init = pd.read_csv(val_df_path_init, 
                              #names=[u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', 
                              #       u'Xmax', u'Ymax', u'Category']
                              )
        
        #########
        # post process
        print ("len df_init:", len(df_init))
        df_init.index = np.arange(len(df_init))
        
        # clean out low probabilities
        print ("minimum retained threshold:",  min_retain_prob)
        bad_idxs = df_init[df_init['Prob'] < min_retain_prob].index
        if len(bad_idxs) > 0:
            print ("bad idxss:", bad_idxs)
            df_init.drop(df_init.index[bad_idxs], inplace=True)

        # clean out bad categories
        df_init['Category'] = df_init['Category'].values.astype(int)
        good_cats = label_map_dict.keys()
        print ("Allowed categories:", good_cats)
        #print ("df_init0['Category'] > np.max(good_cats)", df_init['Category'] > np.max(good_cats))
        #print ("df_init0[df_init0['Category'] > np.max(good_cats)]", df_init[df_init['Category'] > np.max(good_cats)])
        bad_idxs2 = df_init[df_init['Category'] > np.max(good_cats)].index
        if len(bad_idxs2) > 0:
            print ("label_map_dict:", label_map_dict)
            print ("df_init['Category']:", df_init['Category'] )
            print ("bad idxs2:", bad_idxs2)
            df_init.drop(df_init.index[bad_idxs2], inplace=True)
        
        # set index as sequential
        df_init.index = np.arange(len(df_init))
        
        #df_init = df_init0[df_init0['Category'] <= np.max(good_cats)]
        #if (len(df_init) != len(df_init0)):
        #    print (len(df_init0) - len(df_init), "rows cleaned out")
        
        # tf_infer_cmd outputs integer categories, update to strings
        df_init['Category'] = [label_map_dict[ktmp] for ktmp in df_init['Category'].values]
              
        print ("len df_init after filtering:", len(df_init))

        # augment dataframe columns
        df_tot = post_process.augment_df(df_init, 
                   valid_testims_dir_tot=valid_testims_dir_tot,
                   slice_sizes=slice_sizes,
                   valid_slice_sep=valid_slice_sep,
                   edge_buffer_valid=edge_buffer_valid,
                   max_edge_aspect_ratio=max_edge_aspect_ratio,
                   valid_box_rescale_frac=valid_box_rescale_frac,
                   rotate_boxes=rotate_boxes,
                   verbose=True)

    else:
        # post-process
        #df_tot = post_process_yolt_valid_create_df(args)
        df_tot = post_process.post_process_yolt_valid_create_df(yolt_valid_classes_files, 
                   log_file, 
                   valid_testims_dir_tot=valid_testims_dir_tot,
                   slice_sizes=slice_sizes,
                   valid_slice_sep=valid_slice_sep,
                   edge_buffer_valid=edge_buffer_valid,
                   max_edge_aspect_ratio=max_edge_aspect_ratio,
                   valid_box_rescale_frac=valid_box_rescale_frac,
                   rotate_boxes=rotate_boxes)
    
    ###########################################
    # plot
    
    # save to csv
    #df_tot.to_csv(val_df_path_aug, index=False) 
    
    return df_tot

    #post_proccess_make_plots(args, df_tot, verbose=True)
        
#    # refine and plot
#    refine_dic = parse_tfrecord.refine_and_plot_df(df_tot, 
#                           groupby='Image_Path', 
#                           label_map_dict=label_map_dict, 
#                           slice_sizes=slice_sizes,
#                           outdir=results_dir, 
#                           plot_thresh=plot_thresh, 
#                           nms_overlap_thresh=nms_overlap_thresh,
#                           show_labels=show_labels, 
#                           alpha_scaling=alpha_scaling,
#                           plot_line_thickness=plot_line_thickness,
#                           plot=True,
#                           verbose=False)
#    
#    # remove or zip valid_split_dirs to save space
#    for valid_split_dir_tmp in valid_split_dir_list:
#        if os.path.exists(valid_split_dir_tmp):
#            # compress image chip dirs if desired
#            if keep_valid_slices.upper() == 'TRUE':
#                print ("Compressing image chips...")
#                shutil.make_archive(valid_split_dir_tmp, 'zip', 
#                                    valid_split_dir_tmp)    
#            # remove unzipped folder
#            print ("Removing valid_split_dir_tmp:", valid_split_dir_tmp)
#            # make sure that valid_split_dir_tmp hasn't somehow been shortened
#            #  (don't want to remove "/")
#            if len(valid_split_dir_tmp) < len(results_dir):
#                print ("valid_split_dir_tmp too short!!!!:", valid_split_dir_tmp)
#                return
#            else:
#                shutil.rmtree(valid_split_dir_tmp, ignore_errors=True)
#                
#    ## zip image files
#    #print "Zipping image files..."
#    #for f in os.listdir(args.results_dir):
#    #    print "file:", f
#    #    if f.endswith(args.extension_list):
#    #        ftot = os.path.join(args.results_dir, f)
#    #        os.system('gzip ' + ftot)
#            
#    return refine_dic

################################################################################
#def refine_valid(df_tot, valid_split_dir_list=[],
#                 groupby='Image_Path', label_map_dict='', 
#                 sliced=True, results_dir='',
#                 plot_thresh=0.33, nms_overlap_thresh=0.5, show_labels=False,
#                 alpha_scaling=False, plot_line_thickness=1, make_plots=True,
#                 keep_valid_slices=False,
#                 verbose=False):
#
#    # refine and plot
#    df_refine = post_process.refine_and_plot_df(df_tot, 
#                           groupby=groupby, 
#                           label_map_dict=label_map_dict, 
#                           sliced=sliced,
#                           outdir=results_dir, 
#                           plot_thresh=plot_thresh, 
#                           nms_overlap_thresh=nms_overlap_thresh,
#                           show_labels=show_labels, 
#                           alpha_scaling=alpha_scaling,
#                           plot_line_thickness=plot_line_thickness,
#                           plot=make_plots,
#                           verbose=verbose)
#                 
#    # remove or zip valid_split_dirs to save space
#    if len(valid_split_dir_list) > 0:
#        for valid_split_dir_tmp in valid_split_dir_list:
#            if os.path.exists(valid_split_dir_tmp):
#                # compress image chip dirs if desired
#                if keep_valid_slices:
#                    print ("Compressing image chips...")
#                    shutil.make_archive(valid_split_dir_tmp, 'zip', 
#                                        valid_split_dir_tmp)    
#                # remove unzipped folder
#                print ("Removing valid_split_dir_tmp:", valid_split_dir_tmp)
#                # make sure that valid_split_dir_tmp hasn't somehow been shortened
#                #  (don't want to remove "/")
#                if len(valid_split_dir_tmp) < len(results_dir):
#                    print ("valid_split_dir_tmp too short!!!!:", valid_split_dir_tmp)
#                    return
#                else:
#                    shutil.rmtree(valid_split_dir_tmp, ignore_errors=True)
#                    
#    ## zip image files
#    #print "Zipping image files..."
#    #for f in os.listdir(args.results_dir):
#    #    print "file:", f
#    #    if f.endswith(args.extension_list):
#    #        ftot = os.path.join(args.results_dir, f)
#    #        os.system('gzip ' + ftot)
#            
#    return df_refine
#         

###############################################################################
###############################################################################
def execute(args):
    
    print ("\nSIMRDWN now...\n")
    os.chdir(args.simrdwn_dir)
    #t0 = time.time()
        
    # make dirs
    os.mkdir(args.results_dir)
    os.mkdir(args.log_dir)

    # create log file, init to the contents in this file
    print ("Date string:", args.date_string)
    os.system('echo ' + str(args.date_string) + ' > ' + args.log_file)      
    os.system('cat ' + args.this_file + ' >> ' + args.log_file)      
    args_str = '"\nArgs: ' +  str(args) + '\n"'
    print (args_str)  
    os.system('echo ' + args_str + ' >> ' + args.log_file)

    # copy this file (yolt_run.py) as well as config, plot file to results_dir
    shutil.copy2(args.this_file, args.log_dir)
    #shutil.copy2(args.yolt_plot_file, args.log_dir)
    #shutil.copy2(args.tf_plot_file, args.log_dir)
    print ("log_dir:", args.log_dir)

    print ("\nlabel_map_dict:", args.label_map_dict)
    print ("\nlabel_map_dict_tot:", args.label_map_dict_tot)
    #print ("object_labels:", args.object_labels)
    print ("\nyolt_object_labels:", args.yolt_object_labels)
    print ("yolt_classnum:", args.yolt_classnum)
    
    # save labels to log_dir
    #pickle.dump(args.object_labels, open(args.log_dir \
    #                                    + 'labels_list.pkl', 'wb'), protocol=2)
    with open (args.labels_log_file, "wb") as fp:
        for ob in args.yolt_object_labels:
           fp.write(ob+"\n")

    # set YOLT values, if desired
    if args.framework.upper() == 'YOLT':
       
        # copy files to log dir
        shutil.copy2(args.yolt_plot_file, args.log_dir)
        shutil.copy2(args.yolt_cfg_file_in, args.log_dir)
        os.system('cat ' + args.yolt_cfg_file_tot + ' >> ' + args.log_file )      
        # print config values
        print ("yolt_cfg_file:", args.yolt_cfg_file_in)
        if args.mode.upper() in ['TRAIN', 'COMPILE']:
            print ("Updating yolt params in files...")
            replace_yolt_vals_train_compile(yolt_dir=args.yolt_dir, 
                              mode=args.mode, 
                              yolt_cfg_file_tot=args.yolt_cfg_file_tot,
                              yolt_final_output=args.yolt_final_output,
                              yolt_classnum=args.yolt_classnum,
                              nbands=args.nbands,
                              max_batches=args.max_batches,
                              batch_size=args.batch_size,
                              subdivisions=args.subdivisions,
                              boxes_per_grid=args.boxes_per_grid,
                              yolt_input_width=args.yolt_input_width,
                              yolt_input_height=args.yolt_input_height,
                              use_GPU=args.use_GPU,
                              use_opencv=args.use_opencv,
                              use_CUDNN=args.use_CUDNN)
            #replace_yolt_vals(args)    
            # print a few values...
            print ("Final output layer size:", args.yolt_final_output)
            #print ("side size:", args.side)
            print ("batch_size:", args.batch_size)
            print ("subdivisions:", args.subdivisions)

        if args.mode.upper() == 'COMPILE':
            print ("Recompiling yolt...")
            recompile_darknet(args.yolt_dir) 
            return
     
        # set yolt command  
        yolt_cmd = yolt_command(yolt_cfg_file_tot=args.yolt_cfg_file_tot,
                 weight_file_tot=args.weight_file_tot,
                 results_dir=args.results_dir,
                 log_file=args.log_file,
                 yolt_loss_file=args.yolt_loss_file,
                 mode=args.mode,
                 yolt_object_labels_str=args.yolt_object_labels_str,
                 yolt_classnum=args.yolt_classnum,
                 nbands=args.nbands,
                 gpu=args.gpu,
                 single_gpu_machine=args.single_gpu_machine,
                 yolt_train_images_list_file_tot=args.yolt_train_images_list_file_tot,
                 valid_splitims_locs_file=args.valid_splitims_locs_file,
                 yolt_nms_thresh=args.yolt_nms_thresh,
                 min_retain_prob = args.min_retain_prob)
        
        if args.mode.upper() == 'TRAIN':
            print ("yolt_train_cmd:", yolt_cmd)
            #train_cmd_tot = yolt_cmd
            train_cmd1 = yolt_cmd
            #train_cmd2 = ''
        
        # set second validation command
        elif args.mode.upper() == 'VALID':
            valid_cmd_tot = yolt_cmd
        if len(args.label_map_path2) > 0:
            valid_cmd_tot2 = yolt_command(yolt_cfg_file_tot=args.yolt_cfg_file_tot2,
                 weight_file_tot=args.weight_file_tot2,
                 results_dir=args.results_dir,
                 log_file=args.log_file,
                 mode=args.mode,
                 yolt_object_labels_str=args.yolt_object_labels_str2,
                 classnum=args.yolt_classnum2,
                 nbands=args.nbands,
                 gpu=args.gpu,
                 single_gpu_machine=args.single_gpu_machine,
                 valid_splitims_locs_file=args.valid_splitims_locs_file2,
                 yolt_nms_thresh=args.yolt_nms_thresh,
                 min_retain_prob = args.min_retain_prob)

        else:
            valid_cmd_tot2 = ''
        
    # set tensor flow object detection API values
    else:
        
        if args.mode.upper() == 'TRAIN':
            if not os.path.exists(args.tf_model_output_directory):
                os.mkdir(args.tf_model_output_directory)
            # copy plot file to output dir
            shutil.copy2(args.tf_plot_file, args.log_dir)

            print ("Updating tf_config...")
            update_tf_train_config(args.tf_cfg_train_file, args.tf_cfg_train_file_out,
                             label_map_path=args.label_map_path, 
                             train_tf_record=args.train_tf_record, 
                             train_val_tf_record=args.train_val_tf_record, 
                             batch_size=args.batch_size,
                             num_steps=args.max_batches)
            # define train command
            cmd_train_tf = tf_train_cmd(args.tf_cfg_train_file_out, args.results_dir)
                    #, args.log_file)
                    
            # export command
            cmd_export_tf = ''
#            cmd_export_tf = tf_export_model_cmd(args.tf_cfg_train_file_out, 
#                                             args.results_dir, 
#                                             args.tf_model_output_directory)
#                                             #num_steps=args.max_batches)


#            # https://unix.stackexchange.com/questions/47230/how-to-execute-multiple-command-using-nohup/47249
#            # tot_cmd = nohup sh -c './cmd2 >result2 && ./cmd1 >result1' &
#            train_cmd_tot = 'nohup sh -c ' + "'" \
#                                + cmd_train_tf + ' >> ' + args.log_file \
#                                + ' && ' \
#                                + cmd_export_tf + ' >> ' + args.log_file \
#                                + "'" + ' & tail -f ' + args.log_file                    

            train_cmd1 = 'nohup ' + cmd_train_tf + ' >> ' + args.log_file \
                            + ' & tail -f ' + args.log_file + ' &'
            #train_cmd2 = 'nohup ' +  cmd_export_tf + ' >> ' + args.log_file \
            #                + ' & tail -f ' + args.log_file #+ ' &'

            # forget about nohup since we're inside docker?
            #train_cmd1 = cmd_train_tf 
            #train_cmd2 = cmd_export_tf 
    
        # Validate
        else:

            # define inference (validation) command   (output to csv)
            valid_cmd_tot = tf_infer_cmd_dual(inference_graph_path=args.inference_graph_path_tot, 
                          input_file_list=args.valid_splitims_locs_file,
                          in_tfrecord_path=args.valid_presliced_tfrecord_tot, 
                          out_tfrecord_path=args.valid_tfrecord_out,
                          output_csv_path=args.val_df_path_init,
                          min_thresh=args.min_retain_prob,
                          BGR2RGB=args.BGR2RGB,
                          use_tfrecords=args.use_tfrecords,
                          infer_src_path=path_simrdwn_core)
            
            # if using dual classifiers
            if len(args.label_map_path2) > 0:
                # check if model exists, if not, create it.
                if not os.path.exists(args.inference_graph_path_tot2):
                    inference_graph_path_tmp = os.path.dirname(args.inference_graph_path_tot2)
                    cmd_tmp = 'python  ' \
                                + args.src_dir + '/export_model.py ' \
                                + '--results_dir ' + inference_graph_path_tmp
                    t1 = time.time()
                    print ("Running", cmd_tmp, "...\n")
                    os.system(cmd_tmp)
                    t2 = time.time()
                    cmd_time_str = '"Length of time to run command: ' \
                                    +  cmd_tmp + ' ' \
                                    + str(t2 - t1) + ' seconds\n"'
                    print (cmd_time_str)  
                    os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
                # set inference command
                valid_cmd_tot2 = tf_infer_cmd_dual(inference_graph_path=args.inference_graph_path_tot2, 
                          input_file_list=args.valid_splitims_locs_file2,
                          output_csv_path=args.val_df_path_init2,
                          min_thresh=args.min_retain_prob,
                          GPU=args.gpu,
                          BGR2RGB=args.BGR2RGB,
                          use_tfrecords=args.use_tfrecords,
                          infer_src_path=path_simrdwn_core)        
            else:
                valid_cmd_tot2 = ''
                
                
    ### Execute
    if args.mode.upper() == 'TRAIN':
        
        t1 = time.time()
        print ("Running", train_cmd1, "...\n\n")
        os.system(train_cmd1)
        #utils.run_cmd(train_cmd1)
        t2 = time.time()
        cmd_time_str = '"Length of time to run command: ' \
                        +  train_cmd1 + ' ' \
                        + str(t2 - t1) + ' seconds\n"'
        print (cmd_time_str)  
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
    
    
        # export trained model, if using tf object detection api
        if 2 < 1 and (args.framework.upper() != 'YOLT'):
            cmd_export_tf = tf_export_model_cmd(args.tf_cfg_train_file_out, 
                                             args.results_dir, 
                                             args.tf_model_output_directory)
            train_cmd2 = cmd_export_tf

            t1 = time.time()
            print ("Running", train_cmd2, "...\n\n")
            #utils.run_cmd(train_cmd2)
            os.system(train_cmd2)
            t2 = time.time()
            cmd_time_str = '"Length of time to run command: ' \
                            +  train_cmd2 + ' ' \
                            + str(t2 - t1) + ' seconds\n"'
            print (cmd_time_str)  
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        #t1 = time.time()
        #print ("Running", train_cmd_tot, "...\n\n")
        #os.system(train_cmd_tot)
        #t2 = time.time()
        #cmd_time_str = '"Length of time to run command: ' \
        #                +  train_cmd_tot + ' ' \
        #                + str(t2 - t1) + ' seconds\n"'
        #print (cmd_time_str)  
        #os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)



    # need to split file for valid first, then run command
    elif args.mode.upper() == 'VALID':
        
        t3 = time.time()
        # load presliced data, if desired
        if len(args.valid_presliced_list) > 0:
            print ("Loading args.valid_presliced_list:", args.valid_presliced_list_tot)
            ftmp = open(args.valid_presliced_list_tot, 'r')
            valid_files_locs_list = [line.strip() for line in ftmp.readlines()]
            ftmp.close()
            valid_split_dir_list = []
            print ("len valid_files_locs_list:", len(valid_files_locs_list))
        elif len(args.valid_presliced_tfrecord_part) > 0:
            print ("Using", args.valid_presliced_tfrecord_part)
            valid_split_dir_list = []
        # split large validion files
        else:
            print ("Prepping validation files")
            valid_files_locs_list, valid_split_dir_list =\
                    prep_valid_files(args.results_dir, args.log_file, 
                             args.valid_ims_list, 
                             args.valid_testims_dir_tot, 
                             args.valid_splitims_locs_file,
                             slice_sizes=args.slice_sizes,
                             slice_overlap=args.slice_overlap,
                             valid_slice_sep=args.valid_slice_sep,
                             zero_frac_thresh=args.zero_frac_thresh,
                             )
            # return if only interested in prepping
            if bool(args.valid_prep_only):
                print ("Convert to tfrecords...")
                TF_RecordPath = os.path.join(args.results_dir, 'valid_splitims.tfrecord')
                preprocess_tfrecords.yolt_imlist_to_tf(args.valid_splitims_locs_file, 
                                           args.label_map_dict, TF_RecordPath,
                                           TF_PathVal='', val_frac=0.0, 
                                           convert_dict={}, verbose=False)

                print ("Done prepping valid files, ending")
                return


        # check if trained model exists, if not, create it.
        if (args.framework.upper() != 'YOLT') and \
            (not (os.path.exists(args.inference_graph_path_tot)) or \
                    (args.overwrite_inference_graph != 0)):
            print ("Creating args.inference_graph_path_tot:", 
                   args.inference_graph_path_tot, "...")
            
            # remove "saved_model" directory
            saved_dir = os.path.join(
                    os.path.dirname(args.inference_graph_path_tot), 'saved_model')
            print ("Removing", saved_dir, "so we can overwrite it...")
            if os.path.exists(saved_dir):
                shutil.rmtree(saved_dir, ignore_errors=True)

            trained_dir_tmp = os.path.dirname(os.path.dirname(args.inference_graph_path_tot))
            cmd_tmp = tf_export_model_cmd(trained_dir=trained_dir_tmp)
            #cmd_tmp = 'python  ' \
            #            + args.src_dir + '/export_model.py ' \
            #            + '--results_dir=' + inference_graph_path_tmp
            t1 = time.time()
            print ("Running", cmd_tmp, "...\n")
            os.system(cmd_tmp)
            t2 = time.time()
            cmd_time_str = '"Length of time to run command: ' \
                            +  cmd_tmp + ' ' \
                            + str(t2 - t1) + ' seconds\n"'
            print (cmd_time_str)  
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)




        df_tot = run_valid(infer_cmd=valid_cmd_tot, 
              framework=args.framework, 
              results_dir=args.results_dir, 
              log_file=args.log_file,
              #valid_files_locs_list=valid_files_locs_list,
              #valid_presliced_tfrecord_tot=args.valid_presliced_tfrecord_tot,
              valid_tfrecord_out = args.valid_tfrecord_out,
              slice_sizes=args.slice_sizes,
              valid_testims_dir_tot=args.valid_testims_dir_tot,
              yolt_valid_classes_files=args.yolt_valid_classes_files,
              label_map_dict=args.label_map_dict,
              val_df_path_init=args.val_df_path_init,
              #val_df_path_aug=args.val_df_path_aug,
              min_retain_prob=args.min_retain_prob,
              valid_slice_sep=args.valid_slice_sep,
              edge_buffer_valid=args.edge_buffer_valid,
              max_edge_aspect_ratio=args.max_edge_aspect_ratio,
              valid_box_rescale_frac=args.valid_box_rescale_frac,
              rotate_boxes=args.rotate_boxes)
        
        # save to csv
        df_tot.to_csv(args.val_df_path_aug, index=False) 
        # get number of files
        n_files = len(np.unique(df_tot['Loc_Tmp'].values))
        #n_files = str(len(valid_files_locs_list)
        t4 = time.time()
        cmd_time_str = '"Length of time to run valid for ' \
                        + str(n_files) + ' files = ' \
                        + str(t4 - t3) + ' seconds\n"'
        print (cmd_time_str)  
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
        
              
#        refine_dic = run_valid(args.framework, args.results_dir, args.log_file, 
#              valid_files_locs_list, valid_split_dir_list,
#              infer_cmd_tf=infer_cmd_tf, 
#              yolt_cmd=yolt_command,
#              valid_testims_dir_tot=args.valid_testims_dir_tot,
#              yolt_valid_classes_files=args.yolt_valid_classes_files,
#              val_df_path_init=args.val_df_path_init,
#              val_df_path_aug=args.val_df_path_aug,
#              slice_sizes=args.slice_sizes,
#              valid_slice_sep=args.valid_slice_sep,
#              zero_frac_thresh=args.zero_frac_thresh,
#              edge_buffer_valid=args.edge_buffer_valid,
#              max_edge_aspect_ratio=args.max_edge_aspect_ratio,
#              valid_box_rescale_frac=args.valid_box_rescale_frac,
#              rotate_boxes=args.rotate_boxes,
#              plot_thresh=args.plot_thresh[0],
#              nms_overlap_thresh=args.nms_overlap_thresh,
#              keep_valid_slices=args.keep_valid_slices,
#              show_labels=bool(args.show_labels),
#              alpha_scaling=bool(args.alpha_scaling),
#              plot_line_thickness=args.plot_line_thickness
#              )


        # run again, if desired
        if len(args.weight_file2) > 0:
            
            t5 = time.time()
            # split large validion files
            print ("Prepping validation files")
            valid_files_locs_list2, valid_split_dir_list2 =\
                    prep_valid_files(args.results_dir, args.log_file, 
                             args.valid_ims_list, 
                             args.valid_testims_dir_tot, 
                             args.valid_splitims_locs_file2,
                             slice_sizes=args.slice_sizes2,
                             slice_overlap=args.slice_overlap,
                             valid_slice_sep=args.valid_slice_sep,
                             zero_frac_thresh=args.zero_frac_thresh,
                             )
    
            df_tot2 = run_valid(infer_cmd=valid_cmd_tot2, 
                  framework=args.framework, 
                  results_dir=args.results_dir, 
                  log_file=args.log_file,
                  valid_files_locs_list=valid_files_locs_list2,
                  slice_sizes=args.slice_sizes,
                  valid_testims_dir_tot=args.valid_testims_dir_tot2,
                  yolt_valid_classes_files=args.yolt_valid_classes_files2,
                  label_map_dict=args.label_map_dict2,
                  val_df_path_init=args.val_df_path_init2,
                  #val_df_path_aug=args.val_df_path_aug2,
                  valid_slice_sep=args.valid_slice_sep,
                  edge_buffer_valid=args.edge_buffer_valid,
                  max_edge_aspect_ratio=args.max_edge_aspect_ratio,
                  valid_box_rescale_frac=args.valid_box_rescale_frac,
                  rotate_boxes=args.rotate_boxes)
            
            # save to csv
            df_tot2.to_csv(args.val_df_path_aug2, index=False) 
            t6 = time.time()
            cmd_time_str = '"Length of time to run valid' + ' ' \
                            + str(t6 - t5) + ' seconds\n"'
            print (cmd_time_str)  
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
            
            #Update category numbers of df_tot2 so that they aren't the same
            #    as df_tot?  Shouldn't need to since categories are strings
            
            #Combine df_tot and df_tot2
            df_tot = pd.concat([df_tot, df_tot2])
            valid_split_dir_list = valid_split_dir_list \
                                        + valid_split_dir_list2
            
            #Create new label_map_dict with all categories (done in init_args)
            
        else:
            pass
        
        
        # refine and plot
        t8 = time.time()
        if len(np.append(args.slice_sizes, args.slice_sizes2)) > 0:
            sliced = True
        else:
            sliced = False
        print ("validation data sliced?", sliced)
        
#        # refine 
#        df_refine = refine_valid(df_tot, 
#                                  valid_split_dir_list=valid_split_dir_list,
#                                  groupby='Image_Path', 
#                                  label_map_dict=args.label_map_dict_tot, 
#                                  sliced=sliced, 
#                                  results_dir=args.results_dir,
#                                  plot_thresh=args.plot_thresh_tmp, 
#                                  nms_overlap_thresh=args.nms_overlap_thresh, 
#                                  show_labels=False,
#                                  alpha_scaling=False, 
#                                  plot_line_thickness=1, 
#                                  make_plots=args.valid_make_pngs,
#                                  keep_valid_slices=args.keep_valid_slices,
#                                  verbose=False)                  
#        cmd_time_str = '"Length of time to run refine_valid()' + ' ' \
#                        + str(time.time() - t8) + ' seconds\n"'
#        print (cmd_time_str)  
#        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
#
#
#        # save df_refine
#        df_refine.to_csv(args.val_prediction_df_refine_tot)
#        # save refine_dic 
#        #pickle.dump(refine_dic, open(args.val_prediction_pkl_tot, "wb"))
        

        # refine for each plot_thresh
        for plot_thresh_tmp in args.plot_thresh:
            print ("Plotting at:", plot_thresh_tmp)
            groupby='Image_Path'
            groupby_cat='Category'
            df_refine = post_process.refine_df(df_tot, 
                 groupby=groupby, 
                 groupby_cat=groupby_cat,
                 nms_overlap_thresh=args.nms_overlap_thresh,  
                 plot_thresh=plot_thresh_tmp, 
                 verbose=False)
            # make some output plots, if desired
            if args.n_valid_output_plots > 0:
                post_process.plot_refined_df(df_refine, groupby=groupby, 
                        label_map_dict=args.label_map_dict_tot, 
                        outdir=args.results_dir, 
                        plot_thresh=plot_thresh_tmp, 
                        show_labels=bool(args.show_labels), 
                        alpha_scaling=bool(args.alpha_scaling), 
                        plot_line_thickness=2,
                        print_iter=5,
                        n_plots=args.n_valid_output_plots,
                        verbose=False)
                        
#            # refine and plot
#            df_refine = post_process.refine_and_plot_df(df_tot, 
#                           groupby='Image_Path', 
#                           label_map_dict=args.label_map_dict_tot, 
#                           sliced=sliced,
#                           outdir=args.results_dir, 
#                           plot_thresh=plot_thresh_tmp, 
#                           nms_overlap_thresh=args.nms_overlap_thresh,
#                           show_labels=False, 
#                           alpha_scaling=False,
#                           plot_line_thickness=1,
#                           plot=args.valid_make_pngs,
#                           verbose=False)
            
            
            # save df_refine
            outpath_tmp = os.path.join(args.results_dir, 
                                args.val_prediction_df_refine_tot_root_part +
                                '_thresh=' + str(plot_thresh_tmp) + '.csv')
            #df_refine.to_csv(args.val_prediction_df_refine_tot)
            df_refine.to_csv(outpath_tmp)


        cmd_time_str = '"Length of time to run refine_valid()' + ' ' \
                        + str(time.time() - t8) + ' seconds\n"'
        print (cmd_time_str)  
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
                         
        # remove or zip valid_split_dirs to save space
        if len(valid_split_dir_list) > 0:
            for valid_split_dir_tmp in valid_split_dir_list:
                if os.path.exists(valid_split_dir_tmp):
                    # compress image chip dirs if desired
                    if args.keep_valid_slices:
                        print ("Compressing image chips...")
                        shutil.make_archive(valid_split_dir_tmp, 'zip', 
                                            valid_split_dir_tmp)    
                    # remove unzipped folder
                    print ("Removing valid_split_dir_tmp:", valid_split_dir_tmp)
                    # make sure that valid_split_dir_tmp hasn't somehow been shortened
                    #  (don't want to remove "/")
                    if len(valid_split_dir_tmp) < len(args.results_dir):
                        print ("valid_split_dir_tmp too short!!!!:", valid_split_dir_tmp)
                        return
                    else:
                        print ("Removing image chips...")

                        shutil.rmtree(valid_split_dir_tmp, ignore_errors=True)
                    
 


#        # Run second classifier, if desired
#        if len(args.weight_file2) > 0:
#            
#            t3 = time.time()
#            # split large validion files
#            print ("Prepping validation files")
#            valid_files_locs_list2, valid_split_dir_list2 =\
#                    prep_valid_files(args.results_dir, args.log_file, 
#                             args.valid_ims_list, 
#                             args.valid_testims_dir_tot, 
#                             args.valid_splitims_locs_file2,
#                             slice_sizes=args.slice_sizes2,
#                             slice_overlap=args.slice_overlap,
#                             valid_slice_sep=args.valid_slice_sep,
#                             zero_frac_thresh=args.zero_frac_thresh,
#                             )
#    
#            refine_dic = run_valid(args.framework, args.results_dir, args.log_file, 
#                  valid_files_locs_list2, valid_split_dir_list2,
#                  infer_cmd_tf=infer_cmd_tf, 
#                  yolt_cmd=yolt_command,
#                  valid_testims_dir_tot=args.valid_testims_dir_tot,
#                  yolt_valid_classes_files=args.yolt_valid_classes_files,
#                  val_df_path_init=args.val_df_path_init,
#                  val_df_path_aug=args.val_df_path_aug,
#                  slice_sizes=args.slice_sizes,
#                  valid_slice_sep=args.valid_slice_sep,
#                  zero_frac_thresh=args.zero_frac_thresh,
#                  edge_buffer_valid=args.edge_buffer_valid,
#                  max_edge_aspect_ratio=args.max_edge_aspect_ratio,
#                  valid_box_rescale_frac=args.valid_box_rescale_frac,
#                  rotate_boxes=args.rotate_boxes,
#                  plot_thresh=args.plot_thresh[0],
#                  nms_overlap_thresh=args.nms_overlap_thresh,
#                  keep_valid_slices=args.keep_valid_slices,
#                  show_labels=bool(args.show_labels),
#                  alpha_scaling=bool(args.alpha_scaling),
#                  plot_line_thickness=args.plot_line_thickness
#                  )
#            # save refine_dic 
#            pickle.dump(refine_dic, open(args.val_prediction_pkl, "wb"))
#            cmd_time_str = '"Length of time to run valid' + ' ' \
#                            + str(time.time() - t3) + ' seconds\n"'
#            print (cmd_time_str)  
#            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
    

        cmd_time_str = '"Length of time to run valid' + ' ' \
                        + str(time.time() - t3) + ' seconds\n"'
        print (cmd_time_str)  
    
    
    os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
    
        
    print ("\nNo honeymoon. This is business.")
    
    return


###############################################################################
def main():

    ### Construct argument parser
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument('--framework', type=str, default='yolt',
                        help="object detection framework [yolt, ssd, faster_rcnn]")
    parser.add_argument('--mode', type=str, default='test',
                        help="[compile, test, train, valid]")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU number, set < 0 to turn off GPU support")
    parser.add_argument('--single_gpu_machine', type=int, default=0,
                        help="Switch to use a machine with just one gpu")
    parser.add_argument('--nbands', type=int, default=3,
                        help="Number of input bands (e.g.: for RGB use 3)")
    parser.add_argument('--outname', type=str, default='tmp',
                        help="unique name of output")
    parser.add_argument('--label_map_path', type=str, 
                        default='',
                        help="Object classes, /raid/local/src/simrdwn/data/class_labels_airplane_boat_car.pbtxt")
    parser.add_argument('--weight_dir', type=str, default='/raid/local/src/simrdwn/yolt/input_weights',
                        help="Directory holding trained weights")
    parser.add_argument('--weight_file', type=str, default='yolo.weights',
                        help="Input weight file")

    # training settings
    parser.add_argument('--yolt_train_images_list_file', type=str, default='',
                        help="file holding training image names, should be in " \
                            "simrdwn_dir/data/")
    parser.add_argument('--max_batches', type=int, default=60000,
                        help="Max number of training batches")    
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of images per batch")
    parser.add_argument('--yolt_input_width', type=int, default=416,
                        help="Size of image to input to YOLT [n-boxes * 32: " \
                        + "415, 544, 608, 896")
    parser.add_argument('--yolt_input_height', type=int, default=416,
                        help="Size of image to input to YOLT")    
    # TF api specific settings
    parser.add_argument('--tf_cfg_train_file', type=str, default='',
                        help="Configuration file for training")
    parser.add_argument('--train_tf_record', type=str, default='',
                        help="tfrecord for training")
    parser.add_argument('--train_val_tf_record', type=str, default='',
                        help="tfrecord for validation during training")
    #parser.add_argument('--train_tf_imsize', type=int, default=416,
    #                    help="Input image size")
    # yolt specific
    parser.add_argument('--yolt_object_labels_str', type=str, default='',
                        help="yolt labels str: car,boat,giraffe")    
                        
    # valid settings
    parser.add_argument('--train_model_path', type=str, default='',
                        help="Location of trained model")  
    parser.add_argument('--use_tfrecords', type=int, default=1,
                        help="Switch to use tfrecords for infernece")  
    parser.add_argument('--valid_presliced_tfrecord_part', type=str, default='',
                        help="Location of presliced training data tfrecord " \
                        + " if empty us valid_presliced_list")  
    parser.add_argument('--valid_presliced_list', type=str, default='',
                        help="Location of presliced training data list "  \
                        + " if empty, use tfrecord")  
    parser.add_argument('--valid_testims_dir', type=str, default='',
                        help="Location of validation images")  
    parser.add_argument('--slice_sizes_str', type=str, default='416',
                        help="Proposed pixel slice sizes for valid, will be split"\
                           +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--edge_buffer_valid', type=int, default=-1000,
                        help="Buffer around slices to ignore boxes (helps with"\
                            +" truncated boxes and stitching) set <0 to turn off"\
                            +" if not slicing test ims")
    parser.add_argument('--max_edge_aspect_ratio', type=float, default=3,
                        help="Max aspect ratio of any item within the above "\
                            +" buffer")
    parser.add_argument('--slice_overlap', type=float, default=0.35,
                        help="Overlap fraction for sliding window in valid")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="Overlap threshold for non-max-suppresion in python"\
                            +" (set to <0 to turn off)")
    parser.add_argument('--valid_box_rescale_frac', type=float, default=1.0,
                        help="Defaults to 1, rescale output boxes if training"\
                            + " boxes are the wrong size")    
    parser.add_argument('--valid_slice_sep', type=str, default='__',
                        help="Character(s) to split validation image file names")
    parser.add_argument('--val_df_root_init', type=str, default='valid_predictions_init.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_df_root_aug', type=str, default='valid_predictions_aug.csv',
                        help="Results in dataframe format")
    parser.add_argument('--valid_splitims_locs_file_root', type=str, default='valid_splitims_input_files.txt',
                        help="Root of valid_splitims_locs_file")
    parser.add_argument('--valid_prep_only', type=int, default=0,
                        help="Switch to only prep files, not run anything")
    parser.add_argument('--BGR2RGB', type=int, default=0,
                        help="Switch to flip training files to RGB from cv2 BGR")      
    parser.add_argument('--overwrite_inference_graph', type=int, default=0,
                        help="Switch to always overwrite inference graph")      
    parser.add_argument('--min_retain_prob', type=float, default=0.025,
                        help="minimum probability to retain for validation")      
    
    # valid, specific to YOLT
    parser.add_argument('--yolt_nms_thresh', type=float, default=0.0,
                        help="Defaults to 0.5 in yolt.c, set to 0 to turn off "\
                            +" nms in C")

    # valid plotting
    parser.add_argument('--plot_thresh_str', type=str, default='0.3',
                        help="Proposed thresholds to try for valid, will be split"\
                           +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])") 
    parser.add_argument('--show_labels', type=int, default=0,
                        help="Switch to use show object labels")
    parser.add_argument('--alpha_scaling', type=int, default=0,
                        help="Switch to scale box alpha with probability")
    parser.add_argument('--show_valid_plots', type=int, default=0,
                        help="Switch to show plots in real time in validation")
    #parser.add_argument('--plot_names', type=int, default=0,
    #                    help="Switch to show plots names in validation")
    parser.add_argument('--rotate_boxes', type=int, default=0,
                        help="Attempt to rotate output boxes using hough lines")
    parser.add_argument('--plot_line_thickness', type=int, default=3,
                        help="Thickness for valid output bounding box lines")
    parser.add_argument('--n_valid_output_plots', type=int, default=10,
                        help="Switch to save validation pngs")
    parser.add_argument('--valid_make_legend_and_title', type=int, default=1,
                        help="Switch to make legend and title")    
    parser.add_argument('--valid_im_compression_level', type=int, default=6,
                        help="Compression level for output images."\
                            + " 1-9 (9 max compression")    
    parser.add_argument('--keep_valid_slices', type=int, default=0,
                        help="Switch to retain sliced valid files")


    # random YOLT specific settings
    parser.add_argument('--yolt_cfg_file', type=str, default='yolo.cfg',
                        help="Configuration file for network, in cfg directory")
    parser.add_argument('--subdivisions', type=int, default=4,
                        help="Subdivisions per batch")
    parser.add_argument('--use_opencv', type=str, default='1',
                        help="1 == use_opencv")
    parser.add_argument('--boxes_per_grid', type=int, default=5,
                        help="Bounding boxes per grid cell")
    
    # YOLT test settings
    parser.add_argument('--yolt_test_im', type=str, default='person.jpg',
                        help="test image, in data_dir")
    parser.add_argument('--yolt_test_thresh', type=float, default=0.2,
                        help="prob thresh for plotting outputs")
    parser.add_argument('--yolt_test_labels', type=str, default='coco.names',
                        help="test labels, in data_dir")


    # second validation classifier 
    parser.add_argument('--train_model_path2', type=str, default='',
                        help="Location of trained model")  
    parser.add_argument('--label_map_path2', type=str, 
                        default='',
                        help="Object classes")
    parser.add_argument('--weight_dir2', type=str, default='/raid/local/src/simrdwn/yolt/input_weights',
                        help="Directory holding trained weights")
    parser.add_argument('--weight_file2', type=str, default='',
                        help="Input weight file for second inference scale")
    parser.add_argument('--slice_sizes_str2', type=str, default='0',
                        help="Proposed pixel slice sizes for valid2 == second"\
                            + "weight file.  Will be split"\
                            +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--plot_thresh_str2', type=str, default='0.3',
                        help="Proposed thresholds to try for valid2, will be split"\
                           +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--inference_graph_path2', type=str, default='/raid/local/src/simrdwn/outputs/ssd/output_inference_graph/frozen_inference_graph.pb',
                        help="Location of inference graph for tensorflow " \
                        + "object detection API")
    parser.add_argument('--yolt_cfg_file2', type=str, default='yolo.cfg',
                        help="YOLT configuration file for network, in cfg directory")
    parser.add_argument('--val_df_root_init2', type=str, default='valid_predictions_init2.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_df_root_aug2', type=str, default='valid_predictions_aug2.csv',
                        help="Results in dataframe format")
    parser.add_argument('--valid_splitims_locs_file_root2', type=str, default='valid_splitims_input_files2.txt',
                        help="Root of valid_splitims_locs_file")
    
    # total valid
    parser.add_argument('--val_df_root_tot', type=str, default='valid_predictions_tot.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_prediction_df_refine_tot_root_part', type=str, 
                        default='valid_predictions_refine',
                        help="Refined results in dataframe format")


    # Defaults that rarely should need changed
    parser.add_argument('--simrdwn_dir', type=str, default='/raid/simrdwn/',
                        help="path to package /cosmiq/yolt2/ ")  
    parser.add_argument('--multi_band_delim', type=str, default='#',
                        help="Delimiter for multiband data")
    parser.add_argument('--zero_frac_thresh', type=float, default=0.5,
                        help="If less than this value of an image chip is blank,"\
                            + " skip it")
    parser.add_argument('--str_delim', type=str, default=',',
                        help="Delimiter for string lists")
    
    

    args = parser.parse_args()
    args = update_args(args)
    execute(args)
    
###############################################################################
###############################################################################    
if __name__ == "__main__":
    
    print ("\nPermit me to introduce myself...\n" \
            "Well, I’m glad we got that out of the way.\n")
    main()
    
###############################################################################
###############################################################################

