#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

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
# import logging
# import tensorflow as tf

import utils
import post_process
import add_geo_coords
import parse_tfrecord
import preprocess_tfrecords
import slice_im

sys.stdout.flush()
##########################


###############################################################################
def update_args(args):
    """
    Update args (mostly paths)

    Arguments
    ---------
    args : argparse
        Input args passed to simrdwn

    Returns
    -------
    args : argparse
        Updated args
    """

    ###########################################################################
    # CONSTRUCT INFERRED VALUES
    ###########################################################################

    ##########################
    # GLOBAL VALUES
    # set directory structure

    # args.src_dir = os.path.dirname(os.path.realpath(__file__))
    args.core_dir = os.path.dirname(os.path.realpath(__file__))
    args.this_file = os.path.join(args.core_dir, 'simrdwn.py')
    args.simrdwn_dir = os.path.dirname(os.path.dirname(args.core_dir))
    args.results_topdir = os.path.join(args.simrdwn_dir, 'results')
    args.tf_cfg_dir = os.path.join('tf', 'cfg')
    args.yolt_plot_file = os.path.join(args.core_dir, 'yolt_plot_loss.py')
    args.tf_plot_file = os.path.join(args.core_dir, 'tf_plot_loss.py')

    # if train_data_dir is not a full directory, set it as within simrdwn
    if args.train_data_dir.startswith('/'):
        pass
    else:
        args.train_data_dir = os.path.join(args.simrdwn_dir, 'data/train_data')

    # keep raw testims dir if it starts with a '/'
    if args.testims_dir.startswith('/'):
        args.testims_dir_tot = args.testims_dir
    else:
        args.testims_dir_tot = os.path.join(args.simrdwn_dir,
                                            'data/test_images',
                                            args.testims_dir)
    # print ("os.listdir(args.testims_dir_tot:",
    #   os.listdir(args.testims_dir_tot))
    # ensure test ims exist
    if (args.mode.upper() == 'TEST') and \
            (not os.path.exists(args.testims_dir_tot)):
        raise ValueError("Test images directory does not exist: "
                         "{}".format(args.testims_dir_tot))

    if args.framework.upper().startswith('YOLT'):
        args.yolt_dir = os.path.join(args.simrdwn_dir, args.framework)
    else:
        args.yolt_dir = os.path.join(args.simrdwn_dir, 'yolt')
    args.yolt_weight_dir = os.path.join(args.yolt_dir, 'input_weights')
    args.yolt_cfg_dir = os.path.join(args.yolt_dir, 'cfg')

    ##########################################
    # Get datetime and set outlog file
    args.now = datetime.datetime.now()
    if bool(args.append_date_string):
        args.date_string = args.now.strftime('%Y_%m_%d_%H-%M-%S')
        # print "Date string:", date_string
        args.res_name = args.mode + '_' + args.framework + '_' + args.outname \
            + '_' + args.date_string
    else:
        args.date_string = ''
        args.res_name = args.mode + '_' + args.framework + '_' + args.outname

    args.results_dir = os.path.join(args.results_topdir, args.res_name)
    args.log_dir = os.path.join(args.results_dir, 'logs')
    args.log_file = os.path.join(args.log_dir, args.res_name + '.log')
    args.yolt_loss_file = os.path.join(args.log_dir, 'yolt_loss.txt')
    args.labels_log_file = os.path.join(args.log_dir, 'labels_list.txt')

    # set total location of test image file list
    args.test_presliced_list_tot = os.path.join(
        args.results_topdir, args.test_presliced_list)
    # args.test_presliced_list_tot = os.path.join(args.simrdwn_dir,
    #   args.test_presliced_list)
    if len(args.test_presliced_tfrecord_path) > 0:
        args.test_presliced_tfrecord_tot = os.path.join(
            args.results_topdir, args.test_presliced_tfrecord_path,
            'test_splitims.tfrecord')
        args.test_tfrecord_out = os.path.join(
            args.results_dir, 'predictions.tfrecord')
    else:
        args.test_presliced_tfrecord_tot = ''
        args.test_tfrecord_out = ''

    if len(args.test_presliced_list) > 0:
        args.test_splitims_locs_file = args.test_presliced_list_tot
    else:
        args.test_splitims_locs_file = os.path.join(
            args.results_dir, args.test_splitims_locs_file_root)
    # args.test_tfrecord_file = os.path.join(args.results_dir,
    #   args.test_tfrecord_root)
    # args.val_prediction_pkl = os.path.join(args.results_dir,
    #   args.test_prediction_pkl_root)
    # args.val_df_tfrecords_out = os.path.join(args.results_dir,
    #   'predictions.tfrecord')
    args.val_df_path_init = os.path.join(
        args.results_dir, args.val_df_root_init)
    args.val_df_path_aug = os.path.join(args.results_dir, args.val_df_root_aug)

    args.inference_graph_path_tot = os.path.join(
        args.results_topdir, args.train_model_path,
        'frozen_model/frozen_inference_graph.pb')

    # and yolt cfg file
    args.yolt_cfg_file_tot = os.path.join(args.log_dir, args.yolt_cfg_file)

    # weight and cfg files
    # TRAIN
    if args.mode.upper() == 'TRAIN':
        args.weight_file_tot = os.path.join(
            args.yolt_weight_dir, args.weight_file)
        # assume weights are in weight_dir, and cfg in cfg_dir
        args.yolt_cfg_file_in = os.path.join(
            args.yolt_cfg_dir, args.yolt_cfg_file)
        args.tf_cfg_train_file = os.path.join(
            args.tf_cfg_dir, args.tf_cfg_train_file)
    # TEST
    else:
        args.weight_file_tot = os.path.join(
            args.results_topdir, args.train_model_path, args.weight_file)
        args.tf_cfg_train_file = os.path.join(
            args.results_topdir, args.train_model_path,  # 'logs',
            args.tf_cfg_train_file)

        # assume weights and cfg are in the training dir
        args.yolt_cfg_file_in = os.path.join(os.path.dirname(
            args.weight_file_tot), 'logs/', args.yolt_cfg_file)

    # set training files (assume files are in train_data_dir unless a full
    #  path is given)
    if args.yolt_train_images_list_file.startswith('/'):
        args.yolt_train_images_list_file_tot = args.yolt_train_images_list_file
    else:
        args.yolt_train_images_list_file_tot = os.path.join(
            args.train_data_dir, args.yolt_train_images_list_file)

    # train tf record
    if args.train_tf_record.startswith('/'):
        pass
    else:
        args.train_tf_record = os.path.join(
            args.train_data_dir, args.train_tf_record)

    ##########################
    # set tf cfg file out
    tf_cfg_base = os.path.basename(args.tf_cfg_train_file)
    # tf_cfg_root = tf_cfg_base.split('.')[0]
    args.tf_cfg_train_file_out = os.path.join(args.log_dir, tf_cfg_base)
    args.tf_model_output_directory = os.path.join(
        args.results_dir, 'frozen_model')
    # args.tf_model_output_directory = os.path.join(args.results_dir,
    #   tf_cfg_root)

    ##########################
    # set possible extensions for image files
    args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                           '.jpg', '.JPEG', '.jpeg']

    # args.test_make_pngs = bool(args.test_make_pngs)
    args.test_make_legend_and_title = bool(args.test_make_legend_and_title)
    args.keep_test_slices = bool(args.keep_test_slices)
    args.test_add_geo_coords = bool(args.test_add_geo_coords)

    # set cuda values
    # if args.gpu >= 0:
    if args.gpu != "-1":
        args.use_GPU, args.use_CUDNN = 1, 1
    else:
        args.use_GPU, args.use_CUDNN = 0, 0

    # update label_map_path, if needed
    if (args.label_map_path.startswith('/')) or (len(args.label_map_path) == 0):
        pass
    else:
        args.label_map_path = os.path.join(args.train_data_dir,
                                           args.label_map_path)

    # make label_map_dic (key=int, value=str), and reverse
    if len(args.label_map_path) > 0:
        args.label_map_dict = preprocess_tfrecords.load_pbtxt(
            args.label_map_path, verbose=False)
        # ensure dict is 1-indexed
        if min(list(args.label_map_dict.keys())) != 1:
            print("Error: label_map_dict (", args.label_map_path, ") must"
                  " be 1-indexed")
            return
    else:
        args.label_map_dict = {}

    # retersed labels
    args.label_map_dict_rev = {v: k for k, v in args.label_map_dict.items()}
    # args.label_map_dict_rev = {v: k for k,v
    #   in args.label_map_dict.iteritems()}
    # print ("label_map_dict:", args.label_map_dict)

    # infer lists from args
    if len(args.yolt_object_labels_str) == 0:
        args.yolt_object_labels = [args.label_map_dict[ktmp] for ktmp in
                                   sorted(args.label_map_dict.keys())]
        args.yolt_object_labels_str = ','.join(args.yolt_object_labels)
    else:
        args.yolt_object_labels = args.yolt_object_labels_str.split(',')
        # also set label_map_dict, if it's empty
        if len(args.label_map_path) == 0:
            for itmp, val in enumerate(args.yolt_object_labels):
                args.label_map_dict[itmp] = val
            args.label_map_dict_rev = {v: k for k,
                                       v in args.label_map_dict.items()}
            # args.label_map_dict_rev = {v: k for k,v in
            #   args.label_map_dict.iteritems()}

    # set total dict
    args.label_map_dict_tot = copy.deepcopy(args.label_map_dict)
    args.label_map_dict_rev_tot = copy.deepcopy(args.label_map_dict_rev)

    args.yolt_classnum = len(args.yolt_object_labels)

    # for yolov2
    args.yolt_final_output = 1 * 1 * \
        args.boxes_per_grid * (args.yolt_classnum + 4 + 1)
    # for yolov3
    # make sure num boxes is divisible by 3
    if args.framework.upper() == 'YOLT3' and args.boxes_per_grid % 3 != 0:
        print("for YOLT3, boxes_per_grid must be divisble by 3!")
        print("RETURNING!")
        return
    args.yolov3_filters = int(args.boxes_per_grid /
                              3 * (args.yolt_classnum + 4 + 1))

    # plot thresh and slice sizes
    args.plot_thresh = np.array(
        args.plot_thresh_str.split(args.str_delim)).astype(float)
    args.slice_sizes = np.array(
        args.slice_sizes_str.split(args.str_delim)).astype(int)

    # set test list
    try:
        if args.nbands == 3:
            # print ("os.listdir(args.testims_dir_tot:",
            #   os.listdir(args.testims_dir_tot))
            args.test_ims_list = [f for f in os.listdir(args.testims_dir_tot)
                                  if f.endswith(tuple(args.extension_list))]
            # print("args.test_ims_list:", args.test_ims_list)
        else:
            args.test_ims_list = [f for f in os.listdir(args.testims_dir_tot)
                                  if f.endswith('#1.png')]
    except:
        args.test_ims_list = []
    # print ("test_ims_list:", args.test_ims_list)
    # more test files
    args.rotate_boxes = bool(args.rotate_boxes)
    args.yolt_test_classes_files = [os.path.join(args.results_dir, l + '.txt')
                                    for l in args.yolt_object_labels]

    ##########################
    # get second test classifier values
    args.slice_sizes2 = []
    if len(args.label_map_path2) > 0:

        # label dict
        args.label_map_dict2 = preprocess_tfrecords.load_pbtxt(
            args.label_map_path2, verbose=False)
        args.label_map_dict_rev2 = {v: k for k,
                                    v in args.label_map_dict2.items()}
        # args.label_map_dict_rev2 = {v: k for k,v in
        #   args.label_map_dict2.iteritems()}

        # to update label_map_dict just adds second classifier to first
        nmax_tmp = max(args.label_map_dict.keys())
        for ktmp, vtmp in args.label_map_dict2.items():
            # for ktmp, vtmp in args.label_map_dict2.iteritems():
            args.label_map_dict_tot[ktmp+nmax_tmp] = vtmp
        args.label_map_dict_rev_tot = {
            v: k for k, v in args.label_map_dict_tot.items()}
        # args.label_map_dict_rev_tot = {v: k for k,v in
        #   args.label_map_dict_tot.iteritems()}

        # infer lists from args
        args.yolt_object_labels2 = [
            args.label_map_dict2[ktmp]
            for ktmp in sorted(args.label_map_dict2.keys())]
        args.yolt_object_labels_str2 = ','.join(args.yolt_object_labels2)

        # set classnum and final output
        args.yolt_classnum2 = len(args.yolt_object_labels2)
        # for yolov2
        args.yolt_final_output2 = 1 * 1 * \
            args.boxes_per_grid * (args.yolt_classnum2 + 4 + 1)
        # for yolov3
        args.yolov3_filters2 = int(
            args.boxes_per_grid / 3 * (args.yolt_classnum2 + 4 + 1))

        # plot thresh and slice sizes
        args.plot_thresh2 = np.array(
            args.plot_thresh_str2.split(args.str_delim)).astype(float)
        args.slice_sizes2 = np.array(
            args.slice_sizes_str2.split(args.str_delim)).astype(int)

        # test files2
        args.yolt_test_classes_files2 = [
            os.path.join(args.results_dir, l + '.txt')
            for l in args.yolt_object_labels2]
        if len(args.test_presliced_list2) > 0:
            args.test_presliced_list_tot2 = os.path.join(
                args.simrdwn_dir, args.test_presliced_list2)
        else:
            args.test_splitims_locs_file2 = os.path.join(
                args.results_dir, args.test_splitims_locs_file_root2)
        args.test_tfrecord_out2 = os.path.join(
            args.results_dir, 'predictions2.tfrecord')
        args.val_df_path_init2 = os.path.join(
            args.results_dir, args.val_df_root_init2)
        args.val_df_path_aug2 = os.path.join(
            args.results_dir, args.val_df_root_aug2)
        args.weight_file_tot2 = os.path.join(
            args.results_topdir, args.train_model_path, args.weight_file2)
        args.yolt_cfg_file_tot2 = os.path.join(
            args.log_dir, args.yolt_cfg_file2)

        if args.mode == 'test':
            args.yolt_cfg_file_in2 = os.path.join(os.path.dirname(
                args.weight_file_tot2), 'logs/', args.yolt_cfg_file2)
        else:
            args.yolt_cfg_file_in2 = os.path.join(
                args.yolt_cfg_dir, args.yolt_cfg_file2)

        args.inference_graph_path_tot2 = os.path.join(
            args.results_topdir, args.train_model_path2,
            'frozen_model/frozen_inference_graph.pb')

    # total test
    args.val_df_path_tot = os.path.join(args.results_dir, args.val_df_root_tot)
    args.val_prediction_df_refine_tot = os.path.join(
        args.results_dir, args.val_prediction_df_refine_tot_root_part
        + '_thresh=' + str(args.plot_thresh[0]))

    # if evaluating spacenet
    if len(args.building_csv_file) > 0:
        args.building_csv_file = os.path.join(
            args.results_dir, args.building_csv_file)

    ##########################
    # Plotting params
    args.figsize = (12, 12)
    args.dpi = 300

    return args


###############################################################################
def update_tf_train_config(config_file_in, config_file_out,
                           label_map_path='',  train_tf_record='',
                           train_input_width=416, train_input_height=416,
                           train_val_tf_record='', num_steps=10000,
                           batch_size=32,
                           verbose=False):
    """
    Edit tf trainig config file to reflect proper paths and parameters

    Notes
    -----
    For details on how to set up the pipeline, see:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md 
    For example .config files:
        https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
        also located at: /raid/cosmiq/simrdwn/configs

    Arguments
    ---------
    config_file_in : str
        Path to input config file.
    config_file_out : str
        Path to output config file.
    label_map_path : str
        Path to label map file (required).  Defaults to ``''`` (empty).
    ...
        
    """

    #############
    # for now, set train_val_tf_record to train_tf_record!
    train_val_tf_record = train_tf_record
    #############

    # load pbtxt
    label_map_dict = preprocess_tfrecords.load_pbtxt(
        label_map_path, verbose=False)
    n_classes = len(list(label_map_dict.keys()))

    # print ("config_file_in:", config_file_in)
    fin = open(config_file_in, 'r')
    fout = open(config_file_out, 'w')
    line_minus_two = ''
    line_list = []
    for i, line in enumerate(fin):
        if verbose:
            print(i, line)
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
        # resizer
        elif line.strip().startswith('height:'):
            line_out = '        height: ' + str(train_input_height) + '\n'
        elif line.strip().startswith('width:'):
            line_out = '        width: ' + str(train_input_width) + '\n'
        # n classes
        elif line.strip().startswith('num_classes:'):
            line_out = '    num_classes: ' + str(n_classes) + '\n'

        else:
            line_out = line
        fout.write(line_out)

    fin.close()
    fout.close()


###############################################################################
def tf_train_cmd(tf_cfg_train_file, results_dir, max_batches=10000):
    """
    Train a model with tensorflow object detection api
    Example:
    python /opt/tensorflow-models/research/object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=/raid/local/src/simrdwn/tf/configs/ssd_inception_v2_simrdwn.config \
        --train_dir=/raid/local/src/simrdwn/outputs/ssd >> \
            train_ssd_inception_v2_simrdwn.log & tail -f train_ssd_inception_v2_simrdwn.log
    """

    # suffix = ' >> ' + log_file + ' & tail -f ' + log_file
    # suffix =  >> ' + log_file
    suffix = ''

    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md
    # PIPELINE_CONFIG_PATH={path to pipeline config file}
    # MODEL_DIR={path to model directory}
    # NUM_TRAIN_STEPS=50000
    # SAMPLE_1_OF_N_EVAL_EXAMPLES=1
    # python object_detection/model_main.py \
    #    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    #    --model_dir=${MODEL_DIR} \
    #    --num_train_steps=${NUM_TRAIN_STEPS} \
    #    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    #    --alsologtostderr
    cmd_arg_list = [
        'python',
        '/tensorflow/models/research/object_detection/model_main.py',
        '--pipeline_config_path=' + tf_cfg_train_file,
        '--model_dir=' + results_dir,
        '--num_train_steps=' + str(int(max_batches)),
        '--sample_1_of_n_eval_examples={}'.format(1),
        '--alsologtostderr',
        suffix
    ]

    # old version of tensorflow
    # cmd_arg_list = [
    #        'python',
    #        '/opt/tensorflow-models/research/object_detection/train.py',
    #        '--logtostderr',
    #        '--pipeline_config_path=' + tf_cfg_train_file,
    #        '--train_dir=' + results_dir,
    #        suffix
    #        ]

    cmd = ' '.join(cmd_arg_list)

    return cmd


###############################################################################
def tf_export_model_cmd(trained_dir='', tf_cfg_train_file='pipeline.config',
                        model_output_root='frozen_model', verbose=False):
    """Export trained model with tensorflow object detection api"""

    # get max training batches completed
    checkpoints_tmp = [ftmp for ftmp in os.listdir(trained_dir)
                       if ftmp.startswith('model.ckpt')]
    # print ("checkpoints tmp:", checkpoints_tmp)
    nums_tmp = [int(z.split('model.ckpt-')[-1].split('.')[0])
                for z in checkpoints_tmp]
    # print ("nums_tmp:", nums_tmp)
    num_max_tmp = np.max(nums_tmp)
    if verbose:
        print("tf_export_model_cmd() - checkpoints_tmp:", checkpoints_tmp)
        print("tf_export_model_cmd() - num_max_tmp:", num_max_tmp)

    cmd_arg_list = [
        'python',
        '/tensorflow/models/research/object_detection/export_inference_graph.py',
        # '/opt/tensorflow-models/research/object_detection/export_inference_graph.py',
        '--input_type image_tensor',
        '--pipeline_config_path=' + \
        os.path.join(trained_dir, tf_cfg_train_file),
        '--trained_checkpoint_prefix=' + \
        os.path.join(trained_dir, 'model.ckpt-' + str(num_max_tmp)),
        # '--trained_checkpoint_prefix=' + os.path.join(results_dir, 'model.ckpt-' + str(num_steps)),
        '--output_directory=' + \
        os.path.join(trained_dir, model_output_root)
    ]

    cmd = ' '.join(cmd_arg_list)
    if verbose:
        print("tf_export_model_cmd() - output cmd:", cmd)


    return cmd


###############################################################################
def tf_infer_cmd_dual(inference_graph_path='',
                      input_file_list='',
                      in_tfrecord_path='',
                      out_tfrecord_path='',
                      use_tfrecords=0,
                      min_thresh=0.05,
                      GPU=0,
                      BGR2RGB=0,
                      output_csv_path='',
                      infer_src_path='/raid/local/src/simrdwn/core'):
    """
    Run infer_detections.py with the given input tfrecord or input_file_list

    Infer output tfrecord
    Example:
        python /raid/local/src/simrdwn/src/infer_detections.py \
                --input_tfrecord_paths=/raid/local/src/simrdwn/data/qgis_labels_car_boat_plane_val.tfrecord \
                --inference_graph=/raid/local/src/simrdwn/outputs/ssd/output_inference_graph/frozen_inference_graph.pb \
                --output_tfrecord_path=/raid/local/src/simrdwn/outputs/ssd/val_detections_ssd.tfrecord 
     df.to_csv(outfile_df)
    """

    cmd_arg_list = [
        'python',
        infer_src_path + '/' + 'infer_detections.py',
        '--inference_graph=' + inference_graph_path,
        '--GPU=' + str(GPU)
        ]
    if bool(use_tfrecords):
        cmd_arg_list.extend(['--use_tfrecord=' + str(use_tfrecords)])
        
    cmd_arg_list.extend([
        # first method, with tfrecords
        '--input_tfrecord_paths=' + in_tfrecord_path,
        '--output_tfrecord_path=' + out_tfrecord_path,
        # second method, with file list
        '--input_file_list=' + input_file_list,
        '--BGR2RGB=' + str(BGR2RGB),
        '--output_csv_path=' + output_csv_path,
        '--min_thresh=' + str(min_thresh)
    ])
    cmd = ' '.join(cmd_arg_list)

    return cmd


###############################################################################
def yolt_command(framework='yolt2',
                 yolt_cfg_file_tot='',
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
                 test_splitims_locs_file='',
                 test_im_tot='',
                 test_thresh=0.2,
                 yolt_nms_thresh=0,
                 min_retain_prob=0.025):
    """
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
    //char *test_image = (argc >10) ? argv[10]: 0;
    char *test_list_loc = (argc > 10) ? argv[10]: 0;
    char *names_str = (argc > 11) ? argv[11]: 0;
    int len_names = (argc > 12) ? atoi(argv[12]): 0;
    int nbands = (argc > 13) ? atoi(argv[13]): 0;
    char *loss_file = (argc > 14) ? argv[14]: 0;
    """

    ##########################
    # set gpu command
    if single_gpu_machine == 1:  # use_aware:
        gpu_cmd = ''
    else:
        gpu_cmd = '-i ' + str(gpu)
        # gpu_cmd = '-i ' + str(3-args.gpu) # originally, numbers were reversed
    ngpus = len(gpu.split(','))

    ##########################
    # SET VARIABLES ACCORDING TO MODE (SET UNNECCESSARY VALUES TO 0 OR NULL)
    # set train prams (and prefix, and suffix)
    if mode == 'train':
        mode_str = 'train'
        train_ims = yolt_train_images_list_file_tot
        prefix = 'nohup'
        suffix = ' >> ' + log_file + ' & tail -f ' + log_file
    else:
        train_ims = 'null'
        prefix = ''
        suffix = ' 2>&1 | tee -a ' + log_file

    # set test deprecated params
    if mode == 'test_deprecated':
        test_im = test_im_tot
        test_thresh = test_thresh
    else:
        test_im = 'null'
        test_thresh = 0

    # set test params
    if mode == 'test':
        mode_str = 'valid'
        # test_image = args.test_image_tmp
        test_list_loc = test_splitims_locs_file
    else:
        # test_image = 'null'
        test_list_loc = 'null'

    ##########################

    c_arg_list = [
        prefix,
        './' + framework.lower() + '/darknet',
        gpu_cmd,
        framework,  # 'yolt2',
        mode_str,
        yolt_cfg_file_tot,
        weight_file_tot,
        test_im,
        str(test_thresh),
        str(yolt_nms_thresh),
        train_ims,
        results_dir,
        test_list_loc,
        yolt_object_labels_str,
        str(yolt_classnum),
        str(nbands),
        yolt_loss_file,
        str(min_retain_prob),
        str(ngpus),
        suffix
    ]

    cmd = ' '.join(c_arg_list)

    print("Command:\n", cmd)

    return cmd


###############################################################################
def recompile_darknet(yolt_dir):
    """compile darknet"""
    os.chdir(yolt_dir)
    cmd_compile0 = 'make clean'
    cmd_compile1 = 'make'

    print(cmd_compile0)
    utils._run_cmd(cmd_compile0)

    print(cmd_compile1)
    utils._run_cmd(cmd_compile1)


###############################################################################
def replace_yolt_vals_train_compile(framework='yolt2',
                                    yolt_dir='',
                                    mode='train',
                                    yolt_cfg_file_tot='',
                                    yolt_final_output='',
                                    yolt_classnum=2,
                                    nbands=3,
                                    max_batches=512,
                                    batch_size=16,
                                    subdivisions=4,
                                    boxes_per_grid=5,
                                    train_input_width=416,
                                    train_input_height=416,
                                    yolov3_filters=0,
                                    use_GPU=1,
                                    use_opencv=1,
                                    use_CUDNN=1):
    """
    For either training or compiling,
    edit cfg file in darknet to allow for custom models
    editing of network layers must be done in vi, this function just changes
    parameters such as window size, number of trianing steps, etc
    """

    print("Replacing YOLT vals...")

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
        utils._run_cmd('cp ' + yoltm + ' ' + yoltm + '_v0')
        # write new file over old
        utils._run_cmd('mv ' + yoltm_tmp + ' ' + yoltm)

    #################
    # cfg file
    elif mode == 'train':
        yoltcfg = yolt_cfg_file_tot
        # print ("\n\nyolt_cfg_file_tot:", yolt_cfg_file_tot)
        yoltcfg_tmp = yoltcfg + 'tmp'
        f1 = open(yoltcfg, 'r')
        f2 = open(yoltcfg_tmp, 'w')
        # read in reverse because we want to edit the last output length
        s = f1.readlines()
        s.reverse()
        sout = []

        fixed_output = False
        for i, line in enumerate(s):
            # print ("line:", line)

            if i > 3:
                lm4 = sout[i-4]
            else:
                lm4 = ''

            # if line.strip().startswith('side='):
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

            # replace num in yolov2
            elif (framework.upper() == 'YOLT2') and (line.strip().startswith('num=')):
                line_out = 'num=' + str(boxes_per_grid) + '\n'
            elif (framework.upper() in ['YOLT2', 'YOLT3']) and line.strip().startswith('width='):
                line_out = 'width=' + str(train_input_width) + '\n'
            elif (framework.upper() in ['YOLT2', 'YOLT3']) and line.strip().startswith('height='):
                line_out = 'height=' + str(train_input_height) + '\n'
            # change final output, and set fixed to true
            # elif (line.strip().startswith('output=')) and (not fixed_output):
            #    line_out = 'output=' + str(final_output) + '\n'
            #    fixed_output=True
            elif (framework.upper() == 'YOLT2') and (line.strip().startswith('filters=')) and (not fixed_output):
                line_out = 'filters=' + str(yolt_final_output) + '\n'
                fixed_output = True

            # line before a yolo layer should have 3 * (n_classes + 5) filters
            elif (framework.upper() == 'YOLT3') and (line.strip().startswith('filters=')) and (lm4.startswith('[yolo]')):
                line_out = 'filters=' + str(yolov3_filters) + '\n'
                print(i, "lm4:", lm4, "line_out:", line_out)
                print("args.yolov3_filters", yolov3_filters)
                # return

            else:
                line_out = line
            sout.append(line_out)

        sout.reverse()
        for line in sout:
            f2.write(line)

        f1.close()
        f2.close()

        # copy old yoltcfg?
        utils._run_cmd('cp ' + yoltcfg + ' ' + yoltcfg[:-4] + 'orig.cfg')
        # write new file over old
        utils._run_cmd('mv ' + yoltcfg_tmp + ' ' + yoltcfg)
    #################

    else:
        return


###############################################################################
def split_test_im(im_root_with_ext, testims_dir_tot, results_dir,
                  log_file,
                  slice_sizes=[416],
                  slice_overlap=0.2,
                  test_slice_sep='__',
                  zero_frac_thresh=0.5,
                  ):
    """
    Split files for test step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')

    1. get image path (args.test_image_tmp) from image root name
            (args.test_image_tmp)
    2. slice test image and move to results dir
    """

    # get image root, make sure there is no extension
    im_root = im_root_with_ext.split('.')[0]
    im_path = os.path.join(testims_dir_tot, im_root_with_ext)

    # slice test plot into manageable chunks

    # slice (if needed)
    if slice_sizes[0] > 0:
        # if len(args.slice_sizes) > 0:
        # create test_splitims_locs_file
        # set test_dir as in results_dir
        test_split_dir = os.path.join(results_dir,  im_root + '_split' + '/')
        test_dir_str = '"test_split_dir: ' + test_split_dir + '\n"'
        print("test_dir:", test_dir_str[1:-2])
        os.system('echo ' + test_dir_str + ' >> ' + log_file)
        # print "test_split_dir:", test_split_dir

        # clean out dir, and make anew
        if os.path.exists(test_split_dir):
            if (not test_split_dir.startswith(results_dir)) \
                    or len(test_split_dir) < len(results_dir) \
                    or len(test_split_dir) < 10:
                print("test_split_dir too short!!!!:", test_split_dir)
                return
            shutil.rmtree(test_split_dir, ignore_errors=True)
        os.mkdir(test_split_dir)

        # slice
        for s in slice_sizes:
            slice_im.slice_im(im_path, im_root,
                              test_split_dir, s, s,
                              zero_frac_thresh=zero_frac_thresh,
                              overlap=slice_overlap,
                              slice_sep=test_slice_sep)
            test_files = [os.path.join(test_split_dir, f) for
                          f in os.listdir(test_split_dir)]
        n_files_str = '"Num files: ' + str(len(test_files)) + '\n"'
        print(n_files_str[1:-2])
        os.system('echo ' + n_files_str + ' >> ' + log_file)

    else:
        test_files = [im_path]
        test_split_dir = os.path.join(results_dir, 'nonsense')

    return test_files, test_split_dir


###############################################################################
def prep_test_files(results_dir, log_file, test_ims_list,
                    testims_dir_tot, test_splitims_locs_file,
                    slice_sizes=[416],
                    slice_overlap=0.2,
                    test_slice_sep='__',
                    zero_frac_thresh=0.5,
                    ):
    """Split images and save split image locations to txt file"""

    # split test images, store locations
    t0 = time.time()
    test_split_str = '"Splitting test files...\n"'
    print(test_split_str[1:-2])
    os.system('echo ' + test_split_str + ' >> ' + log_file)
    print("test_ims_list:", test_ims_list)

    test_files_locs_list = []
    test_split_dir_list = []
    # !! Should make a tfrecord when we split files, instead of doing it later
    for i, test_base_tmp in enumerate(test_ims_list):
        iter_string = '"\n' + str(i+1) + ' / ' + \
            str(len(test_ims_list)) + '\n"'
        print(iter_string[1:-2])
        os.system('echo ' + iter_string + ' >> ' + log_file)
        # print "\n", i+1, "/", len(args.test_ims_list)

        # dirty hack: ignore file extensions for now
        # test_base_tmp_noext = test_base_tmp.split('.')[0]
        # test_base_string = '"test_base_tmp_noext:' \
        #                    + str(test_base_tmp_noext) + '\n"'
        test_base_string = '"test_file: ' + str(test_base_tmp) + '\n"'
        print(test_base_string[1:-2])
        os.system('echo ' + test_base_string + ' >> ' + log_file)

        # split data
        # test_files_list_tmp, test_split_dir_tmp = split_test_im(test_base_tmp, args)
        test_files_list_tmp, test_split_dir_tmp = \
            split_test_im(test_base_tmp, testims_dir_tot,
                          results_dir, log_file,
                          slice_sizes=slice_sizes,
                          slice_overlap=slice_overlap,
                          test_slice_sep=test_slice_sep,
                          zero_frac_thresh=zero_frac_thresh)
        # add test_files to list
        test_files_locs_list.extend(test_files_list_tmp)
        test_split_dir_list.append(test_split_dir_tmp)

    # swrite test_files_locs_list to file (file = test_splitims_locs_file)
    print("Total len test files:", len(test_files_locs_list))
    print("test_splitims_locs_file:", test_splitims_locs_file)
    # write list of files to test_splitims_locs_file
    with open(test_splitims_locs_file, "w") as fp:
        for line in test_files_locs_list:
            if not line.endswith('.DS_Store'):
                fp.write(line + "\n")

    t1 = time.time()
    cmd_time_str = '"\nLength of time to split test files: ' \
        + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str[1:-2])
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)

    return test_files_locs_list, test_split_dir_list


###############################################################################
def run_test(framework='YOLT2',
             infer_cmd='',
             results_dir='',
             log_file='',
             n_files=0,
             test_tfrecord_out='',
             slice_sizes=[416],
             testims_dir_tot='',
             yolt_test_classes_files='',
             label_map_dict={},
             val_df_path_init='',
             test_slice_sep='__',
             edge_buffer_test=1,
             max_edge_aspect_ratio=4,
             test_box_rescale_frac=1.0,
             rotate_boxes=False,
             min_retain_prob=0.05,
             test_add_geo_coords=True,
             verbose=False
             ):
    """Evaluate multiple large images"""

    t0 = time.time()
    # run command
    print("Running", infer_cmd)
    os.system('echo ' + infer_cmd + ' >> ' + log_file)
    os.system(infer_cmd)  # run_cmd(outcmd)
    t1 = time.time()
    cmd_time_str = '"\nLength of time to run command: ' + infer_cmd \
        + ' for ' + str(n_files) + ' cutouts: ' \
        + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str[1:-1])
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)

    if framework.upper() not in ['YOLT2', 'YOLT3']:

        # if we ran inference with a tfrecord, we must now parse that into
        #   a dataframe
        if len(test_tfrecord_out) > 0:
            df_init = parse_tfrecord.tf_to_df(
                test_tfrecord_out, max_iter=500000,
                label_map_dict=label_map_dict, tf_type='test',
                output_columns=['Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                u'Xmax', u'Ymax', u'Category'],
                # replace_paths=()
                )
            # use numeric categories
            label_map_dict_rev = {v: k for k, v in label_map_dict.items()}
            # label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}
            df_init['Category'] = [label_map_dict_rev[vtmp]
                                   for vtmp in df_init['Category'].values]
            # save to file
            df_init.to_csv(val_df_path_init)
        else:
            print("Read in val_df_path_init:", val_df_path_init)
            df_init = pd.read_csv(val_df_path_init
                                  # names=[u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                  #       u'Xmax', u'Ymax', u'Category']
                                  )

        #########
        # post process
        print("len df_init:", len(df_init))
        df_init.index = np.arange(len(df_init))

        # clean out low probabilities
        print("minimum retained threshold:",  min_retain_prob)
        bad_idxs = df_init[df_init['Prob'] < min_retain_prob].index
        if len(bad_idxs) > 0:
            print("bad idxss:", bad_idxs)
            df_init.drop(df_init.index[bad_idxs], inplace=True)

        # clean out bad categories
        df_init['Category'] = df_init['Category'].values.astype(int)
        good_cats = list(label_map_dict.keys())
        print("Allowed categories:", good_cats)
        # print ("df_init0['Category'] > np.max(good_cats)", df_init['Category'] > np.max(good_cats))
        # print ("df_init0[df_init0['Category'] > np.max(good_cats)]", df_init[df_init['Category'] > np.max(good_cats)])
        bad_idxs2 = df_init[df_init['Category'] > np.max(good_cats)].index
        if len(bad_idxs2) > 0:
            print("label_map_dict:", label_map_dict)
            print("df_init['Category']:", df_init['Category'])
            print("bad idxs2:", bad_idxs2)
            df_init.drop(df_init.index[bad_idxs2], inplace=True)

        # set index as sequential
        df_init.index = np.arange(len(df_init))

        # df_init = df_init0[df_init0['Category'] <= np.max(good_cats)]
        # if (len(df_init) != len(df_init0)):
        #    print (len(df_init0) - len(df_init), "rows cleaned out")

        # tf_infer_cmd outputs integer categories, update to strings
        df_init['Category'] = [label_map_dict[ktmp]
                               for ktmp in df_init['Category'].values]

        print("len df_init after filtering:", len(df_init))

        # augment dataframe columns
        df_tot = post_process.augment_df(
            df_init,
            testims_dir_tot=testims_dir_tot,
            slice_sizes=slice_sizes,
            slice_sep=test_slice_sep,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            test_box_rescale_frac=test_box_rescale_frac,
            rotate_boxes=rotate_boxes,
            verbose=True)

    else:
        # post-process
        # df_tot = post_process_yolt_test_create_df(args)
        df_tot = post_process.post_process_yolt_test_create_df(
            yolt_test_classes_files, log_file,
            testims_dir_tot=testims_dir_tot,
            slice_sizes=slice_sizes,
            slice_sep=test_slice_sep,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            test_box_rescale_frac=test_box_rescale_frac,
            rotate_boxes=rotate_boxes)

    ###########################################
    # plot

    # add geo coords to eall boxes?
    if test_add_geo_coords and len(df_tot) > 0:
        ###########################################
        # !!!!! Skip?
        # json = None
        ###########################################
        df_tot, json = add_geo_coords.add_geo_coords_to_df(
            df_tot, create_geojson=False, inProj_str='epsg:4326',
            outProj_str='epsg:3857', verbose=verbose)
    else:
        json = None

    return df_tot, json


###############################################################################
def prep(args):
    """Prep data for train or test

    Arguments
    ---------
    args : Namespace
        input arguments

    Returns
    -------
    train_cmd1 : str
        Training command
    test_cmd_tot : str
        Testing command
    test_cmd_tot2 : str
        Testing command for second scale (optional)
    """

    # initialize commands to null strings
    train_cmd1, test_cmd_tot, test_cmd_tot2 = '', '', ''

    print("\nSIMRDWN now...\n")
    os.chdir(args.simrdwn_dir)
    print("cwd:", os.getcwd())
    # t0 = time.time()

    # make dirs
    os.mkdir(args.results_dir)
    os.mkdir(args.log_dir)

    # create log file
    print("Date string:", args.date_string)
    os.system('echo ' + str(args.date_string) + ' > ' + args.log_file)
    # init to the contents in this file?
    # os.system('cat ' + args.this_file + ' >> ' + args.log_file)
    args_str = '"\nArgs: ' + str(args) + '\n"'
    print(args_str[1:-1])
    os.system('echo ' + args_str + ' >> ' + args.log_file)

    # copy this file (yolt_run.py) as well as config, plot file to results_dir
    shutil.copy2(args.this_file, args.log_dir)
    # shutil.copy2(args.yolt_plot_file, args.log_dir)
    # shutil.copy2(args.tf_plot_file, args.log_dir)
    print("log_dir:", args.log_dir)

    # print ("\nlabel_map_dict:", args.label_map_dict)
    print("\nlabel_map_dict_tot:", args.label_map_dict_tot)
    # print ("object_labels:", args.object_labels)
    print("yolt_object_labels:", args.yolt_object_labels)
    print("yolt_classnum:", args.yolt_classnum)

    # save labels to log_dir
    # pickle.dump(args.object_labels, open(args.log_dir \
    #                                + 'labels_list.pkl', 'wb'), protocol=2)
    with open(args.labels_log_file, "w") as fp:
        for ob in args.yolt_object_labels:
            fp.write(str(ob) + "\n")

    # set YOLT values, if desired
    if (args.framework.upper() == 'YOLT2') \
            or (args.framework.upper() == 'YOLT3'):

        # copy files to log dir
        shutil.copy2(args.yolt_plot_file, args.log_dir)
        shutil.copy2(args.yolt_cfg_file_in, args.log_dir)
        os.system('cat ' + args.yolt_cfg_file_tot + ' >> ' + args.log_file)
        # print config values
        print("yolt_cfg_file:", args.yolt_cfg_file_in)
        if args.mode.upper() in ['TRAIN', 'COMPILE']:
            print("Updating yolt params in files...")
            replace_yolt_vals_train_compile(
                framework=args.framework,
                yolt_dir=args.yolt_dir,
                mode=args.mode,
                yolt_cfg_file_tot=args.yolt_cfg_file_tot,
                yolt_final_output=args.yolt_final_output,
                yolt_classnum=args.yolt_classnum,
                nbands=args.nbands,
                yolov3_filters=args.yolov3_filters,
                max_batches=args.max_batches,
                batch_size=args.batch_size,
                subdivisions=args.subdivisions,
                boxes_per_grid=args.boxes_per_grid,
                train_input_width=args.train_input_width,
                train_input_height=args.train_input_height,
                use_GPU=args.use_GPU,
                use_opencv=args.use_opencv,
                use_CUDNN=args.use_CUDNN)
            # replace_yolt_vals(args)
            # print a few values...
            print("Final output layer size:", args.yolt_final_output)
            # print ("side size:", args.side)
            print("batch_size:", args.batch_size)
            print("subdivisions:", args.subdivisions)

        if args.mode.upper() == 'COMPILE':
            print("Recompiling yolt...")
            recompile_darknet(args.yolt_dir)
            return

        # set yolt command
        yolt_cmd = yolt_command(
            args.framework, yolt_cfg_file_tot=args.yolt_cfg_file_tot,
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
            test_splitims_locs_file=args.test_splitims_locs_file,
            yolt_nms_thresh=args.yolt_nms_thresh,
            min_retain_prob=args.min_retain_prob)

        if args.mode.upper() == 'TRAIN':
            print("yolt_train_cmd:", yolt_cmd)
            # train_cmd_tot = yolt_cmd
            train_cmd1 = yolt_cmd
            # train_cmd2 = ''
        # set second test command
        elif args.mode.upper() == 'TEST':
            test_cmd_tot = yolt_cmd
        else:
            print("Error: Unknown execution type (should be train or test)")
            return

        if len(args.label_map_path2) > 0:
            test_cmd_tot2 = yolt_command(
                args.framework, yolt_cfg_file_tot=args.yolt_cfg_file_tot2,
                weight_file_tot=args.weight_file_tot2,
                results_dir=args.results_dir,
                log_file=args.log_file,
                mode=args.mode,
                yolt_object_labels_str=args.yolt_object_labels_str2,
                classnum=args.yolt_classnum2,
                nbands=args.nbands,
                gpu=args.gpu,
                single_gpu_machine=args.single_gpu_machine,
                test_splitims_locs_file=args.test_splitims_locs_file2,
                yolt_nms_thresh=args.yolt_nms_thresh,
                min_retain_prob=args.min_retain_prob)

        else:
            test_cmd_tot2 = ''

    # set tensor flow object detection API values
    else:

        if args.mode.upper() == 'TRAIN':
            if not os.path.exists(args.tf_model_output_directory):
                os.mkdir(args.tf_model_output_directory)
            # copy plot file to output dir
            shutil.copy2(args.tf_plot_file, args.log_dir)

            print("Updating tf_config...")
            update_tf_train_config(
                args.tf_cfg_train_file, args.tf_cfg_train_file_out,
                label_map_path=args.label_map_path,
                train_tf_record=args.train_tf_record,
                train_val_tf_record=args.train_val_tf_record,
                train_input_width=args.train_input_width,
                train_input_height=args.train_input_height,
                batch_size=args.batch_size,
                num_steps=args.max_batches)
            # define train command
            cmd_train_tf = tf_train_cmd(args.tf_cfg_train_file_out,
                                        args.results_dir,
                                        args.max_batches)

            # export command
            # cmd_export_tf = ''
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

            # train_cmd1 = cmd_train_tf
            train_cmd1 = 'nohup ' + cmd_train_tf + ' >> ' + args.log_file \
                + ' & tail -f ' + args.log_file + ' &'

            # train_cmd2 = 'nohup ' +  cmd_export_tf + ' >> ' + args.log_file \
            #                + ' & tail -f ' + args.log_file #+ ' &'

            # forget about nohup since we're inside docker?
            # train_cmd1 = cmd_train_tf
            # train_cmd2 = cmd_export_tf

        # test
        else:

            # define inference (test) command   (output to csv)
            test_cmd_tot = tf_infer_cmd_dual(
                inference_graph_path=args.inference_graph_path_tot,
                input_file_list=args.test_splitims_locs_file,
                in_tfrecord_path=args.test_presliced_tfrecord_tot,
                out_tfrecord_path=args.test_tfrecord_out,
                output_csv_path=args.val_df_path_init,
                min_thresh=args.min_retain_prob,
                BGR2RGB=args.BGR2RGB,
                use_tfrecords=args.use_tfrecords,
                infer_src_path=args.core_dir)

            # if using dual classifiers
            if len(args.label_map_path2) > 0:
                # check if model exists, if not, create it.
                if not os.path.exists(args.inference_graph_path_tot2):
                    inference_graph_path_tmp = os.path.dirname(
                        args.inference_graph_path_tot2)
                    cmd_tmp = 'python  ' \
                        + args.core_dir + '/export_model.py ' \
                        + '--results_dir ' + inference_graph_path_tmp
                    t1 = time.time()
                    print("Running", cmd_tmp, "...\n")
                    os.system(cmd_tmp)
                    t2 = time.time()
                    cmd_time_str = '"Length of time to run command: ' \
                        + cmd_tmp + ' ' \
                        + str(t2 - t1) + ' seconds\n"'
                    print(cmd_time_str[1:-1])
                    os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
                # set inference command
                test_cmd_tot2 = tf_infer_cmd_dual(
                    inference_graph_path=args.inference_graph_path_tot2,
                    input_file_list=args.test_splitims_locs_file2,
                    output_csv_path=args.val_df_path_init2,
                    min_thresh=args.min_retain_prob,
                    GPU=args.gpu,
                    BGR2RGB=args.BGR2RGB,
                    use_tfrecords=args.use_tfrecords,
                    infer_src_path=args.core_dir)
            else:
                test_cmd_tot2 = ''

    return train_cmd1, test_cmd_tot, test_cmd_tot2


###############################################################################
def execute(args, train_cmd1, test_cmd_tot, test_cmd_tot2=''):
    """
    Execute train or test

    Arguments
    ---------
    args : Namespace
        input arguments
    train_cmd1 : str
        Training command
    test_cmd_tot : str
        Testing command
    test_cmd_tot2 : str
        Testing command for second scale (optional)

    Returns
    -------
    None
    """

    # Execute
    if args.mode.upper() == 'TRAIN':

        t1 = time.time()
        print("Running", train_cmd1, "...\n\n")

        os.system(train_cmd1)
        # utils._run_cmd(train_cmd1)
        t2 = time.time()
        cmd_time_str = '"Length of time to run command: ' \
            + train_cmd1 + ' ' \
            + str(t2 - t1) + ' seconds\n"'
        print(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        # export trained model, if using tf object detection api?
        if 2 < 1 and (args.framework.upper() not in ['YOLT2', 'YOLT3']):
            cmd_export_tf = tf_export_model_cmd(
                args.tf_cfg_train_file_out,
                tf_cfg_train_file=args.tf_cfg_train_file,
                model_output_root='frozen_model')
            # args.results_dir,
            # args.tf_model_output_directory,
            # tf_cfg_train_file=args.tf_cfg_train_file)
            train_cmd2 = cmd_export_tf

            t1 = time.time()
            print("Running", train_cmd2, "...\n\n")
            # utils._run_cmd(train_cmd2)
            os.system(train_cmd2)
            t2 = time.time()
            cmd_time_str = '"Length of time to run command: ' \
                + train_cmd2 + ' ' \
                + str(t2 - t1) + ' seconds\n"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

    # need to split file for test first, then run command
    elif args.mode.upper() == 'TEST':

        t3 = time.time()
        # load presliced data, if desired
        if len(args.test_presliced_list) > 0:
            print("Loading args.test_presliced_list:",
                  args.test_presliced_list_tot)
            ftmp = open(args.test_presliced_list_tot, 'r')
            test_files_locs_list = [line.strip() for line in ftmp.readlines()]
            ftmp.close()
            test_split_dir_list = []
            print("len test_files_locs_list:", len(test_files_locs_list))
        elif len(args.test_presliced_tfrecord_path) > 0:
            print("Using", args.test_presliced_tfrecord_path)
            test_split_dir_list = []
        # split large test files
        else:
            print("Prepping test files")
            test_files_locs_list, test_split_dir_list =\
                prep_test_files(args.results_dir, args.log_file,
                                args.test_ims_list,
                                args.testims_dir_tot,
                                args.test_splitims_locs_file,
                                slice_sizes=args.slice_sizes,
                                slice_overlap=args.slice_overlap,
                                test_slice_sep=args.test_slice_sep,
                                zero_frac_thresh=args.zero_frac_thresh,
                                )
            # return if only interested in prepping
            if (bool(args.test_prep_only)) \
                    and (bool(args.use_tfrecords)):
                    # or (args.framework.upper() not in ['YOLT2', 'YOLT3']):
                print("Convert to tfrecords...")
                TF_RecordPath = os.path.join(
                    args.results_dir, 'test_splitims.tfrecord')
                preprocess_tfrecords.yolt_imlist_to_tf(
                    args.test_splitims_locs_file,
                    args.label_map_dict, TF_RecordPath,
                    TF_PathVal='', val_frac=0.0,
                    convert_dict={}, verbose=False)
                print("Done prepping test files, ending")
                return

        # check if trained model exists, if not, create it.
        if (args.framework.upper() not in ['YOLT2', 'YOLT3']) and \
            (not (os.path.exists(args.inference_graph_path_tot)) or
             (args.overwrite_inference_graph != 0)):
            print("Creating args.inference_graph_path_tot:",
                  args.inference_graph_path_tot, "...")

            # remove "saved_model" directory
            saved_dir = os.path.join(
                os.path.dirname(args.inference_graph_path_tot), 'saved_model')
            print("Removing", saved_dir, "so we can overwrite it...")
            if os.path.exists(saved_dir):
                shutil.rmtree(saved_dir, ignore_errors=True)

            trained_dir_tmp = os.path.dirname(
                os.path.dirname(args.inference_graph_path_tot))
            cmd_tmp = tf_export_model_cmd(
                trained_dir=trained_dir_tmp,
                tf_cfg_train_file=args.tf_cfg_train_file)

            # cmd_tmp = 'python  ' \
            #            + args.core_dir + '/export_model.py ' \
            #            + '--results_dir=' + inference_graph_path_tmp
            t1 = time.time()
            print("Running export command:", cmd_tmp, "...\n")
            os.system(cmd_tmp)
            t2 = time.time()
            cmd_time_str = '"Length of time to run command: ' \
                + cmd_tmp + ' ' \
                + str(t2 - t1) + ' seconds\n"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        df_tot, json = run_test(infer_cmd=test_cmd_tot,
                                framework=args.framework,
                                results_dir=args.results_dir,
                                log_file=args.log_file,
                                # test_files_locs_list=test_files_locs_list,
                                # test_presliced_tfrecord_tot=args.test_presliced_tfrecord_tot,
                                test_tfrecord_out=args.test_tfrecord_out,
                                slice_sizes=args.slice_sizes,
                                testims_dir_tot=args.testims_dir_tot,
                                yolt_test_classes_files=args.yolt_test_classes_files,
                                label_map_dict=args.label_map_dict,
                                val_df_path_init=args.val_df_path_init,
                                # val_df_path_aug=args.val_df_path_aug,
                                min_retain_prob=args.min_retain_prob,
                                test_slice_sep=args.test_slice_sep,
                                edge_buffer_test=args.edge_buffer_test,
                                max_edge_aspect_ratio=args.max_edge_aspect_ratio,
                                test_box_rescale_frac=args.test_box_rescale_frac,
                                rotate_boxes=args.rotate_boxes,
                                test_add_geo_coords=args.test_add_geo_coords)

        if len(df_tot) == 0:
            print("No detections found!")
        else:
            # save to csv
            df_tot.to_csv(args.val_df_path_aug, index=False)
            # get number of files
            n_files = len(np.unique(df_tot['Loc_Tmp'].values))
            # n_files = str(len(test_files_locs_list)
            t4 = time.time()
            cmd_time_str = '"Length of time to run test for ' \
                + str(n_files) + ' files = ' \
                + str(t4 - t3) + ' seconds\n"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        # run again, if desired
        if len(args.weight_file2) > 0:

            t5 = time.time()
            # split large testion files
            print("Prepping test files")
            test_files_locs_list2, test_split_dir_list2 =\
                prep_test_files(args.results_dir, args.log_file,
                                args.test_ims_list,
                                args.testims_dir_tot,
                                args.test_splitims_locs_file2,
                                slice_sizes=args.slice_sizes2,
                                slice_overlap=args.slice_overlap,
                                test_slice_sep=args.test_slice_sep,
                                zero_frac_thresh=args.zero_frac_thresh,
                                )

            df_tot2 = run_test(infer_cmd=test_cmd_tot2,
                               framework=args.framework,
                               results_dir=args.results_dir,
                               log_file=args.log_file,
                               test_files_locs_list=test_files_locs_list2,
                               slice_sizes=args.slice_sizes,
                               testims_dir_tot=args.testims_dir_tot2,
                               yolt_test_classes_files=args.yolt_test_classes_files2,
                               label_map_dict=args.label_map_dict2,
                               val_df_path_init=args.val_df_path_init2,
                               # val_df_path_aug=args.val_df_path_aug2,
                               test_slice_sep=args.test_slice_sep,
                               edge_buffer_test=args.edge_buffer_test,
                               max_edge_aspect_ratio=args.max_edge_aspect_ratio,
                               test_box_rescale_frac=args.test_box_rescale_frac,
                               rotate_boxes=args.rotate_boxes,
                               test_add_geo_coords=args.test_add_geo_coords)

            # save to csv
            df_tot2.to_csv(args.val_df_path_aug2, index=False)
            t6 = time.time()
            cmd_time_str = '"Length of time to run test' + ' ' \
                + str(t6 - t5) + ' seconds\n"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

            # Update category numbers of df_tot2 so that they aren't the same
            #    as df_tot?  Shouldn't need to since categories are strings

            # Combine df_tot and df_tot2
            df_tot = pd.concat([df_tot, df_tot2])
            test_split_dir_list = test_split_dir_list \
                + test_split_dir_list2

            # Create new label_map_dict with all categories (done in init_args)

        else:
            pass

        # refine and plot
        t8 = time.time()
        if len(np.append(args.slice_sizes, args.slice_sizes2)) > 0:
            sliced = True
        else:
            sliced = False
        print("test data sliced?", sliced)

        # refine for each plot_thresh (if we have detections)
        if len(df_tot) > 0:
            for plot_thresh_tmp in args.plot_thresh:
                print("Plotting at:", plot_thresh_tmp)
                groupby = 'Image_Path'
                groupby_cat = 'Category'
                df_refine = post_process.refine_df(df_tot,
                                                   groupby=groupby,
                                                   groupby_cat=groupby_cat,
                                                   nms_overlap_thresh=args.nms_overlap_thresh,
                                                   plot_thresh=plot_thresh_tmp,
                                                   verbose=False)
                # make some output plots, if desired
                if len(args.building_csv_file) > 0:
                    building_csv_file_tmp = args.building_csv_file.split('.')[0] \
                        + '_plot_thresh_' + str(plot_thresh_tmp).replace('.', 'p') \
                        + '.csv'
                else:
                    building_csv_file_tmp = ''
                if args.n_test_output_plots > 0:
                    post_process.plot_refined_df(df_refine, groupby=groupby,
                                                 label_map_dict=args.label_map_dict_tot,
                                                 outdir=args.results_dir,
                                                 plot_thresh=plot_thresh_tmp,
                                                 show_labels=bool(
                                                     args.show_labels),
                                                 alpha_scaling=bool(
                                                     args.alpha_scaling),
                                                 plot_line_thickness=args.plot_line_thickness,
                                                 print_iter=5,
                                                 n_plots=args.n_test_output_plots,
                                                 building_csv_file=building_csv_file_tmp,
                                                 shuffle_ims=bool(
                                                     args.shuffle_val_output_plot_ims),
                                                 verbose=False)
    
                # geo coords?
                if bool(args.test_add_geo_coords):
                    df_refine, json = add_geo_coords.add_geo_coords_to_df(
                        df_refine,
                        create_geojson=bool(args.save_json),
                        inProj_str='epsg:32737', outProj_str='epsg:3857',
                        # inProj_str='epsg:4326', outProj_str='epsg:3857',
                        verbose=False)
    
                # save df_refine
                outpath_tmp = os.path.join(args.results_dir,
                                           args.val_prediction_df_refine_tot_root_part +
                                           '_thresh=' + str(plot_thresh_tmp) + '.csv')
                # df_refine.to_csv(args.val_prediction_df_refine_tot)
                df_refine.to_csv(outpath_tmp)
                print("Num objects at thresh:", plot_thresh_tmp, "=",
                      len(df_refine))
                # save json
                if bool(args.save_json) and (len(json) > 0):
                    output_json_path = os.path.join(args.results_dir,
                                                    args.val_prediction_df_refine_tot_root_part +
                                                    '_thresh=' + str(plot_thresh_tmp) + '.GeoJSON')
                    json.to_file(output_json_path, driver="GeoJSON")
    
            cmd_time_str = '"Length of time to run refine_test()' + ' ' \
                + str(time.time() - t8) + ' seconds"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        # remove or zip test_split_dirs to save space
        if len(test_split_dir_list) > 0:
            for test_split_dir_tmp in test_split_dir_list:
                if os.path.exists(test_split_dir_tmp):
                    # compress image chip dirs if desired
                    if args.keep_test_slices:
                        print("Compressing image chips...")
                        shutil.make_archive(test_split_dir_tmp, 'zip',
                                            test_split_dir_tmp)
                    # remove unzipped folder
                    print("Removing test_split_dir_tmp:", test_split_dir_tmp)
                    # make sure that test_split_dir_tmp hasn't somehow been shortened
                    #  (don't want to remove "/")
                    if len(test_split_dir_tmp) < len(args.results_dir):
                        print("test_split_dir_tmp too short!!!!:",
                              test_split_dir_tmp)
                        return
                    else:
                        print("Removing image chips...")

                        shutil.rmtree(test_split_dir_tmp, ignore_errors=True)

        cmd_time_str = '"Total Length of time to run test' + ' ' \
            + str(time.time() - t3) + ' seconds\n"'
        print(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

    # print ("\nNo honeymoon. This is business.")
    print("\n\n\nWell, I'm glad we got that out of the way.\n\n\n\n")

    return


###############################################################################
def main():

    # Construct argument parser
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--framework', type=str, default='yolt2',
                        help="object detection framework [yolt2, 'yolt3', ssd, faster_rcnn]")
    parser.add_argument('--mode', type=str, default='test',
                        help="[compile, test, train, test]")
    parser.add_argument('--gpu', type=str, default="0",
                        help="GPU number, set < 0 to turn off GPU support " \
                        "to use multiple, use '0,1'")
    parser.add_argument('--single_gpu_machine', type=int, default=0,
                        help="Switch to use a machine with just one gpu")
    parser.add_argument('--nbands', type=int, default=3,
                        help="Number of input bands (e.g.: for RGB use 3)")
    parser.add_argument('--outname', type=str, default='tmp',
                        help="unique name of output")
    parser.add_argument('--label_map_path', type=str,
                        default='',
                        help="Object classes, if not starts with '/', "
                        "assume it's in train_data_dir")
    parser.add_argument('--weight_file', type=str, default='yolo.weights',
                        help="Input weight file")
    parser.add_argument('--append_date_string', type=int, default=1,
                        help="Switch to append date to results filename")

    # training settings
    parser.add_argument('--train_data_dir', type=str, default='',
                        help="folder holding training image names, if empty "
                        "simrdwn_dir/data/")
    parser.add_argument('--yolt_train_images_list_file', type=str, default='',
                        help="file holding training image names, should be in "
                        "simrdwn_dir/data/")
    parser.add_argument('--max_batches', type=int, default=60000,
                        help="Max number of training batches")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of images per batch")
    parser.add_argument('--train_input_width', type=int, default=416,
                        help="Size of image to input to YOLT [n-boxes * 32: "
                        + "415, 544, 608, 896")
    parser.add_argument('--train_input_height', type=int, default=416,
                        help="Size of image to input to YOLT")
    # TF api specific settings
    parser.add_argument('--tf_cfg_train_file', type=str, default='',
                        help="Configuration file for training")
    parser.add_argument('--train_tf_record', type=str, default='',
                        help="tfrecord for training, assumed to be in training_dir")
    parser.add_argument('--train_val_tf_record', type=str, default='',
                        help="tfrecord for test during training")
    
    # yolt specific
    # yolt_object_labels_str is now redundant, and only label_map_path is needed
    parser.add_argument('--yolt_object_labels_str', type=str, default='',
                        help="yolt labels str: car,boat,giraffe")

    # test settings
    parser.add_argument('--train_model_path', type=str, default='',
                        help="Location of trained model")
    parser.add_argument('--use_tfrecords', type=int, default=0,
                        help="Switch to use tfrecords for inference")
    parser.add_argument('--test_presliced_tfrecord_path', type=str, default='',
                        help="Location of presliced training data tfrecord "
                        + " if empty us test_presliced_list")
    parser.add_argument('--test_presliced_list', type=str, default='',
                        help="Location of presliced training data list "
                        + " if empty, use tfrecord")
    parser.add_argument('--testims_dir', type=str, default='test_images',
                        help="Location of test images (look within simrdwn_dir unless begins with /)")
    parser.add_argument('--slice_sizes_str', type=str, default='416',
                        help="Proposed pixel slice sizes for test, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])"
                        + "(Set to < 0 to not slice")
    parser.add_argument('--edge_buffer_test', type=int, default=-1000,
                        help="Buffer around slices to ignore boxes (helps with"
                        + " truncated boxes and stitching) set <0 to turn off"
                        + " if not slicing test ims")
    parser.add_argument('--max_edge_aspect_ratio', type=float, default=3,
                        help="Max aspect ratio of any item within the above "
                        + " buffer")
    parser.add_argument('--slice_overlap', type=float, default=0.35,
                        help="Overlap fraction for sliding window in test")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="Overlap threshold for non-max-suppresion in python"
                        + " (set to <0 to turn off)")
    # parser.add_argument('--extra_pkl', type=str, default='',
    #                    help="External pkl to load on plots")
    parser.add_argument('--test_box_rescale_frac', type=float, default=1.0,
                        help="Defaults to 1, rescale output boxes if training"
                        + " boxes are the wrong size")
    parser.add_argument('--test_slice_sep', type=str, default='__',
                        help="Character(s) to split test image file names")
    parser.add_argument('--val_df_root_init', type=str, default='test_predictions_init.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_df_root_aug', type=str, default='test_predictions_aug.csv',
                        help="Results in dataframe format")
    parser.add_argument('--test_splitims_locs_file_root', type=str, default='test_splitims_input_files.txt',
                        help="Root of test_splitims_locs_file")
    parser.add_argument('--test_prep_only', type=int, default=0,
                        help="Switch to only prep files, not run anything")
    parser.add_argument('--BGR2RGB', type=int, default=0,
                        help="Switch to flip training files to RGB from cv2 BGR")
    parser.add_argument('--overwrite_inference_graph', type=int, default=0,
                        help="Switch to always overwrite inference graph")
    parser.add_argument('--min_retain_prob', type=float, default=0.025,
                        help="minimum probability to retain for test")
    parser.add_argument('--test_add_geo_coords', type=int, default=1,
                        help="switch to add geo coords to test outputs")

    # test, specific to YOLT
    parser.add_argument('--yolt_nms_thresh', type=float, default=0.0,
                        help="Defaults to 0.5 in yolt.c, set to 0 to turn off "
                        + " nms in C")

    # test plotting
    parser.add_argument('--plot_thresh_str', type=str, default='0.3',
                        help="Proposed thresholds to try for test, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--show_labels', type=int, default=0,
                        help="Switch to use show object labels")
    parser.add_argument('--alpha_scaling', type=int, default=0,
                        help="Switch to scale box alpha with probability")
    parser.add_argument('--show_test_plots', type=int, default=0,
                        help="Switch to show plots in real time in test")
    parser.add_argument('--save_json', type=int, default=1,
                        help="Switch to save a json in test")

    # parser.add_argument('--plot_names', type=int, default=0,
    #                    help="Switch to show plots names in test")
    parser.add_argument('--rotate_boxes', type=int, default=0,
                        help="Attempt to rotate output boxes using hough lines")
    parser.add_argument('--plot_line_thickness', type=int, default=2,
                        help="Thickness for test output bounding box lines")
    parser.add_argument('--n_test_output_plots', type=int, default=10,
                        help="Switch to save test pngs")
    parser.add_argument('--test_make_legend_and_title', type=int, default=1,
                        help="Switch to make legend and title")
    parser.add_argument('--test_im_compression_level', type=int, default=6,
                        help="Compression level for output images."
                        + " 1-9 (9 max compression")
    parser.add_argument('--keep_test_slices', type=int, default=0,
                        help="Switch to retain sliced test files")
    parser.add_argument('--shuffle_val_output_plot_ims', type=int, default=0,
                        help="Switch to shuffle images for plotting, if 0, images are sorted")

    # random YOLT specific settings
    parser.add_argument('--yolt_cfg_file', type=str, default='yolo.cfg',
                        help="Configuration file for network, in cfg directory")
    parser.add_argument('--subdivisions', type=int, default=4,
                        help="Subdivisions per batch")
    parser.add_argument('--use_opencv', type=str, default='1',
                        help="1 == use_opencv")
    parser.add_argument('--boxes_per_grid', type=int, default=5,
                        help="Bounding boxes per grid cell")

    # if evaluating spacenet data
    parser.add_argument('--building_csv_file', type=str, default='',
                        help="csv file for spacenet outputs")

    # second test classifier
    parser.add_argument('--train_model_path2', type=str, default='',
                        help="Location of trained model")
    parser.add_argument('--label_map_path2', type=str,
                        default='',
                        help="Object classes")
    parser.add_argument('--weight_file2', type=str, default='',
                        help="Input weight file for second inference scale")
    parser.add_argument('--slice_sizes_str2', type=str, default='0',
                        help="Proposed pixel slice sizes for test2 == second"
                        + "weight file.  Will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--plot_thresh_str2', type=str, default='0.3',
                        help="Proposed thresholds to try for test2, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--inference_graph_path2', type=str, default='/raid/local/src/simrdwn/outputs/ssd/output_inference_graph/frozen_inference_graph.pb',
                        help="Location of inference graph for tensorflow "
                        + "object detection API")
    parser.add_argument('--yolt_cfg_file2', type=str, default='yolo.cfg',
                        help="YOLT configuration file for network, in cfg directory")
    parser.add_argument('--val_df_root_init2', type=str, default='test_predictions_init2.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_df_root_aug2', type=str, default='test_predictions_aug2.csv',
                        help="Results in dataframe format")
    parser.add_argument('--test_splitims_locs_file_root2', type=str, default='test_splitims_input_files2.txt',
                        help="Root of test_splitims_locs_file")
    # parser.add_argument('--test_prediction_pkl_root2', type=str, default='val_refine_preds2.pkl',
    #                   help="Root of test pickle")

    # total test
    parser.add_argument('--val_df_root_tot', type=str, default='test_predictions_tot.csv',
                        help="Results in dataframe format")
    parser.add_argument('--val_prediction_df_refine_tot_root_part', type=str,
                        default='test_predictions_refine',
                        help="Refined results in dataframe format")

    # Defaults that rarely should need changed
    parser.add_argument('--multi_band_delim', type=str, default='#',
                        help="Delimiter for multiband data")
    parser.add_argument('--zero_frac_thresh', type=float, default=0.5,
                        help="If less than this value of an image chip is "
                        + "blank, skip it")
    parser.add_argument('--str_delim', type=str, default=',',
                        help="Delimiter for string lists")

    args = parser.parse_args()

    # check framework
    if args.framework.upper() == 'YOLT':
        raise ValueError("args.framework must specify YOLT2, YOLT3, or "
                         "a TensorFlow model, not YOLT!")
    args = update_args(args)
    train_cmd1, test_cmd_tot, test_cmd_tot2 = prep(args)
    execute(args, train_cmd1, test_cmd_tot, test_cmd_tot2)


###############################################################################
###############################################################################
if __name__ == "__main__":

    print("\n\n\nPermit me to introduce myself...\n")
    main()

###############################################################################
###############################################################################
