#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:23:34 2017

@author: avanetten


Adapted from:
https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

sorted(example.features.feature.keys())
[u'image/detection/bbox/xmax',
 u'image/detection/bbox/xmin',
 u'image/detection/bbox/ymax',
 u'image/detection/bbox/ymin',
 u'image/detection/class/label',
 u'image/detection/class/text',
 u'image/detection/label',
 u'image/detection/score',
 u'image/encoded',
 u'image/filename',
 u'image/format',
 u'image/height',
 u'image/key/sha256',
 u'image/object/bbox/xmax',
 u'image/object/bbox/xmin',
 u'image/object/bbox/ymax',
 u'image/object/bbox/ymin',
 u'image/object/class/label',
 u'image/object/class/text',
 u'image/object/difficult',
 u'image/object/truncated',
 u'image/object/view',
 u'image/score',
 u'image/source_id',
 u'image/width']


object size:
    sys.getsizeof(x)
"""


from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import time
import os

###################
import sys
#utils_path = '/cosmiq/simrdwn/src'
utils_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(utils_path)
#import visualization_utils
import preprocess_tfrecords
import simrdwn
import post_process
reload(simrdwn)
###################



###############################################################################
def df_to_txtfiles(df, out_dir):
    '''Validation output is a file such as car.txt, with entries:
          ['Loc_Tmp', 'Prob',  'Xmin', 'Ymin', 'Xmax', 'Ymax']
    get_records() returns a dataframe with columns:
        ['Category', 'Location', 'Prob',  'Xmin',  'Ymin', 'Xmax', 'Ymax']
    group this dataframe and output validation text files
    !! ACTUALLY, WE CAN SKIP THIS STEP AND JUST AUGMENT THE DATAFRAME,
        SIMILAR TO SIMRDWN.AUGMENT_DF() !!
    '''
    
    #group = df.groupby('Category')
    #for itmp,g in enumerate(group):
    #    category = g[0]
    #    data_all_classes = g[1]
    
    pass
        
    
def tf_to_df(tfrecords_filename, max_iter=50000, 
                label_map_dict={}, 
                tf_type='test',
                output_columns = ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category'],
                replace_paths=(
                               ('/raid/local/src/yolt2/', 
                               '/cosmiq/yolt2/'),
                               ('/raid/data/ave/qgis_labels/',
                                '/cosmiq/qgis_labels/')
                               ),
                ):
    '''Convert inference tfrecords file to pandas dataframe
    if tf_type=test, assume this is an inference tfrecord,
      else, it will be a training tfrecord'''
    t0 = time.time()
    
    
    print ("\nTransforming tfrecord to dataframe...")
    #reconstructed_images = []
    df_data = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for i,string_record in enumerate(record_iterator):
        
        example = tf.train.Example()
        
        example.ParseFromString(string_record)

        #print ("example:", example)
        
        if i == 0:
            print("example.features.feature.keys()", sorted(example.features.feature.keys()))
                
        if i > max_iter:
            break

        height = int(example.features.feature['image/height']
                                     .int64_list.value[0])
        width = int(example.features.feature['image/width']
                                    .int64_list.value[0])
        
        #a = example.features.feature['image/detection/score']
        if tf_type == 'test':
            xmins = np.array(example.features.feature['image/detection/bbox/xmin']
                                         .float_list.value)
            ymins = np.array(example.features.feature['image/detection/bbox/ymin']
                                         .float_list.value)
            xmaxs = np.array(example.features.feature['image/detection/bbox/xmax']
                                         .float_list.value)
            ymaxs = np.array(example.features.feature['image/detection/bbox/ymax']
                                         .float_list.value)
            classes_int = np.array(example.features.feature['image/detection/label']
                                         .int64_list.value)
            scores = np.array(example.features.feature['image/detection/score']
                                         .float_list.value)

        else:
            xmins = np.array(example.features.feature['image/object/bbox/xmin']
                                         .float_list.value)
            ymins = np.array(example.features.feature['image/object/bbox/ymin']
                                         .float_list.value)
            xmaxs = np.array(example.features.feature['image/object/bbox/xmax']
                                         .float_list.value)
            ymaxs = np.array(example.features.feature['image/object/bbox/ymax']
                                         .float_list.value)
            classes_int = np.array(example.features.feature['image/object/class/label']
                                         .int64_list.value)
            scores = np.ones(len(classes_int))

        
        # convert from fractions to pixel coords
        xmins = (width * xmins).astype(int)
        xmaxs = (width * xmaxs).astype(int)
        ymins = (height * ymins).astype(int)
        ymaxs = (height * ymaxs).astype(int)
        #boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)
        classes_int_str = classes_int.astype(str)  
        #classes_text = np.array(example.features.feature['image/detection/class/text']
        #                             .bytes_list.value)

        classes_str, classes_legend_str = classes_int_str, classes_int_str     
        if len(label_map_dict.keys()) > 0:
            classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
            classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
        
        img_loc_string = (example.features.feature['image/filename']
                                      .bytes_list.value[0]) 
        if (i % 100) == 0:
            print("\n", i, "Image Location:", img_loc_string)   
            print ("  xmins:", xmins)
            print ("  classes_str:", classes_str)
        #print ("classes_int:", classes_int)
        
        # update path, if desired       
        if len(replace_paths) > 0:
            for item in replace_paths:
                (init_str, out_str) = item
                if img_loc_string.startswith(init_str):
                    img_loc_string = img_loc_string.replace(init_str, out_str)
                    break
                else:
                     continue
            
        # update data
        # output_columns =  ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']
        for j in range(len(xmins)):
            out_row = [img_loc_string,  #img_loc_string[j],
                       scores[j],
                       xmins[j],
                       ymins[j],
                       xmaxs[j],
                       ymaxs[j],
                       #classes_int_str[j]]
                       classes_str[j]]

            df_data.append(out_row)
            

#        
#        img_string = (example.features.feature['image/encoded']
#                                      .bytes_list
#                                      .value[0])
#        img_1d = np.fromstring(img_string, dtype=np.uint8)
#        reconstructed_img = img_1d.reshape((height, width, 3))    
#        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
#        # Annotations don't have depth (3rd dimension)
#        reconstructed_annotation = annotation_1d.reshape((height, width))
#        reconstructed_images.append((reconstructed_img, reconstructed_annotation))
    

            
    df_init = pd.DataFrame(df_data, columns=output_columns)
    #print("\ndf_init.columns:", df_init.columns)

    print ("len dataframe:", len(df_init))
    print ("Time to transform", len(df_init), "tfrecords to dataframe:", \
           time.time() - t0, "seconds")
    return df_init


###############################################################################    
###############################################################################     
def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tfrecords_filename', type=str, default='/cosmiq/simrdwn/tmp/val_detections_ssd.tfrecord',  
                      help="tfrecords file")
    parser.add_argument('--outdir', type=str, default='/cosmiq/simrdwn/tmp/images_ssd',
                        help="Output file location")
    parser.add_argument('--pbtxt_filename', type=str, default='/cosmiq/simrdwn/data/class_labels_airplane_boat_car.pbtxt',
                        help="Class dictionary")
    parser.add_argument('--tf_type', type=str, default='test',
                        help="weather the tfrecord is for test or train")
    parser.add_argument('--slice_val_images', type=int, default=0,
                        help="Switch for if validaion images are sliced")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Print a lot o stuff?")
    
    #### Plotting settings
    parser.add_argument('--plot_thresh', type=float, default=0.33,  
                        help="Threshold for plotting boxes, set < 0 to skip plotting")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,  
                        help="IOU threshold for non-max-suppresion, skip if < 0")
    parser.add_argument('--make_box_labels', type=int, default=1,
                        help="If 1, make print label above each box")
    parser.add_argument('--scale_alpha', type=int, default=1,
                        help="If 1, scale box opacity with confidence")
    parser.add_argument('--plot_line_thickness', type=int, default=1,
                        help="If 1, scale box opacity with confidence")


    args = parser.parse_args()
    print ("args:", args)
    t0 = time.time()
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        
    # make label_map_dic (key=int, value=str), and reverse
    label_map_dict = preprocess_tfrecords.load_pbtxt(args.pbtxt_filename, verbose=False)
    #label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}
    
    # convert tfrecord to dataframe
    df_init0 = tf_to_df(tfrecords_filename=args.tfrecords_filename,
                       label_map_dict=label_map_dict,
                       tf_type=args.tf_type)
    #df_init = tf_to_df(tfrecords_filename=args.tfrecords_filename, 
    #            outdir=args.outdir, plot_thresh=args.plot_thresh,
    #            label_map_dict=label_map_dict,
    #            show_labels = bool(args.make_box_labels),
    #            alpha_scaling = bool(args.scale_alpha),
    #            plot_line_thickness=args.plot_line_thickness)
    t1 = time.time()
    print ("Time to run tf_to_df():", t1-t0, "seconds")
    print("df_init.columns:", df_init0.columns)
    
    # filter out low confidence detections
    df_init = df_init0.copy()[df_init0['Prob'] >= args.plot_thresh]
    
    # augment dataframe columns
    df = post_process.augment_df(df_init, 
               valid_testims_dir_tot='',
               slice_sizes=[0],
               valid_slice_sep='__',
               edge_buffer_valid=0,
               max_edge_aspect_ratio=4,
               valid_box_rescale_frac=1.0,
               rotate_boxes=False,
               verbose=bool(args.verbose))
    print("len df:", len(df))
    print("df.columns:", df_init.columns)
    print("df.iloc[0[:", df.iloc[0])
    outfile_df = os.path.join(args.outdir, '00_dataframe.csv')
    df.to_csv(outfile_df)
    
    # plot 
    if args.plot_thresh > 0:
        post_process.refine_and_plot_df(df, label_map_dict=label_map_dict, 
                 outdir=args.outdir, 
                 #slice_sizes=[0],
                 sliced=bool(args.slice_val_images),
                 plot_thresh=args.plot_thresh, 
                 nms_overlap_thresh=args.nms_overlap_thresh,
                 show_labels=args.make_box_labels, 
                 alpha_scaling=args.scale_alpha, 
                 plot_line_thickness=args.plot_line_thickness,
                 verbose=bool(args.verbose))
    
    print ("Plots output to:", args.outdir)
    print ("Time to get and plot records:", time.time() - t0, "seconds")



###############################################################################
if __name__ == "__main__":
    main()
    
    '''
	    python /cosmiq/simrdwn/src/parse_tfrecord.py \
	        --pbtxt_filename /cosmiq/simrdwn/data/class_labels_airplane_boat_car.pbtxt \
	        --tfrecords_filename /cosmiq/simrdwn/tmp/val_detections_ssd.tfrecord \
	        --outdir /cosmiq/simrdwn/tmp/images_ssd \
	        --plot_thresh 0.5 \
		   --make_box_labels 0 \
            --nms_overlap_thresh 0.9 \
			--scale_alpha 1            

    '''
    
    