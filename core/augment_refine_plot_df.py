#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:26:01 2018

@author: avanetten
"""

from __future__ import print_function
import pandas as pd
#import numpy as np
import argparse
import time
import sys
import os

###################
path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_simrdwn_core)
#import visualization_utils
import preprocess_tfrecords
import simrdwn
import post_process
reload(simrdwn)
###################

###############################################################################     
def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outdir', type=str, default='/cosmiq/simrdwn/tmp/images_ssd',
                        help="Output file location")
    parser.add_argument('--pbtxt_filename', type=str, default='/cosmiq/simrdwn/data/class_labels_airplane_boat_car.pbtxt',
                        help="Class dictionary")
    parser.add_argument('--df_csv', type=str, default='',
                        help="dataframe csv")
    parser.add_argument('--df_csv_out', type=str, default='',
                        help="output dataframe csv")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Print a lot o stuff?")
    
    #### Plotting settings
    parser.add_argument('--slice_val_images', type=int, default=0,
                        help="Switch for if validaion images are sliced")
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

    header = ['Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category']

    # make label_map_dic (key=int, value=str), and reverse
    label_map_dict = preprocess_tfrecords.load_pbtxt(args.pbtxt_filename, verbose=False)
    #label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
            
    # read dataframe
    df_init = pd.read_csv(args.df_csv, names=header)
    # tf_infer_cmd outputs integer categories, update to strings
    df_init['Category'] = [label_map_dict[ktmp] for ktmp in df_init['Category'].values]


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
    print("df.columns:", df_init.columns)
    print("df.iloc[0[:", df.iloc[0])
    
    outfile_df = args.df_csv_out
    #outfile_df = args.df_csv.split('.')[0] + '_aug.csv'
    #outfile_df = os.path.join(args.outdir, '00_dataframe.csv')
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