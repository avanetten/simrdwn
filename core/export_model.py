#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:15:25 2018

@author: avanetten

Export model from checkpoint

"""


from __future__ import print_function
import argparse
import time
import sys
import os

###################
path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_simrdwn_core)
import simrdwn
reload(simrdwn)
###################


###############################################################################     
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='',
                        help="results directory")
    args = parser.parse_args()

    # inferred values
    args.tf_model_output_directory = os.path.join(args.results_dir, 
                                                  'frozen_model')
    args.tf_cfg_train_file_out = os.path.join(args.results_dir,
                                                  'pipeline.config')
    print ("args:", args)
    
    # define command
    cmd_export_tf = simrdwn.tf_export_model_cmd(args.results_dir, )
#    cmd_export_tf = simrdwn.tf_export_model_cmd(args.tf_cfg_train_file_out, 
#                                     args.results_dir, 
#                                     args.tf_model_output_directory)

    
    # Execute
    print ("Running", cmd_export_tf, "...\n\n")
    t0 = time.time()
    os.system(cmd_export_tf)
    t1 = time.time()
    cmd_time_str = '"Length of time to run command: ' \
                    +  cmd_export_tf + ' ' \
                    + str(t1 - t0) + ' seconds\n"'
    print (cmd_time_str)  
    print ("output_dir:", args.tf_model_output_directory)

    return

###############################################################################
if __name__ == "__main__":
    main()
    
'''


python /raid/local/src/simrdwn/core/export_model.py \
    --results_dir /raid/local/src/simrdwn/results/train_faster_rcnn_resnet101_3class_v5_2018_02_22_16-40-52/


'''