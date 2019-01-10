#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:42:00 2017

@author: avanetten
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os
import shutil
#import scipy.signal
#import scipy.interpolate

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

###############################################################################
def plot_loss(df, figsize=(8,6), batchsize=64, 
                 N_train_ims=2418, twin_axis=False, ylim_perc_max=98,
                 rolling_mean_window=100, plot_file='', dpi=200, 
                 sample_size=20, verbose=True):
    '''if loss file only has two columns: batch_num and loss'''
    
    batch0 = df['Batch_Num'].values
    loss0 = df['Loss'].values
    #N_seen = batch * batchsize
    #epoch = 1.*N_seen / N_train_ims
    
    # subsample every N items
    batch = batch0[0:len(batch0):sample_size]
    loss = loss0[0:len(loss0):sample_size]
    
    # ylimit
    #loss_clip = np.clip(loss, np.percentile(loss, 0.01), np.percentile(loss, 0.98))
    #ymin_plot = max(0,  np.mean(loss_clip) - 2*np.std(loss_clip))
    #ymax_plot = np.mean(loss_clip) + 2*np.std(loss_clip)
    #ylim = (ymin_plot, ymax_plot)
    ylim = (0.9*np.min(loss), np.percentile(loss, ylim_perc_max))
    
    if verbose:
        print ("batch:", batch)
        print ("loss:", loss)
        print ("ylim:", ylim)
    
    # plot
    fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))        
    #ax.plot(epoch, loss, color='blue', alpha=0.7,
    ax.plot(batch, loss, color='blue', alpha=0.7,
            linewidth=2, solid_capstyle='round', zorder=2)
    #ax.scatter(epoch, loss, color='cyan', alpha=0.3)
    
    # horizintal line at minumum loss
    ax.axhline(y=np.min(loss), c='orange', alpha=0.3, linestyle='--')

    # filter
    #filt = scipy.signal.medfilt(loss, kernel_size=99)
    #ax.plot(epoch, filt, color='red', linestyle='--')

    # spline
    #filt = scipy.interpolate.UnivariateSpline(epoch, loss)
    #ax.plot(epoch, filt(epoch), color='red', linestyle='--')
    
    # better, just take moving average
    #Series.rolling(window=150,center=False).mean()
    df2 = pd.DataFrame(loss, columns=['Loss'])
    roll_mean = df2['Loss'].rolling(window=rolling_mean_window, center=False).mean()
    #roll_mean = pd.rolling_mean(df['Loss'], window=rolling_mean_window)
    
    ax.plot(batch[int(1.1*rolling_mean_window): ], roll_mean[int(1.1*rolling_mean_window): ], 
            color='red', linestyle='--', alpha=0.85)
    #ax.plot(epoch[int(1.1*rolling_mean_window): ], roll_mean[int(1.1*rolling_mean_window): ], 
    #        color='red', linestyle='--', alpha=0.85)

    ax.set_ylim(ylim)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.grid(color='gray', alpha=0.4, linestyle='--')
    #plt.axis('equal')
    #ax.set_title('YOLT Loss')  
    
    ax.set_title('TF Loss')  
    plt.tight_layout()
        
    if len(plot_file) > 0:
        plt.savefig(plot_file, dpi=dpi)

    return

###############################################################################
def main():
    
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('--path', type=str, default='/raid/local/src/yolt2/results/',
    #                    help="path to package")
    parser.add_argument('--res_dir', type=str, default='oops',
                        help="results")
    parser.add_argument('--rolling_mean_window', type=int, default=100,
                        help="Window for rolling mean")
    parser.add_argument('--ylim_perc_max', type=float, default=95,
                        help="Data percentile for max of y axis")
    parser.add_argument('--sep', type=str, default=',',
                        help="csv separator")
    parser.add_argument('--verbose', type=int, default=0,
                        help="verbose if == 1")
    #parser.add_argument('--batchsize', type=int, default=8,
    #                    help="Training epochs")
    #parser.add_argument('--N_train_ims', type=int, default=2418,
    #                    help="Number of images in training corpus")
    args = parser.parse_args()

    verbose = bool(args.verbose)
    # set directories
    #res_dir = os.path.join(args.path, args.res_dir)
    if args.res_dir == 'oops':
        #res_dir = os.get_cwd()
        res_dir = os.path.dirname(os.path.realpath(__file__))

    else:
        res_dir = args.res_dir
        
    #log_dir = os.path.join(res_dir, 'logs')
    log_dir = res_dir #os.path.join(res_dir, 'logs')

    # get log file, e.g.: train_ssd_inception_v2_3class_gpu0_2018_03_06_21-36-04.log
    log_file = [f for f in os.listdir(log_dir) if f.endswith('.log')][0]
    log_path = os.path.join(log_dir, log_file)

    print ("res_dir:", res_dir)
    print ("log_dir:", log_dir)
    print ("log_path:", log_path)

    # set plot name
    plot_file = os.path.join(log_dir, 'tf_loss_plot.png')
    loss_file_p = os.path.join(log_dir, 'tf_loss_for_plotting.txt')
    print ("loss_file_p:", loss_file_p)

    # copy file because it's probably being actively written to
    #cmd = 'cp ' + loss_file + ' ' + loss_file_p
    #print "copy command:", cmd
    #os.system(cmd)
    shutil.copy2(log_path, loss_file_p)

    # read in lines (INFO:tensorflow:global step 39997: loss = 2.4568 (0.747 sec/step))
    out_list = []
    ftmp = open(log_path, "r")
    for line in ftmp:
        if line.startswith("INFO:tensorflow:global step"):
            spl0 = line.split("INFO:tensorflow:global step ")[-1]
            step = spl0.split(":")[0]
            spl1 = spl0.split("loss = ")
            loss = spl1[-1].split(" ")[0]
            if verbose:
                print ("line:", line)
                print ("  spl0:", spl0)
                print ("  spl1:", spl1)
                print ("  step:", step)
                print ("  loss:", loss)
            out_list.append([int(step), float(loss)])
    ftmp.close()
    
    # create dataframe
    df = pd.DataFrame(out_list, columns=['Batch_Num', 'Loss'])
    
    # make plot    
    plot_loss(df, plot_file=plot_file, ylim_perc_max=args.ylim_perc_max)


###############################################################################
if __name__ == "__main__":
    main()
