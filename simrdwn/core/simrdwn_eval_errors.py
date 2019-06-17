#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:56:38 2018

@author: avanetten

Evaluate error bars of predictions
"""
from __future__ import print_function
import pandas as pd
import scipy.stats
import numpy as np
from . import simrdwn_eval
from .. utils import weighted_avg_and_std


###############################################################################
def compute_f1(tp, fn, fp):
    '''Compute f1 for scalar inputs'''
    try:
        precision = 1.*tp / (tp + fp)
        recall = 1.*tp / (tp + fn)
    except:
        return 0
    if (precision + recall) > 0:
        f1 = 2. * precision * recall / (precision + recall)
    else:
        f1 = 0
    return f1


###############################################################################
def compute_df1(tp_vec, fn_vec, fp_vec, errs=[], verbose=False):
    # def compute_df1(df_, errs=[], verbose=False):
    '''
    F1 uncertainty can be computed from
    https://en.wikipedia.org/wiki/F1_score
    and
    https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    errs should be a list of [tp_std, fn_std, fp_std]
    '''

    # vectors
    # tp_vec = df_.TP.values
    # fn_vec = df_.FN.values
    # fp_vec = df_.FP.values
    # F1_vec = df_.F1.values
    # F1_std = np.std(F1_vec)

    # stds
    if len(errs) == 0:
        tp_std = np.std(tp_vec)
        fn_std = np.std(fn_vec)
        fp_std = np.std(fp_vec)
    else:
        [tp_std, fn_std, fp_std] = errs

    # mean values
    tp = np.mean(tp_vec)
    fn = np.mean(fn_vec)
    fp = np.mean(fp_vec)

    # partial derivaties
    dtp_df1 = 2.0 * (fn + fp) / (2.0*tp + fn + fp)**2
    dfn_df1 = (-2.0 * tp) / (2.0*tp + fn + fp)**2
    dfp_df1 = (-2.0 * tp) / (2.0*tp + fn + fp)**2

    # get covariances
    X = np.vstack((tp_vec, fn_vec, fp_vec))
    cov = np.cov(X)
    cov_tp_fn = cov[0][1]
    cov_tp_fp = cov[0][2]
    cov_fn_fp = cov[1][2]

    # get df1
    # var =    (dtp_df1**2 * tp_std**2) \
    #       + (dfn_df1**2 * fn_std**2) \
    #       + (dfp_df1**2 * fp_std**2) \
    #       + (2.0 * dtp_df1 * dfn_df1 * cov_tp_fn) \
    #       + (2.0 * dtp_df1 * dfp_df1 * cov_tp_fp) \
    #       + (2.0 * dfn_df1 * dfp_df1 * cov_fn_fp)

    cov_coeff = 2.0  # 2.0    # cov_coeff should be 2.0, set to 0 to ignore
    delta_f1 = np.sqrt(dtp_df1**2 * tp_std**2 +
                       + dfn_df1**2 * fn_std**2 +
                       + dfp_df1**2 * fp_std**2 +
                       + cov_coeff * dtp_df1 * dfn_df1 * cov_tp_fn +
                       + cov_coeff * dtp_df1 * dfp_df1 * cov_tp_fp +
                       + cov_coeff * dfn_df1 * dfp_df1 * cov_fn_fp)

    if verbose:
        print("df1:", delta_f1)
        # print "F1_std:", F1_std

    return delta_f1


###############################################################################
def bootstrap_f1(df_, n_bootstraps=5000, bootstrap_len_mult=1, verbose=False,
                 super_verbose=False):
    '''Bootstrap F1 error from lists of tp, fp, fn'''

    tp_vec = df_.TP.values
    fn_vec = df_.FN.values
    fp_vec = df_.FP.values
    F1_vec = df_.F1.values

    # get sem of F1_vec?
    F1_sem = scipy.stats.sem(F1_vec)

    # weighted std
    weights = tp_vec + fn_vec
    m0, std0, var0 = weighted_avg_and_std(F1_vec, weights)

    idxs = np.arange(len(tp_vec))
    len_bootstrap_sample = int(bootstrap_len_mult*len(idxs))
    #len_bootstrap_sample = 5

    f1_array = []
    fp_array = []
    fn_array = []
    tp_array = []
    for i in range(n_bootstraps):
        if super_verbose:
            print(i, "/", n_bootstraps)

        # define bootstrap indexes
        if n_bootstraps == 1:
            idxs_boot = idxs
        else:
            #idxs_boot = modular_classifier_prep.bootstrap_resample(idxs, n=len_bootstrap)
            idxs_boot = np.random.random_integers(
                0, len(idxs)-1, len_bootstrap_sample)
            # print idxs_boot

        # return data from indexes
        tpv_tmp, fnv_tmp, fpv_tmp = \
            tp_vec[idxs_boot], fn_vec[idxs_boot], fp_vec[idxs_boot]
        # compute total f1 from those temp vectors
        tp, fn, fp = np.sum(tpv_tmp), np.sum(fnv_tmp), np.sum(fpv_tmp)
        #tp, fn, fp = np.mean(tpv_tmp), np.mean(fnv_tmp), np.mean(fpv_tmp)

        f1_tmp = compute_f1(tp, fn, fp)
        if super_verbose:
            print("tp, fn, fp, f1_tmp:", tp, fn, fp, f1_tmp)
        f1_array.append(f1_tmp)
        fp_array.append(fp)
        fn_array.append(fn)
        tp_array.append(tp)

    # get std of array
    boot_f1_std = np.std(f1_array)
    boot_fp_std = np.std(fp_array)
    boot_fn_std = np.std(fn_array)
    boot_tp_std = np.std(tp_array)

    if verbose:
        print("F1_std:", np.std(F1_vec))
        print("F1_std_weighted:", std0)
        print("F1_sem:", F1_sem)
        # print "propagated_std:", compute_df1(df_)
        print("propagated std:", compute_df1(tp_array, fn_array, fp_array,
                                             errs=[boot_tp_std, boot_fn_std,
                                                   boot_fp_std],
                                             verbose=False))
        print("boot_f1_std:", boot_f1_std)
        print("boot_fp_std:", boot_fp_std)
        print("boot_fn_std:", boot_fn_std)
        print("boot_tp_std:", boot_tp_std)

    return boot_f1_std, boot_fp_std, boot_fn_std, boot_tp_std, \
        f1_array, fp_array, fn_array, tp_array


###############################################################################
def construct_df_scores_from_precision_recall_df(df_precision_recall_tot,
                                                 outfile='', iou_thresh=0.25,
                                                 cats='', verbose=False):
    '''
    Assume the input dataframe has been filtered for desired categories
    and image roots
    '''

    categories = np.sort(np.unique(df_precision_recall_tot['Category'].values))
    detect_threshes = np.sort(
        np.unique(df_precision_recall_tot['Threshold'].values))

    score_list = []
    # iterate through categories
    for i, category in enumerate(categories):

        df_filt_cat = df_precision_recall_tot[df_precision_recall_tot['Category'] == category]

        # iterate through detect threshes to create scores
        for j, detect_thresh in enumerate(detect_threshes):
            if verbose:
                print("detect_thresh:", detect_thresh)

            # keep only desired thresh
            df_filt = df_filt_cat[df_filt_cat['Threshold'] == detect_thresh]

            # get true pos, false pos, false net
            n_gt_sum = np.sum(df_filt['n_ground_truth_boxes'].values)
            n_true_pos_tot = np.sum(df_filt['n_true_pos'].values)
            n_false_pos_tot = np.sum(df_filt['n_false_pos'].values)
            n_false_neg_tot = np.sum(df_filt['n_false_neg'].values)

            # compute f1, precision, recall for all images at the given threshold
            f1, precision, recall = simrdwn_eval.compute_f1_precision_recall(
                n_true_pos_tot, n_false_pos_tot, n_false_neg_tot)
            # always add to score list
            if (2 > 1):
                # only add if positive values (this screws up means!!!)
                # if (precision > 0) and (recall > 0):
                score_list.append([n_gt_sum, detect_thresh, f1,
                                   precision, recall, category, iou_thresh])

    # score_list = np.array(score_list)
    # save dataframe
    df_scores = pd.DataFrame(score_list, columns=['N_Ground_Truth',
                                                  'Detect_Thresh',
                                                  'F1',
                                                  'Precision',
                                                  'Recall',
                                                  'Category',
                                                  'IOU_Thresh'])
    # df_score_fname = os.path.join(args.out_dir,
    #                         'iou_thresh_' + str(args.iou_thresh).replace('.', 'p')
    #                         + '_' + category
    #                         + '_precision_recall_df.csv')
    if len(outfile) > 0:
        df_scores.to_csv(outfile)

    return df_scores


###############################################################################
def bootstrap_mAP(df_precision_recall, n_bootstraps=5000,
                  bootstrap_len_mult=1, iou_thresh=0.25,
                  f1_thresh=0.15, outfile='',
                  verbose=False, super_verbose=False):
    '''Bootstrap F1 and mAP error from output of simrdwn_eval.py, the
    precision_recall_tot dataframe'''

    name_col = 'im_root'
    categories = ['All'] + \
        list(np.sort(np.unique(df_precision_recall['Category'].values)))

    # filter out categories (since generating samples is the slow part,
    #  compute for all cats at once)
    # cat_col = 'Category'
    # if len(cats) > 0:
    #    df = df_precision_recall.loc[df_precision_recall[cat_col].isin(cats)]
    # else:
    #    df = df_precision_recall
    df = df_precision_recall

    # get number of unique images
    image_list = df[name_col].values
    n_images = len(np.unique(image_list))
    idxs = np.arange(n_images)
    len_bootstrap_sample = int(bootstrap_len_mult*len(idxs))
    # len_bootstrap_sample = 5

    # iterate through bootstraps
    # initialize dictionaries
    f1_dic, map_dic = {}, {}
    for cat in categories:
        f1_dic[cat], map_dic[cat] = [], []
    # f1_list, map_list = [], []
    for i in range(n_bootstraps):
        if verbose:
            print(i, "/", n_bootstraps, outfile.split('_eval')
                  [0].split('results')[1])

        # define bootstrap indexes
        if n_bootstraps == 1:
            idxs_boot = idxs
        else:
            # idxs_boot = modular_classifier_prep.bootstrap_resample(idxs, n=len_bootstrap)
            idxs_boot = np.random.random_integers(
                0, len(idxs)-1, len_bootstrap_sample)
        # print ("idxs_boot:", idxs_boot)

        # get image names that correpond to idxs
        im_names_boot = image_list[idxs_boot]

        # filter dataframe for image names (can't use .isin because we are sampling with replacement!!!)
        #   https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
        # df_filt = df.loc[df[name_col].isin(im_names_boot)]
        # instead, iteratively create dataframe
        df_filt_data = []
        for j, im_name_tmp in enumerate(im_names_boot):
            # if j > 4:   break
            df_filt_data.extend(df.loc[df[name_col] == im_name_tmp].values)
        df_filt = pd.DataFrame(df_filt_data, columns=df.columns.values)
        # print ("df.columns.values:", df.columns.values)
        # print ("df_filt_data[0]:", df_filt_data[0])
        # print ("df_filt.iloc[0]:", df_filt.iloc[0])
        # print ("len df_filt:", len(df_filt))

        df_scores_boot = construct_df_scores_from_precision_recall_df(
            df_filt, outfile='', iou_thresh=iou_thresh, verbose=super_verbose)
        # print ("df_scores_boot)
        # compute f1, map for each category
        for cat in categories:
            mAP_boot, _, f1_boot, _ = simrdwn_eval.compute_map(
                df_scores_boot, f1_thresh=f1_thresh, category=cat,
                verbose=super_verbose)
            map_dic[cat].append(mAP_boot)
            f1_dic[cat].append(f1_boot)
            print("  cat:", cat, "boot mAP:", mAP_boot, "boot F1:", f1_boot)
        # map_list.append(mAP_boot)
        # f1_list.append(f1_boot)

    # get stds of arrays
    out_data = []
    for cat in categories:
        map_mean_tmp = np.mean(map_dic[cat])
        map_std_tmp = np.std(map_dic[cat])
        f1_mean_tmp = np.mean(f1_dic[cat])
        f1_std_tmp = np.std(f1_dic[cat])
        out_data.append([map_mean_tmp, map_std_tmp, f1_mean_tmp, f1_std_tmp,
                         cat, iou_thresh, n_bootstraps])
    print("out_data:", out_data)
    df_out = pd.DataFrame(out_data,
                          columns=['map_mean', 'map_std', 'f1_mean', 'f1_std',
                                   'Category', 'IOU_Thresh', 'N_Bootstraps'])

    # boot_map_std = np.std(map_list)
    # boot_f1_std = np.std(f1_list)
    # if verbose:
    #    print "boot_map_std:", boot_map_std
    #   print "boot_f1_std:", boot_f1_std

    if len(outfile) > 0:
        df_out.to_csv(outfile)

    return df_out
