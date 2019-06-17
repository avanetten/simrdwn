#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:02:42 2016

@author: avanetten


Evaluate f1 score of simrdwn results compared to ground truth

"""

from __future__ import print_function

import os
import pickle
import cv2
import sys
import time
import random
import sklearn
import argparse
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.wkt
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy import optimize
from shapely.geometry import Polygon
# import math
# import sys

import post_process

# path_simrdwn_core = os.getcwd()
path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path_simrdwn_core, '..', 'utils'))
import utils
from utils import parse_cowc, parse_shapefile
from utils import weighted_avg_and_std, twinx_function, piecewise_linear


##############################################################################
def plot_rects_eval(im_test_c, boxes_rect_rot_coords, figsize=(10, 10),
                    color=(255, 165, 0), thickness=1, verbose=False):
    # http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
    vis = im_test_c.copy()
    fig, ax = plt.subplots(figsize=figsize)
    img_mpl = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ax.imshow(img_mpl)

    t0 = time.time()

    for i, coords_rot0 in enumerate(boxes_rect_rot_coords):
        if type(coords_rot0) == np.ndarray:
            coords_rot = coords_rot0
            coords1 = coords_rot.reshape((-1, 1, 2))

        else:
            coords_rot = np.array(
                list(coords_rot0.exterior.coords), np.int32)[:-1]
            coords1 = coords_rot.reshape((-1, 1, 2))

        if verbose:
            print("coords_rot0:", coords_rot0)
            print("coords_rot:", coords_rot)
            print("coords1:", coords1)

        # plot rotated rect
        # coords1 = coords_rot.reshape((-1,1,2))
        cv2.polylines(img_mpl, [coords1], True, color, thickness=thickness)

    plt.axis('off')
    plt.tight_layout()

    print("Time to plot rects:", time.time() - t0, "seconds")

    return fig, ax, img_mpl


###############################################################################
def compute_f1_precision_recall(n_true_pos, n_false_pos, n_false_neg):
    '''Compute precision and recall'''
    try:
        recall = 1.*n_true_pos / (n_true_pos + n_false_neg)
    except:
        recall = 0.0
    try:
        precision = 1.*n_true_pos / (n_true_pos + n_false_pos)
    except:
        precision = 0.0
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


###############################################################################
def compute_performance(ground_truth_boxes, boxes_rect_rot_coords,
                        iou_thresh=0.5, im_test_c=None, plot_name=None,
                        figsize=(9, 9), dpi=600, thickness=2,
                        label_font_width=2, shape_method='rect',
                        valid_im_compression_level=6,
                        plot_nums=True, colorf1=(255, 255, 255),
                        verbose=False):
    '''
    boxes_rect_rot_coords should be in format [[x0, y0], [x1, y1] ... ]
    mean average precision
    http://toblerity.org/shapely/manual.html
    object.bounds
        Returns a (minx, miny, maxx, maxy) tuple (float values) that bounds
        the object.
        outvals = [ f1, precision, recall, len(ground_truth_boxes),
                len(boxes_rect_rot_coords), len(false_neg_gt),
                len(false_pos_prop), len(true_pos_gt) ]
   '''

    t0 = time.time()

    # keep_set = set([])
    # rem_set = set([])
    true_pos_prop = set([])
    false_neg_gt = set([])
    true_pos_gt = set([])
    true_pos_pair = set([])
    outstring = '\nCompute F1 between ground truth and proposals\n'
    print(outstring)

    # if ground truth boxes are empty, everything is a false positive
    if len(ground_truth_boxes) == 0:
        false_pos_prop = set(range(len(boxes_rect_rot_coords)))

    else:
        # convert ground_truth_boxes to polygons
        if type(ground_truth_boxes[0]) == shapely.geometry.polygon.Polygon:
            # get bounding box...
            polys_gt = []
            for g in ground_truth_boxes:
                (x0, y0, x1, y1) = g.bounds
                if verbose:
                    print("gt_box: g.bounds:", g.bounds)
                polys_gt.append(shapely.geometry.box(x0, y0, x1, y1, ccw=True))
            # polys_gt = ground_truth_boxes
        else:
            polys_gt = [Polygon(c) for c in ground_truth_boxes]

        centroids_gt = [np.asarray(p.centroid) for p in polys_gt]
        areas_gt = [p.area for p in polys_gt]
        bounds_gt = [p.bounds for p in polys_gt]
        sizes_gt = [[p[2]-p[0], p[3]-p[1]] for p in bounds_gt]
        radii_gt = [np.sqrt(size[0]**2 + size[1]**2) for size in sizes_gt]

    #    # compute sizes of boxes
    #    gt_arr = np.asarray(ground_truth_boxes)
    #    widths_gt = []
    #    for item in gt_arr:
    #        x_tmp = itme[:,0]
    #        y_tmp = itme[:,1]
    #        dx_tmp = np.max(x_tmp) - np.min(x_tmp)
    #        dy_tmp = np.max(y_tmp) - np.min(y_tmp)
    #        r = np.sqrt( dx_tmp**2 + dy_tmp**2 )
    #        widths_gt.append(r)

        # convert ground_truth_boxes to polygons
        if type(boxes_rect_rot_coords[0]) == shapely.geometry.polygon.Polygon:
            polys_prop = []
            for g in boxes_rect_rot_coords:
                (x0, y0, x1, y1) = g.bounds
                if verbose:
                    print("prop bounds:", g.bounds)
                polys_prop.append(shapely.geometry.box(
                    x0, y0, x1, y1, ccw=True))
            # polys_prop = boxes_rect_rot_coords
        else:
            polys_prop = [Polygon(c) for c in boxes_rect_rot_coords]
        centroids_prop = [np.asarray(p.centroid) for p in polys_prop]
        areas_prop = [p.area for p in polys_prop]

        # create ball tree of proposed regions
        t1 = time.time()
        tree = sklearn.neighbors.BallTree(np.asarray(centroids_prop))
        # tree = sklearn.neighbors.KDTree(centroids)
        # p0 = "Time to form tree: " +  str(time.time() - t1) +  "seconds\n"
        # print p0
        # outstring += p0

        # iterate through groud truth boxes
        width_mult = 1.5
        # iterate through ground truth boxes
        for i, (poly, centroid, area, radius0) in enumerate(zip(polys_gt, centroids_gt, areas_gt, radii_gt)):

            #        # skip if we've already flagged this index
            #        if i in rem_set or i in keep_set:
            #        	#print ("skipping i:", i
            #        	continue

            # get nearest neigbhors
            radius = width_mult * radius0  # np.sqrt( size[0]**2 + size[1]**2 )
            centroid_reshape = centroid.reshape(1, -1)
            inds0 = tree.query_radius(centroid_reshape, r=radius)[0]
            if verbose:
                print("i:", i)
                print("  radius:", radius)
                print("  inds0:", inds0)

            # remove any inds already in false_pos_prop or true_pos_prop
            inds = set(inds0) - true_pos_prop
            # print ("inds", inds

            # if empty, add to false_neg, continue
            if len(inds) == 0:
                false_neg_gt.add(i)
                continue

            # else, check for overlap of closest items
            jaccard_l = []
            area_l = []
            jaccard_inds_l = []

            for itmp in inds:
                poly_tmp = polys_prop[itmp]
                intersection = poly.intersection(poly_tmp).area
                union = poly.union(poly_tmp).area
                if union == 0:
                    jaccard = 0
                else:
                    jaccard = intersection / union
                jaccard_inds_l.append(itmp)
                jaccard_l.append(jaccard)
                area_tmp = areas_prop[itmp]
                area_l.append(area_tmp)

            # print ("jaccard", jaccard_l
            # get max jaccard
            idx_max = np.argmax(jaccard_l)
            prop_ind = jaccard_inds_l[idx_max]
            jaccard_max = jaccard_l[idx_max]
            if verbose:
                print("  max jaccard:", jaccard_max)
                print("  prop_ind:", prop_ind)

            # check if max jaccard over threshold
            if jaccard_max > iou_thresh:
                true_pos_gt.add(i)
                true_pos_prop.add(prop_ind)
                true_pos_pair.add((i, prop_ind))
            else:
                false_neg_gt.add(i)
        false_pos_prop = set(range(len(boxes_rect_rot_coords))) - true_pos_prop

    #######
#    print ("Num ground_truth boxes:", len(ground_truth_boxes)
#    print ("Num proposed boxes:", len(boxes_rect_rot_coords)
#    print ("Num false negatives:", len(false_neg_gt)
#    print ("Num false positives:", len(false_pos_prop)
#    print ("Num true positives:", len(true_pos_gt)
    p1 = "Num ground_truth boxes: " + str(len(ground_truth_boxes)) + "\n" +\
        "Num proposed boxes: " + str(len(boxes_rect_rot_coords)) + "\n" +\
        "Num false negatives: " + str(len(false_neg_gt)) + "\n" +\
        "Num false positives: " + str(len(false_pos_prop)) + "\n" +\
        "Num true positives: " + str(len(true_pos_gt)) + "\n"
    print(p1)
    outstring += p1
    try:
        recall = 1.*len(true_pos_gt) / (len(true_pos_gt) + len(false_neg_gt))
    except:
        recall = 0.0
    try:
        precision = 1.*len(true_pos_gt) / \
            (len(true_pos_gt) + len(false_pos_prop))
    except:
        precision = 0.0
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
#    print ("recall:", recall)
#    print ("precision:", precision)
#    print ("f1:", f1)
#    print ("Total time to run evaluation:", time.time() - t0, "seconds")

    p2 = "Recall: " + str(recall) + "\n" +\
        "Precision: " + str(precision) + "\n" +\
        "F1: " + str(f1) + "\n" +\
        "Total time to run evaluation: " + str(time.time() - t0) + " seconds\n"
    print(p2)
    outstring += p2

    true_pos_prop = list(true_pos_prop)
    false_pos_prop = list(false_pos_prop)
    false_neg_gt = list(false_neg_gt)
    true_pos_gt = list(true_pos_gt)
    true_pos_pair = list(true_pos_pair)

    # plot
    if plot_name:
        p2p1 = "Create plot: " + plot_name + '\n'
        print(p2p1)
        outstring += p2p1
        legend_dic = {'Ground Truth': (0, 0, 255),  # blue
                      # goldenrod # gold (255,215,0), (255,255,0), # yellow
                      'False Negative': (218, 165, 32),
                      'False Positive': (255, 0, 0),  # red
                      'True Positive': (0, 200, 0)  # (0,255,0) # green
                      }
        legend_text_dic = {'Ground Truth': ' (' + str(len(ground_truth_boxes)) + ')',
                           'False Negative': ' (' + str(len(false_neg_gt)) + ')',
                           'False Positive': ' (' + str(len(false_pos_prop)) + ')',
                           'True Positive': ' (' + str(len(true_pos_gt)) + ')'
                           }

        tp_p = [boxes_rect_rot_coords[i] for i in true_pos_prop]
        fp_p = [boxes_rect_rot_coords[i] for i in false_pos_prop]
        #tp_gt = ground_truth_boxes
        fn_gt = [ground_truth_boxes[i] for i in false_neg_gt]

        arrt = [(fn_gt, 'False Negative'),
                (fp_p, 'False Positive'), (tp_p, 'True Positive')]

        # plot ground truth (blue)
        if len(ground_truth_boxes) > 0:
            color = legend_dic['Ground Truth']
            p2p2 = "Plot Ground truth rects\n"
            print(p2p2)
            outstring += p2p2
            fig, ax, img_mpl = plot_rects_eval(
                im_test_c, ground_truth_boxes, color=color, 
                thickness=thickness, figsize=figsize)

        else:
            fig, ax = plt.subplots(figsize=figsize)
            img_mpl = cv2.cvtColor(im_test_c, cv2.COLOR_BGR2RGB)

        # plot other labels
        p2p3 = "Plot other labels\n"
        print(p2p3)
        outstring += p2p3
        for z in arrt:
            (data, label) = z
            # print ("data", data
            print("label", label)
            # plot false negatives (yellow)
            for coords_rot in data:
                color = legend_dic[label]
                # plot rotated rect

                if type(coords_rot) == np.ndarray:
                    coords_rot_arr = coords_rot
                    coords1 = coords_rot_arr.reshape((-1, 1, 2))

                else:
                    coords_rot_arr = np.array(
                        list(coords_rot.exterior.coords), np.int32)[:-1]
                    coords1 = coords_rot_arr.reshape((-1, 1, 2))

                if verbose:
                    print("coords_rot:", coords_rot)
                    print("coords_rot_arr:", coords_rot_arr)
                    print("coords1:", coords1)

                #coords_rot_arr = np.array(coords_rot)
                #coords1 = coords_rot_arr.reshape((-1,1,2))
                # print ("coords", coords1
                cv2.polylines(img_mpl, [coords1], True,
                              color, thickness=thickness)

#        # plot false negatives (yellow)
#        for coords_rot in fn_gt:
#            color = (255,255,0)
#            thickness=2
#            # plot rotated rect
#            coords1 = coords_rot.reshape((-1,1,2))
#            cv2.polylines(img_mpl, [coords1], True, color, thickness=thickness)
#
#        # plot false positives (red)
#        for coords_rot in fp_p:
#            color = (255,0,0)
#            thickness=2
#            # plot rotated rect
#            coords1 = coords_rot.reshape((-1,1,2))
#            cv2.polylines(img_mpl, [coords1], True, color, thickness=thickness)
#        # plot true positives (green)
#        for coords_rot in tp_p:
#            color = (0,255,0)
#            thickness=2
#            # plot rotated rect
#            coords1 = coords_rot.reshape((-1,1,2))
#            cv2.polylines(img_mpl, [coords1], True, color, thickness=thickness)

        # add border
        # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
        # top, bottom, left, right - border width in number of pixels in corresponding directions
        h, w = img_mpl.shape[:2]
        # use s for scaling
        s = min(h, w)
        print("Input image size:", img_mpl.shape)
        border = (0, int(s*0.07), 0, 0)  # (40, 0, 0, 200)
        border_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_TRIPLEX  # FONT_HERSHEY_SIMPLEX #_SIMPLEX _TRIPLEX
        # font_mult_l sets font_size for different image sizes
        if plot_nums:
            font_mult_l = [0.5, 0.3, 0.32]  # 0.28]#0.32]
        else:
            font_mult_l = [0.6, 0.42, 0.45]

        if s < 2000:
            font_size = font_mult_l[0] * (s / 500)
        elif s > 6000:
            font_size = font_mult_l[1] * (s / 500)
        else:
            font_size = font_mult_l[2] * (s / 500)

        if s <= 1000:
            label_font_width = 1
        elif s > 6000:
            label_font_width = 8
        else:
            label_font_width = 2

        bdiff = int(0.1*border[1])
        #font_width = 1
        #text_offset = [3, 10]

        img_mpl = cv2.copyMakeBorder(img_mpl, border[0], border[1], border[2], border[3],
                                     cv2.BORDER_CONSTANT, value=border_color)
        # add legend
        #xpos = img_mpl.shape[1] - border[3] + 15
        #ydiff = border[0]
        ypos = img_mpl.shape[0] - int(border[1]*0.45)
        if plot_nums:
            xdiff = int(w/4.05)
        else:
            xdiff = int(w/4.2)
        for itmp, labelt in enumerate(sorted(legend_dic.keys())):
            colort = legend_dic[labelt]
            # print ("itmp, labelt, colort", itmp, labelt, colort
            if plot_nums:
                text_append = legend_text_dic[labelt]
            else:
                text_append = ''
            text = '- ' + labelt + text_append  # str(k) + ': ' + labelt
            # vertical legend
            #xpos = xpos
            #ypos = border[0] + (2+itmp) * ydiff
            # horizontal legend
            #xpos = int(1.5 * border[1]) + (itmp) * xdiff
            if plot_nums:
                xpos = int(0.3 * border[1]) + (itmp) * xdiff
            else:
                xpos = int(0.7 * border[1]) + (itmp) * xdiff

            ypos = ypos
            try:
                cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, font_size,
                            colort, label_font_width, cv2.CV_AA)  # cv2.LINE_AA)#cv2.CV_AA)#
            except:
                cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font,
                            font_size, colort, label_font_width, cv2.LINE_AA)
        # legend box (vertical)
        #cv2.rectangle(img_mpl, (xpos-5, 2*border[0]), (img_mpl.shape[1]-10, ypos+int(0.75*ydiff)), (0,0,0), label_font_width)
        # legend box (horizontal)
        cv2.rectangle(img_mpl, (bdiff, h + bdiff), (img_mpl.shape[1]-bdiff, h + int(
            0.9*border[1]) - bdiff), (0, 0, 0), label_font_width)

        # title
        # title = figname.split('/')[-1].split('_')[0] + ':  Plot Threshold = ' + str(plot_thresh) # + ': thresh=' + str(plot_thresh)
        #title_pos = (border[0], int(border[0]*0.66))
        # cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), label_font_width, cv2.CV_AA)#, cv2.LINE_AA)

        # plot F1 in top right
        f1_text = 'F1=' + str(round(f1, 2))
        xf1, yf1 = int(0.83*w), int(0.04*h)  # int(0.82*w), int(0.05*h)
        # xf1, yf1 = int(0.83*w), int(0.05*h)  #int(0.82*w), int(0.05*h)
        #xf1, yf1 = int(0.88*w), int(0.04*h)

        #f1_text = '*F1='  + str(round(f1, 2))
        #xf1, yf1 = int(0.85*w), int(0.04*h)

        if plot_nums:
            f1_size_mult = 1.9
        else:
            f1_size_mult = 1.5

        try:
            cv2.putText(img_mpl, f1_text, (xf1, yf1), font, f1_size_mult*font_size,
                        colorf1, label_font_width, cv2.CV_AA)  # cv2.LINE_AA)#cv2.CV_AA)#
        except:
            cv2.putText(img_mpl, f1_text, (xf1, yf1), font, f1_size_mult *
                        font_size, colorf1, label_font_width, cv2.LINE_AA)  # cv2.CV_AA)#

        # plt.tight_layout()
        plt.axis('off')
        plt.tight_layout()
        plt.close('all')
        #plt.savefig(plot_name, dpi=dpi)

        if len(plot_name) > 0:
            # save high resolution
            #plt.savefig(figname, dpi=dpi)
            img_mpl_out = cv2.cvtColor(img_mpl, cv2.COLOR_BGR2RGB)  # img_mpl #
            #cv2.imwrite(plot_name, img_mpl_out)
            cv2.imwrite(plot_name, img_mpl_out,  [
                        cv2.IMWRITE_PNG_COMPRESSION, valid_im_compression_level])

    #outvals = [f1, precision, recall, num_ground_truth, num_proposed, num_false_neg, num_false_pos, num_true_pos]
    outvals = [f1, precision, recall, len(ground_truth_boxes),
               len(boxes_rect_rot_coords), len(false_neg_gt),
               len(false_pos_prop), len(true_pos_gt)]

    return true_pos_prop, false_pos_prop, false_neg_gt, true_pos_gt, true_pos_pair, outstring, outvals


###############################################################################
def get_gdf_tot(truth_dir, extension_list, enforce_rects=True, extra_pkls=[],
                verbose=False):
    '''
    Gather geodataframes from truth_dir, assume the following schema:
        image:       image_test.tif
        shpf_file:   image_test_boat.shp
                     image_test_airplane.shp...
    if enforce_rects, cast ground truth geometry to a bounding box
    Return a geodataframe of all the ground truth geometries
    '''

    print("Executing get_gdf_tot()...")
    shp_files = [f for f in os.listdir(truth_dir) if f.endswith('.shp')]
    for i, shp_file in enumerate(shp_files):

        # get image
        image_root0 = '_'.join(shp_file.split('.')[0].split('_')[:-1])
        found = False
        for ext in extension_list:
            if os.path.exists(os.path.join(truth_dir, image_root0 + ext)):
                image_root = image_root0 + ext
                found = True
                break
        if not found:
            print("image file not found for shapefile:", shp_file)
            break

        if verbose:
            print("shape file:", shp_file)
            print("image_root:", image_root)
        shp_file_tot = os.path.join(truth_dir, shp_file)
        image_file_tot = os.path.join(truth_dir, image_root)

        # transform crs of image file?
        # parse_shapefile.transform_crs(image_file_tot)

        # get geodataframe
        gdf, _ = parse_shapefile.get_gdf_pix_coords(shp_file_tot, image_file_tot,
                                                    category='',
                                                    max_aspect_ratio=3,
                                                    line_padding=0.1,
                                                    enforce_rects=enforce_rects,
                                                    verbose=verbose)
        if verbose:
            print("gdf.columns:", gdf.columns)
        # check that pixel coords are > 0
        if np.min(gdf['xmin'].values) < 0:
            if verbose:
                print("x pixel coords < 0:", np.min(gdf['xmin'].values))

        if np.min(gdf['ymin'].values) < 0:
            if verbose:
                print("y pixel coords < 0:", np.min(gdf['ymin'].values))

        if i == 0:
            gdf_tot = gdf
        else:
            gdf_tot = gdf_tot.append(gdf)
    gdf_tot.index = np.arange(len(gdf_tot))

    # add extra pkls
    # if len(extra_pkls) == 0:
    #    for pkl in extra_pkls:
    #        df_tmp = pd.read_pickle(pkl)
    #        gdf_tot = gdf_tot.append(df_tmp)
    #    gdf_tot.index = np.arange(len(gdf_tot))

    return gdf_tot


'''
get_gdf_tot_cowc(truth_dir, annotation_suffix='_Annotated_Cars.png',
                     category='car', yolt_box_size=10,
                     verbose=False)
'''


###############################################################################
def eval_f1(gdf_truth, df_prop, im_root, category, detect_thresh=0.5,
            iou_thresh=0.5, nms_overlap=0.5, plot_file='', log_file='',
            colorf1=(0, 255, 255), line_thickness=2, figsize=(12, 12),
            show_intermediate_plots=False, out_ext='.tif',
            verbose=False):
    '''Evaluate f1 for the given image and category
    compute_performance returns many values, but outvals is the most useful:
            outvals = [ f1, precision, recall, len(ground_truth_boxes), 
                len(boxes_rect_rot_coords), len(false_neg_gt), 
                len(false_pos_prop), len(true_pos_gt) ]
    return this value from eval_f1
    skip if empty proposals or ground truth
   '''

    # get truth subset
    gdf_truth_filt = gdf_truth[(gdf_truth['Image_Root'] == im_root) &
                               (gdf_truth['Category'] == category)]
    # get proposal subset
    df_prop_filt = df_prop[(df_prop['Image_Root'] == im_root) &
                           (df_prop['Category'] == category) &
                           (df_prop['Prob'] >= detect_thresh)]
    if verbose:
        print("len(gdf_truth_filt):", len(gdf_truth_filt))
        print("len(df_prop_filt):", len(df_prop_filt))
        print("  gdf_truth_filt.loc[0]:", gdf_truth_filt.iloc[0])
        print("  df_prop_filt.loc[0]:", df_prop_filt.iloc[0])

    # return if empty ground_truth
    if (len(gdf_truth_filt) == 0):
        return 'Empty', 8*[0]
    # return if empty proposals
    if (len(gdf_truth_filt) == 0) or (len(df_prop_filt) == 0):
        return 'No Proposals', [0, 0, 0, len(gdf_truth_filt), 0, len(gdf_truth_filt), 0, 0]
    # return if emtpy proposals or gt
    # if (len(gdf_truth_filt) == 0) or (len(df_prop_filt) == 0):
    #    return 'Empty', [0,0,0, len(gdf_truth_filt), 0, 0, 0, 0] #8*[0]

    image_path = gdf_truth_filt['Image_Path'].values[0]

    #########
    # logging
    # create log file
    logstring = "EVALUATE...\n\n"
    status0 = "\nEval for: " + image_path + '\n' + \
        "detect_thresh: " + str(detect_thresh) + '\n' + \
        "IOU true_thresh: " + str(iou_thresh) + '\n' + \
        "category: " + str(category)
    print(status0)
    logstring += status0
    #########

    ground_truth_boxes = gdf_truth_filt['geometry_poly_pixel'].values
    proposed_boxes = df_prop_filt['Geometry'].values

    if nms_overlap > 0:
        boxes = [p.bounds for p in proposed_boxes]
        boxes_nms, boxes_tot_nms, nms_idxs \
            = post_process.non_max_suppression(boxes, nms_overlap)
        proposed_boxes = proposed_boxes[nms_idxs]
    if verbose:
        print("num boxes (after nms):", len(proposed_boxes))

    #########
    # plots
    if not os.path.exists(image_path):
        print("Path DNE!:", image_path, "skipping plots")
        plot_file = ''
        # return
    if len(plot_file) > 0:
        # Read in image and proposal files
        im_test_c = cv2.imread(image_path, 1)  # color
    else:
        im_test_c = None
    #print ("image_path:", image_path)
    #print ("im_test_c.shape:", im_test_c.shape)

    #########
    # Compare ground truth and predicted
    if verbose:
        print("plot_file:", plot_file)
    # Need to write copute_map() print statements to file
    true_pos_prop, false_pos_prop, false_neg_gt, \
        true_pos_gt, true_pos_pair, status3, outvals3 = \
        compute_performance(ground_truth_boxes, proposed_boxes,
                            iou_thresh=iou_thresh, im_test_c=im_test_c,
                            plot_name=plot_file, figsize=figsize,
                            thickness=line_thickness, colorf1=colorf1,
                            verbose=False)

    logstring += status3

    # write to log file
    if len(log_file) > 0:
        with open(log_file, "a") as text_file:
            text_file.write(logstring)

    if not show_intermediate_plots:
        plt.close('all')

    return status3, outvals3


###############################################################################
def make_performace_plot_mpl(df, groupby, x_val, y_val, y_val_mean=-1,
                             y_val_error=-1, outdir='', linspace_len=100,
                             show_legend=False, verbose=False, plot_type='line',
                             stat_weights='Number of Cars (Ground Truth)',
                             plot_dyn_err=False, interp_method='cubic',
                             plot_style='seaborn-ticks', figsize=(9, 6),
                             y_val_alpha=0.2, error_alpha=0.5, scatter_alpha=0.2,
                             dpi=500, xticklabel_pad=-0.1, twinx_func=False,
                             fit_func=False, show_grid=True, custom_x_axis=True):
    '''https://stackoverflow.com/questions/4805048/how-to-get-different-colored-lines-for-different-plots-in-a-single-figure
    '''

    plt.style.use(plot_style)  # ('fivethirtyeight')#('ggplot')
    colormap = plt.cm.gist_ncar

    error_plus = linspace_len * [y_val_mean + error_val]
    error_minus = linspace_len * [mean_val - error_val]

    fig, ax = plt.subplots(figsize=figsize)
    group = df.groupby(groupby)

    # set colors
    plt.gca().set_color_cycle([colormap(i)
                               for i in np.linspace(0, 0.9, len(group))])

    y_vals = []
    labels = []
    y_arr = []
    weights = []
    for g in group:
        label = g[0]
        x = g[1][x_val]
        y = g[1][y_val]
        y_vals.extend(y)
        y_arr.append(y.values)
        labels.append(label)
        # get weight of first item (should all be the same)
        if stat_weights != '':
            weights.append(g[1][stat_weights].values[0])
        if verbose:
            print("label:", label)
            print("x:", x)
            print("y:", y)
        if plot_type == 'line':
            ax.plot(x, y, label=label, alpha=y_val_alpha)
        ax.scatter(x, y, label=None, s=5, color='gray', alpha=scatter_alpha)

    # if desired, plot error band
    if y_val_error != -1:
        x_error = np.linspace(
            np.min(df[x_val])-1, np.max(df[x_val])+1, linspace_len)
        plt.fill_between(x_error, error_plus, error_minus,
                         alpha=error_alpha, color='red')

    # get error for each x val
    y_arr = np.array(y_arr)
    #mean_y = np.mean(y_arr, axis=1)
    means_w, stds_w = [], []
    for itmp in range(len(x)):
        ytmp = y_arr[:, itmp]
        mean_w, std_w, var_w = weighted_avg_and_std(ytmp, weights)
        means_w.append(mean_w)
        stds_w.append(std_w)
    means_w, stds_w = np.array(means_w), np.array(stds_w)
    if plot_dyn_err:
        # x_error = np.linspace(np.min(df[x_val]), np.max(df[x_val]),
        #                      linspace_len)
        x_error = np.linspace(df[x_val].values[0], df[x_val].values[-1],
                              linspace_len)

        # interpolate (linear)
        if interp_method == 'linear':
            mean_interp = np.interp(x_error, x, means_w)
            err_interp = np.interp(x_error, x, stds_w)
            if x.values[0] > x.values[-1]:
                mean_interp = np.interp(x_error[::-1], x[::-1], means_w[::-1])
                err_interp = np.interp(x_error[::-1], x[::-1], stds_w[::-1])
                mean_interp = mean_interp[::-1]
                err_interp = err_interp[::-1]
        #interpolate (cubic)
        else:
            order = 3
            mean_interp = scipy.interpolate.spline(x, means_w, x_error,
                                                   order=order)
            err_interp = scipy.interpolate.spline(x, stds_w, x_error,
                                                  order=order)
            #mean_interp = scipy.interpolate.splrep(x, means_w, x_error)
            #err_interp = scipy.interpolate.spline(x, stds_w, x_error)
            #tck = scipy.interpolate.splrep(x, means_w, s=0)
            #mean_interp = scipy.interpolate.splev(x_error, tck, der=0)

        if fit_func:
            fit_color = 'cyan'
            fit_alpha = 0.6
            # https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
            # def piecewise_linear(x, x0, y0, k1, k2):
            #    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
            # fit guess
            fit_guess = [0.6, 0.85, -0.25, -0.25]   # if x axis is gsd
            fit_guess = [5, 0.85, 0.1, 0.01]        # if x axis is pixel size
            print("x:", x.values)
            print("means_w:", means_w)
            p, e = optimize.curve_fit(fit_func, x.values, means_w,
                                      p0=fit_guess)
            #p , e = optimize.curve_fit(piecewise_linear, x_error, mean_interp)
            print(
                "p params: [x_inflection, y_inflection, slope_less_than_inflection, slope_greater_than_inflection]")
            print("p:", p)
            # print ("e", e
            xd = np.linspace(np.min(x.values), np.max(x.values), 100)
            yd = piecewise_linear(xd, *p)
            print("xd.shape:", xd.shape)
            print("yd.shape:", yd.shape)
            plt.plot(xd, yd, c=fit_color, linestyle=':',
                     linewidth=2.5, alpha=fit_alpha)
            # show inflection point (do this later)
            #plt.scatter(p[0], p[1], c=fit_color, alpha=0.8, s=15)

#            #https://stackoverflow.com/questions/19955686/fit-a-curve-for-data-made-up-of-two-distinct-regimes
#            def two_lines(x, a, b, c, d):
#                one = a*x + b
#                two = c*x + d
#                return np.maximum(one, two)
#            y = means_w
#            pw0 = (-0.2, 0.9, -0.5, 1.7) # a guess for slope, intercept, slope, intercept
#            pw, cov = opimize.curve_fit(two_lines, x, y, pw0)
#            plt.plot(x, two_lines(x, *pw), ':', c='gray')

# plot model, if desired
#            z = np.polyfit(x_error, mean_interp, 1)
#            print ("slope, y-intercept:", z
#            p = np.poly1d(z)
#            plt.plot(x_error, p(x_error), linewidth=1, c='gray', linestyle='--' )

            # compute slope in twinx
            if twinx_func:
                xbreak, ybreak = p[0], p[1]
                xtwinbreak = twinx_function(xbreak, raw=True)
                xtwin0, xtwin1 = twinx_function(
                    xd[0], raw=True), twinx_function(xd[-1], raw=True)
                slope0 = (ybreak - yd[0]) / (xtwinbreak - xtwin0)
                slope1 = (yd[-1] - ybreak) / (xtwin1 - xtwinbreak)
                print("xd[0], xbreak, xd[-1]:", xd[0], xbreak, xd[-1])
                print("xtwin0, xtwinbreak, xtwin1:",
                      xtwin0, xtwinbreak, xtwin1)
                print("yd[0], ybreak, yd[-1]:", yd[0], ybreak, yd[-1])
                print("axis twin slope gsd < break:", slope0)
                print("axis twin slope gsd > break:", slope1)

        # plot mean and error
        mean_color = 'blue'  # 'gray'#'gray'
        err_color = 'red'
        #ax.scatter(x_error, mean_interp, color=mean_color, s=4)
        ax.plot(x_error, mean_interp, '--', label='Weighted Mean', color=mean_color,
                linewidth=2.6, alpha=0.8)
        plt.fill_between(x_error, mean_interp+err_interp,
                         mean_interp-err_interp, alpha=0.5,
                         color=err_color)
        if fit_func:
            # show inflection point
            plt.scatter(p[0], p[1], c=fit_color, alpha=fit_alpha, s=20)

    for (xtmp, ytmp, errtmp) in zip(x, means_w, stds_w):
        print("x, y, err:", xtmp, ytmp, errtmp)
    print("mean_err:", np.mean(stds_w))

    axes = plt.gca()
    dx = np.max(x) - np.min(x)
    axes.set_xlim([np.min(x)-0.03*dx, np.max(x)+0.03*dx])
    #axes.set_xlim([-0.5, len(df)-0.5])
    axes.set_ylim([0.0, 1.1*np.max(y_vals)])
    if plot_dyn_err:
        axes.set_ylim([0.0, 1.05*np.max(mean_interp + err_interp)])

    ax.set_ylabel(y_val)
    ax.set_xlabel(x_val)

    ax.yaxis.set_tick_params(labelsize=8)
    #ax.set_yticklabels(ax.yaxis.get_majorticklabels(), fontsize=8)
    if custom_x_axis:
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=50, fontsize=8)
        ax.tick_params(axis='x', which='major', pad=xticklabel_pad)

    if show_grid:
        ax.grid(b=True, which='major', color='gray',
                alpha=0.075, linestyle=':')
        #gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        # for line in gridlines:
        #    line.set_linestyle('-.')

    if twinx_func:
        # https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
        print("x", x)
        print("twinx_func(x):", twinx_function(x))
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels(twinx_function(x))
        ax2.set_xlabel(r"Object Pixel Size")
        ax2.set_xticklabels(twinx_function(x), rotation=50, fontsize=8)
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.tick_params(axis='x', which='major', pad=xticklabel_pad)

    #ax.set_xticks(range(len(valid_dirs)), valid_dirs)
    title = 'YOLT2 ON COWC CARS'
    if twinx_func:
        title_yloc = 1.35
        #ax.set_title(title, y=title_yloc)

    else:
        title_yloc = 1
        ax.set_title(title, y=title_yloc)

    if show_legend:
        ax.legend(labels, ncol=1, loc='center right',
                  bbox_to_anchor=[1.1, 0.5],
                  columnspacing=1.0, labelspacing=0.0,
                  handletextpad=0.0, handlelength=1.5,
                  fancybox=True, shadow=True,
                  fontsize='x-small')

        # ax.legend(fontsize='small')

    # if just one validation file...
    # plt.xticks(df[x_val], im_roots, rotation=70)#'vertical')

    plt.tight_layout()
    plt.show()

    # print 'outdir:', outdir
    figname = outdir + \
        'total_df_yoltthresh=' + detect_thresh + '_' + groupby + '_' + x_val + '_' +\
        y_val + '_' + suffix + '.png'

    if len(outdir) > 0:
        # plt.savefig(figname, bbox_extra_artists=(title), bbox_inches='tight',
        #            dpi=dpi)
        plt.savefig(figname, dpi=dpi)


###############################################################################
def refine_precision_recall(precision, recall, verbose=False):
    '''Refine precisio nand recall for plotting'''

    x = recall
    y = precision

    if (len(x) == 0) or (len(y) == 0):
        return x, y, 0, 0

    # reverse order
    x = x[::-1]
    y = y[::-1]

#    # reorder?
#    order = np.argsort(x)
#    x = x[order]
#    y = y[order]

    # final index should be max recall?
    max_idx = np.argmax(x)
    x = x[:max_idx]
    y = y[:max_idx]

    # cut off early zeros
    idxs_pos = np.where(x > 0)
    x = x[idxs_pos]
    y = y[idxs_pos]

    # add a couple data points that correspond to the edge of the plot
    if len(x) == 0:
        x = [0]
        y = [0]
    else:
        x = np.concatenate(([0],     x, [x[-1]]))
        y = np.concatenate(([y[0]], y, [0]))

    # compute area
    area_nptrapz = np.trapz(y, x=x)

    try:
        area_sklearn = sklearn.metrics.auc(x, y)
    except:
        area_sklearn = -1

    if verbose:
        print("x:", x)
        print("y:", y)
        print("Area_nptrapz:", area_nptrapz)
        print("Area sklearn:", area_sklearn)

    return x, y, area_nptrapz, area_sklearn


###############################################################################
def plot_precision_recall(precision, recall, outfile='', figsize=(6, 6),
                          title='', dpi=300):
    '''plot precision recall curve
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    https://sanchom.wordpress.com/tag/average-precision/
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.step.html
    '''

    if (len(recall) == 0) or np.max(recall) <= 0:
        return 0, 0, 0

    # plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    #df.plot(x='precision', y='recall')
    #precision, recall = df['precision'].values, df['recall'].values

    # recall should be 1 at the start.  This is because super low thresholds
    # screw up bounding boxes due to nms. Clip these values for plotting
    #start_idx = np.argmax(recall >= 1.0)
    #precision = precision[start_idx:]
    #recall = recall[start_idx:]

    x, y, area_nptrapz, area_sklearn = refine_precision_recall(
        precision, recall)

    #########
    # plots
    plt.step(x, y, color='b', alpha=0.2, where='post')
    #plt.plot(x,y, color='red')
    # plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                 color='b')

    # plot f1 lines and values
    #f_scores = np.linspace(0.2, 0.8, num=4)
    f_scores = np.linspace(0.4, 0.9, num=5)
    #lines = []
    #labels = []
    for f_score in f_scores:
        xf1 = np.linspace(0.01, 1, 500)
        yf1 = f_score * xf1 / (2 * xf1 - f_score)
        # plot f1 lines
        #l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        l, = ax.plot(xf1[(yf1 >= 0) & (yf1 <= 1)],
                     yf1[(yf1 >= 0) & (yf1 <= 1)], color='gray', alpha=0.2)
        # plot f1 annotations
        #plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        #ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[int(len(x)*0.9)] + 0.02))
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(
            0.86, yf1[int(len(xf1)*0.9)] + 0.02))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
#    ax.set_ylim([0.0, 1.05])
#    ax.set_xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #          average_precision))
    # print 'precision:', precision
    # print 'recall:', recall

    if len(title) > 0:
        title += ':  AP=' + str(np.round(area_nptrapz, 2))
        ax.set_title(title)

    plt.tight_layout()

    if len(outfile) > 0:
        plt.savefig(outfile, dpi=dpi)

#    # for output
#    # add a couple data points that correspond to the edge of the plot
#    #  set max recall and precision == 0, and visa versa
#    # add a data point of max precision, 0 recall
#    precision = np.append(precision, [precision[-1]] )#.extend(np.max(precision))
#    recall = np.append(recall, [0]) #recall.extend(0)
#    # add a data point of max recall, 0 precision at the start
#    recall = np.append([recall[0]], recall )
#    precision = np.append([0], precision)

    recall_out = x
    precision_out = y
    return precision_out, recall_out, area_sklearn


###############################################################################
def plot_precision_recall_multi(df_scores_tot, outfile='', figsize=(4, 4),
                                title='', dpi=300, linewidth=2,
                                linealpha=0.4, f1xy=0.86, f1loc_buff=0.02,
                                show_legend=True, cat_colors=[],
                                verbose=False):
    '''plot precision recall curve for multiple csvs
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.step.html
'''

    # plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)
    linestyles = 20*['-', '--', '-.', ':']
    # https://stackoverflow.com/questions/33337989/how-to-draw-more-type-of-lines-in-matplotlib/33338727
    dashList = 20*[(5, 2), (4, 1), (3, 2), (1, 1), (0, 0)]
    # dashList = 20*[(1,1),(2,1)]  # for yolt cars and cars_2x
    #dashList = 20*[(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]

    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11
    weight = 'medium'  # 'semibold'
    # controls default text sizes
    plt.rc('font', size=SMALL_SIZE, weight=weight)
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    cats = np.sort(np.unique(df_scores_tot['Category']))
    if verbose:
        print("Categories:", cats)
    areas = []
    for i, category in enumerate(cats):
        linestyle = linestyles[i]
        df_filt = df_scores_tot[df_scores_tot['Category'] == category]
        precision, recall = df_filt['Precision'].values, df_filt['Recall'].values
        x, y, area_nptrapz, area_sklearn = refine_precision_recall(
            precision, recall)
        areas.append(area_nptrapz)
        if verbose:
            print("Category:", category)
            print("area_nptrapz:", area_nptrapz)
            print("area_sklearn:", area_sklearn)
            print("precision average:", np.mean(precision))

        #########
        # plots
        if len(cats) < 16:  # 6:
            legend_text = category + ': AP=' + str(np.round(area_nptrapz, 2))
            #legend_text = category
            # plt.step(x, y, where='post', alpha=linealpha, label=legend_text,
            #     linestyle=linestyle, linewidth=linewidth)
            if len(cat_colors) > 0:  # use preset colors
                plt.step(x, y, where='post', alpha=linealpha, label=legend_text,
                         linestyle='--', linewidth=linewidth, dashes=dashList[i], color=cat_colors[i])
            else:
                plt.step(x, y, where='post', alpha=linealpha, label=legend_text,
                         linestyle='--', linewidth=linewidth, dashes=dashList[i])
        else:
            if len(cat_colors) > 0:  # use preset colors
                plt.step(x, y, where='post', alpha=linealpha,
                         linestyle=linestyle, linewidth=linewidth, color=cat_colors[i])
            else:
                plt.step(x, y, where='post', alpha=linealpha,
                         linestyle=linestyle, linewidth=linewidth)

        #plt.plot(x,y, color='red')
        # plt.fill_between(recall, precision, step='post', alpha=0.2,
        #                 color='b')

        # plot f1 lines and values
        #f_scores = np.linspace(0.2, 0.8, num=4)
        f_scores = np.linspace(0.4, 0.9, num=5)
        for f_score in f_scores:
            xf1 = np.linspace(0.01, 1, 500)
            yf1 = f_score * xf1 / (2 * xf1 - f_score)
            # plot f1 lines
            #l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            l, = ax.plot(xf1[(yf1 >= 0) & (yf1 <= 1)], yf1[(
                yf1 >= 0) & (yf1 <= 1)], color='gray', alpha=0.15)
            # plot f1 annotations
            #plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            #ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[int(len(x)*0.9)] + 0.02))
            ax.annotate('f1={0:0.1f}'.format(f_score), xy=(
                f1xy, yf1[int(len(xf1)*0.9)] + f1loc_buff))

    mAP = np.mean(areas)
    print("mAP:", mAP)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.03])
    ax.set_xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #          average_precision))
    # print 'precision:', precision
    # print 'recall:', recall

    if len(title) > 0:
        title += ':  mAP=' + str(np.round(mAP, 2))
        # fig.suptitle(title)
        ax.set_title(title)

    if show_legend:
        # plt.rc('font', size=10, weight=weight)          # controls default text sizes
        if len(cats) < 6:
            plt.legend(fontsize='small')
        else:
            plt.legend(fontsize='x-small')

    plt.tight_layout()

    if len(outfile) > 0:
        plt.savefig(outfile, dpi=dpi)


###############################################################################
def compute_map(df_scores_tot, category='All', f1_thresh=0.15, verbose=False):
    '''
    Akin to plot_precision_recall_multi, but no plotting
    f1_thresh is the threshold to use for optimum f1
    if desired, use only certain categories
    stds are not very useful and shouldn't be used for publication, since
     they are rough aggregates and if only once class is used std = 0
    '''

    # if we don't pass in any cats, use all of em
    if category == 'All':
        cats_list = np.sort(np.unique(df_scores_tot['Category']))
    else:
        cats_list = [category]

    if verbose:
        print("Categories:", cats_list)

    areas = []
    f1s = []
    for i, category in enumerate(cats_list):
        df_filt = df_scores_tot[df_scores_tot['Category'] == category]
        precision, recall = df_filt['Precision'].values, df_filt['Recall'].values
        x, y, area_nptrapz, area_sklearn = refine_precision_recall(
            precision, recall)
        areas.append(area_nptrapz)
        # get f1s
        if f1_thresh > 0:  # and len(cats) > 0:
            f1s_tmp = df_filt[df_filt['Detect_Thresh']
                              == f1_thresh]['F1'].values
            #print ("f1s_tmp:", f1s_tmp)
            f1s.extend(f1s_tmp)

        if verbose:
            print("Category:", category)
            print("area_nptrapz:", area_nptrapz)
            print("area_sklearn:", area_sklearn)
            print("precision average:", np.mean(precision))
            print("mean f1:", np.mean(f1s_tmp))

    mAP = np.mean(areas)
    mAP_std = np.std(areas)

    if verbose:
        print("category:", category, "f1s:", f1s)

#    # compute f1s (this one onlyl works if cats=[])
#    if f1_thresh > 0 and len(:
#        f1s = df_scores_tot[df_scores_tot['Detect_Thresh'] == f1_thresh]['F1'].values
#        f1_mean = np.mean(f1s)
#        f1_std = np.std(f1s)
#    else:
#        f1_mean, f1_std = 0, 0

    if len(f1s) > 1:
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s)
    elif len(f1s) == 1:
        f1_mean = f1s[0]
        f1_std = 0
    else:
        f1_mean, f1_std = 0, 0

    return mAP, mAP_std, f1_mean, f1_std


###############################################################################
def execute(args):

    print("Execute simrdwn_eval.py...")

    ##############
    # ground truth

    # find shape files
    if len(args.test_truth_pkl) == 0:
        shp_files = [f for f in os.listdir(
            args.truth_dir) if f.endswith('.shp')]
        # if shape files exist, extract ground truth from them
        if len(shp_files) > 0:
            gdf_truth = get_gdf_tot(args.truth_dir, args.extension_list,
                                    verbose=args.verbose)

        # else assume it's the cars dataset
        else:

            print("yolt_box_size for cars:", args.yolt_car_box_size)

            df_path = os.path.join(args.truth_dir, '_truth_df.csv')
            if not os.path.exists(df_path):
                print("Creating gdf_truth for cowc cars...")
                gdf_truth = parse_cowc.get_gdf_tot_cowc(args.truth_dir,
                                                        annotation_suffix='_Annotated_Cars.png',
                                                        category='car',
                                                        yolt_box_size=args.yolt_car_box_size,
                                                        verbose=args.verbose)
            else:
                print("Loading gdf_truth from:", df_path, "...")
                gdf_truth = pd.read_csv(df_path)
                # print ("gdf_truth[geometry_poly_pixel]:", gdf_truth['geometry_poly_pixel']
                print("type(gdf_truth[geometry_poly_pixel].values[0])):", type(
                    gdf_truth['geometry_poly_pixel'].values[0]))
                # turn wkt string into shapely geom
                gdf_truth['geometry_pixel'] = [shapely.wkt.loads(
                    z) for z in gdf_truth['geometry_pixel']]
                gdf_truth['geometry_poly_pixel'] = [shapely.wkt.loads(
                    z) for z in gdf_truth['geometry_poly_pixel']]
                print("refined type(gdf_truth[geometry_poly_pixel].values[0])):", type(
                    gdf_truth['geometry_poly_pixel'].values[0]))

    # parse xview dataset
    else:
        with open(args.test_truth_pkl, "rb") as input_file:
            gdf_truth = pickle.load(input_file)

    # print ("len(gdf_tot):", len(gdf_truth)
    # print ("gdf_truth.iloc[len(gdf_truth)-1]:", gdf_truth.iloc[len(gdf_truth)-1]
    print("len(gdf_truth):", len(gdf_truth))

    im_roots = np.unique(gdf_truth['Image_Root'])
    # optional: rename im_roots to be a subset of images?
    # im_roots = ['5.tif', '18.tif', '20.tif', '24.tif', '31.tif', '38.tif',
    #            '40.tif', '41.tif', '43.tif', '46.tif', '46.tif',
    #            '53.tif', '69.tif']
    #im_roots = [z for z in os.listdir(args.prediction_dir) if z.endswith('.tif')]
    # clean out im_roots names?
    # ...

    print("im_roots:", im_roots)

    categories = np.sort(np.unique(gdf_truth['Category']))
    print("ground truth categories:", categories)

    unique, counts = np.unique(
        gdf_truth['Category'].values, return_counts=True)
    print("Ground Truth Category counts")
    print(np.asarray((unique, counts)).T)

    ##############
    # proposals

    print("\nGather proposals...")
    prop_df_file = os.path.join(
        args.prediction_dir, 'valid_predictions_aug.csv')
    df_prop = pd.read_csv(prop_df_file)
    print("df_prop.columns:", df_prop.columns)
    # print ("len(df_prop):", len(df_prop)

    unique, counts = np.unique(df_prop['Category'].values, return_counts=True)
    print("Proposal Category counts")
    print(np.asarray((unique, counts)).T)

    # Index([u'Unnamed: 0', u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax',
    #       u'Category', u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY',
    #       u'Upper', u'Left', u'Height', u'Width', u'Pad', u'Image_Path'],
    #      dtype='object')

    print("Create proposal geometry column...")
    t1 = time.time()
    poly_list = []
    for idx, row in df_prop.iterrows():
        xmin, xmax = row['Xmin_Glob'], row['Xmax_Glob']
        ymin, ymax = row['Ymin_Glob'], row['Ymax_Glob']
        geom = shapely.geometry.box(xmin, ymin, xmax, ymax)
        poly_list.append(geom)
    df_prop['Geometry'] = poly_list
    t2 = time.time()
    print("Time to create geometry column:", t2-t1, "seconds")
    print("len(df_prop):", len(df_prop))
    prop_categories = np.sort(np.unique(df_prop['Category']))
    print("prop categories:", prop_categories)

    #  We need to run post_process.refine_and_plot_df(plot=False) to do nms for
    #    each subsample of the predicted boxes at the given thresh

    # get minimum prop
    min_prob = np.min(df_prop['Prob'].values)
    # update detect_threshes if it's automated
    min_thresh = max(min_prob, args.detect_thresh_min)
    if len(args.detect_threshes_str) > 0:
        args.detect_threshes = np.array(
            args.detect_threshes_str.split(',')).astype(float)
    else:
        args.detect_threshes = np.arange(min_thresh,
                                         args.detect_thresh_max,
                                         args.detect_thresh_delta)

    ##############
    # test the proposal?
    if args.run_prop_test == 1:

        # refine plot_df
        print("\n\nRunning proposal test...")
        print("im_roots:", im_roots)
        groupby = 'Image_Root'
        groupby_cat = 'Category'

        df_prop_refine = post_process.refine_df(df_prop,
                                                groupby=groupby,
                                                groupby_cat=groupby_cat,
                                                nms_overlap_thresh=args.nms_overlap,
                                                plot_thresh=args.detect_thresh_plot,
                                                verbose=False)

#        df_prop_refine = post_process.refine_and_plot_df(df_prop,
#                           groupby='Image_Path',
#                           sliced=True,
#                           outdir=args.out_dir,
#                           plot_thresh=args.detect_thresh_plot,
#                           nms_overlap_thresh=args.nms_overlap,
#                           plot=False,
#                           verbose=False)

        print("len df_prop_refine:", len(df_prop_refine))
        # pick random image, make sure objects exist in said image
        selected_im = False
        iter_tmp = 0
        while not selected_im:
            im_root = im_roots[np.random.randint(len(im_roots))]
            iter_tmp += 1
            if iter_tmp > len(im_roots):
                break

            print("Test im_root:", im_root)
            # figure out which categories are in the ground truth of the image
            # get truth subset
            gdf_truth_filt = gdf_truth[(gdf_truth['Image_Root'] == im_root)]
            cats_present_truth = np.unique(gdf_truth_filt['Category'].values)
            #category = categories[np.random.randint(len(categories))]
            print("cats_present_truth:", cats_present_truth)

            # get proposal subset
            df_prop_filt = df_prop[(df_prop['Image_Root'] == im_root) &
                                   (df_prop['Prob'] >= args.detect_thresh_plot)]
            cats_present_prop = np.unique(df_prop_filt['Category'].values)
            print("cats_present_prop:", cats_present_prop)
            # see if any categories exist in both truth and proposal
            cats_present = list(
                set(cats_present_truth).intersection(set(cats_present_prop)))
            print("Categories present in both truth and prop:", cats_present)
            if len(cats_present) > 0:
                # cats_present[np.random.randint(len(cats_present))]
                category = random.choice(cats_present)
                selected_im = True

        ##########
        # Optional: manually set category and im_root
        # gqis test
        #category = 'airplane'
        # im_root = '054956943030_01_assembly_17_02_BaghdadAir.tif'#'013022232122_buildings_coast_boats_airport_validate_crop2.tif' #'054593918020_01_assembly_3_5_LondonCityAir.tif'

        #category = 'boat'
        #im_root = 'AOI1.tif' #
        #im_root = '20160211_112333_0b0d_visual_suez.tif'
        #im_root = '20151225_003442_0c53_visual_crop.tif'
        #
        ##im_root, category = '054593918020_01_assembly_3_5_LondonCityAir.tif', 'airplane'
        #im_root, category = '054956943030_01_assembly_17_02_BaghdadAir.tif', 'airplane'
        # im_root = '20151225_003442_0c53_visual_crop.tif'  # shapefile gives negative pixel values!
        #im_root, category = 'WV03_03102015_R1C1_panama_boats.tif', 'boat'

        #im_root = '53.tif'
        #category= 'Small_Car'
        ##########

        # Plot random file
        log_file = os.path.join(args.out_dir, ''  # args.prediction_dir_part
                                + '_' + im_root.split('.')[0]
                                + '_ground_truth_im_' + category
                                + '_iou_thresh_' + \
                                str(args.iou_thresh).replace('.', 'p')
                                + '_' + str(args.detect_thresh_plot).replace('.', 'p') + '.log')
        plot_file = os.path.join(args.out_dir, ''  # args.prediction_dir_part
                                 + '_' + im_root.split('.')[0]
                                 + '_ground_truth_im_' + category
                                 + '_iou_thresh_' + \
                                 str(args.iou_thresh).replace('.', 'p')
                                 + '_' + str(args.detect_thresh_plot).replace('.', 'p') + '.png')
        print("im_root:", im_root)
        print("category:", category)
        print("detect_thresh_plot:", args.detect_thresh_plot)
        print("run prop test plot_file:", plot_file)
        print("run prop test log_file:", log_file)
        # try:
        log, vals = eval_f1(gdf_truth, df_prop_refine, im_root,
                            category,
                            detect_thresh=args.detect_thresh_plot,
                            iou_thresh=args.iou_thresh,
                            nms_overlap=args.nms_overlap,
                            plot_file=plot_file,
                            log_file=log_file,
                            colorf1=args.colorf1,
                            line_thickness=args.line_thickness,
                            figsize=args.figsize,
                            out_ext=args.out_ext,
                            verbose=True,  # args.verbose,
                            show_intermediate_plots=False)
        # except:
        #    print ("test failed")

        # return

    # return

        ##############
    #    # Get precision-recall curve for single image, and category
    #    #if args.run_single_eval == 1:
    #    print ("\n\nRunning precision recall curve for single image and category..."
    #    category = categories[np.random.randint(len(categories))]
    #    im_root = im_roots[np.random.randint(len(im_roots))]
    #    #category = 'boat' #categories[np.random.randint(len(categories))]
    #    #im_root = 'AOI1.tif' #im_roots[np.random.randint(len(im_roots))]
    #
    #    im_root = '18.tif' #'47.tif'
    #    category= 'Small_Car'

        print("\n\nGet precision-recall curve for single image, and category")
        print("im_root:", im_root)
        print("category:", category)

        groupby = 'Image_Root'
        groupby_cat = 'Category'
        val_tot = []
        for i, detect_thresh in enumerate(args.detect_threshes):
            # get only subsample of df_prop
            # We need to run post_process.refine_and_plot_df(plot=False) to do nms
            # for each subsample of the predicted boxes at the given thresh
            print("detect_thresh:", detect_thresh)
            df_prop_refine = post_process.refine_df(df_prop,
                                                    groupby=groupby,
                                                    groupby_cat=groupby_cat,
                                                    nms_overlap_thresh=args.nms_overlap,
                                                    plot_thresh=detect_thresh,
                                                    verbose=False)
    #        df_prop_refine = post_process.refine_and_plot_df(df_prop,
    #                           groupby='Image_Path',
    #                           sliced=True,
    #                           outdir=args.out_dir,
    #                           plot_thresh=detect_thresh,
    #                           nms_overlap_thresh=args.nms_overlap,
    #                           plot=False,
    #                           verbose=False)

            log, vals = eval_f1(gdf_truth, df_prop_refine, im_root, category,
                                detect_thresh=detect_thresh,
                                iou_thresh=args.iou_thresh,
                                nms_overlap=args.nms_overlap,
                                plot_file='', log_file='',
                                verbose=False)
            # add threshold and category, and im_root
            vals.extend([detect_thresh, category, im_root])
            val_tot.append(vals)

        #val_tot = np.array(val_tot)
        df = pd.DataFrame(val_tot, columns=args.out_columns)
        precision, recall = df['Precision'].values, df['Recall'].values
        plot_file = os.path.join(args.out_dir, ''  # args.prediction_dir_part
                                 + '_' + im_root.split('.')[0]
                                 + '_iou_thresh_' + \
                                 str(args.iou_thresh).replace('.', 'p')
                                 + '_' + category + '_' + 'precision_recall.png')
        df_file = os.path.join(args.out_dir, ''  # args.prediction_dir_part
                               + '_' + im_root.split('.')[0]
                               + '_iou_thresh_' + \
                               str(args.iou_thresh).replace('.', 'p')
                               + '_' + category + '_' + 'precision_recall.csv')

        prec, rec, area = plot_precision_recall(precision, recall, outfile=plot_file,
                                                figsize=(10, 10), title=category)
        #ap = compute_ap(precision, recall)
        # print ("AP:", ap
        df.to_csv(df_file)
        print("im_root:", im_root)
        print("category:", category)

        # return


#    ##############
#    # Get performance dataframe for all images and categories
#    groupby='Image_Path'
#    groupby_cat = 'Category'
#    val_tot = []
#    count = 0
#    t3 = time.time()
#    for i,im_root in enumerate(im_roots):
#        print ("\n\n im_root:", im_root)
#        for j,category in enumerate(categories):
#            print ("\ncategory:", category)
#            for k,detect_thresh in enumerate(args.detect_threshes):
#                # get only subsample of df_prop
#                # We need to run post_process.refine_and_plot_df(plot=False) to do nms
#                # for each subsample of the predicted boxes at the given thresh
#                print ("detect_thresh:", detect_thresh)
#                df_prop_refine = post_process.refine_df(df_prop,
#                         groupby=groupby,
#                         groupby_cat=groupby_cat,
#                         nms_overlap_thresh=args.nms_overlap,
#                         plot_thresh=detect_thresh,
#                         verbose=False)
#
#                log, vals = eval_f1(gdf_truth, df_prop_refine, im_root, category,
#                                detect_thresh=detect_thresh,
#                                iou_thresh=args.iou_thresh,
#                                nms_overlap=args.nms_overlap,
#                                plot_file='', log_file='',
#                                verbose=False)
#                # add threshold and category, and im_root
#                vals.extend([detect_thresh, category, im_root])
#                val_tot.append(vals)
#                count += 1
#
#    t4 = time.time()
#    print ("Time to compute", count, "f1s:", t4-t3, "seconds")
#    #val_tot = np.array(val_tot)
#    df_perf = pd.DataFrame(val_tot, columns=args.out_columns)
#    precision, recall = df_perf['precision'].values, df_perf['recall'].values
#    plot_file = os.path.join(args.out_dir, 'precision_recall_tot.png')
#    df_file = os.path.join(args.out_dir,  'precision_recall_tot.csv')
#    plot_precision_recall(precision, recall, outfile=plot_file, figsize=(6,6))
#    df_perf.to_csv(df_file)

    ##############
    print("\n\n\n\n Get performance dataframe for all thresholds, images, and categories (if desired)")
    #im_roots = np.unique(gdf_truth['Image_Root'])
    print("im_roots:", im_roots)
    print("categories:", categories)

    groupby = 'Image_Root'
    groupby_cat = 'Category'
    val_tot = []
    log_tot = ''
    count = 0
    len_scores = 0
    df_scores_list = []

    ##############
    # OPTIONAL, DEFINE CATEGORIES
    #categories = ['Private_Boat']
    ##############

    im_roots = np.unique(gdf_truth['Image_Root'])
    # optional: rename im_roots to be a subset of images?
    # im_roots = ['5.tif', '18.tif', '20.tif', '24.tif', '31.tif', '38.tif',
    #            '40.tif', '41.tif', '43.tif', '46.tif', '46.tif',
    #            '53.tif', '69.tif']

    t3 = time.time()

    # computing post_process.refine_df() is slow, so precompute all of the
    #  refined dfs for each detect_thresh, so we don't keep recomputing below
    df_prop_refine_dict = {}
    print("\n\nPrecomputing df_prop_refine list...")
    for jtmp, detect_thresh in enumerate(args.detect_threshes):
        print("detect_thresh:", detect_thresh)
        # get only subsample of df_prop
        # We need to run post_process.refine_and_plot_df(plot=False) to do nms
        # for each subsample of the predicted boxes at the given thresh
        df_prop_refine = post_process.refine_df(df_prop,
                                                groupby=groupby,
                                                groupby_cat=groupby_cat,
                                                nms_overlap_thresh=args.nms_overlap,
                                                plot_thresh=detect_thresh,
                                                verbose=False)
        df_prop_refine_dict[detect_thresh] = df_prop_refine

    for i, category in enumerate(categories):
        print("\n\ncategory:", category)
        #cat_data = []
        score_list = []
        #f1_list = []
        for j, detect_thresh in enumerate(args.detect_threshes):
            print("detect_thresh:", detect_thresh)
            n_true_pos_tot, n_false_pos_tot, n_false_neg_tot = 0, 0, 0

            # retrieve df_prop_refine
            df_prop_refine = df_prop_refine_dict[detect_thresh]
            # get only subsample of df_prop
            # We need to run post_process.refine_and_plot_df(plot=False) to do nms
            # for each subsample of the predicted boxes at the given thresh
            # df_prop_refine = post_process.refine_df(df_prop,
            #         groupby=groupby,
            #         groupby_cat=groupby_cat,
            #         nms_overlap_thresh=args.nms_overlap,
            #         plot_thresh=detect_thresh,
            #         verbose=False)

            n_gt_sum = 0
            # iterate through im_roots
            for i, im_root in enumerate(im_roots):
                # skip certain image roots
                if im_root in args.skip_imroots_for_ap_calc:
                    print("Skipping im_root:", im_root)
                    continue

                else:
                    print("im_root:", im_root)

                log, vals = eval_f1(gdf_truth, df_prop_refine, im_root, category,
                                    detect_thresh=detect_thresh,
                                    iou_thresh=args.iou_thresh,
                                    nms_overlap=args.nms_overlap,
                                    plot_file='', log_file='',
                                    verbose=False)

                [f1, precision, recall, n_ground_truth_boxes,
                    n_prop_boxes, n_false_neg, n_false_pos, n_true_pos] = vals
                n_gt_sum += n_ground_truth_boxes

                # add threshold and category, and im_root for output array
                vals.extend([detect_thresh, category, im_root])
                val_tot.append(vals)
                count += 1

                # add positives and negatives if log is not empty
                if log != 'Empty':
                    log_tot += log
                    n_true_pos_tot += n_true_pos
                    n_false_pos_tot += n_false_pos
                    n_false_neg_tot += n_false_neg
                    #vals_list.append([detect_thresh, im_root, category] + vals)

            # compute f1, precision, recall for all images at the given threshold
            f1, precision, recall = compute_f1_precision_recall(n_true_pos_tot,
                                                                n_false_pos_tot,
                                                                n_false_neg_tot)
            print("\n\ndetect_thresh:", detect_thresh)
            print("n_true_pos_tot:", n_true_pos_tot)
            print("n_false_pos_tot:", n_false_pos_tot)
            print("n_false_neg_tot:", n_false_neg_tot)
            print("precision:", precision)
            print("recall:", recall)

            # if (precision > 0) and (recall > 0):
            if 2 > 1:
                score_list.append([n_gt_sum, detect_thresh,
                                   f1, precision, recall])
                #f1_list.append([n_ground_truth_boxes, f1])

        score_list = np.array(score_list)

        # skip if score_list is empty?
        if len(score_list) == 0:
            score_list = [0, detect_thresh, 0, 0, 0]
            f1, precision, recall = [0], [0], [0]
            # continue

        else:
            f1, precision, recall = score_list[:, 2],  score_list[:, 3], \
                score_list[:, 4]

        # get weighted mean f1
        #f1_arr = np.array(f1_list)
        #nboxes_tmp, f1s_tmp = f1_arr[:,0], f1_arr[:,1]
        #(mean_f1, std_f1, var_f1) = weighted_avg_and_std(f1s_tmp, nboxes_tmp)
        # print ("Weighted Mean F1:", mean_f1
        # print ("Weighted STD F1:", std_f1

        plot_file = os.path.join(args.out_dir,
                                 'iou_thresh_' +
                                 str(args.iou_thresh).replace('.', 'p')
                                 + '_' + category
                                 + '_precision_recall_tot.png')
        prec, rec, a = plot_precision_recall(precision, recall, outfile=plot_file,
                                             title=category, dpi=300)
        #ap = compute_ap(precision, recall)
        #print ("AP:", ap)

        # save dataframe
        df_scores = pd.DataFrame(score_list, columns=['N_Ground_Truth',
                                                      'Detect_Thresh',
                                                      'F1',
                                                      'Precision',
                                                      'Recall'])
        df_scores['Category'] = category
        df_scores['IOU_Thresh'] = args.iou_thresh
        df_score_fname = os.path.join(args.out_dir,
                                      'iou_thresh_' +
                                      str(args.iou_thresh).replace('.', 'p')
                                      + '_' + category
                                      + '_precision_recall_df.csv')
        df_scores.to_csv(df_score_fname)
        df_scores_list.append(df_scores)

        # create or append to total dataframe
        if len_scores == 0:
            df_scores_tot = df_scores
            len_scores += len(df_scores)

        else:
            df_scores_tot = df_scores_tot.append(df_scores, ignore_index=True)

    if len(df_scores_list) > 0:
        df_scores_tot = pd.concat(df_scores_list, ignore_index=True)
    else:
        df_scores_tot = pd.DataFrame([[1, 0, 0, 0, 0, category, 0.5]],
                                     columns=['N_Ground_Truth',
                                              'Detect_Thresh',
                                              'F1',
                                              'Precision',
                                              'Recall',
                                              'Category',
                                              'IOU_Thresh'])

    t4 = time.time()
    print("Time to compute", count, "f1s:", t4-t3, "seconds")
    df_perf = pd.DataFrame(val_tot, columns=args.out_columns)
    df_file = os.path.join(args.out_dir, 'iou_thresh_' +
                           str(args.iou_thresh).replace('.', 'p') + '_precision_recall_tot.csv')
    df_perf.to_csv(df_file)
    df_plot_file = os.path.join(
        args.out_dir,  '_map' + '_' + str(args.iou_thresh).replace('.', 'p') + '.png')
    plot_precision_recall_multi(df_scores_tot, outfile=df_plot_file,
                                figsize=(6, 6),
                                title='', dpi=300)

    print("Save total scores file")
    df_scores_file = os.path.join(
        args.out_dir, 'iou_thresh_' + str(args.iou_thresh).replace('.', 'p') + '_scores_tot.csv')
    df_scores_tot.to_csv(df_scores_file)

    if len(args.extra_csvs_for_tot_ap_plot) > 0:
        df_scores_file2 = os.path.join(args.out_dir, 'scores_tot_append.csv')
        df_plot_file = os.path.join(
            args.out_dir,  'precision_recall_append.png')
        # append dataframes
        df_scores_tot_new = df_scores_tot
        for dloc in args.extra_csvs_for_tot_ap_plot:
            df_tmp = pd.read_csv(dloc)
            df_scores_tot_new = df_scores_tot_new.append(df_tmp)
        # save datafrme and plot
        df_scores_tot_new.to_csv(df_scores_file2)
        plot_precision_recall_multi(df_scores_tot_new, outfile=df_plot_file,
                                    figsize=(6, 6),
                                    title='', dpi=300)

    return


###############################################################################
def main():

    # Construct argument parser
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--truth_dir', type=str, default='/validation/all',
                        help="Location of qgis ground truth labels"
                        + "os.path.join(yolt_dir, 'test_images/validation/all/")
    parser.add_argument('--prediction_dir_part', type=str, default='yolt_3class_2018_03_06_16-55-33',
                        help="Location of predictions (within results dir")
    parser.add_argument('--out_dir', type=str, default='',
                        help="Location of output, if null, make a "
                        + "subdirectory in prediction_dir named '_eval'")
    parser.add_argument('--skip_imroots_for_ap_calc', type=str, default='',
                        help="comma separated value of imroots to skip for "
                        + "average precision calculation")
    parser.add_argument('--extra_csvs_for_tot_ap_plot_part', type=str, default='',
                        help="comma separated values of locations of score "
                        + "csvs for plot_precisin_recal_multi()")

    parser.add_argument('--detect_threshes_str', type=str, default='',
                        help="Detection threshold, set to null if using "
                        + "the range defined below")
    parser.add_argument('--detect_thresh_min', type=float, default=0.05,
                        help="Min detection threshold")
    parser.add_argument('--detect_thresh_max', type=float, default=0.95,
                        help="Max detection threshold")
    parser.add_argument('--detect_thresh_delta', type=float, default=0.05,
                        help="Detection threshold range delta")
    parser.add_argument('--detect_thresh_plot', type=float, default=0.1,
                        help="Detection threshold")

    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help="Minimum IoU for detection")
    parser.add_argument('--nms_overlap', type=float, default=0.5,
                        help="Overlap for non-max-suppression")

    parser.add_argument('--run_prop_test', type=int, default=1,
                        help="Switch to run prop test")
    parser.add_argument('--out_ext', type=str, default='.tif',
                        help="Extension for output plots")
    parser.add_argument('--line_thickness', type=int, default=2,
                        help="Plot line thickness")
    parser.add_argument('--verbose_switch', type=int, default=0,
                        help="Switch to use verbose")

    parser.add_argument('--test_truth_pkl', type=str, default='',
                        help="test gdf truth pkl, set to '' to skip. "
                        + " See xview/ave_xview_val_prcoessing.ipynb")

    args = parser.parse_args()
    t0 = time.time()

    # infer a few values
    args.simrdwn_dir = os.path.dirname(path_simrdwn_core)
    args.prediction_topdir = os.path.join(args.simrdwn_dir, 'results')
    args.prediction_dir = os.path.join(args.prediction_topdir,
                                       args.prediction_dir_part)
    # split into lists
    args.skip_imroots_for_ap_calc = args.skip_imroots_for_ap_calc.split(',')
    extra_csvs_for_tot_ap_plot_part = args.extra_csvs_for_tot_ap_plot_part.split(
        ',')
    if len(args.extra_csvs_for_tot_ap_plot_part) > 0:
        args.extra_csvs_for_tot_ap_plot = [os.path.join(args.prediction_topdir, dtmp)
                                           for dtmp in extra_csvs_for_tot_ap_plot_part]
    else:
        args.extra_csvs_for_tot_ap_plot = []

    # set output directory
    if len(args.out_dir) == 0:
        args.out_dir = os.path.join(args.prediction_dir, '_eval')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print("\nOutput directory:", args.out_dir)

    # args.categories = args.categories_str.split(',')
    if len(args.detect_threshes_str) > 0:
        args.detect_threshes = np.array(
            args.detect_threshes_str.split(',')).astype(float)
    else:
        args.detect_threshes = np.arange(args.detect_thresh_min,
                                         args.detect_thresh_max,
                                         args.detect_thresh_delta)

    args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                           '.jpg', '.JPEG', '.jpeg']

    args.out_columns = ['F1', 'Precision', 'Recall', 'n_ground_truth_boxes',
                        'n_prop_boxes', 'n_false_neg', 'n_false_pos', 'n_true_pos',
                        'Threshold', 'Category', 'im_root']
    args.verbose = bool(args.verbose_switch)

    # plot settings
    args.colorf1 = (0, 255, 255)
    args.figsize = (8, 8)

    # set yolt training box size for cars
    car_size = 3      # meters
    GSD = 0.3          # meters
    args.yolt_car_box_size = np.rint(car_size/GSD)
    #print ("yolt_box_size:", args.yolt_box_size)

    # Planet images sometimes have the wrong crs, so might need to convert them
    # see parse_shapefile.transform_crs

    # execute
    execute(args)

    # just load in file to make plots
    df_scores_file = os.path.join(
        args.out_dir, 'iou_thresh_' + str(args.iou_thresh).replace('.', 'p') + '_scores_tot.csv')
    df_scores_tot = pd.read_csv(df_scores_file)
    df_plot_file = os.path.join(
        args.out_dir,  '_map' + '_'
        + str(args.iou_thresh).replace('.', 'p') + '.png')
    plot_precision_recall_multi(df_scores_tot, outfile=df_plot_file,
                                figsize=(6, 6),
                                title='xview', dpi=300)

    print("\nOutput directory:", args.out_dir)
    print("\nTotal time to run:", time.time() - t0, "seconds")

    return


###############################################################################
if __name__ == "__main__":
    main()
