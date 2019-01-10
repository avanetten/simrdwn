#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:05:43 2018

@author: avanetten
"""

from __future__ import print_function

from osgeo import ogr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#import random
import pickle
import math
import time
import csv
import cv2
import sys
import os

path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
# import slice_im, convert, post_process scripts
sys.path.append(path_simrdwn_core)


###############################################################################
def get_global_coords(row, 
                      edge_buffer_valid=0,
                      max_edge_aspect_ratio=4,
                      valid_box_rescale_frac=1.0,
                      rotate_boxes=False,
                      ):
    '''Get global coords of bounding box prediction from dataframe row
            #columns:Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', 
            #                u'Xmax', u'Ymax', u'Category', 
            #                u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', 
            #                u'Upper', u'Left', u'Height', u'Width', u'Pad', 
            #                u'Image_Path']
    '''

    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height']
    pad = row['Pad']
    
    # skip if near edge (set edge_buffer_valid < 0 to skip)
    if edge_buffer_valid > 0:
        if ((float(xmin0) < edge_buffer_valid) or 
            (float(xmax0) > (sliceWidth - edge_buffer_valid)) or                      
            (float(ymin0) < edge_buffer_valid) or 
            (float(ymax0) > (sliceHeight - edge_buffer_valid)) ):
            #print ("Too close to edge, skipping", row, "...")
            return [], []

    # skip if near edge and high aspect ratio (set edge_buffer_valid < 0 to skip)
    if edge_buffer_valid > 0:
        if ((float(xmin0) < edge_buffer_valid) or 
                (float(xmax0) > (sliceWidth - edge_buffer_valid)) or                      
                (float(ymin0) < edge_buffer_valid) or 
                (float(ymax0) > (sliceHeight - edge_buffer_valid)) ):
            # compute aspect ratio
            dx = xmax0 - xmin0
            dy = ymax0 - ymin0
            if (1.*dx/dy > max_edge_aspect_ratio) or (1.*dy/dx > max_edge_aspect_ratio):
                #print ("Too close to edge, and high aspect ratio, skipping", row, "...")
                return [], []
    
    
    # set min, max x and y for each box, shifted for appropriate
    #   padding                
    xmin = max(0, int(round(float(xmin0)))+left - pad) 
    xmax = min(vis_w - 1, int(round(float(xmax0)))+left - pad)
    ymin = max(0, int(round(float(ymin0)))+upper - pad)
    ymax = min(vis_h - 1, int(round(float(ymax0)))+upper - pad)
    
    # rescale output box size if desired, might want to do this
    #    if the training boxes were the wrong size
    if valid_box_rescale_frac != 1.0:
        dl = valid_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    # rotate boxes, if desird
    if rotate_boxes:   
        # import vis            
        vis = cv2.imread(row['Image_Path'], 1)  # color
        #vis_h,vis_w = vis.shape[:2]
        gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        #lines = cv2.HoughLines(edges,1,np.pi/180,50)
        coords = rotate_box(xmin, xmax, ymin, ymax, canny_edges)  

    # set bounds, coords
    bounds = [xmin, xmax, ymin, ymax]
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    
    # check that nothing is negative
    if np.min(bounds) < 0:
        print ("part of bounds < 0:", bounds)
        print (" row:", row)
        return
    if (xmax > vis_w) or (ymax > vis_h):
        print ("part of bounds > image size:", bounds)
        print (" row:", row)
        return
    
    return bounds, coords        
      
###############################################################################
def augment_df(df, 
               valid_testims_dir_tot='',
               slice_sizes=[416],
               valid_slice_sep='__',
               edge_buffer_valid=0,
               max_edge_aspect_ratio=4,
               valid_box_rescale_frac=1.0,
               rotate_boxes=False,
               verbose=False
               ):
    '''Add columns to dataframe, assume input columns are:
        ['Loc_Tmp', 'Prob','Xmin', 'Ymin', 'Xmax', 'Ymax', 'Category']
    output columns:
        # df.columns:
        # Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category',
        # u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', u'Upper', u'Left',
        # u'Height', u'Width', u'Pad', u'Image_Path'],
        # dtype='object')
    '''
    
    
    extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG', 
                           '.jpg', '.JPEG', '.jpeg'] 
    t0  = time.time()
    print ("Augmenting dataframe of initial length:", len(df), "...")
    # extract image root
    df['Image_Root_Plus_XY'] = [f.split('/')[-1] for f in df['Loc_Tmp']]

    #print "df:", df
    # parse out image root and location
    im_roots, im_locs = [], []
    for j,f in enumerate(df['Image_Root_Plus_XY'].values):
        
        if (j % 10000) == 0:
            print (j)
        
        ext = f.split('.')[-1]
        # get im_root, (if not slicing ignore '|')
        if slice_sizes[0] > 0:
            im_root_tmp = f.split(valid_slice_sep)[0]
            xy_tmp = f.split(valid_slice_sep)[-1]
        else:
            im_root_tmp, xy_tmp = f, '0_0_0_0_0_0_0'
        if im_root_tmp == xy_tmp:
            xy_tmp = '0_0_0_0_0_0_0'
        im_locs.append(xy_tmp)   
        
        if not '.' in im_root_tmp:
            im_roots.append(im_root_tmp + '.' + ext)
        else:
            im_roots.append(im_root_tmp)
    
    if verbose:
        print ("loc_tmp[:3]", df['Loc_Tmp'].values[:3])
        print ("im_roots[:3]", im_roots[:3])
        print ("im_locs[:3]", im_locs[:3])

    df['Image_Root'] = im_roots
    df['Slice_XY'] = im_locs
    # get positions
    df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
    df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
    df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
    df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
    df['Pad'] = [float(sl.split('_')[4].split('.')[0]) for sl in df['Slice_XY'].values]
    df['Im_Width'] = [float(sl.split('_')[5].split('.')[0]) for sl in df['Slice_XY'].values]
    df['Im_Height'] = [float(sl.split('_')[6].split('.')[0]) for sl in df['Slice_XY'].values]
    
    print ("  set image path, make sure the image exists...")
    im_paths_list = []
    im_roots_update = []
    for ftmp in df['Image_Root'].values:
        # get image path
        im_path = os.path.join(valid_testims_dir_tot, ftmp.strip())
        if os.path.exists(im_path):
            im_roots_update.append(os.path.basename(im_path))
            im_paths_list.append(im_path)
        # if this path doesn't exist, see if other extensions might work
        else:
            found = False
            for ext in extension_list:
                im_path_tmp = im_path.split('.')[0] + ext
                if os.path.exists(im_path_tmp):
                    im_roots_update.append(os.path.basename(im_path_tmp))
                    im_paths_list.append(im_path_tmp)
                    found = True
                    break
            if not found:
                print ("im_path not found with valid extensions:", im_path)
                print ("   im_path_tmp:", im_path_tmp)
    # update columns
    df['Image_Path'] = im_paths_list
    df['Image_Root'] = im_roots_update

    #df['Image_Path'] = [os.path.join(valid_testims_dir_tot, f.strip()) for f
    #                    in df['Image_Root'].values]

    print ("  add in global location of each row")
    # if slicing, get global location from filename
    if slice_sizes[0] > 0:
        x0l, x1l, y0l, y1l = [], [], [], []
        bad_idxs = []
        for index, row in df.iterrows():
            #bounds, coords = get_global_coords(args, row)
            bounds, coords = get_global_coords(row, 
                                               edge_buffer_valid=edge_buffer_valid,
                                               max_edge_aspect_ratio=max_edge_aspect_ratio,
                                               valid_box_rescale_frac=valid_box_rescale_frac,
                                               rotate_boxes=rotate_boxes)
            if len(bounds) == 0 and len(coords) == 0:
                bad_idxs.append(index)
                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
            else:
                [xmin, xmax, ymin, ymax] = bounds
            x0l.append(xmin)
            x1l.append(xmax)
            y0l.append(ymin)
            y1l.append(ymax)
        df['Xmin_Glob'] = x0l
        df['Xmax_Glob'] = x1l
        df['Ymin_Glob'] = y0l
        df['Ymax_Glob'] = y1l
    # if not slicing, global coords are equivalent to local coords
    else:
        df['Xmin_Glob'] = df['Xmin'].values
        df['Xmax_Glob'] = df['Xmax'].values
        df['Ymin_Glob'] = df['Ymin'].values
        df['Ymax_Glob'] = df['Ymax'].values
        bad_idxs = []
        
    
    # remove bad_idxs
    if len(bad_idxs) > 0:
        print ("removing bad idxs:", bad_idxs)
        df = df.drop(df.index[bad_idxs])  
            
    print ("Time to augment dataframe of length:", len(df), "=",
           time.time() - t0, "seconds")    
    return df
    

###############################################################################
def post_process_yolt_valid_create_df(yolt_valid_classes_files, log_file, 
               valid_testims_dir_tot='',
               slice_sizes=[416],
               valid_slice_sep='__',
               edge_buffer_valid=0,
               max_edge_aspect_ratio=4,
               valid_box_rescale_frac=1.0,
               rotate_boxes=False):
    '''take output files and create df
    # df.columns:
        # Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category',
        # u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', u'Upper', u'Left',
        # u'Height', u'Width', u'Pad', u'Image_Path'],
        # dtype='object')
        
    # test
    #args.yolt_valid_classes_files = ['/cosmiq/yolt2/results/valid_yolt2_explore1_cfg=ave_19x19_2017_04_25_22-47-05/building.txt']
    '''
 
    # parse out files, create df
    df_tot = []
    
    #str0 = '"args.yolt_valid_classes_files: ' + str(args.valid_results) + '\n"'
    
    for i,vfile in enumerate(yolt_valid_classes_files):

        valid_base_string = '"valid_file: ' + str(vfile) + '\n"'
        print (valid_base_string[1:-2])
        os.system('echo ' + valid_base_string + ' >> ' + log_file)
        
        cat = vfile.split('/')[-1].split('.')[0]
        # load into dataframe
        df = pd.read_csv(vfile, sep=' ', names=['Loc_Tmp', 'Prob', 
                                                       'Xmin', 'Ymin', 'Xmax',
                                                       'Ymax'])
        # set category
        df['Category'] = len(df) * [cat]
        
        # augment
        df = augment_df(df, 
               valid_testims_dir_tot=valid_testims_dir_tot,
               slice_sizes=slice_sizes,
               valid_slice_sep=valid_slice_sep,
               edge_buffer_valid=edge_buffer_valid,
               max_edge_aspect_ratio=max_edge_aspect_ratio,
               valid_box_rescale_frac=valid_box_rescale_frac,
               rotate_boxes=rotate_boxes)
        
#        # extract image root
#        df['Image_Root_Plus_XY'] = [f.split('/')[-1] for f in df['Loc_Tmp']]
#
#        #print "df:", df
#        # parse out image root and location
#        im_roots, im_locs = [], []
#        for j,f in enumerate(df['Image_Root_Plus_XY'].values):
#            ext = f.split('.')[-1]
#            # get im_root, (if not slicing ignore '|')
#            if args.slice_sizes[0] > 0:
#                im_root_tmp = f.split(args.valid_slice_sep)[0]
#                xy_tmp = f.split(args.valid_slice_sep)[-1]
#            else:
#                im_root_tmp, xy_tmp = f, '0_0_0_0_0_0_0'
#
#            if im_root_tmp == xy_tmp:
#                #xy_tmp = '0_0_0_0_0'
#                xy_tmp = '0_0_0_0_0_0_0'
#            im_locs.append(xy_tmp)   
#            if not '.' in im_root_tmp:
#                im_roots.append(im_root_tmp + '.' + ext)
#            else:
#                im_roots.append(im_root_tmp)
#
#        df['Image_Root'] = im_roots
#        df['Slice_XY'] = im_locs
#        # get positions
#        df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
#        df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
#        df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
#        df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
#        df['Pad'] = [float(sl.split('_')[4].split('.')[0]) for sl in df['Slice_XY'].values]
#        df['Im_Width'] = [float(sl.split('_')[5].split('.')[0]) for sl in df['Slice_XY'].values]
#        df['Im_Height'] = [float(sl.split('_')[6].split('.')[0]) for sl in df['Slice_XY'].values]
#        
#        # set image path
#        df['Image_Path'] = [os.path.join(args.valid_testims_dir_tot, f) for f
#                            in df['Image_Root'].values]
#
#        # add in global location of each row
#        x0l, x1l, y0l, y1l = [], [], [], []
#        bad_idxs = []
#        for index, row in df.iterrows():
#            bounds, coords = get_global_coords(args, row)
#            if len(bounds) == 0 and len(coords) == 0:
#                bad_idxs.append(index)
#                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
#            else:
#                [xmin, xmax, ymin, ymax] = bounds
#            x0l.append(xmin)
#            x1l.append(xmax)
#            y0l.append(ymin)
#            y1l.append(ymax)
#        df['Xmin_Glob'] = x0l
#        df['Xmax_Glob'] = x1l
#        df['Ymin_Glob'] = y0l
#        df['Ymax_Glob'] = y1l
#        
#        # remove bad_idxs
#        if len(bad_idxs) > 0:
#            print ("removing bad idxs:", bad_idxs)
#            df = df.drop(df.index[bad_idxs])        
            
        # append to total df
        if i == 0:
            df_tot = df
        else:
            df_tot = df_tot.append(df, ignore_index=True)
        

       
    return df_tot
          

###############################################################################
def post_proccess_make_plots(args, df, verbose=False):
    
    '''Original version for creating plots with YOLT'''
    
    # create dictionary of plot_thresh lists
    thresh_poly_dic = {}
    for plot_thresh_tmp in args.plot_thresh:        
        # initilize to empty
        thresh_poly_dic[np.around(plot_thresh_tmp, decimals=2)] = []
    if verbose:
        print ("thresh_poly_dic:", thresh_poly_dic)
        
    # group 
    group = df.groupby('Image_Path')
    for itmp,g in enumerate(group):
        im_path = g[0]
        im_root_noext = im_path.split('/')[-1].split('.')[0]
        data_all_classes = g[1]
        
            
        print ("\n", itmp, "/", len(group), "Analyzing Image:", im_path)

        # plot validation outputs    
        #for plot_thresh_tmp in plot_thresh:
        for plot_thresh_tmp in thresh_poly_dic.keys():
            if args.valid_make_pngs.upper() != 'FALSE':
                figname_val = os.path.join(args.results_dir, im_root_noext \
                            + '_valid_thresh=' + str(plot_thresh_tmp) + '.png')
                            #+ '_valid_thresh=' + str(plot_thresh_tmp) + '.jpeg')

            else:
                figname_val = ''
            pkl_val = os.path.join(args.results_dir, im_root_noext \
                        + '_boxes_thresh=' + str(plot_thresh_tmp) + '.pkl')

            ############
            # make plot
            out_list = plot_vals_deprecated(args, im_path, data_all_classes, pkl_val, 
                                 figname_val, plot_thresh_tmp)                                  
            ############
            
            # convert to wkt format for buildings
            if (len(args.object_labels)==1) and \
                                    (args.object_labels[0]=='building'):
                for i,row in enumerate(out_list):
                    [xmin, ymin, xmax, ymax, coords, filename, textfile, prob, 
                         color0, color1, color2, labeltmp, labeltmp_full] = row
                    im_name0 = filename.split('/')[-1].split('.')[0]
                    if args.multi_band_delim in im_root_noext:
                        im_name1 = im_name0[15:].split('#')[0]
                    else:
                       # im_name1 = im_name0[6:]
                        im_name1 = im_name0[15:]
                    wkt_row = building_polys_to_csv(im_name1, str(i), 
                                                    coords,
                                                    #[xmin,ymin,xmax,ymax],
                                                    conf=prob)                      
                    thresh_poly_dic[plot_thresh_tmp].append(wkt_row)
            
            # save out_list
            if len(out_list) > 0:
                out_list_f = os.path.join(args.results_dir, im_root_noext \
                            + '_plot_vals_deprecated_outlist.csv')
                with open(out_list_f, "wb") as f:
                    writer = csv.writer(f)
                    writer.writerows(out_list)
                

    # save thresh_poly_dic
    if len(args.object_labels) == 1 and args.object_labels[0] == 'building':
        for plot_thresh_tmp in thresh_poly_dic.keys():
            csv_name = os.path.join(args.results_dir, 'predictions_' \
                                        + str(plot_thresh_tmp) + '.csv')
            print ("Saving wkt buildings to file:", csv_name, "...")
            # save to csv
            #print "thresh_poly_dic:", thresh_poly_dic
            #print "thresh_poly_dic[plot_thresh_tmp]:", thresh_poly_dic[plot_thresh_tmp]
            with open(csv_name, 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for j,line in enumerate(thresh_poly_dic[plot_thresh_tmp]):
                    print (j, line)
                    writer.writerow(line)            

    return



###############################################################################
def plot_vals_deprecated(args, im_path, data_all_classes, outpkl, figname, plot_thresh,
              verbose=False):
    '''
    Iterate through files and plot on valid_im
    see modular_sliding_window.py
    each line of input data is: [file, xmin, ymin, xmax, ymax]
    /cosmiq/yolt2/test_images/scs_0.2_split/slice_scs_0.2_0_1024.jpg 0.003840 0.553413 1.505103 1.309415 2.035146
    outlist has format: [xmin, ymin, xmax, ymax, filename, file_v, prob, 
                         color0, color1, color2, labeltmp, labeltmp_full] = b
    '''
    

    ##########################################
    # import slice_im, convert, post_process scripts
    sys.path.append(os.path.join(args.yolt_dir, 'scripts'))
    import yolt_post_process

    
    ####################################f######
    # APPEARANCE SETTINGS
    # COLORMAP
    if len(args.object_labels) > 1:
        # boat/plane colormap    
        colormap = [(255, 0,   0),
                    (0,   255, 0),
                    (0,   0,   255),
                    (255, 255, 0),
                    (0,   255, 255),
                    (255, 0,   255),
                    (0,   0,   255),
                    (127, 255, 212),
                    (72,  61,  139),
                    (255, 127, 80),
                    (199, 21,  133),
                    (255, 140, 0),
                    (0, 165, 255)] 
    else:
        # airport colormap
        colormap = [(0, 165, 255), (0, 165, 255)]
            
    # TEXT FONT
    # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
    font = cv2.FONT_HERSHEY_TRIPLEX  #FONT_HERSHEY_SIMPLEX 
    font_size = 0.4
    font_width = 1
    text_offset = [3, 10]          

    # add border
    # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
    # top, bottom, left, right - border width in number of pixels in corresponding directions
    border = (40, 0, 0, 200) 
    border_color = (255,255,255)
    label_font_width = 1 
    ##########################################
    
    # import vis            
    vis = cv2.imread(im_path, 1)  # color
    vis_h,vis_w = vis.shape[:2]
    #fig, ax = plt.subplots(figsize=args.figsize)
    img_mpl = vis #cv2.cvtColor(vis, cv2.COLOR_BGR2RGB
    
    if args.rotate_boxes:
        gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        #lines = cv2.HoughLines(edges,1,np.pi/180,50)

    out_list = []
    boxes = []
    boxes_nms = []
    legend_dic = {}
    
    if verbose:
        print ("data_all_classes.columns:", data_all_classes.columns)
    
    # group by category
    group2 = data_all_classes.groupby('Category')
    for i,(category, plot_df) in enumerate(group2):
        print ("Plotting category:", category)
        label_int = args.object_labels.index(category)
        color = colormap[label_int]#
        print ("color:", color)
        label = str(label_int)
        label_str = args.object_labels[label_int] #label_root0.split('_')[-1]
        print ("label:", label, 'label_str:', label_str)
        legend_dic[label_int] = (label_str, color)
        
        for index, row in plot_df.iterrows():
            #columns:Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', 
            #                u'Xmax', u'Ymax', u'Category', 
            #                u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', 
            #                u'Upper', u'Left', u'Height', u'Width', u'Pad', 
            #                u'Image_Path']
            
            filename, prob0, xmin0, ymin0, xmax0, ymax0, category, image_root_pxy, \
                image_root, slice_xy, upper, left, sliceHeight, sliceWidth, \
                pad, im_width, im_height, image_path, xmin, xmax, ymin, ymax = row
            #filename, prob0, xmin0, ymin0, xmax0, ymax0, category, image_root_pxy, \
            #    image_root, slice_xy, upper, left, sliceHeight, sliceWidth, \
            #    pad, image_path = row            
            prob = float(prob0)
            if prob >= plot_thresh:
                                    
                # skip if near edge (set edge_buffer_valid < 0 to skip)
                if args.edge_buffer_valid > 0:
                    if ((float(xmin0) < args.edge_buffer_valid) or 
                        (float(xmax0) > (sliceWidth - args.edge_buffer_valid)) or                      
                        (float(ymin0) < args.edge_buffer_valid) or 
                        (float(ymax0) > (sliceHeight - args.edge_buffer_valid)) ):
                        print ("Too close to edge, skipping", row, "...")
                        continue
                
#                # below is accomplished when df is created
#                # set min, max x and y for each box, shifted for appropriate
#                #   padding                
#                xmin = max(0, int(round(float(xmin0)))+left - pad) 
#                xmax = min(vis_w, int(round(float(xmax0)))+left - pad)
#                ymin = max(0, int(round(float(ymin0)))+upper - pad)
#                ymax = min(vis_h, int(round(float(ymax0)))+upper - pad)
#                
#                # rescale output box size if desired, might want to do this
#                #    if the training boxes were the wrong size
#                if args.valid_box_rescale_frac != 1.0:
#                    dl = args.valid_box_rescale_frac
#                    xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
#                    dx = dl*(xmax - xmin) / 2
#                    dy = dl*(ymax - ymin) / 2
#                    x0 = xmid - dx
#                    x1 = xmid + dx
#                    y0 = ymid - dy
#                    y1 = ymid + dy
#                    xmin, xmax, ymin, ymax = x0, x1, y0, y1
#
                # set coords
                coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], 
                          [xmin, ymax]]
                if args.rotate_boxes:
                    coords = yolt_post_process.rotate_box(xmin, xmax, ymin, 
                                                          ymax, canny_edges)                    
    
                out_row = [xmin, ymin, xmax, ymax, coords, filename, category, prob, 
                           color[0], color[1], color[2], label, label_str]

                #out_list.append(out_row)
                boxes.append(out_row)
                # add to plot?
                # could add a function to scale thickness with prob
                if args.nms_overlap_thresh <= 0:
                    
                    if not args.rotate_boxes:
                        cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), 
                                      (color), args.plot_line_thickness)   
                        # plot text
                        if args.plot_names:
                            try:
                                cv2.putText(img_mpl, label, (int(xmin)
                                    +text_offset[0], int(ymin)+text_offset[1]), 
                                    font, font_size, color, font_width, 
                                    cv2.CV_AA)#, cv2.LINE_AA)
                            except:
                                cv2.putText(img_mpl, label, (int(xmin)
                                    +text_offset[0], int(ymin)+text_offset[1]), 
                                    font, font_size, color, font_width, 
                                    cv2.LINE_AA)
                                
                    else:
                        # plot rotated rect
                        coords1 = coords.reshape((-1,1,2))
                        cv2.polylines(img_mpl, [coords1], True, color, 
                                      thickness=args.plot_line_thickness)
                                
                        
    # apply non-max-suppresion on total 
    if args.nms_overlap_thresh > 0:
        boxes_nms, boxes_tot_nms, nms_idxs = non_max_suppression(boxes, 
                                                            args.nms_overlap_thresh)
        out_list = boxes_tot_nms
        # plot
        #for itmp,b in enumerate(boxes_nms):
        #    [xmin, ymin, xmax, ymax] = b
        #    color = out_list[itmp][-1]
        #    cv2. rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)
        for itmp,b in enumerate(boxes_tot_nms):
            [xmin, ymin, xmax, ymax, coords, filename, v, prob, color0, 
                             color1, color2, labeltmp, labeltmp_full] = b
            color = (int(color0), int(color1), int(color2))
            
            if not args.rotate_boxes:
                cv2.rectangle(img_mpl, (int(xmin), int(ymin)), (int(xmax), 
                                        int(ymax)), (color), 
                                        args.plot_line_thickness)
                if args.plot_names:
                    cv2.putText(img_mpl, labeltmp, (int(xmin)+text_offset[0], 
                                                    int(ymin)+text_offset[1]), 
                                                    font, font_size, color, 
                                                    font_width, 
                                                    cv2.CV_AA)#, cv2.LINE_AA)

            else:
                # plot rotated rect
                coords1 = coords.reshape((-1,1,2))
                cv2.polylines(img_mpl, [coords1], True, color, 
                              thickness=args.plot_line_thickness)                    
    else:
        out_list = boxes
        
    # add extra classifier pickle, if desired
    if args.extra_pkl:
        labeltmp = 'airport'
        extra_idx = len(colormap) - 1
        [out_list_ex, boxes_ex, boxes_nms_ex] \
                    = pickle.load(open(args.extra_pkl, 'rb'))
        for itmp,b in enumerate(out_list_ex):
            [xmin, ymin, xmax, ymax, filename, v, prob,color0,color1,color2] = b
            color_ex = colormap[extra_idx]
            cv2.rectangle(img_mpl, (int(xmin), int(ymin)), (int(xmax), 
                                    int(ymax)), (color_ex), 
                                    2*args.plot_line_thickness)
            if args.plot_names:
                cv2.putText(img_mpl, labeltmp, (int(xmin)+text_offset[0], 
                                                int(ymin)+text_offset[1]), 
                                                font, font_size, color_ex, 
                                                font_width,  
                                                cv2.CV_AA)#cv2.LINE_AA)

                legend_dic[extra_idx] = (labeltmp, color_ex)

    # add legend and border, if desired
    if args.valid_make_legend_and_title.upper() == 'TRUE':

        # add border
        # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
        # top, bottom, left, right - border width in number of pixels in 
        # corresponding directions
        img_mpl = cv2.copyMakeBorder(img_mpl, border[0], border[1], border[2], 
                                     border[3], 
                                     cv2.BORDER_CONSTANT,value=border_color)

        xpos = img_mpl.shape[1] - border[3] + 15
        ydiff = border[0]
        for itmp, k in enumerate(sorted(legend_dic.keys())):
            labelt, colort = legend_dic[k]                             
            text = '- ' + labelt #str(k) + ': ' + labelt
            ypos = border[0] + (2+itmp) * ydiff
            cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 
                        1.5*font_size, colort, label_font_width, 
                        cv2.CV_AA)#cv2.LINE_AA)
    
        # legend box
        cv2.rectangle(img_mpl, (xpos-5, 2*border[0]), (img_mpl.shape[1]-10, 
                      ypos+int(0.75*ydiff)), (0,0,0), label_font_width)   
                                          
        # title                                  
        title_pos = (border[0], int(border[0]*0.66))
        title = figname.split('/')[-1].split('_')[0] + ':  Plot Threshold = ' \
                        + str(plot_thresh) # + ': thresh=' + str(plot_thresh)
        cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), 
                    label_font_width,  
                    cv2.CV_AA)#cv2.LINE_AA)

    print ("Saving to files", outpkl, figname, "...")
 
    if len(outpkl) > 0:
        pickle.dump([out_list, boxes, boxes_nms], open(outpkl, 'wb'), protocol=2)
    
    if len(figname) > 0:   
        # save high resolution
        #plt.savefig(figname, dpi=args.dpi)  
        img_mpl_out = img_mpl #= cv2.cvtColor(img_mpl, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(figname, img_mpl_out)
        # compress?
        cv2.imwrite(figname, img_mpl_out,  [cv2.IMWRITE_PNG_COMPRESSION, args.valid_im_compression_level])

        if args.show_valid_plots:
            #plt.show()
            cmd = 'eog ' + figname + '&'
            os.system(cmd)
   
    return out_list
    
    
###############################################################################
def non_max_suppression(boxes, overlapThresh):
    '''
    Non max suppression (assume boxes = [[xmin, ymin, xmax, ymax, ...\
                             sometiems extra cols are: filename, v, prob, color]]
    # http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Malisiewicz et al.
    see modular_sliding_window.py, functions non_max_suppression, \
            non_max_supression_rot
    '''
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []
    
    boxes_tot = boxes#np.asarray(boxes)
    boxes = np.asarray([b[:4] for b in boxes])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    outboxes = boxes[pick].astype("int")
    #outboxes_tot = boxes_tot[pick]
    outboxes_tot = [boxes_tot[itmp] for itmp in pick]
    
    return outboxes, outboxes_tot, pick


###############################################################################
def refine_df(df, groupby='Loc_Tmp', 
                 groupby_cat='Category',
                 nms_overlap_thresh=0.5,  plot_thresh=0.5,
                 sliced=True, # not needed anymore...
                 verbose=True):
    '''Plot bounding boxes stored in dataframe
    SHOULD GROUP BY CATEGORY PRIOR TO NMS SO THAT AIRPLANES CAN BE INSIDE
    AIRPORTS'''
    
    print ("Running refine_df()...")
    t0 = time.time()

    # group by image, and plot
    group = df.groupby(groupby)
    count = 0
    #refine_dic = {}
    print_iter = 1
    df_idxs_tot = []
    for i,g in enumerate(group):
        
        img_loc_string = g[0]
        data_all_classes = g[1] 
        
        #image_root = data_all_classes['Image_Root'].values[0]
        if (i % print_iter) == 0 and verbose:
            print (i+1, "/", len(group), "Processing image:", img_loc_string)
            print ("  num boxes:", len(data_all_classes))
        
        #image = cv2.imread(img_loc_string, 1)
        #if verbose:
        #    print ("  image.shape:", image.shape)
        
        #boxes_im, scores_im, classes_im = [], [], []
        # groupby category as well so that detections can be overlapping of 
        # different categories (i.e.: a helicopter on a boat)
        group2 = data_all_classes.groupby(groupby_cat)
        for j,g2 in enumerate(group2):
            
            class_str = g2[0]
            data = g2[1]   
            df_idxs = data.index.values
            #classes_str = np.array(len(data) * [class_str])
            scores = data['Prob'].values

            if (i % print_iter) == 0 and verbose:                
                print ("    Category:", class_str)
                print ("    num boxes:", len(data))
                #print ("    scores:", scores)
    
            xmins = data['Xmin_Glob'].values
            ymins = data['Ymin_Glob'].values
            xmaxs = data['Xmax_Glob'].values
            ymaxs = data['Ymax_Glob'].values

#            if not sliced:  
#                xmins = data['Xmin'].values
#                ymins = data['Ymin'].values
#                xmaxs = data['Xmax'].values
#                ymaxs = data['Ymax'].values
#            else:
#                xmins = data['Xmin_Glob'].values
#                ymins = data['Ymin_Glob'].values
#                xmaxs = data['Xmax_Glob'].values
#                ymaxs = data['Ymax_Glob'].values
    
            ## set legend str? 
            #if len(label_map_dict.keys()) > 0:
            #    classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
            #    classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
            #else:
            #    classes_str = classes_int_str
            #    classes_legend_str = classes_str    
        
            # filter out low probs
            high_prob_idxs = np.where(scores >= plot_thresh)
            scores = scores[high_prob_idxs]
            #classes_str = classes_str[high_prob_idxs]
            xmins = xmins[high_prob_idxs]
            xmaxs = xmaxs[high_prob_idxs]
            ymins = ymins[high_prob_idxs]
            ymaxs = ymaxs[high_prob_idxs]
            df_idxs = df_idxs[high_prob_idxs]
    
            boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)
            
            if verbose:
                print ("len boxes:", len(boxes))
            
            ###########
            # NMS
            if nms_overlap_thresh > 0:
            
                ## try tf nms (always returns an empty list!)
                ## https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
                #boxes_tf = tf.convert_to_tensor(boxes, np.float32)
                #scores_tf = tf.convert_to_tensor(scores, np.float32)
                #nms_idxs = tf.image.non_max_suppression(boxes_tf, scores_tf, 
                #                                        max_output_size=1000,
                #                                        iou_threshold=0.5)
                #selected_boxes = tf.gather(boxes_tf, nms_idxs)
                #print ("  len boxes:", len(boxes))
                #print ("  nms idxs:", nms_idxs)
                #print ("  selected boxes:", selected_boxes)
        
                # Try nms with pyimagesearch algorightm
                # assume boxes = [[xmin, ymin, xmax, ymax, ...
                #   might want to split by class because we could have a car inside
                #   the bounding box of a plane, for example
                boxes_nms_input = np.stack((xmins, ymins, xmaxs, ymaxs), axis=1)
                _, _, good_idxs =  non_max_suppression(boxes_nms_input, 
                                                     overlapThresh=nms_overlap_thresh)
                if verbose:
                    print ("num boxes_all:", len(xmins))
                    print ("num good_idxs:", len(good_idxs))
                boxes = boxes[good_idxs]
                scores = scores[good_idxs]
                df_idxs = df_idxs[good_idxs]
                #classes = classes_str[good_idxs]
                
            df_idxs_tot.extend(df_idxs)
            count += len(df_idxs)
            
    
    #print ("len df_idxs_tot:", len(df_idxs_tot))
    df_idxs_tot_final = np.unique(df_idxs_tot)
    #print ("len df_idxs_tot unique:", len(df_idxs_tot))

    # create dataframe
    if verbose:
        print ("df idxs::", df.index)
        print ("df_idxs_tot_final:", df_idxs_tot_final)
    df_out = df.loc[df_idxs_tot_final]
    
    t1 = time.time()
    print ("Inintial length:", len(df), "Final length:", len(df_out))
    print ("Time to run refine_df():", t1-t0, "seconds")
    return df_out  #refine_dic
            
            
            
###############################################################################          
def plot_refined_df(df, groupby='Loc_Tmp', label_map_dict={}, 
                 outdir='', plot_thresh=0.5,
                 show_labels=True, alpha_scaling=True, plot_line_thickness=2, 
                 legend_root='00_colormap_legend.png',
                 plot=True, skip_empty=False, print_iter=1, n_plots=100000,
                 building_csv_file='',
                 shuffle_ims=False, verbose=True):
           
    '''Plot refined dataframe}'''

    print ("Running plot_refined_df...")
    t0 = time.time()
    # get colormap, if plotting
    outfile_legend = os.path.join(outdir, legend_root)
    colormap, color_dict = make_color_legend(outfile_legend, label_map_dict)
    
    # group by image, and plot
    if shuffle_ims:
        group = df.groupby(groupby, sort=False)
    else:
        group = df.groupby(groupby)
    #print_iter = 1
    for i,g in enumerate(group):
        
        # break if we already met the number of plots to create
        if (i >= n_plots) and (len(building_csv_file) == 0):
            break
        
        img_loc_string = g[0]
        
        #if '740351_3737289' not in img_loc_string:
        #    continue
        
        data_all_classes = g[1] 
        image = cv2.imread(img_loc_string, 1)
        
        #image_root = data_all_classes['Image_Root'].values[0]
        im_root = os.path.basename(img_loc_string)
        im_root_no_ext, ext = im_root.split('.')
        outfile = os.path.join(outdir, im_root_no_ext + '_thresh=' \
                               + str(plot_thresh) + '.' + ext)

        if (i % print_iter) == 0 and verbose:
            print (i+1, "/", len(group), "Processing image:", img_loc_string)
            print ("  num boxes:", len(data_all_classes))
        #if verbose:
            print ("  image.shape:", image.shape)
        

        xmins = data_all_classes['Xmin_Glob'].values
        ymins = data_all_classes['Ymin_Glob'].values
        xmaxs = data_all_classes['Xmax_Glob'].values
        ymaxs = data_all_classes['Ymax_Glob'].values
        classes = data_all_classes['Category']
        scores = data_all_classes['Prob']

        boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

        # make plots if we are below the max
        if i < n_plots:
            plot_rects(image, boxes, scores, classes=classes,
                  plot_thresh=plot_thresh, 
                  color_dict=color_dict, #colormap=colormap,
                  outfile=outfile,
                  show_labels=show_labels,
                  alpha_scaling=alpha_scaling,
                  plot_line_thickness=plot_line_thickness,
                  verbose=verbose)
        


    t1 = time.time()
    print ("Time to run plot_refined_df():", t1-t0, "seconds")
    return



###############################################################################
def refine_and_plot_df(df, groupby='Loc_Tmp', label_map_dict={}, 
                 sliced=True, groupby_cat='Category',
                 outdir='', plot_thresh=0.33, nms_overlap_thresh=0.5, 
                 show_labels=True, alpha_scaling=True, plot_line_thickness=2, 
                 out_cols = [u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', 
                             u'Ymax', u'Category', 'Image_Root'],
                 legend_root='00_colormap_legend.png',
                 plot=True, skip_empty=False,
                 extra_columns=[],
                 verbose=True):
    '''Plot bounding boxes stored in dataframe
    SHOULD GROUP BY CATEGORY PRIOR TO NMS SO THAT AIRPLANES CAN BE INSIDE
    AIRPORTS'''
    
    print ("Running refine_and_plot_df()...")
    t0 = time.time()

    # group by image, and plot
    group = df.groupby(groupby)
    count = 0
    #refine_dic = {}
    out_list=[]
    print_iter = 1
    for i,g in enumerate(group):
        
        img_loc_string = g[0]
        data_all_classes = g[1] 
        
        image_root = data_all_classes['Image_Root'].values[0]

        if (i % print_iter) == 0 and verbose:
            print (i+1, "/", len(group), "Processing image:", img_loc_string)
            print ("  num boxes:", len(data_all_classes))
        
        #image = cv2.imread(img_loc_string, 1)
        #if verbose:
        #    print ("  image.shape:", image.shape)
        
        boxes_im, scores_im, classes_im = [], [], []
        # groupby category as well so that detections can be overlapping of 
        # different categories (i.e.: a helicopter on a boat)
        group2 = data_all_classes.groupby(groupby_cat)
        for j,g2 in enumerate(group2):
            
            class_str = g2[0]
            data = g2[1]            
            #classes_str = np.array(len(data) * [class_str])
            scores = data['Prob'].values

            if (i % print_iter) == 0 and verbose:                
                print ("    Category:", class_str)
                print ("    num boxes:", len(data))
                #print ("    scores:", scores)
    
            if not sliced:  
                xmins = data['Xmin'].values
                ymins = data['Ymin'].values
                xmaxs = data['Xmax'].values
                ymaxs = data['Ymax'].values
            else:
                xmins = data['Xmin_Glob'].values
                ymins = data['Ymin_Glob'].values
                xmaxs = data['Xmax_Glob'].values
                ymaxs = data['Ymax_Glob'].values
    
            ## set legend str? 
            #if len(label_map_dict.keys()) > 0:
            #    classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
            #    classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
            #else:
            #    classes_str = classes_int_str
            #    classes_legend_str = classes_str    
        
            # filter out low probs
            high_prob_idxs = np.where(scores >= plot_thresh)
            scores = scores[high_prob_idxs]
            #classes_str = classes_str[high_prob_idxs]
            xmins = xmins[high_prob_idxs]
            xmaxs = xmaxs[high_prob_idxs]
            ymins = ymins[high_prob_idxs]
            ymaxs = ymaxs[high_prob_idxs]
    
            boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)
            
            if verbose:
                print ("len boxes:", len(boxes))
            
            ###########
            # NMS
            if nms_overlap_thresh > 0:
            
                ## try tf nms (always returns an empty list!)
                ## https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
                #boxes_tf = tf.convert_to_tensor(boxes, np.float32)
                #scores_tf = tf.convert_to_tensor(scores, np.float32)
                #nms_idxs = tf.image.non_max_suppression(boxes_tf, scores_tf, 
                #                                        max_output_size=1000,
                #                                        iou_threshold=0.5)
                #selected_boxes = tf.gather(boxes_tf, nms_idxs)
                #print ("  len boxes:", len(boxes))
                #print ("  nms idxs:", nms_idxs)
                #print ("  selected boxes:", selected_boxes)
        
                # Try nms with pyimagesearch algorightm
                # assume boxes = [[xmin, ymin, xmax, ymax, ...
                #   might want to split by class because we could have a car inside
                #   the bounding box of a plane, for example
                boxes_nms_input = np.stack((xmins, ymins, xmaxs, ymaxs), axis=1)
                _, _, good_idxs =  non_max_suppression(boxes_nms_input, 
                                                     overlapThresh=nms_overlap_thresh)
                if verbose:
                    print ("num boxes_all:", len(xmins))
                    print ("num good_idxs:", len(good_idxs))
                boxes = boxes[good_idxs]
                scores = scores[good_idxs]
                #classes = classes_str[good_idxs]
                
            # create output
            #refine_dic[img_loc_string] = [scores, boxes, classes]
            
            # add to output list
            for score, box in zip(scores, boxes):
                x0, y0, x1, y1 = box
                out_list.append([img_loc_string, score, x0, y0, x1, y1, class_str, image_root])
            
            classes_str = np.array(len(scores) * [class_str])
            #image_roots = np.array(len(scores) * [image_root])
            
            # add to image values
            classes_im.extend(classes_str)
            boxes_im.extend(boxes)
            scores_im.extend(scores)
                
            
        #############
        # Plot
        if plot:    

            # get colormap, if plotting
            outfile_legend = os.path.join(outdir, legend_root)
            colormap, color_dict = make_color_legend(outfile_legend, label_map_dict)

            image = cv2.imread(img_loc_string, 1)
            if verbose:
                print ("  image.shape:", image.shape)
            
            # load image
            #image = cv2.imread(img_loc_string, 1)
    
            # could try to plot with:
            #     https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
            #     using the draw_bouding_boxes on image_array() function
            #     but this is annoying, so instead use plot_rects() above 
            #dlist = [[z] for z in class_ints_str]
            #visualization_utils.draw_bounding_boxes_on_image_array(image, 
            #                                                       boxes,
            #                                                       display_str_list_list=dlist)

            #print ("scores_im:", scores_im)
            if skip_empty:
                z = np.where(scores_im >= plot_thresh)
                if len(z[0]) == 0:
                    print ("Empty image, skip plotting")
                    return

    
            # else, use custom function
            im_root = os.path.basename(img_loc_string)
            #outfile = os.path.join(outdir, im_root)
            im_root_no_ext, ext = im_root.split('.')
            outfile = os.path.join(outdir, im_root_no_ext + '_thresh=' \
                                   + str(plot_thresh) + '.' + ext)
            #print ("image_location:", img_loc_string)
            #print ("  image:", image)
            count += len(boxes_im)
            if verbose:
                print ("outfile:", outfile)
            
            plot_rects(image, boxes_im, scores_im, classes=classes_im,
                  plot_thresh=plot_thresh, 
                  color_dict=color_dict, #colormap=colormap,
                  outfile=outfile,
                  show_labels=show_labels,
                  alpha_scaling=alpha_scaling,
                  plot_line_thickness=plot_line_thickness,
                  skip_empty=skip_empty,
                  verbose=verbose)
            
        #t1 = time.time()
        #print ("Time to plot", count, "boxes:", t1 - t0, "seconds")
        
        
    # create dataframe
    df_out = pd.DataFrame(out_list, columns=out_cols)
    
    t1 = time.time()
    print ("Time to run refine_and_plot_df():", t1-t0, "seconds")
    return df_out  #refine_dic


###############################################################################
def refine_and_plot_df_v0(df, groupby='Loc_Tmp', label_map_dict={}, 
                 sliced=True, #slice_sizes=[0],
                 outdir='', plot_thresh=0.33, nms_overlap_thresh=0.5, 
                 show_labels=True, alpha_scaling=True, plot_line_thickness=2, 
                 plot=True,
                 verbose=True):
    '''Plot bounding boxes stored in dataframe
    SHOULD GROUP BY CATEGORY PRIOR TO NMS SO THAT AIRPLANES CAN BE INSIDE
    AIRPORTS'''
    
    print ("Running plot_df()...")
    t0 = time.time()
    # get colormap, if plotting
    outfile_legend = os.path.join(outdir, '00_colormap_legend.png')
    colormap, color_dict = make_color_legend(outfile_legend, label_map_dict)

    # group by image, and plot
    group = df.groupby(groupby)
    count = 0
    refine_dic = {}
    for i,g in enumerate(group):
        
        img_loc_string = g[0]
        data = g[1] 
        
        if (i % 1) == 0:
            print (i+1, "Plotting image:", img_loc_string)
            print ("  len boxes:", len(data))
            #print ("slice_sizes:", slice_sizes)
        
        #print("data:", data)
        image = cv2.imread(img_loc_string, 1)
        if verbose:
            print ("image.shape:", image.shape)

        if not sliced:  #slice_sizes[0] <= 0:
            xmins = data['Xmin'].values
            ymins = data['Ymin'].values
            xmaxs = data['Xmax'].values
            ymaxs = data['Ymax'].values
        else:
            xmins = data['Xmin_Glob'].values
            ymins = data['Ymin_Glob'].values
            xmaxs = data['Xmax_Glob'].values
            ymaxs = data['Ymax_Glob'].values

        scores = data['Prob'].values
        #classes_int = (data['Category'].data).astype(int)
        classes = data['Category'].values
        classes_str = classes
        
        ## set legend str? 
        #if len(label_map_dict.keys()) > 0:
        #    classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
        #    classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
        #else:
        #    classes_str = classes_int_str
        #    classes_legend_str = classes_str    
    
        # filter out low probs
        high_prob_idxs = np.where(scores >= plot_thresh)
        scores = scores[high_prob_idxs]
        classes = classes[high_prob_idxs]
        classes_str = classes_str[high_prob_idxs]
        xmins = xmins[high_prob_idxs]
        xmaxs = xmaxs[high_prob_idxs]
        ymins = ymins[high_prob_idxs]
        ymaxs = ymaxs[high_prob_idxs]

        boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)
        
        if verbose:
            print ("len boxes:", len(boxes))
        
        ###########
        # NMS
        if nms_overlap_thresh > 0:
        
            ## try tf nms (always returns an empty list!)
            ## https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
            #boxes_tf = tf.convert_to_tensor(boxes, np.float32)
            #scores_tf = tf.convert_to_tensor(scores, np.float32)
            #nms_idxs = tf.image.non_max_suppression(boxes_tf, scores_tf, 
            #                                        max_output_size=1000,
            #                                        iou_threshold=0.5)
            #selected_boxes = tf.gather(boxes_tf, nms_idxs)
            #print ("  len boxes:", len(boxes))
            #print ("  nms idxs:", nms_idxs)
            #print ("  selected boxes:", selected_boxes)
    
            # Try nms with pyimagesearch algorightm
            # assume boxes = [[xmin, ymin, xmax, ymax, ...
            #   might want to split by class because we could have a car inside
            #   the bounding box of a plane, for example
            boxes_nms_input = np.stack((xmins, ymins, xmaxs, ymaxs), axis=1)
            _, _, good_idxs =  non_max_suppression(boxes_nms_input, 
                                                 overlapThresh=nms_overlap_thresh)
            if verbose:
                print ("num boxes_all:", len(xmins))
                print ("num good_idxs:", len(good_idxs))
            boxes = boxes[good_idxs]
            scores = scores[good_idxs]
            classes = classes_str[good_idxs]
            
            
        # create output
        refine_dic[img_loc_string] = [scores, boxes, classes]
        
        
        #############
        # Plot
        if plot:    
            
            # load image
            #image = cv2.imread(img_loc_string, 1)
    
            # could try to plot with:
            #     https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
            #     using the draw_bouding_boxes on image_array() function
            #     but this is annoying, so instead use plot_rects() above 
            #dlist = [[z] for z in class_ints_str]
            #visualization_utils.draw_bounding_boxes_on_image_array(image, 
            #                                                       boxes,
            #                                                       display_str_list_list=dlist)
    
            # else, use custom function
            im_root = os.path.basename(img_loc_string)
            outfile = os.path.join(outdir, im_root)
            #print ("image_location:", img_loc_string)
            #print ("  image:", image)
            count += len(boxes)
            if verbose:
                print ("outfile:", outfile)
            
            plot_rects(image, boxes, scores, 
                  classes=classes_str,
                  plot_thresh=plot_thresh, 
                  color_dict=color_dict, #colormap=colormap,
                  outfile=outfile,
                  show_labels=show_labels,
                  alpha_scaling=alpha_scaling,
                  plot_line_thickness=plot_line_thickness,
                  verbose=verbose)
        
     
        
    t1 = time.time()
    print ("Time to plot", count, "boxes:", t1 - t0, "seconds")
    
    return refine_dic

###############################################################################
def make_color_legend(outfile, label_map_dict, auto_assign_colors=True,
                      verbose=False):
    '''Create and save color legend as image'''
        
    
    if auto_assign_colors:
        # automatically assign colors?
        cmap = plt.cm.get_cmap('jet', len(list(label_map_dict.keys())))
        # sometimes label_map_dict starts at 1, instead of 0
        if min(list(label_map_dict.keys())) == 1:
            idx_plus_val = 1
        else:
            idx_plus_val = 0
        colormap = []
        color_dict = {}
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            #hexa = matplotlib.colors.rgb2hex(rgb)
            #cmaplist.append(hexa)
            rgb_tuple = tuple([int(255*z) for z in rgb])
            colormap.append(rgb_tuple)
            color_dict[label_map_dict[i + idx_plus_val]] = rgb_tuple
            
        #for key in label_map_dict.keys():
        #    itmp = key
        #    color = colormap[itmp]
        #    color_dict[label_map_dict[key]] = color
            
    else:
        colormap = [(255, 0,   0),
                (0,   255, 0),
                (0,   0,   255),
                (255, 255, 0),
                (0,   255, 255),
                (255, 0,   255),
                (0,   0,   255),
                (127, 255, 212),
                (72,  61,  139),
                (255, 127, 80),
                (199, 21,  133),
                (255, 140, 0),
                (0, 165, 255)] 

        # manually assign colors?
        # colrs are bgr not rgb https://www.webucator.com/blog/2015/03/python-color-constants-module/
        color_dict = {
              'airplane':   (0,   255, 0),
              'boat':       (0,   0,   255),
              'car':        (255, 255, 0),
              'airport':    (255, 155,   0),
              'building':   (0, 0, 200),
              }

    h, w = 800, 400
    xpos = int(0.2*w)
    ydiff = int(0.05*h)  
    # TEXT FONT
    # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
    font = cv2.FONT_HERSHEY_TRIPLEX  #FONT_HERSHEY_SIMPLEX 
    font_size = 0.4
    label_font_width = 1

    # rescale height so that if we have a long list of categories it fits 
    rescale_h = h * len(label_map_dict.keys()) / 18.
    hprime = max(h, int(rescale_h))
    img_mpl = 255*np.ones((hprime, w, 3))

    try:
        cv2.putText(img_mpl, 'Color Legend', (int(xpos), int(ydiff)), font, 
                    1.5*font_size, (0,0,0), int(1.5*label_font_width), 
                    cv2.CV_AA)
                    #cv2.LINE_AA)
    except:
        cv2.putText(img_mpl, 'Color Legend', (int(xpos), int(ydiff)), font, 
                    1.5*font_size, (0,0,0), int(1.5*label_font_width), 
                    cv2.LINE_AA)

    
    for key in label_map_dict.keys():
        itmp = key
        val = label_map_dict[key]
        color = color_dict[val]
        
        #color = colormap[itmp]
        #color_dict[label_map_dict[key]] = color
        
        text = '- ' + str(key) + ': ' + str(label_map_dict[key])
        ypos = 2* ydiff + itmp * ydiff
        try:
            cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 
                    1.5*font_size, color, label_font_width, 
                    cv2.CV_AA)
                    #cv2.LINE_AA)
        except:
             cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 
                    1.5*font_size, color, label_font_width, 
                    cv2.LINE_AA)
            
    cv2.imwrite(outfile, img_mpl)
    
    if verbose:
        print ("post_process.py - make_color_legend() label_map_dict:", label_map_dict)
        print ("post_process.py - make_color_legend() colormap:", colormap)
        print ("post_process.py - make_color_legend() color_dict:", color_dict)
        
    return colormap, color_dict


###############################################################################
def plot_rects(im, boxes, scores, classes=[], outfile='', plot_thresh=0.3,
              color_dict={},  #colormap=[(0,0,0)], 
              plot_line_thickness=2, show_labels=True, 
              label_alpha_scale=0.85, compression_level=7,
              alpha_scaling=True, show_plots=False, skip_empty=False,
              resize_factor=1,
              verbose=False, super_verbose=False):
    '''Plot boxes in image
    if alpha_scaling, scale box opacity with probability
    if show_labels, plot the label above each box
    extremely slow if alpha_scaling = True
    '''

    ##################################
    # label settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.3
    font_width = 1
    display_str_height = 3
    # upscale plot_line_thickness
    plot_line_thickness *= resize_factor
    ##################################

    if verbose:
        print ("color_dict:", color_dict)
    output = im
    h,w = im.shape[:2]
    nboxes = 0
                         
    # scale alpha with prob can be extremely slow since we're overlaying a
    #  a fresh image for each box, need to bin boxes and then plot. Instead,
    #  bin the scores, then plot

    # if alpha scaling, bin by scores
    if alpha_scaling:
        # if alpha scaling, bin by scores
        if verbose:
            print ("Binning scores in plot_rects()...")
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html
        bins = np.linspace(0, 1, 11)   # define a step of 0.1 between 0 and 1
        inds = np.digitize(scores, bins)   # bin that each element belongs to
        unique_inds = np.sort(np.unique(inds))
        for bin_ind in unique_inds:
            alpha_val = bins[bin_ind]
            boxes_bin = boxes[bin_ind == inds]
            scores_bin = scores[bin_ind == inds]
            classes_bin = classes[bin_ind == inds]

            if verbose:
                print ("bin_ind:", bin_ind)
                print ("alpha_val:", alpha_val)
                print ("scores_bin.shape:", scores_bin.shape)
            
            # define overlay alpha
            # rescale to be between 0.3 and 1 (alpha_val starts at 0.1)
            alpha = 0.2 + 0.8*alpha_val
            #alpha = min(0.95, alpha_val+0.1)
            overlay = np.zeros(im.shape).astype(np.uint8)     #overlay = im_raw.copy()

            # for labels, if desired, make labels a bit dimmer 
            alpha_prime = max(0.25, label_alpha_scale * alpha)
            overlay1 = np.zeros(im.shape).astype(np.uint8)

            for box, score, classy in zip(boxes_bin, scores_bin, classes_bin):
              
                if score >= plot_thresh:
                    nboxes += 1
                    [ymin, xmin, ymax, xmax] = box
                    left, right, top, bottom = xmin, xmax, ymin, ymax
                    
                    # check boxes
                    if (left < 0) or (right > (w-1)) or (top < 0) or (bottom > (h-1)):
                        print ("box coords out of bounds...")
                        print ("  im.shape:", im.shape)
                        print ("  left, right, top, bottom:", left, right, top, bottom)
                        return
                    
                    if (right < left) or (bottom < top) :
                        print ("box coords reversed?...")
                        print ("  im.shape:", im.shape)
                        print ("  left, right, top, bottom:", left, right, top, bottom)
                        return
                    
                   
                    # get label and color
                    classy_str = str(classy) + ': ' + str(int(100*float(score))) + '%'
                    color = color_dict[classy]
         
                    if super_verbose:
                        #print ("  box:", box)
                        print ("  left, right, top, bottom:", left, right, top, bottom)
                        print ("   classs:", classy)
                        print ("   score:", score)
                        print ("   classy_str:", classy_str)
                        print ("   color:", color)


                    # add rectangle to overlay
                    cv2.rectangle(overlay, (int(left), int(bottom)), (int(right), 
                                            int(top)), color, 
                                            plot_line_thickness,
                                            lineType=1)#cv2.CV_AA) 

                    # plot categories too?
                    if show_labels:
                        # adapted from visuatlizion_utils.py
                        # get location
                        display_str = classy_str  # or classy, whch is '1 = airplane'
                        # If the total height of the display strings added to the top of the bounding
                        # box exceeds the top of the image, stack the strings below the bounding box
                        # instead of above.
                        #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                        # Each display_str has a top and bottom margin of 0.05x.
                        total_display_str_height = (1 + 2 * 0.05) * display_str_height
                        if top > total_display_str_height:
                            text_bottom = top
                        else:
                            text_bottom = bottom + total_display_str_height
                        # Reverse list and print from bottom to top.
                        (text_width, text_height), _ = cv2.getTextSize(display_str, 
                                                                      fontFace=font,
                                                                      fontScale=font_size,
                                                                      thickness=font_width) #5, 5#font.getsize(display_str)
                        margin = np.ceil(0.1 * text_height)
                        
                        # get rect and text coords,
                        rect_top_left = (int(left - (plot_line_thickness - 1) * margin), 
                                                 int(text_bottom - text_height - (plot_line_thickness + 3) * margin ))
                        rect_bottom_right = (int(left + text_width + margin), 
                                                 int(text_bottom - (plot_line_thickness * margin)))
                        text_loc = (int(left + margin), 
                                     int(text_bottom  - (plot_line_thickness + 2) * margin))

    
                        # plot
                        # if desired, make labels a bit dimmer 
                        cv2.rectangle(overlay1, rect_top_left, rect_bottom_right, 
                                                color, -1)
                        cv2.putText(overlay1, display_str, text_loc, 
                                        font, font_size, (0,0,0), font_width, 
                                        cv2.CV_AA)
                                        #cv2.LINE_AA)
 
    
            # for the bin, combine overlay and original image              
            overlay_alpha = (alpha * overlay).astype(np.uint8)
            if verbose:
                print ("overlay.shape:", overlay.shape)
                print ("overlay_alpha.shape:", overlay_alpha.shape)
                print ("overlay.dtype:", overlay.dtype)
                print ("min, max, overlay", np.min(overlay), np.max(overlay))
                #print ("output.shape:", output.shape)
                #print ("output.dtype:", output.dtype)
            # simply sum the two channels?
            # Reduce the output image where the overaly is non-
            # to use masks, see https://docs.opencv.org/3.1.0/d0/d86/tutorial_py_image_arithmetics.html
            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            yup = np.nonzero(overlay_gray)
            output_tmp = output.astype(float)
            output_tmp[yup] *= (1.0 - alpha)
            output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha)
    
            # add labels, if desired
            if show_labels:
                overlay_alpha1 = (alpha_prime * overlay1).astype(np.uint8)
                overlay_gray1 = cv2.cvtColor(overlay1, cv2.COLOR_BGR2GRAY)
                yup = np.nonzero(overlay_gray1)
                output_tmp = output.astype(float)
                output_tmp[yup] *= (1.0 - alpha_prime)
                output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha1)    


    # no alpha scaling, much simpler to plot           
    else:
         
        for box, score, classy in zip(boxes, scores, classes):
      
            if score >= plot_thresh:
                nboxes += 1
                [ymin, xmin, ymax, xmax] = box
                left, right, top, bottom = xmin, xmax, ymin, ymax
               
                # get label and color
                classy_str = str(classy) + ': ' + str(int(100*float(score))) + '%'
                color = color_dict[classy]
    
                if verbose:
                    #print ("  box:", box)
                    print ("  left, right, top, bottom:", left, right, top, bottom)
                    print ("   classs:", classy)
                    print ("   score:", score)
        
                # add rectangle
                cv2.rectangle(output, (int(left), int(bottom)), (int(right), 
                                            int(top)), color, 
                                            plot_line_thickness)
                
                # plot categories too?
                if show_labels:
                    # adapted from visuatlizion_utils.py
                    # get location
                    display_str = classy_str  # or classy, whch is '1 = airplane'
                    # If the total height of the display strings added to the top of the bounding
                    # box exceeds the top of the image, stack the strings below the bounding box
                    # instead of above.
                    #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                    # Each display_str has a top and bottom margin of 0.05x.
                    total_display_str_height = (1 + 2 * 0.05) * display_str_height
                    if top > total_display_str_height:
                        text_bottom = top
                    else:
                        text_bottom = bottom + total_display_str_height
                    # Reverse list and print from bottom to top.
                    (text_width, text_height), _ = cv2.getTextSize(display_str, 
                                                                  fontFace=font,
                                                                  fontScale=font_size,
                                                                  thickness=font_width) #5, 5#font.getsize(display_str)
                    margin = np.ceil(0.1 * text_height)
                    
                    # get rect and text coords,
                    rect_top_left = (int(left - (plot_line_thickness - 1) * margin), 
                                             int(text_bottom - text_height - (plot_line_thickness + 3) * margin ))
                    rect_bottom_right = (int(left + text_width + margin), 
                                             int(text_bottom - (plot_line_thickness * margin)))
                    text_loc = (int(left + margin), 
                                 int(text_bottom  - (plot_line_thickness + 2) * margin))
    
                    # annoying notch between label box and bounding box, 
                    #    caused by rounded lines, so if
                    #    alpha is high, move everything down a smidge
                    if (not alpha_scaling) or ((alpha > 0.75) and (plot_line_thickness > 1)):
                        rect_top_left = (rect_top_left[0], int(rect_top_left[1] + margin))
                        rect_bottom_right = (rect_bottom_right[0], int(rect_bottom_right[1] + margin))
                        text_loc = (text_loc[0], int(text_loc[1] + margin))

                    cv2.rectangle(output, rect_top_left, rect_bottom_right, 
                                            color, -1)
                    cv2.putText(output, display_str, text_loc, 
                                        font, font_size, (0,0,0), font_width, 
                                        cv2.CV_AA)
                                        #cv2.LINE_AA)
    
    # resize, if desired
    if resize_factor != 1:
        height, width = output.shape[:2]
        output = cv2.resize(output,(width/resize_factor, height/resize_factor),
                           interpolation=cv2.INTER_CUBIC)


    if skip_empty and nboxes==0:
        return
    else:
        cv2.imwrite(outfile, output, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])

    if show_plots:
        #plt.show()
        cmd = 'eog ' + outfile + '&'
        os.system(cmd)
    
    return



###############################################################################
def plot_rects_v0(im, boxes, scores, classes=[], outfile='', plot_thresh=0.3,
              color_dict={},  #colormap=[(0,0,0)], 
              plot_line_thickness=2, show_labels=True, 
              label_alpha_scale=0.85, compression_level=7,
              alpha_scaling=True, show_plots=False, skip_empty=False,
              verbose=False):
    '''Plot boxes in image
    if alpha_scaling, scale box opacity with probability
    if show_labels, plot the label above each box
    extremely slow if alpha_scaling = True
    '''

    ##################################
    # label settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.3
    font_width = 1
    display_str_height = 3
    ##################################

    if verbose:
        print ("color_dict:", color_dict)
    output = im
    h,w = im.shape[:2]
    nboxes = 0
    for box, score, classy in zip(boxes, scores, classes):
      
        if score >= plot_thresh:
            nboxes += 1
            [ymin, xmin, ymax, xmax] = box
            left, right, top, bottom = xmin, xmax, ymin, ymax
            #(left, right, top, bottom) = (int(xmin * w), int(xmax * w),
            #                          int(ymin * h), int(ymax * h))
           
            # get label and color
            # updated method
            classy_str = str(classy) + ': ' + str(int(100*float(score))) + '%'
            color = color_dict[classy]
            # original method
            #classy_int, classy_str0 = int(classy.split(' ')[0]), classy.split(' ')[-1]
            #classy_str = classy_str0 + ': ' + str(int(100*float(score))) + '%'
            #color = colormap[classy_int]

            if verbose:
                #print ("  box:", box)
                print ("  left, right, top, bottom:", left, right, top, bottom)
                print ("   classs:", classy)
                print ("   score:", score)
    
                 
            # scale alpha with prob  (extremely slow since we're overlaying a
            #  a fresh image for each box, need to bin boxes and then plot )
            if alpha_scaling:
                alpha = score

                #classy_str = classy_str0 + ': ' + str(int(100*alpha)) + '%'
                overlay = np.zeros(im.shape).astype(np.uint8)     #overlay = im_raw.copy()
                cv2.rectangle(overlay, (int(left), int(bottom)), (int(right), 
                                        int(top)), color, 
                                        plot_line_thickness,
                                        lineType=1)#cv2.CV_AA)                
                overlay_alpha = (alpha * overlay).astype(np.uint8)
    
                if verbose:
                    print ("alpha:", alpha)
                    print ("color:", color)
                    print ("classy_str:", classy_str)
                    print ("overlay.shape:", overlay.shape)
                    print ("overlay_alpha.shape:", overlay_alpha.shape)
                    print ("overlay.dtype:", overlay.dtype)
                    print ("min, max, overlay", np.min(overlay), np.max(overlay))
                    #print ("output.shape:", output.shape)
                    #print ("output.dtype:", output.dtype)

                # simply sum the two channels?
                # Reduce the output image where the overaly is non-
                # to use masks, see https://docs.opencv.org/3.1.0/d0/d86/tutorial_py_image_arithmetics.html
                overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                yup = np.nonzero(overlay_gray)
                output_tmp = output.astype(float)
                output_tmp[yup] *= (1.0 - alpha)
                output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha)

                ##############
                # manually update each pixel (slow, but gives desired results)
                #pos_rows, pos_cols, pos_channels = np.nonzero(overlay)
                #for (row, col) in zip(pos_rows, pos_cols):
                #    output[row][col] = ((1.0-alpha) * output[row][col]).astype(np.uint8)
                #output = cv2.add(output, overlay_alpha)

                #############
                # using addWeighted is easiest, though since we have multiple
                # boxes, the original image must have alpha'= 1, instead of 
                # alpha' = 1-alpha as in pyimagesearch blog:
                #  https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
                # adding the two images gives the appearance of high opacity
                #   output = cv2.add(output, overlay_alpha)
                #   since ove water the color would still be (255, 0, 255)
                #cv2.addWeighted(overlay, alpha, output, 1, 0, output)
                #cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output) *from pyimagesearch

            
            else:
                cv2.rectangle(output, (int(left), int(bottom)), (int(right), 
                                        int(top)), color, 
                                        plot_line_thickness)
            
            # plot categories too?
            if show_labels:
                # adapted from visuatlizion_utils.py
                # get location
                display_str = classy_str  # or classy, whch is '1 = airplane'
                # If the total height of the display strings added to the top of the bounding
                # box exceeds the top of the image, stack the strings below the bounding box
                # instead of above.
                #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                # Each display_str has a top and bottom margin of 0.05x.
                total_display_str_height = (1 + 2 * 0.05) * display_str_height
                if top > total_display_str_height:
                    text_bottom = top
                else:
                    text_bottom = bottom + total_display_str_height
                # Reverse list and print from bottom to top.
                (text_width, text_height), _ = cv2.getTextSize(display_str, 
                                                              fontFace=font,
                                                              fontScale=font_size,
                                                              thickness=font_width) #5, 5#font.getsize(display_str)
                margin = np.ceil(0.1 * text_height)
                
                # get rect and text coords,
                rect_top_left = (int(left - (plot_line_thickness - 1) * margin), 
                                         int(text_bottom - text_height - (plot_line_thickness + 3) * margin ))
                rect_bottom_right = (int(left + text_width + margin), 
                                         int(text_bottom - (plot_line_thickness * margin)))
                text_loc = (int(left + margin), 
                             int(text_bottom  - (plot_line_thickness + 2) * margin))

                # annoying notch between label box and bounding box, 
                #    caused by rounded lines, so if
                #    alpha is high, move everything down a smidge
                if (not alpha_scaling) or ((alpha > 0.75) and (plot_line_thickness > 1)):
                    rect_top_left = (rect_top_left[0], int(rect_top_left[1] + margin))
                    rect_bottom_right = (rect_bottom_right[0], int(rect_bottom_right[1] + margin))
                    text_loc = (text_loc[0], int(text_loc[1] + margin))


                if alpha_scaling:
                    # box
                    overlay1 = np.zeros(im.shape).astype(np.uint8)

                    # plot
                    # if desired, make labels a bit dimmer 
                    alpha_prime = label_alpha_scale * alpha
                    cv2.rectangle(overlay1, rect_top_left, rect_bottom_right, 
                                            color, -1)
                    overlay_alpha1 = (alpha_prime * overlay1).astype(np.uint8)
                    overlay_gray1 = cv2.cvtColor(overlay1, cv2.COLOR_BGR2GRAY)
                    yup = np.nonzero(overlay_gray1)
                    output_tmp = output.astype(float)
                    output_tmp[yup] *= (1.0 - alpha_prime)
                    output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha1)    
                    # text
                    try:
                        cv2.putText(output, display_str, text_loc, 
                                        font, font_size, (0,0,0), font_width, 
                                        cv2.CV_AA)   
                                        #cv2.LINE_AA)
                    except:
                        cv2.putText(output, display_str, text_loc, 
                                        font, font_size, (0,0,0), font_width, 
                                        cv2.LINE_AA)

#                    # tight boxes
#                    cv2.rectangle(overlay1, (int(left - (plot_line_thickness - 1) * margin), 
#                                             int(text_bottom - text_height - (plot_line_thickness + 2) * margin )), 
#                                            (int(left + text_width), 
#                                             int(text_bottom - (plot_line_thickness * margin))), 
#                                            color, -1)
#                    overlay_alpha1 = (alpha * overlay1).astype(np.uint8)
#                    overlay_gray1 = cv2.cvtColor(overlay1, cv2.COLOR_BGR2GRAY)
#                    yup = np.nonzero(overlay_gray1)
#                    output_tmp = output.astype(float)
#                    output_tmp[yup] *= (1.0 - alpha)
#                    output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha1)    
#                    # text
#                    cv2.putText(output, display_str, 
#                                (int(left), 
#                                 int(text_bottom  - (plot_line_thickness + 1) * margin)), 
#                                        font, font_size, (0,0,0), font_width, 
#                                        cv2.CV_AA)                    

                else:
                    cv2.rectangle(output, rect_top_left, rect_bottom_right, 
                                            color, -1)
                    cv2.putText(output, display_str, text_loc, 
                                        font, font_size, (0,0,0), font_width, 
                                        cv2.CV_AA)
                                        #cv2.LINE_AA)

                    #cv2.rectangle(output, (int(left-margin), int(text_bottom - text_height - 3 * margin)), 
                    #                        (int(left + text_width), int(text_bottom)), color, -1)
                    #cv2.putText(output, display_str, 
                    #            (int(left), int(text_bottom  - 2 * margin)), 
                    #                    font, font_size, (0,0,0), font_width, 
                    #                    cv2.CV_AA)


    if skip_empty and nboxes==0:
        return
    else:
        cv2.imwrite(outfile, output, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])

    if show_plots:
        #plt.show()
        cmd = 'eog ' + outfile + '&'
        os.system(cmd)
    
    return

###############################################################################
### Rotated boxes
###############################################################################    
def rotatePoint(centerPoint, point, angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise
    #http://stackoverflow.com/questions/20023209/python-function-for-rotating-2d-objects
    """
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = (temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle), 
                  temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point
#rotatePoint( (1,1), (2,2), 45)       

###############################################################################    
def rescale_angle(angle_rad):
    '''transform theta to angle between -45 and 45 degrees
    expect input angle to be between 0 and pi radians'''
    angle_deg = round(180. * angle_rad / np.pi, 2)
 
    if angle_deg >= 0. and angle_deg <= 45.:
        angle_out = angle_deg
    elif angle_deg > 45. and angle_deg < 90.:
        angle_out = angle_deg - 90.
    elif angle_deg >= 90. and angle_deg < 135:
        angle_out = angle_deg - 90.
    elif angle_deg >= 135 and angle_deg < 180:
        angle_out = angle_deg - 180.
    else:
        print ("Unexpected angle in rescale_angle() ", \
               "[should be from 0-pi radians]")
        return
        
    return angle_out

###############################################################################
def rotate_box(xmin, xmax, ymin, ymax, canny_edges, verbose=False):
    '''Rotate box'''
    
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    centerPoint = (np.mean([xmin, xmax]), np.mean([ymin, ymax]))
    
    # get canny edges in desired window
    win_edges = canny_edges[ymin:ymax, xmin:xmax]
    
    # create hough lines
    hough_lines = cv2.HoughLines(win_edges,1,np.pi/180,20)
    if hough_lines is not None:   
        #print "hough_lines:", hough_lines
        # get primary angle
        line = hough_lines[0]
        if verbose:
            print (" hough_lines[0]",  line)
        if len(line) > 1:
            rho, theta = line[0].flatten()
        else:
            rho, theta = hough_lines[0].flatten()
    else:
        theta = 0.
    # rescale to between -45 and +45 degrees
    angle_deg = rescale_angle(theta)
    if verbose:
        print ("angle_deg_rot:", angle_deg)
    # rotated coords
    coords_rot = np.asarray([rotatePoint(centerPoint, c, angle_deg) for c in
                             coords], dtype=np.int32)

    return coords_rot
