#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:37:56 2018

@author: avanetten

Transform data from:
    https://gdo152.llnl.gov/cowc/
"""

from __future__ import print_function
import shapely.geometry
import pandas as pd
import numpy as np
import argparse
import shapely
import cv2
import os

###############################################################################
# FROM YOLT/SCRIPS/CONVERT.PY
def convert(size, box):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmid = (box[0] + box[1])/2.0
    ymid = (box[2] + box[3])/2.0
    w0 = box[1] - box[0]
    h0 = box[3] - box[2]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x,y,w,h)
    
###############################################################################
# FROM YOLT/SCRIPS/CONVERT.PY
def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x,y,w,h = box
    dw = 1./size[0]
    dh = 1./size[1]
    
    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh
    
    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]
    

###############################################################################
def gt_boxes_from_cowc_png(gt_c, yolt_box_size, verbose=False):

    '''
    Get ground truth locations from cowc ground_truth image
    input:
        gt_c is cowc label image
        yolt_box_size is the size of each car in pixels
    outputs:
        box_coords = [x0, x1, y0, y1]
        yolt_coords = convert.conver(box_coords)
    '''
        
    win_h, win_w = gt_c.shape[:2]

    # find locations of labels (locs => (h, w))
    label_locs = zip(*np.where(gt_c > 0))
    
    # skip if label_locs is empty
    if len(label_locs) == 0:
        if verbose:
            print ("Label empty")
        return [], []    
                
    if verbose:
        print ("Num cars:", len(label_locs))
        
    # else, create yolt labels from car locations
    # make boxes around cars
    box_coords = []
    yolt_coords = []
    grid_half = yolt_box_size/2
    for i,l in enumerate(label_locs):
        
        if verbose and (i % 100) == 0:
            print (i, "/", len(label_locs))
            
        ymid, xmid = l
        x0, y0, x1, y1 = xmid - grid_half, ymid - grid_half, \
                         xmid + grid_half, ymid + grid_half
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, gt_c.shape[1]-1)
        y1 = min(y1, gt_c.shape[0]-1)
        box_i = [x0, x1, y0, y1]
        box_coords.append(box_i)
        # Input to convert: image size: (w,h), box: [x0, x1, y0, y1]
        yolt_co_i = convert((win_w, win_h), box_i)
        yolt_coords.append(yolt_co_i)

    box_coords = np.array(box_coords)
    yolt_coords = np.array(yolt_coords)     

    return box_coords, yolt_coords     

###############################################################################
def cowc_box_coords_to_gdf(box_coords, image_path, category, verbose=False):
    '''Convert box_coords to geodataframe, assume schema:      
        box_coords = [x0, x1, y0, y1]
        Adapted from parse_shapefile.py'''
        
    pix_geom_poly_list = []
    for i,b in enumerate(box_coords):
        if verbose and ((i % 100) == 0):
            print ("  ", i, "box:", b)
        [x0, x1, y0, y1] = b
        out_coords = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
        points = [shapely.geometry.Point(coord) for coord in out_coords]
        pix_poly = shapely.geometry.Polygon([[p.x, p.y] for p in points])
        pix_geom_poly_list.append(pix_poly)

    df_shp = pd.DataFrame(pix_geom_poly_list, columns=['geometry_poly_pixel'])
                
    #df_shp['geometry_poly_pixel'] = pix_geom_poly_list
    df_shp['geometry_pixel'] = pix_geom_poly_list
    df_shp['geometry_poly_pixel'] = pix_geom_poly_list
    df_shp['xmin'] = box_coords[:,0]
    df_shp['xmax'] = box_coords[:,1]
    df_shp['ymin'] = box_coords[:,2]
    df_shp['ymax'] = box_coords[:,3]
    df_shp['shp_file'] = ''
    df_shp['Category'] = category
    df_shp['Image_Path'] = image_path
    df_shp['Image_Root'] = image_path.split('/')[-1]

    df_shp.index = np.arange(len(df_shp))

    return df_shp
    
###############################################################################
def cowc_to_gdf(label_image_path, image_path, 
                category, yolt_box_size, rescale_to_int=True, verbose=False):
    '''yolt_box_size is size of car in pixels
    rescale ground truth box locations to correct image size if raw image is 
    a different shape than the label image'''
    
    gt_c = cv2.imread(label_image_path, 0)
    box_coords_init, yolt_coords = gt_boxes_from_cowc_png(gt_c, yolt_box_size, 
                                                     verbose=verbose)
    # rescale yolt_coords to dimensions of input image
    im = cv2.imread(image_path, 0)
    h,w = im.shape[:2]
    print ("gt_c.shape:", gt_c.shape)
    print ("im.shape:", im.shape)
    
    if im.shape != gt_c.shape:
        boxes_rescale = []
        for yb in yolt_coords:
            box_tmp_init = convert_reverse((w,h), yb)
            # rescale to ints
            if rescale_to_int:
                box_tmp = [np.rint(itmp) for itmp in box_tmp_init]
            else:
                box_tmp = box_tmp_init
            boxes_rescale.append(box_tmp)
        
        box_coords = np.asarray(boxes_rescale)
        
    else:
        box_coords = box_coords_init
        
    df = cowc_box_coords_to_gdf(box_coords, image_path, category, 
                                verbose=verbose)
    
    return df


###############################################################################
def get_gdf_tot_cowc(truth_dir, image_dir='',
                     annotation_suffix='_Annotated_Cars.png',
                     category='car', yolt_box_size=10, outfile_df='',
                     verbose=False):
    '''
    yolt_box_size is car size in pixels
    '''
    
    print ("Executing get_gdf_tot_cowc()...")
    
    gt_files = [f for f in os.listdir(truth_dir) if f.endswith(annotation_suffix)]    
    for i,gt_file in enumerate(gt_files):   
        basename_annotated = os.path.basename(gt_file)
        basename = basename_annotated.split(annotation_suffix)[0] + '.png'
        label_image_path = os.path.join(truth_dir, basename_annotated)
        
        # if image_dir is provided
        if len(image_dir) > 0:
            image_path = os.path.join(image_dir, basename)
        else:
            image_path = os.path.jin(truth_dir, basename)
        print (i, "label_image_path:", label_image_path)
        print (i, "image_path:", image_path)
        
        gdf = cowc_to_gdf(label_image_path, image_path, category, 
                          yolt_box_size, verbose=verbose)

        if verbose:
            print ("gdf.columns:", gdf.columns)
        # check that pixel coords are > 0
        if np.min(gdf['xmin'].values) < 0:
            if verbose:
                print ("x pixel coords < 0:", np.min(gdf['xmin'].values))
            
        if np.min(gdf['ymin'].values) < 0:
            if verbose:
                print ("y pixel coords < 0:", np.min(gdf['ymin'].values)  )                 
        
        if i == 0:
            gdf_tot = gdf
        else:
            gdf_tot = gdf_tot.append(gdf)
    gdf_tot.index = np.arange(len(gdf_tot))
    
    if len(outfile_df) > 0:
        gdf_tot.to_csv(outfile_df)
                
    return gdf_tot


###############################################################################
def main():
    
    ### Construct argument parser
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument('--truth_dir', type=str, default='/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC',
                        help="Location of  ground truth labels")
    parser.add_argument('--image_dir', type=str, default='',
                        help="Location of  images, look in truth dir if == ''")
    parser.add_argument('--out_dir', type=str, default='',
                        help="Location of output df, if '', use truth_dir")
    parser.add_argument('--annotation_suffix', type=str, default='_Annotated_Cars.png',
                        help="Suffix of annoation files")
    parser.add_argument('--category', type=str, default='car',
                        help="default category")
    parser.add_argument('--input_box_size', type=int, default=10,
                        help="Default input car size, in pixels")
    parser.add_argument('--verbose', type=int, default=0,
                        help="verbose switch")
    args = parser.parse_args()
    
    if len(args.out_dir) == 0:
        args.out_dir = args.image_dir
    outfile_df = os.path.join(args.out_dir, '_truth_df.csv')
    verbose = bool(args.verbose)
    
    get_gdf_tot_cowc(args.truth_dir,
                     args.image_dir,
                     annotation_suffix = args.annotation_suffix, 
                     category=args.category,
                     yolt_box_size=args.input_box_size,
                     outfile_df=outfile_df,
                     verbose=verbose)
                     

###############################################################################
if __name__ == "__main__":
    main()

'''
python /Users/avanetten/Documents/cosmiq/simrdwn/core/parse_cowc.py \
    --truth_dir=/Users/avanetten/Documents/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC \
    --image_dir=/Users/avanetten/Documents/cosmiq/simrdwn/test_images/cowc_utah_raw_0p3GSD \
    --input_box_size=10 \
    --verbose=1

'''
