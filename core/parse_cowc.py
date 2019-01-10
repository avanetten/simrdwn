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
import shutil
import pickle
import time
import cv2
import sys
import os

path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_simrdwn_core)
import yolt_data_prep_funcs

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
        yolt_co_i = yolt_data_prep_funcs.convert((win_w, win_h), box_i)
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
            box_tmp_init = yolt_data_prep_funcs.convert_reverse((w,h), yb)
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
def gt_dic_from_box_coords(box_coords):
    '''
    box_coords are of form:
        box_coords = [x0, x1, y0, y1]
    output should be of form:
    x1l0, y1l0 = lineData['pt1X'].astype(int), lineData['pt1Y'].astype(int)
    x2l0, y2l0 = lineData['pt2X'].astype(int), lineData['pt2Y'].astype(int)
    x3l0, y3l0 = lineData['pt3X'].astype(int), lineData['pt3Y'].astype(int)
    x4l0, y4l0 = lineData['pt4X'].astype(int), lineData['pt4Y'].astype(int) 
    assume pt1 is stern, pt2 is bow, pt3 and pt4 give width

    '''
                
    box_coords = np.array(box_coords)
    out_dic = {}

    out_dic['pt1X'] = box_coords[:,0]
    out_dic['pt1Y'] = box_coords[:,2]

    # set p2 as diagonal from p1
    out_dic['pt2X'] = box_coords[:,1] #box_coords[:,1]
    out_dic['pt2Y'] = box_coords[:,3] #box_coords[:,2]

    out_dic['pt3X'] = box_coords[:,1] #box_coords[:,1]
    out_dic['pt3Y'] = box_coords[:,2] #box_coords[:,3]

    out_dic['pt4X'] = box_coords[:,0]
    out_dic['pt4Y'] = box_coords[:,3]

    return out_dic   

###############################################################################
def slice_im_cowc(input_im, input_mask, outname_root, outdir_im, outdir_label, 
             classes_dic, category, yolt_box_size, 
             sliceHeight=256, sliceWidth=256, 
             zero_frac_thresh=0.2, overlap=0.2, pad=0, verbose=False,
             box_coords_dir='', yolt_coords_dir=''):
    '''
    ADAPTED FROM YOLT/SCRIPTS/SLICE_IM.PY
    Assume input_im is rgb
    Slice large satellite image into smaller pieces, 
    ignore slices with a percentage null greater then zero_fract_thresh'''

    image = cv2.imread(input_im, 1)  # color
    gt_image = cv2.imread(input_mask, 0)
    category_num = classes_dic[category]
    
    im_h, im_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth
    
    # if slice sizes are large than image, pad the edges
    if sliceHeight > im_h:
        pad = sliceHeight - im_h
    if sliceWidth > im_w:
        pad = max(pad, sliceWidth - im_w)
    # pad the edge of the image with black pixels
    if pad > 0:    
        border_color = (0,0,0)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, 
                                 cv2.BORDER_CONSTANT, value=border_color)

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y in range(0, im_h, dy):#sliceHeight):
        for x in range(0, im_w, dx):#sliceWidth):
            n_ims += 1
            # extract image
            # make sure we don't go past the edge of the image
            if y + sliceHeight > im_h:
                y0 = im_h - sliceHeight
            else:
                y0 = y
            if x + sliceWidth > im_w:
                x0 = im_w - sliceWidth
            else:
                x0 = x
            
            window_c = image[y0:y0 + sliceHeight, x0:x0 + sliceWidth]
            gt_c = gt_image[y0:y0 + sliceHeight, x0:x0 + sliceWidth]
            win_h, win_w = window_c.shape[:2]
            
            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

            # find threshold of image that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret,thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            #print ("zero_frac", zero_fra   
            # skip if image is mostly empty
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print ("Zero frac too high at:", zero_frac)
                continue 
            
            box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, 
                                                             yolt_box_size, 
                                                             verbose=verbose)
            # continue if no coords
            if len(box_coords) == 0:
                continue
            
            
            #  save          
            outname_part = 'slice_' + outname_root + \
            '_' + str(y0) + '_' + str(x0) + '_' + str(win_h) + '_' + str(win_w) +\
            '_' + str(pad)
            outname_im = os.path.join(outdir_im, outname_part + '.png')
            txt_outpath = os.path.join(outdir_label, outname_part + '.txt')
            
            # save yolt ims
            if verbose:
                print ("image output:", outname_im)
            cv2.imwrite(outname_im, window_c)
            
            # save yolt labels
            txt_outfile = open(txt_outpath, "w")
            if verbose:
                print ("txt output:" + txt_outpath)
            for bb in yolt_coords:
                outstring = str(category_num) + " " + " ".join([str(a) for a in bb]) + '\n'
                if verbose:
                    print ("outstring:", outstring)
                txt_outfile.write(outstring)
            txt_outfile.close()

            # if desired, save coords files
            # save box coords dictionary so that yolt_eval.py can read it                                
            if len(box_coords_dir) > 0: 
                coords_dic = gt_dic_from_box_coords(box_coords)
                outname_pkl = os.path.join(box_coords_dir, outname_part + '_' + category + '.pkl')
                pickle.dump(coords_dic, open(outname_pkl, 'wb'), protocol=2)
            if len(yolt_coords_dir) > 0:  
                outname_pkl = os.path.join(yolt_coords_dir, outname_part + '_' + category + '.pkl')
                pickle.dump(yolt_coords, open(outname_pkl, 'wb'), protocol=2)

            n_ims_nonull += 1

    print ("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull, \
            "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print ("Time to slice", input_im, time.time()-t0, "seconds")
      
    return

###############################################################################        
def plot_gt_boxes(im_file, label_file, yolt_box_size,
                  figsize=(10,10), color=(0,0,255), thickness=2):
    '''
    plot ground truth boxes overlaid on image
    '''
    
    im = cv2.imread(im_file)
    gt_c = cv2.imread(label_file, 0)
    box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, yolt_box_size,
                                                     verbose=False)
    
    img_mpl = im
    for b in box_coords:
        [xmin, xmax, ymin, ymax] = b

        cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)    


###############################################################################
def main():
    
    ### Construct argument parser
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument('--truth_dir', type=str, default='/Users/avanetten/Documents/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC',
                        help="Location of  ground truth labels")
    parser.add_argument('--simrdwn_data_dir', type=str, default='/cosmiq/simrdwn/data/',
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
                     
    # create image list
    yolt_im_list_loc = os.path.join(args.out_dir, 'cowc_training_list.txt')
    im_ext = '.png'
    print ("\nsave image list to:", yolt_im_list_loc)
    with open(yolt_im_list_loc, 'w') as file_handler:
        for item in os.listdir(args.image_dir):
            if item.endswith(im_ext):
                outpath_tmp = os.path.join(args.image_dir, item)
                file_handler.write("{}\n".format(outpath_tmp))
    # copy image list to simrdwn_data_dir
    shutil.copy2(yolt_im_list_loc, args.simrdwn_data_dir)
  
                     
# im_locs = [output_loc+'images/' + ftmp for ftmp in os.listdir(output_loc+'images') if ftmp.endswith('.tif')]
# f = open(im_list_name, 'wb')
# for item in im_locs:
#     f.write("%s\n" % item)
# f.close()
#
#
# # make image list
# im_ext = '.png'
# print ("\n\nsave image list to:", yolt_im_list_loc)
# with open(yolt_im_list_loc, 'w') as file_handler:
#     for item in os.listdir(out_dir_im):
#         if item.endswith(im_ext):
#             outpath_tmp = os.path.join(out_dir_im, item)
#             file_handler.write("{}\n".format(outpath_tmp))


###############################################################################
if __name__ == "__main__":
    main()

'''
python /Users/avanetten/Documents/cosmiq/simrdwn/core/parse_cowc.py \
    --truth_dir=/Users/avanetten/Documents/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC \
    --image_dir=/Users/avanetten/Documents/cosmiq/simrdwn/test_images/cowc_utah_raw_0p3GSD \
    --input_box_size=10 \
    --verbose=1

python /Users/avanetten/Documents/cosmiq/simrdwn/core/parse_cowc.py \
    --truth_dir=/Users/avanetten/Documents/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC \
    --image_dir=/Users/avanetten/Documents/cosmiq/simrdwn/test_images/cowc_utah_raw_0p3GSD_rescale \
    --input_box_size=20 \
    --verbose=1
'''
