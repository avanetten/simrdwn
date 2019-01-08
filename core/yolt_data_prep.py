#!/usr/bin/env python2
# -*- coding: utf-8 -*-"""
Created on Thu Jul  7 12:37:50 2016

@author: avanetten


This is a very rough preprocessing script.  It is intended to be run in blocks
(delineated by #%%).
"""



import os
import sys
import cv2
import math
import shutil
import numpy as np
import pandas as pd
import json
import glob
import pickle
import time 
import random
import subprocess
import operator
#import tifffile as tiff
from skimage import exposure
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from subprocess import Popen, PIPE, STDOUT
from matplotlib.collections import PatchCollection
from osgeo import gdal, osr, ogr, gdalnumeric
from matplotlib.patches import Polygon


#import random
#import time
#import pickle
#from PIL import Image




#%%
###############################################################################
###############################################################################    
###############################################################################
def main():
    return
   
# many more training ims for boats, so don't bother augmenting 
training_sets = ['rio_planes_256', 'rio_planes_312', 'panama_boats_256', 
                 'panama_boats_360', 'planet_airports_1800']
train_name = training_sets[-1]                 
already_labeled = True  # Is the data already labeled with labelImg.py?                                 
run_slice = False       # switch to actually slice up the raw input images
run_augment = False     # switch to augment in python. should always be 
                        #   False, since yolt.c also augments 
                        
#training_sets = ['gbdx_planes']
#train_name = training_sets[0]   
#already_labeled = False  # Is the data already labeled with labelImg.py?                                 
#run_slice = True       # switch to actually slice up the raw input images
#run_augment = True     # switch to augment in python. should always be 
#                        #   False, since yolt.c also augments                 


###############################################################################

##########################
run_relabel = False     # switch to change labels after the fact
make_new_labels = False # switch to create new label ims (labels ims 
                        #   are currently unused)

#yolo.c.train() function augments the data (see data.c.load_data_region()), so no need to augment in python!!!!


yolt_dir = '//cosmiq/yolt2/'
output_loc = '/raid/local/src/yolt2/training_datasets/'  # eventual location of files 
train_images_list_file_loc = yolt_dir + 'data/'
label_image_dir = yolt_dir +  '/data/category_label_images/'

os.chdir(yolt_dir)
sys.path.append(yolt_dir + 'scripts')
import convert, slice_im
reload(convert)
reload(slice_im)
   
###############
# airplanes
if train_name == 'rio_planes_256':
    classes_dic = {"boat": 0, "boat_harbor": 1, "airplane": 2}    
    slice_overlap = 0.45
    zero_frac_thresh=0.2
    sliceHeight, sliceWidth = 256, 256
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = ['airplane_rgb00', 'airplane_rgb01', 'airplane_rgb02', 'airplane_rgb03', 'airplane_rgb04',
                      'airplane_rgb05', 'airplane_rgb06', 'airplane_rgb07', 'airplane_rgb08', 'airplane_rgb09', 'airplane_rgb10']

elif train_name == 'rio_planes_312':
    classes_dic = {"boat": 0, "boat_harbor": 1, "airplane": 2}    
    slice_overlap = 0.45
    zero_frac_thresh=0.2
    sliceHeight, sliceWidth = 312, 312
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = ['airplane_rgb0', 'airplane_rgb1', 'airplane_rgb2', 'airplane_rgb3', 'airplane_rgb4']

elif train_name == 'gbdx_planes':
    classes_dic = {"boat": 0, "boat_harbor": 1, "airplane": 2}    
    slice_overlap = 0.45
    zero_frac_thresh=0.2
    sliceHeight, sliceWidth = 312, 312
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = ['gbdx_054593918020_01_assembly_3_5_london_city_airport0', 
                       'gbdx_054956943030_01_baghdad1', 
                       'gbdx_054956943030_01_baghdad2',
                       'gbdx_054956943030_01_baghdad3',
                       'gbdx_054956943030_01_baghdad4',
                       'gbdx_054956943030_01_baghdad5']
    

    # all training data
#train_file_all = train_images_list_file_loc + name + '_boat_list.txt'
#train_list_all = [train_images_list_file_loc +  name + '_list.txt',
#                  train_images_list_file_loc + 'boat_list3_dev_box.txt']

##############
# boats
elif train_name == "panama_boats_256":
    classes_dic = {"boat": 0, "boat_harbor": 1, "airplane": 2}    
    slice_overlap = 0.45
    zero_frac_thresh=0.2
    sliceHeight, sliceWidth = 256, 256
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = ['WV_wavy', 'WV03_03102015_R1C2_Masked_small']

elif train_name == "panama_boats_360":
    classes_dic = {"boat": 0, "boat_harbor": 1, "airplane": 2}    
    slice_overlap = 0.45
    zero_frac_thresh=0.2
    sliceHeight, sliceWidth = 360, 360
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = ['WV_wavy', 'WV03_03102015_R1C2_crop']

################
## airports
elif train_name == 'planet_airports_1800':
    classes_dic = {"airport_single": 0, "airport_multi": 1, "fuel_depot": 2}    
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    slice_overlap = 0.6
    zero_frac_thresh=0.8
    sliceHeight, sliceWidth = 1800, 1800
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    train_base_list = [f.split('.')[0] for f in os.listdir(train_base_dir + 'ims_input_raw') if f.endswith('.png')]
    # all training data
    #train_file_all = train_images_list_file_loc + train_name + '_list.txt'
    #train_list_all = [train_file_all]


#######################################################################
# infer variables from previous settings
input_images_raw_dir = train_base_dir + 'ims_input_raw/'
split_dir = train_base_dir + train_name + '_split/'
t_dir = train_base_dir + 'training_data/'
labels_dir = t_dir + 'labels/'
images_dir = t_dir + 'images/'

#images_dir = train_base_dir + train_base_out + '_split/'
sample_label_vis_dir = train_base_dir + 'sample_label_vis/'

# make directories
for tmp_dir in [input_images_raw_dir,t_dir,labels_dir,images_dir,split_dir,
                sample_label_vis_dir]:
    try: os.mkdir(tmp_dir)
    except: print ""


#im_locs_for_list = image_dir + name + '/'
#im_locs_for_list = '/home/avanetten/yolt/images/' + name + '/'
#im_locs_for_list = '/raid/local/src/yolt/images/' + name + '/'
im_locs_for_list = output_loc + train_name + '/' + 'training_data/images/'


# Make label images
if make_new_labels:
    print "Making label text images..."
    make_label_images(label_image_dir, new_labels=classes_dic.keys())

##########################
# Labeling and bonunding box settings
# put all images in yolt/images/boat
# put all labels in yolt/labels/boat
# put list of images in yolt/darknet/data/boat_list2_dev_box.txt
##########################    



###############################################################################
# SLICE training images (DON'T DO FOR RE-ANALYSIS!)    

if run_slice:
    # training image should be in darknet/training_datasets/"im_name"
    for train_base in train_base_list:
        #train_im = train_base_dir + train_base + '.png'
        #train_im = train_base_dir + 'ims/' + train_base + '.png'
        train_im = input_images_raw_dir + train_base + '.png'

        slice_im.slice_im(train_im, train_base, split_dir, sliceHeight, sliceWidth,
                          overlap=slice_overlap, zero_frac_thresh=zero_frac_thresh) 
                      
           
        

if not already_labeled:
    print "Need to label images for", train_base_dir, "before continuuing..."
    

else:

    # assume xml files are originally saved in images_dir
    #   move to xml_boxes folder
    xml_dir = train_base_dir + 'xml_boxes/'
    if os.path.exists(xml_dir):
        print xml_dir, "already exists...\n"
    else: 
        
        os.mkdir(xml_dir)
    mv_cmd = 'mv ' + split_dir + '*.xml ' + xml_dir
    print "Move command:", mv_cmd, '\n'
    run_cmd(mv_cmd)
    
    
    ###############################################################################
    # (NO NEED TO AUGMENT, DONE IN yolt.c!)
    # remove previously created augment data (if it exists)
    tmp_dir = train_base_dir + 'augment_oops/'
    
    rm_augment_training_data(labels_dir, images_dir, tmp_dir)
    
    if not os.path.exists(tmp_dir[:-1] + '.zip') and os.path.exists(tmp_dir):
        # zip tmp_dir
        shutil.make_archive(tmp_dir[:-1], 'zip', tmp_dir)
        # rm uncompressed tmp_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    ###############################################################################
    # convert data from xml format to yolt format
    run_cmd('rm ' + xml_dir + '.DS_Store')
    convert.main(train_base_dir, xml_dir, labels_dir, train_name, 
                 classes_dic, im_locs_for_list, split_dir)
    
    # copy images with labels to image_dir
    for label_file in os.listdir(labels_dir):    
        if (label_file == '.DS_Store') or \
            (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt','_rot90.txt','_rot180.txt','_rot270.txt'))):
            continue
        # get image
        print "image loc:", label_file
        root = label_file.split('.')[0]
        im_loc = split_dir + root + '.jpg'
        # copy
        shutil.copy(im_loc, images_dir)
    
    ## zip split_dir to conserve space
    #if not os.path.exists(tmp_dir[:-1] + '.zip'):
    #    shutil.make_archive(split_dir[:-1], 'zip', split_dir)
    #    # rm uncompressed dir
    #    shutil.rmtree(split_dir, ignore_errors=True)
            
        
    # plot boxes
    #plot_training_bboxes(labels_dir, images_dir, ignore_augment=True)
    
    
    ###############################################################################
    # expand data via mirroring, rotation (NO NEED, DONE IN yolt.C!)
    # ...
    
            
    if run_augment:
        augment_ims = augment_training_data(labels_dir, images_dir)
        # update image list
        im_list_loc = train_base_dir + train_name + '_list.txt'
        f = open(im_list_loc, 'wb')
        for item in augment_ims:
            end = item.split('/')[-1]
            out = im_locs_for_list + end
            f.write("%s\n" % out)
        f.close()
    
#%%    
    ###############################################################################
    # plot box labels
    max_plots=200
    thickness=2
    plot_training_bboxes(labels_dir, images_dir, ignore_augment=False,
                         sample_label_vis_dir=sample_label_vis_dir, 
                         max_plots=max_plots, thickness=thickness,
                         verbose=True, ext='.jpg', output_width=500)
    #plt.close("all")
    #%%
    
    ###############################################################################
    # cp _list.txt, label files and image files to appropriate directories
    # get list of images utilzed
    
    # cp _list.txt
    cmd0 = 'cp ' + train_base_dir + '/' + train_name + '_list.txt '  + train_images_list_file_loc
    print cmd0
    run_cmd(cmd0)
    
    ## cp labels and images
    #print "Copying labels and images to ../labels and ../images directory"
    #good_ims = []
    #for label_file in os.listdir(labels_dir):
    #    # copy label to darknet/labels
    #    tmp_dir0 = label_dir + name + '/'
    #    try: 
    #        os.mkdir(tmp_dir0)
    #    except: 
    #        None#print ""
    #    run_cmd('cp ' + labels_dir + label_file + ' '  + tmp_dir0)
    #    
    #    # get image
    #    root = label_file.split('.')[0]
    #    im_loc = train_base_dir + train_base_out + '_split/' + root + '.jpg'
    #    tmp_dir1 = image_dir + name + '/'
    #    try: 
    #        os.mkdir(tmp_dir1)
    #    except: 
    #        None
    #    run_cmd('cp ' + im_loc + ' '  + tmp_dir1)
    #    #print "im_loc", im_loc
    #    good_ims.append(im_loc)
    
    
    ###############################################################################
    # change numeric labels, may be necessary if we want to combine classes
    if run_relabel:
        
        # Relabel airport labels
        indir = '/cosmiq/yolt/labels/airport/'
        indir_rename = '/cosmiq/yolt2/labels/airport_orig/'
        
        # copy to new dir (oly do once!!!)
        #shutil.copytree(indir, indir_rename )
        
        new_label = 3
        files = os.listdir(indir_rename)
        
        for f in files:
            if not f.endswith('.txt'):
                continue
            data = np.loadtxt(indir_rename + f)
            data[0] = int(new_label)
            df = pd.DataFrame(data).T
            df[0] = df[0].astype(int)
            df.to_csv(indir+f, sep=' ', header=False, index=False)
    
    ###############################################################################
    ## combine different lists of sources?
    # BETTER TO DO THIS MANUALLY! by combining lists in darknet/data
    
    #with open(train_file_all, 'wb') as outfile:
    #    for fname in train_list_all:
    #        with open(fname) as infile:
    #            outfile.write(infile.read())\


###############################################################################
###############################################################################
###############################################################################
#%%
### obc data

# images are single band and taken in low light, so equalize histogram
#   http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    
obc_dir = yolt_dir + 'test_images/obc/'
obc_raw_dir = obc_dir + 'obc_raw/'
obc_eq_dir = obc_dir + 'obc_equalize/'
obc_clahe_dir = obc_dir + 'obc_clahe/'
obc_rs_dir = obc_dir + 'obc_rs/'

from skimage import exposure
#http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py

for p in [obc_eq_dir, obc_clahe_dir, obc_rs_dir]:
    if not os.path.exists(p):
        os.mkdir(p)
        

files = glob.glob(os.path.join(obc_raw_dir, '*.tif'))
for i,f in enumerate(files):
    img = cv2.imread(f, 0)
    print "\n", i, "file:", f
    print "   img.shape:", img.shape
    
    basename = os.path.basename(f)
    #print basename
    
    # equalize histogram
    outname_eq = os.path.join(obc_eq_dir, basename)
    print "outname_eq:", outname_eq
    img_eq = cv2.equalizeHist(img)
    cv2.imwrite(outname_eq, img_eq)
    
    # Contrast stretching
    outname_rs = os.path.join(obc_rs_dir, basename)
    print "outname_eq:", outname_rs
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    cv2.imwrite(outname_rs, img_rescale)
   
    
    # create a CLAHE object (Arguments are optional).
    outname_clahe = os.path.join(obc_clahe_dir, basename)
    print "outname_clahe:", outname_clahe
    # cv2
    clahe = cv2.createCLAHE()#tileGridSize=(400, 400))
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    img_clahe = clahe.apply(img)
#    # skimage
#    # Adaptive Equalization
#    img_clahe = exposure.equalize_adapthist(img, kernel_size=(400, 400), 
#                                           clip_limit=0.03)
    cv2.imwrite(outname_clahe, img_clahe)
    
   
###############################################################################
###############################################################################
###############################################################################
#%%
### spacenet data

    # this part is similar to code in sivnet_data_prep.py
    classes_dic = {"building": 0}    
    train_name = 'spacenet'

    imtype = 'iband3'       # [3band, 8band_3pan, iband3, iband]
                            # iband makes 8 1-band images, 
                            # iband3 makes 3 3-band images, repeat final band
    suff = '_' + imtype

    # set yolt paths
    yolt_dir = '/cosmiq/yolt2/'
    deploy_dir_root = '/raid/local/src/yolt2/'
    #deploy_dir_root = yolt_dir#'/raid/local/src/yolt2/'

    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'
    pkl_name_test = train_base_dir + 'ims_coords_list_test' + '.pkl'
    pkl_name_train = train_base_dir + 'ims_coords_list_train' + '.pkl'
    new_schema=False

    #########
    # set paths to input to yolt.c
    t_dir = train_base_dir + 'training_data/'
    labels_dir = t_dir + 'labels' + suff + '/'
    images_dir = t_dir + 'images' + suff + '/'
    sample_label_vis_dir = train_base_dir + 'sample_label_vis' + suff + '/'
    sample_label_vis_dir2 = train_base_dir + 'sample_label_vis2/'

    #########
    train_images_list_file = train_base_dir + 'spacenet_building_list' + suff + '.txt'
    train_images_list_file_loc = yolt_dir + 'data/'
    #deploy_dir_root = '/raid/local/src/yolt2/training_datasets/spacenet/training_data/'
    deploy_dir_root2 = deploy_dir_root + 'training_datasets/'  + train_name + '/training_data/'
    deploy_dir = deploy_dir_root2 + 'images' + suff + '/'
    #########

    # data dirs
    in_data_dir = '/cosmiq/spacenet/data/spacenetFromAWS_08122016/processedData/spacenet_data/'
    maskDir = in_data_dir + 'blupr_mask/'
    vecDir = in_data_dir + 'vectorData/geoJson/'
    im3Dir = in_data_dir + '3band/'
    im8Dir = in_data_dir + '8band/'
    
    # set out dir for image preprocessing 
    imDir = in_data_dir + imtype + '/'

    # import functions
    sys.path.append('/cosmiq/sivnet/src/')
    import sivnet_data_prep
    reload(sivnet_data_prep)
    from sivnet_data_prep import get_contours, plot_contours
    path_to_spacenet_utils = '/cosmiq/git/ave/spacenet_explore/'
    sys.path.extend([path_to_spacenet_utils])
    import spacenet_utilities
    sys.path.append(yolt_dir + 'scripts/')
    import convert
    

    # make dirs
    for f in [images_dir, labels_dir, sample_label_vis_dir, sample_label_vis_dir2]:
        if not os.path.exists(f):
            os.mkdir(f)
    
    #%%    
    ###########################################################################
    # Explore random mask (_vis should have 1 band)
    if new_schema:
        mroot = '3band_AOI_1_RIO_img1011.tif'
    else:
        mroot = '3band_013022223131_Public_img1014.tif'
    m3_file = im3Dir + mroot
    m8_file = im8Dir + '8' + mroot[1:]
    geojson_file = vecDir + mroot[6:-4] + '_Geo.geojson'
    os.chdir(in_data_dir)
    
    print "Sat file (3 band pan sharpened)"
    cmd_t = 'gdalinfo ' + m3_file
    print subprocess.check_output(cmd_t, shell=True)

    print "Sat file (8 band)"
    cmd_t = 'gdalinfo ' + m8_file
    print subprocess.check_output(cmd_t, shell=True)
    
    # explore 8 band
    print "\nCombine 3 band and 8 band..."
    hnew, wnew = 400, 400
    arrt = comb_3band_8band(m3_file, m8_file, wnew, hnew, 
                     verbose=True, show_cv2=False)    
    
    # convert numpy array to geotiff
    hnew, wnew, bandnew = cv2.imread(m3_file).shape
    arrt = comb_3band_8band(m3_file, m8_file, wnew, hnew, 
                     verbose=True, show_cv2=False)    

    test_dir = in_data_dir + 'exploration/'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    outfile = test_dir + mroot + '_np_to_geotiff_test1.tiff'
    np_to_geotiff(arrt, outfile, im_for_geom=m3_file)
    
    # now try each band
    im_refine_method = 'std'#'std'#, 'hist', 'uint16'
    nstd = 2
    for j in range(0, arrt.shape[-1]):
        band = arrt[:, :, j]
        # first three band are pan-sharpened and already scaled
        if j < -3:#3:
            band_rescale = band
        # rescale other bands
        else:
            band_rescale = rescale_intensity(band, nstd=nstd, method=im_refine_method, 
                                             out_range='uint8', verbose=False)
    
        outfile_tmp = outfile.split('.')[0] + '_' + str(j+1) + '.tif'
        # reshape to have 1 band
        new_shape = (band.shape[0], band.shape[1], 1)
        band_reshape = band_rescale.reshape(new_shape)
        cv2.imwrite(outfile_tmp, band_rescale.astype('uint8'))
        #np_to_geotiff(band_reshape, outfile_tmp, im_for_geom=m3_file)
        

    #%%
    
    # create 8 band combined images or indiviual bands
    reextract_bands = False#True
    band_delim = '#'
    iband_as_3band = True#False
    im_refine_method = 'std'      # options: ['std', 'hist', 'uint16', False]
                                  # see rescale_intensity()
    nstd = 3
    save_as_npy = False
    save_only_3band = False#True

    if imtype == '8band_3pan':
        out_ext = '.tif'
    else:
        out_ext = '.png'
    
    if (reextract_bands) and (imtype in ['8band_3pan', 'iband3', 'iband']):
        
        print "/nCreate", imtype,  "files..."
        # make dir
        if not os.path.exists(imDir):
            os.mkdir(imDir)
        # iterate through names
        for i,mroot in enumerate(os.listdir(im3Dir)):

            m3_file = im3Dir + mroot
            m8_file = im8Dir + '8' + mroot[1:]
            # retrieve size of 3 band pan-sharpened file
            gray_tmp = cv2.imread(m3_file, 0)
            htmp, wtmp = gray_tmp.shape[:2]
            # get combined file
            arrt = comb_3band_8band(m3_file, m8_file, wtmp, htmp, 
                                    rgb_first=True,
                                    verbose=False, show_cv2=False)
            if (i % 50) == 0:
                print i, mroot
                print "arrt.shape:", arrt.shape

            # save as np array if desired
            if save_as_npy:
                outfile = imDir + mroot.split('.')[0] + '.npy'
                np.save(outfile, arrt)
                continue

            ############
            # rescale, if desired 
            #arrt_rot = np.rollaxis(arrt, 2, 0)    # for saving with tifffile
            # change type?
            if im_refine_method:
                # http://scikit-image.org/docs/dev/user_guide/data_types.html#rescaling-intensity-values
                list_rescale = []
                # for saving with tifffile
                #for j in range(0, arrt_rot.shape[0]):
                #     band = arrt_rot[j] 
                for j in range(0, arrt.shape[-1]):
                    band = arrt[:, :, j]
                    # first three band are pan-sharpened and already scaled
                    if j < 3:
                        band_rescale = band
                    else:
                        # else, rescale other bands
                        band_rescale = rescale_intensity(band, nstd=nstd, 
                                                     method=im_refine_method, 
                                                     out_range='uint8',
                                                     verbose=False)
                        #band_rescale = exposure.rescale_intensity(band, 
                        #                        in_range=('uint16'), 
                        #                        out_range=('uint8'))
                    list_rescale.append(band_rescale.astype('uint8'))
                    
                #arr_rescale = np.array(arr_rescale).astype('uint8') # tifffile
                arr_rescale = np.dstack(list_rescale)#.astype('uint8')

                # lazy way, just crop
                #arrt_rot = arrt_rot.astype('uint8')
            else:
                arr_rescale = arrt       # arrt_rot # if using tifffile          

            if (i % 50) == 0:
                print "arr_rescale.shape:", arr_rescale.shape
            ############ 
            
            ############
            # save to file.... 
                                       
            # test saves only first three bands (rgb) to see if these
            # can be visualized correctly in qgis
            if save_only_3band:
                np_to_geotiff(arr_rescale[:,:,:3], outfile, im_for_geom=m3_file)
                #tiff.imsave(outfile, arr_rescale[:3])
            
            else:
                if imtype == '8band_3pan':
                    # save 8band file
                    outfile = imDir + '8' + mroot.split('.')[0][1:] + out_ext
                    np_to_geotiff(arr_rescale, outfile, im_for_geom=m3_file)
                    #tiff.imsave(outfile, arr_rescale)  
                    
                elif imtype == 'iband3':
                    # save individual bands in groupds of 3
                    nbandtmp = arr_rescale.shape[-1]
                    outfile = imDir + 'i' + mroot.split('.')[0][1:] + out_ext
                    nout = 1                            
                    for k in range(0, nbandtmp, 3):
                        #print "k", k
                        outfile_tmp = outfile.split('.')[0] + band_delim + str(nout) + out_ext
                        nout += 1
                        # get 1st band
                        banda = arr_rescale[:, :, k]
                        # get 2nd band
                        try:
                            bandb = arr_rescale[:, :, k+1]
                        except:
                            bandb = np.zeros(np.shape(arr_rescale[:, :, k]))
                            #print "bandb == banda"
                        # get 3rd band
                        try:
                            bandc = arr_rescale[:, :, k+2]
                        except:
                            bandc = np.zeros(np.shape(arr_rescale[:, :, k]))#arr_rescale[:, :, k]
                            #print "bandc == banda"
                        # combine into three - band image, reverse order since
                        # cv2 used bgr instead of rgb
                        out_im = np.dstack((bandc, bandb, banda))
                        
                        # save with opencv
                        cv2.imwrite(outfile_tmp, out_im)


                elif imtype == 'iband':
                    # save individual bands
                    outfile = imDir + 'i' + mroot.split('.')[0][1:] + out_ext
                    for k in range(0, arr_rescale.shape[-1]):
                        outfile_tmp = outfile.split('.')[0] + band_delim + str(k+1) + out_ext
                        band = arr_rescale[:, :, k]
                        
                        if iband_as_3band:
                            # save as 3band?
                            band_out = np.dstack((band, band, band))
                            cv2.imwrite(outfile_tmp, band_out)
                        else:
                            # save with opencv
                            cv2.imwrite(outfile_tmp, band)
                        
                        ## save with gdal
                        ## reshape to have 1 band
                        #new_shape = (band.shape[0], band.shape[1], 1)
                        #band_reshape = band.reshape(new_shape)
                        #np_to_geotiff(band_reshape, outfile_tmp, im_for_geom=m3_file)
                        
            ############

#            # save to file using tifffile, instead of gdal (ignores geo data) 
#            if save_as_tiff:
#                outfile = imDir + '8' + mroot.split('.')[0][1:] + '.tif'
#                arrt_rot = np.rollaxis(arrt, 2, 0)
#                # change type?
#                if rescale_to_uint8:
#                    # http://scikit-image.org/docs/dev/user_guide/data_types.html#rescaling-intensity-values
#                    arr_rescale = []
#                    for j in range(0, arrt_rot.shape[0]):
#                        band = arrt_rot[j]
#                        if j < 3:
#                            arr_rescale.append(band)
#                            continue
#                        band_rescale = exposure.rescale_intensity(band, 
#                                                in_range=('uint16'), 
#                                                out_range=('uint8'))
#                        arr_rescale.append(band_rescale)
#                    arr_rescale = np.array(arr_rescale).astype('uint8')
#                    # lazy way, just crop
#                    #arrt_rot = arrt_rot.astype('uint8')
#                else:
#                    arr_rescale = arrt_rot                
#                
#                # test saves only first three bands (rgb) to see if these
#                # can be visualized correctly in qgis
#                if test_save_as_tiff:
#                    arr_new = arr_rescale[:3]
#                    tiff.imsave(outfile, arr_new)
#                
#                else:
#                    tiff.imsave(outfile, arr_rescale)                
                    
        #if i > 35:
        #    break

    #%%
    # get test and train ground truth coords for 3band
    
    rerun_extract_traindata = False
    test_size = 0.1
    
    rasterList_tot = glob.glob(os.path.join(im3Dir, '*.tif'))
    # split to train-test
    rasterList_train, rasterList_test = train_test_split(rasterList_tot, 
                                                         test_size=test_size)

    # output of get_yolt_coords_spacenet is:
    #   rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, 
    #       cont_plot_box
    if rerun_extract_traindata:
        # train
        out_train_list = []
        for i,rasterSrc in enumerate(rasterList_train):   
            if (i % 50) == 0:
                print i, "/", len(rasterList_train)
            row = get_yolt_coords_spacenet(rasterSrc, vecDir, new_schema=False,
                             pixel_ints=True, verbose=False)
            out_train_list.append(row)
        pickle.dump(out_train_list, open(pkl_name_train, 'wb'), protocol=2)

        # test
        out_test_list = []
        for i,rasterSrc in enumerate(rasterList_test):
            if (i % 50) == 0:
                print i , "/", len(rasterList_test)
            row = get_yolt_coords_spacenet(rasterSrc, vecDir, new_schema=False,
                             pixel_ints=True)
            out_test_list.append(row)
        pickle.dump(out_test_list, open(pkl_name_test, 'wb'), protocol=2)
            
            
#    if rerun_extract:
#        # train
#        out_list_train = get_yolt_coords_spacenet_v0(rasterList_train, im3Dir, 
#                                                  vecDir, maskDir='',
#                                                  new_schema=False,
#                                                  outDir=imDir,
#                                                  use8band=use8band)   
#        pickle.dump(out_list_train, open(pkl_name_train, 'wb'), protocol=2)
#
#        # test
#        out_list_test = get_yolt_coords_spacenet_v0(rasterList_test, im3Dir, 
#                                                 vecDir, maskDir='', 
#                                                 new_schema=False,
#                                                 outDir=imDir,
#                                                 use8band=use8band)   
#        pickle.dump(out_list_test, open(pkl_name_test, 'wb'), protocol=2)

    #%%

    # load bounding boxes (for YOLT)
    
    # first load the pickle
    #pkl_name_train = '/cosmiq/yolt2/training_datasets/spacenet/ims_coords_list_train_3band.pkl'
    print "Loading", pkl_name_train, "..."
    t0 = time.time()
    building_list_train = pickle.load(open(pkl_name_train, 'rb'))
    print "Time to load pickle:", time.time() - t0, "seconds"

    # if data moves, change locations in pkl
    rename_loc = False#True
    if rename_loc:
        out = []
        for i,row in enumerate(building_list_train):
            [rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, 
             cont_plot_box] = row
             
            if (i % 50) == 0:
                print i, "Src:", rasterSrc
                
            rast_out = rasterSrc.replace('/cosmiq/blupr_net/spacenet_data/',
                                         '/cosmiq/spacenet/data/spacenetFromAWS_08122016/processedData/spacenet_data/')
            vec_out = vectorSrc.replace('/cosmiq/blupr_net/spacenet_data/',
                                         '/cosmiq/spacenet/data/spacenetFromAWS_08122016/processedData/spacenet_data/')
            row_out = [rast_out, vec_out, pixel_coords, latlon_coords, yolt_coords, 
             cont_plot_box]
            out.append(row_out)
        pickle.dump(out, open(pkl_name_train, 'wb'), protocol=2)

                                           
                                           
    
    #%%
#    # convert to yolt format (deprecated)
#    # make directories
#    for tmp_dir in [train_base_dir, t_dir, labels_dir, images_dir,
#                    sample_label_vis_dir]:
#        try: os.mkdir(tmp_dir)
#        except: print ""                       
#           
#    yolt_list_train = spacenet_yolt_setup(building_list_train, classes_dic, 
#                        labels_dir, images_dir,
#                        sample_label_vis_dir, train_images_list_file,
#                        deploy_dir, dl=0.8)            
#    #building list has length: 6467 though only 3926 images were processed, 
#    #the remainder are empty
#    #Time to setup training data for: 
#    #    /cosmiq/yolt2/training_datasets/spacenet/training_data/images/ 
#    #    of length: 6467 59.3504288197 
#
#    # cp _list.txt to data dir
#    shutil.copy(train_images_list_file, train_images_list_file_loc)
#  
#    # save file
#    pickle.dump(yolt_list_train, open(pkl_name_yolt, 'wb'), protocol=2)
#
#    # skip augment for now, since we'll need to recreate building_list_train
#    #if run_augment:
#    #    augment_ims = augment_training_data(labels_dir, images_dir)
#    #    # update image list
#    #    im_list_loc = train_base_dir + train_name + '_list.txt'
#    #    f = open(im_list_loc, 'wb')
#    #    for item in augment_ims:
#    #        end = item.split('/')[-1]
#    #        out = im_locs_for_list + end
#    #        f.write("%s\n" % out)
#    #    f.close()
      
    # copy files to appropriate locales
    spacenet_yolt_setup(building_list_train, classes_dic, imDir,
                        labels_dir, images_dir,
                        train_images_list_file,
                        deploy_dir, imtype, 
                        maskDir=maskDir,
                        sample_mask_vis_dir=sample_label_vis_dir2)
    # copy to data dir
    #shutil.copy(train_images_list_file, train_images_list_file_loc)

    #%%
    ###############################################################################
    # plot box labels
    max_plots=40
    thickness=1
    plot_training_bboxes(labels_dir, images_dir, ignore_augment=False,
                         sample_label_vis_dir=sample_label_vis_dir, 
                         max_plots=max_plots, thickness=thickness, ext='.png')
    #plt.close("all")

    #%%
    # Downsample
    indir = t_dir + 'images_3band/'
    outdir = t_dir + 'images_3band_2x/'
    inGSD = 0.45
    outGSD = 0.90
    rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=True)

    #%%
    # Downsample graound truth
    indir = '/cosmiq/spacenet/data/spacenetv2/RGB-PanSharpen_8bit/'
    outdir = '/cosmiq/spacenet/data/spacenetv2/RGB-PanSharpen_8bit_0.5mGSD/'
    inGSD = 0.32
    outGSD = 0.50
    rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=True)
    
#%%

"""
Refine and create test images from topcoder competition
"""

indir_test  = '/cosmiq/spacenet/data/spacenet_TestData_topcoder/'
dir_geojson = indir_test+'vectordata/geojson/'
im3Dir = indir_test + '3band/'
im8Dir = indir_test + '8band/'


# make tmp dirs
dir_3tmp = indir_test + '3band_extra/'
dir_8tmp = indir_test + '8band_extra/'
if not os.path.exists(dir_3tmp): os.mkdir(dir_3tmp)
if not os.path.exists(dir_8tmp): os.mkdir(dir_8tmp)

os.chdir(indir_test)

# get list of geojsons, for form Geo_AOI_2_RIO_img15.geojson
# keep everything after Geo_  and before '.'

json_list0 = os.listdir(dir_geojson)
geojson_list = [g[4:].split('.')[0] for g in json_list0 if g.endswith('json')]
geojson_set = set(geojson_list)

# loop through 3band and remove extra files
# file is of form 3band_AOI_2_RIO_img2.tif
for f in os.listdir(im3Dir):
    froot = f[6:].split('.')[0]
    if froot not in geojson_set:
        shutil.move(im3Dir + f, dir_3tmp)
    
# loop through 8band and remove extra files
# file is of form 3band_AOI_2_RIO_img2.tif
for f in os.listdir(im8Dir):
    froot = f[6:].split('.')[0]
    if froot not in geojson_set:
        shutil.move(im8Dir+ f, dir_8tmp)
        
#%%
########
# create iband files
    # create 8 band combined images or indiviual bands
    imDir = indir_test + 'images_iband3/'
    imtype = 'iband3'
    # create 8 band combined images or indiviual bands
    band_delim = '#'
    iband_as_3band = True#False
    im_refine_method = 'std'      # options: ['std', 'hist', 'uint16', False]
                                  # see rescale_intensity()
    nstd = 3
    out_ext = '.png'
    
    if (imtype in ['8band_3pan', 'iband3', 'iband']):
        
        print "/nCreate", imtype,  "files..."
        # make dir
        if not os.path.exists(imDir):
            os.mkdir(imDir)

        # iterate through names
        for i,mroot in enumerate(os.listdir(im3Dir)):

            m3_file = im3Dir + mroot
            m8_file = im8Dir + '8' + mroot[1:]
            # retrieve size of 3 band pan-sharpened file
            gray_tmp = cv2.imread(m3_file, 0)
            htmp, wtmp = gray_tmp.shape[:2]
            # get combined file
            arrt = comb_3band_8band(m3_file, m8_file, wtmp, htmp, 
                                    rgb_first=True,
                                    verbose=False, show_cv2=False)
            if (i % 50) == 0:
                print i, mroot
                print "arrt.shape:", arrt.shape

            ############
            # rescale, if desired 
            #arrt_rot = np.rollaxis(arrt, 2, 0)    # for saving with tifffile
            # change type?
            if im_refine_method:
                # http://scikit-image.org/docs/dev/user_guide/data_types.html#rescaling-intensity-values
                list_rescale = []
                # for saving with tifffile
                for j in range(0, arrt.shape[-1]):
                    band = arrt[:, :, j]
                    # first three band are pan-sharpened and already scaled
                    if j < 3:
                        band_rescale = band
                    else:
                        # else, rescale other bands
                        band_rescale = rescale_intensity(band, nstd=nstd, 
                                                     method=im_refine_method, 
                                                     out_range='uint8',
                                                     verbose=False)
                    list_rescale.append(band_rescale.astype('uint8'))
                    
                arr_rescale = np.dstack(list_rescale)#.astype('uint8')

            else:
                arr_rescale = arrt       # arrt_rot # if using tifffile          

            if (i % 50) == 0:
                print "arr_rescale.shape:", arr_rescale.shape
            ############ 
            
            ############
            # save to file.... 
                                                   
            if imtype == '8band_3pan':
                # save 8band file
                outfile = imDir + '8' + mroot.split('.')[0][1:] + out_ext
                np_to_geotiff(arr_rescale, outfile, im_for_geom=m3_file)
                #tiff.imsave(outfile, arr_rescale)  
                
            elif imtype == 'iband3':
                # save individual bands in groupds of 3
                nbandtmp = arr_rescale.shape[-1]
                outfile = imDir + 'i' + mroot.split('.')[0][1:] + out_ext
                nout = 1                            
                for k in range(0, nbandtmp, 3):
                    #print "k", k
                    outfile_tmp = outfile.split('.')[0] + band_delim + str(nout) + out_ext
                    nout += 1
                    # get 1st band
                    banda = arr_rescale[:, :, k]
                    # get 2nd band
                    try:
                        bandb = arr_rescale[:, :, k+1]
                    except:
                        bandb = np.zeros(np.shape(arr_rescale[:, :, k]))
                        #print "bandb == banda"
                    # get 3rd band
                    try:
                        bandc = arr_rescale[:, :, k+2]
                    except:
                        bandc = np.zeros(np.shape(arr_rescale[:, :, k]))#arr_rescale[:, :, k]
                        #print "bandc == banda"
                    # combine into three - band image, reverse order since
                    # cv2 used bgr instead of rgb
                    out_im = np.dstack((bandc, bandb, banda))
                    
                    # save with opencv
                    cv2.imwrite(outfile_tmp, out_im)


            elif imtype == 'iband':
                # save individual bands
                outfile = imDir + 'i' + mroot.split('.')[0][1:] + out_ext
                for k in range(0, arr_rescale.shape[-1]):
                    outfile_tmp = outfile.split('.')[0] + band_delim + str(k+1) + out_ext
                    band = arr_rescale[:, :, k]
                    
                    if iband_as_3band:
                        # save as 3band?
                        band_out = np.dstack((band, band, band))
                        cv2.imwrite(outfile_tmp, band_out)
                    else:
                        # save with opencv
                        cv2.imwrite(outfile_tmp, band)


    #%%      
###############################################################################

    ###############################################################################
    # copy test data to yolt testing dir
    # now try getting bounding boxes (for YOLT)
    N_test_ims = 10
    test_dir = yolt_dir + 'test_data/'

    # load file list pickle
    print "Loading", pkl_name_train, "..."
    t0 = time.time()
    building_list_test = pickle.load(open(pkl_name_test, 'rb'))
    print "Time to load pickle:", time.time() - t0, "seconds"
    
    im_list = []
    # copy to test dir
    for row in building_list_test[15:15+N_test_ims]:
        name_root, im_loc = row[0], row[1]
        outroot = 'spacenet_' + name_root 
        outname_tot = test_dir + outroot + '.tif'
        shutil.copy(im_loc, outname_tot)  
        im_list.append(outroot)
    
    print "im_list:", im_list

'''
    ['spacenet_3band_013022223132_Public_img2328',
     'spacenet_3band_013022232020_Public_img4330',
     'spacenet_3band_013022232022_Public_img6611',
     'spacenet_3band_013022223133_Public_img3469',
     'spacenet_3band_013022223132_Public_img2190',
     'spacenet_3band_013022223132_Public_img2105',
     'spacenet_3band_013022223132_Public_img2396',
     'spacenet_3band_013022223130_Public_img235',
     'spacenet_3band_013022223133_Public_img2940',
     'spacenet_3band_013022232020_Public_img5034']
    
'''

#%%
###############################################################################
# create a test set with the portion of data not used in training, but the 
# same AOI as the train set (the formal SpaceNet training data is a 
# different AOI so the scores will be lower)
    
test_dir = yolt_dir + 'test_data/'
N_test_ims = 100
truth_file = in_data_dir + 'vectorData/summaryData/public_polygons_solution_3Band.csv'
new_truth_file = train_base_dir + 'public_polygons_solution_3Band_test_subset' + str(N_test_ims) + '.csv'

# read in test list
building_list_test = pickle.load(open(pkl_name_test, 'rb'))
# read in ground truth file
df = pd.read_csv(truth_file)
# iterate through building_list_test and create set of image names
im_roots = []
for i,row in enumerate(building_list_test):
    if i > N_test_ims + 1:
        break
    im_path = row[1]
    im_root = row[0][6:]
    im_roots.append(im_root)
    # copy image to yolt test dir
    shutil.copy(im_path, test_dir)
im_roots_set = set(im_roots[:N_test_ims])
print "Number of test images:", len(im_roots_set)

# iterate through dataframe and keep items in im_roots_set
df_new = df.loc[df['ImageId'].isin(im_roots_set)]
df_new.to_csv(new_truth_file, index=False)

name_list = ['3band_' + t for t in im_roots_set]
print "name_list:", name_list


'''
['013022232022_Public_img6257', '013022232022_Public_img6417', '013022232022_Public_img6250', '013022223132_Public_img1951', '013022223133_Public_img2940', '013022232020_Public_img4977', '013022223132_Public_img2672', '013022223132_Public_img2779', '013022223131_Public_img1492', '013022232020_Public_img4873', '013022232020_Public_img4164', '013022232020_Public_img4464', '013022232020_Public_img4371', '013022223133_Public_img3233', '013022223131_Public_img1191', '013022232023_Public_img6740', '013022232020_Public_img4547', '013022232023_Public_img6808', '013022232200_Public_img7163', '013022223132_Public_img2075', '013022223131_Public_img1487', '013022232020_Public_img4665', '013022232020_Public_img4668', '013022223130_Public_img567', '013022223133_Public_img3593', '013022223132_Public_img2712', '013022223133_Public_img3750', '013022232020_Public_img4802', '013022223132_Public_img2076', '013022223132_Public_img2652', '013022232022_Public_img5630', '013022223131_Public_img1437', '013022232020_Public_img4259', '013022232020_Public_img4652', '013022232022_Public_img6348', '013022223132_Public_img2396', '013022232022_Public_img5089', '013022232200_Public_img7031', '013022232020_Public_img4834', '013022232020_Public_img4830', '013022223131_Public_img839', '013022232022_Public_img5834', '013022223133_Public_img3694', '013022223133_Public_img3697', '013022223133_Public_img3512', '013022223132_Public_img2792', '013022232020_Public_img4531', '013022223131_Public_img1143', '013022223131_Public_img591', '013022232022_Public_img5092', '013022223132_Public_img2328', '013022223132_Public_img2033', '013022223132_Public_img2083', '013022223132_Public_img2085', '013022223133_Public_img3446', '013022232020_Public_img4928', '013022232022_Public_img5790', '013022232020_Public_img5034', '013022232023_Public_img6989', '013022232022_Public_img6611', '013022223131_Public_img583', '013022223132_Public_img1907', '013022223131_Public_img589', '013022232200_Public_img7099', '013022223131_Public_img1442', '013022223133_Public_img3172', '013022223133_Public_img3703', '013022223132_Public_img1535', '013022223132_Public_img2105', '013022232022_Public_img5332', '013022232020_Public_img4937', '013022223133_Public_img2976', '013022232020_Public_img4330', '013022223132_Public_img2435', '013022223132_Public_img2689', '013022223130_Public_img517', '013022223131_Public_img1378', '013022232020_Public_img4518', '013022223133_Public_img3387', '013022223132_Public_img2056', '013022232200_Public_img7142', '013022223130_Public_img235', '013022232022_Public_img6665', '013022223131_Public_img771', '013022223133_Public_img3469', '013022223132_Public_img2423', '013022223133_Public_img3398', '013022223133_Public_img3008', '013022232022_Public_img6280', '013022232022_Public_img5612', '013022223133_Public_img3005', '013022223132_Public_img2190', '013022223132_Public_img2103', '013022232020_Public_img4915', '013022223133_Public_img3097', '013022232022_Public_img5379', '013022223132_Public_img2690', '013022223132_Public_img2560', '013022223130_Public_img536', '013022223132_Public_img2514']
'''       

###############################################################################
#%%
### spacenet v2




#suff = 'AOI_2_Vegas_Train'
#suff = 'AOI_5_Khartoum_Train'
#suff = 'AOI_4_Shanghai_Train'
suff = 'AOI_3_Paris_Train'


loc_yolt = '/cosmiq/spacenet/data/spacenetv2/' + suff + '/'

classes_dic = {"building": 0}    
label_folder = loc_yolt + 'labels/'
label_folder8 = loc_yolt + 'labels8/'
image_folder = loc_yolt + 'images/'
image_folder8 = loc_yolt + 'images8/'
vecDir = loc_yolt + 'geojson/buildings/'
pkl_name_train = loc_yolt + 'ims_coords_list_' +  suff + '.pkl'
train_images_list_file = loc_yolt + 'spacenetv2_650_rgb_list_' + suff + '.txt'
train_images_list_file8 = loc_yolt + 'spacenetv2_650_list_' + suff + '_8band.txt'
#train_images_list_file8 = loc_yolt + 'spacenetv2_650_rgb_list_' + suff + '_pan.txt'
train_images_list_file_gray = loc_yolt + 'spacenetv2_650_rgb_list_' + suff + '_gray.txt'
deploy_dir = '/raid/local/src/yolt2/training_datasets/spacenetv2/' + suff + '/images/'
deploy_dir8 = '/raid/local/src/yolt2/training_datasets/spacenetv2/' + suff + '/images8/'
deploy_dir_gray = '/raid/local/src/yolt2/training_datasets/spacenetv2/' + suff + '/images_gray/'
deploy_dir_pan = '/raid/local/src/yolt2/training_datasets/spacenetv2/' + suff + '/images_pan/'
sample_label_vis_dir = loc_yolt + 'sample_label_vis_dir/' 


#dl = 0.9 # vegas, khartoum
dl = 0.9 # Shanghai
band_delim = '#'
imtype = '3band'# 'iband3'#'iband3' #'3band'#'iband3'#'3band'
if imtype == 'iband3':
    label_folder = label_folder8
    image_folder = image_folder8

for f in [image_folder, label_folder]:
    if not os.path.exists(f):
        os.mkdir(f)

# inspect
yolt_dir = '/cosmiq/yolt2/'
os.chdir(yolt_dir)
sys.path.append(yolt_dir + 'scripts')
import convert, slice_im
reload(convert)
reload(slice_im)

path_to_spacenet_utils = '/cosmiq/git/ave/spacenet_explore/'
sys.path.extend([path_to_spacenet_utils])
import spacenet_utilities
reload(spacenet_utilities)

#%%

#d = label_folder
#for f in os.listdir(d):
#    roots = f.split('_')
#    newroots = roots[:4] + ['8bit'] + [roots[-1]]
#    fout = d + '_'.join(newroots)
#    shutil.move(d+f, fout)
#
#sample_label_vis_dir = loc_yolt + 'sample_label_vis_dir/'
if not os.path.exists(sample_label_vis_dir): os.mkdir(sample_label_vis_dir)
#plot_training_bboxes(label_folder, image_folder, ignore_augment=True,
#                         figsize=(10,10), color=(0,0,255), thickness=2, 
#                         max_plots=100, 
#                         sample_label_vis_dir=sample_label_vis_dir, ext='.tif',
#                         verbose=True, show_plot=False)


# convert to 8bit?
to_8bit = True
if imtype == '3band':
    raw_ims_loc = loc_yolt + 'RGB-PanSharpen/'
    out_folder = image_folder
    #out_folder = loc_yolt + 'RGB-PanSharpen_8bit/'

elif imtype == 'iband3':
    raw_ims_loc = loc_yolt + 'MUL-PanSharpen/'
    out_folder = loc_yolt + 'MUL-PanSharpen_8bit/'

try: os.mkdir(out_folder)
except: print ''
if to_8bit:
    for i,im in enumerate(os.listdir(raw_ims_loc)):
        if (i % 100) == 0:
            print i, im
        rasterImageName = raw_ims_loc + im
        outputRaster = out_folder + im
        convertTo8Bit(rasterImageName, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff')
        # split 8 bband
        if raw_ims_loc.endswith('MUL-PanSharpen/'):
            split_8band(outputRaster, image_folder8, band_delim=band_delim,
                        out_ext='.png')
        
    # remove .xml.
    for zippath in glob.iglob(os.path.join(image_folder, '*.xml')):
        os.remove(zippath)
        
    # remove _8bit folder
    if imtype == 'iband3':
        shutil.rmtree(out_folder, ignore_errors=True)

    # make rgb 8bit folder?
    shutil.copytree(image_folder, loc_yolt + 'RGB-PanSharpen_8bit/')



###
#%%
# create labels
rerun_extract_traindata = True
rasterList_train = glob.glob(os.path.join(image_folder, '*.tif'))
if imtype == 'iband3':
    rasterList_train = glob.glob(os.path.join(raw_ims_loc, '*.tif'))
        

# output of get_yolt_coords_spacenet is:
#   rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, 
#       cont_plot_box
if rerun_extract_traindata:
    # train
    out_train_list = []
    for i,rasterSrc in enumerate(rasterList_train):   
        if (i % 50) == 0:
            print i, "/", len(rasterList_train)
        row = get_yolt_coords_spacenet(rasterSrc, vecDir, new_schema=False,
                         pixel_ints=True, verbose=False, dl=dl)
        out_train_list.append(row)
    pickle.dump(out_train_list, open(pkl_name_train, 'wb'), protocol=2)

# complete setup
print "Loading", pkl_name_train, "..."
t0 = time.time()
building_list_train = pickle.load(open(pkl_name_train, 'rb'))
print "Time to load pickle:", time.time() - t0, "seconds"

# copy files to appropriate locales
label_folder_orig = label_folder[:-1]+'_orig/'
spacenet_yolt_setup(building_list_train, classes_dic, image_folder,
                    label_folder_orig, image_folder,
                    train_images_list_file,
                    deploy_dir, '3band', 
                    maskDir='', sample_mask_vis_dir='')
#
#%%
# remove any boxes with negligible height or width (this may be causing the
#   program to terminate early)
min_width = 0.005
label_folder_orig = label_folder[:-1]+'_orig/'
#if os.path.exists(label_folder):
#    shutil.move(label_folder, label_folder_orig)
#    os.mkdir(label_folder)
try:    os.mkdir(label_folder)
except: print ""
n_files = len(os.listdir(label_folder_orig))
n_cropped = 0
n_deleted = 0
for f in os.listdir(label_folder_orig):
    if not f.endswith('.txt'):
        continue
    ftot = label_folder_orig + f
    df = pd.read_csv(ftot, sep=' ', names=['class', 'x', 'y', 'w', 'h'])
    # keep only rows of desired size
    dfw = df[df.w >= min_width]
    dfwh = dfw[dfw.h >= min_width]
    if len(dfwh) < len(df):
        n_cropped += 1
        print f
        print "  len df:", len(df)
        print "  len dfnew:", len(dfwh)
    # print
    if len(dfwh) > 0:
        outfile = label_folder + f
        dfwh.to_csv(outfile, index=False, header=False, sep=' ')
    else:
        n_deleted += 1
print "n_files:", n_files
print "n_cropped:", n_cropped
print "n_deleted:", n_deleted

#n_files: 3617
#n_cropped: 1183
#n_deleted: 3

#%%
# move unlabeled images (3band)
im_null_dir = loc_yolt + 'images_null_' + imtype + '/'
if not os.path.exists(im_null_dir): os.mkdir(im_null_dir)
if imtype != 'iband3':
    for im in os.listdir(image_folder):
        im_tot = os.path.join(image_folder, im)
        lab_tot = im_tot.split('.')[0].replace('images', 'labels') + '.txt'
        if imtype == 'iband3':
            lab_tot = lab_tot.split('#')[0] + '.txt'
        if not os.path.exists(lab_tot):
            shutil.move(im_tot, im_null_dir)
    
    # remake list
    im_locs = [deploy_dir + f for f in os.listdir(image_folder) if f.endswith('.tif')]
    f = open(train_images_list_file, 'wb')
    for item in im_locs:
        f.write("%s\n" % item)
    f.close()
    shutil.copy(train_images_list_file, '/cosmiq/yolt2/data')

#%%
# 8 band

if imtype == 'iband3':
    # rename labels
    try: os.mkdir(label_folder8)
    except: print ""
    for lab_tmp in os.listdir(label_folder):
        if not lab_tmp.endswith('.txt'):
            continue
        if not band_delim in lab_tmp:
            lnew = 'MUL' + lab_tmp.split('.')[0][3:] + band_delim + '1.txt'
            shutil.copy(label_folder + lab_tmp, label_folder8 + lnew)
    
    # remove anything that doesn't end with #1.txt
    for f in os.listdir(label_folder8):
        if not f.endswith(band_delim + '1.txt'):
            os.remove(label_folder8 + f)
    
    
    # remove images 
    for im in os.listdir(image_folder8):
        if not im.startswith('MUL-'):
            continue
        im_tot = os.path.join(image_folder8, im)
        im_root0 = im.split('/')[-1].split('.')[0][:-2]
        lab_tot = label_folder8 + im_root0  + band_delim + '1.txt'
        if not os.path.exists(lab_tot):
            shutil.move(im_tot, im_null_dir)    
    
    # remake list
    im_locs = [deploy_dir8 + f for f in os.listdir(image_folder8) if f.endswith('1.png')]
    f = open(train_images_list_file8, 'wb')
    for item in im_locs:
        f.write("%s\n" % item)
    f.close()
    shutil.copy(train_images_list_file8, '/cosmiq/yolt2/data')


# create grayscale from rgb (imtype should be '3band')
make_grayscale = True
if make_grayscale and imtype == '3band':
    gray_dir = loc_yolt + 'images_gray/'
    if not os.path.exists(gray_dir):
        os.mkdir(gray_dir)
    for i,im in enumerate(os.listdir(image_folder)):
        if (i % 100) == 0:
            print i, im
        rasterImageName = image_folder + im
        outputRaster = gray_dir + im
        # read rgb as gray, then output
        imgray = cv2.imread(rasterImageName,0)
        cv2.imwrite(outputRaster, imgray)
        
    # remake list
    im_locs = [deploy_dir_gray + ftmp for ftmp in os.listdir(gray_dir) if ftmp.endswith('.tif')]
    f = open(train_images_list_file_gray, 'wb')
    for item in im_locs:
        f.write("%s\n" % item)
    f.close()
    shutil.copy(train_images_list_file_gray, '/cosmiq/yolt2/data')
    
    # duplicate labels/ to labels_gray/
    #label_folder = loc_yolt + 'labels/'
    shutil.copytree(label_folder, loc_yolt + 'labels_gray/')


#%%
# plot box labels
max_plots=40
thickness=1
specific_labels = ['RGB-PanSharpen_AOI_2_Vegas_8bit_img1500.txt']
#specific_labels = ['RGB-PanSharpen_AOI_2_Vegas_8bit_img148.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img4994.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1743.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img4906.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img2176.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img4787.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img5807.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img4987.txt']
#specific_labels = ['RGB-PanSharpen_AOI_2_Vegas_8bit_img3646.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img174.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img319.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1791.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img3095.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img6321.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img569.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img2988.txt']
#specific_labels = ['RGB-PanSharpen_AOI_2_Vegas_8bit_img6266.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img2328.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1520.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1774.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img3373.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1255.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img6062.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img322.txt']
#specific_labels=['RGB-PanSharpen_AOI_2_Vegas_8bit_img884.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img3082.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1359.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img2359.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1334.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img1259.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img3990.txt',
#'RGB-PanSharpen_AOI_2_Vegas_8bit_img6285.txt']
specific_labels = ['RGB-PanSharpen_AOI_2_Vegas_8bit_img998.txt']

plot_training_bboxes(label_folder, image_folder, ignore_augment=False,
                     sample_label_vis_dir=sample_label_vis_dir, 
                     max_plots=max_plots, thickness=thickness, ext='.tif',
                     #specific_labels=specific_labels)
                     specific_labels=[])



#%%

# data location:
# /raid/data/CosmiQ_Challenge_FY2016/TopCoder_Challenge2/spacenetV2_TrainAnnotation650/AOI_2_Vegas_Train/annotations/images
#mv large number of files:
#	ls RGB-PanSharpen__-115.*.tif | sudo xargs -i mv {} images
#	ls RGB-PanSh*.txt | sudo xargs -i mv {} labels

data_loc = '/raid/data/CosmiQ_Challenge_FY2016/TopCoder_Challenge2/spacenetV2_TrainAnnotation544/AOI_2_Vegas_Train/annotations/'
# get list of tifs
yolt_dir = '/raid/local/src/yolt2/'
train_images_list_file_loc = yolt_dir + 'data/'
name = 'spacenetv2_544_rgb'#'spacenetv2_650_rgb'
im_list_name = train_images_list_file_loc + name + '_list.txt'

# copy to train dir
output_loc = '/raid/local/src/yolt2/training_datasets/spacenetv2_544/'  # eventual location of files 
if not os.path.exists(output_loc): os.mkdir(output_loc)
shutil.copytree(data_loc + 'images', output_loc+'images')
shutil.copytree(data_loc + 'labels', output_loc+'labels')

# rename labels
#d = output_loc+'labels/'
#for f in os.listdir(d):
#    roots = f.split('_')
#    newroots = roots[:4] + ['8bit'] + [roots[-1]]
#    fout = d + '_'.join(newroots)
#    shutil.move(d+f, fout)

im_locs = [output_loc+'images/' + ftmp for ftmp in os.listdir(output_loc+'images') if ftmp.endswith('.tif')]
f = open(im_list_name, 'wb')
for item in im_locs:
    f.write("%s\n" % item)
f.close()
 
#%%
### Spacenet  test data

# convert test to grayscale (dev box)
import os, cv2


tdir = '/cosmiq/yolt2/test_images/spacenetv2/AOI_4_Shanghai/'
tloc = tdir + 'jpgs/'
ndir = tdir + 'gray/'

os.chdir(tdir)
if not os.path.exists(ndir): os.mkdir(ndir)
for f in os.listdir(tloc):
    print f
    if f.endswith('.jpg'):
        im = cv2.imread(tloc + f, 0)
        cv2.imwrite(ndir + f, im)
        

#%%    
###############################################################################
###############################################################################
if __name__ == "__main__":
    print "No honeymoon. This is business."
    main()
    
