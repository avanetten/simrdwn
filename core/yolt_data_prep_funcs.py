#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

These functiosn are useful for transforming data into the correct format for
YOLT

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
import time 
import random
import subprocess
import operator
#import pickle
#import tifffile as tiff
from skimage import exposure
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from subprocess import Popen, PIPE, STDOUT
#from matplotlib.collections import PatchCollection
from osgeo import gdal, osr, ogr #, gdalnumeric
#from matplotlib.patches import Polygon

#import geopandas as gpd
#import rasterio as rio
#import affine as af
#import shapely
#import gdal
#import os

#import random
#import time
#import pickle
#from PIL import Image


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
def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    '''
    Convert latitude, longitude coords to pixexl coords.
    From spacenet geotools
    '''

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)

###############################################################################
def run_cmd(cmd):
    p = Popen(cmd, stdout = PIPE, stderr = STDOUT, shell = True)
    while True:
        line = p.stdout.readline()
        if not line: break
        print ( line.replace('\n', '') )
    return

###############################################################################
def make_label_images(root_dir, new_labels=[]):
    '''Create new images of label names'''
    
    cwd = os.getcwd()
    
    # legacy0
    l0 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    # legacyl
    l1 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    # new
    l2 = ["boat", "dock", "boat_harbor", "airport", "airport_single", "airport_multi"]
    
    l = l0 + l1 + l2 + new_labels
    
    #for word in l:
    #    os.system("convert -fill black -background white -bordercolor white -border 4 -font futura-normal -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
    
    # change to label directory
    os.chdir(root_dir)
    for word in l:
        #os.system("convert -fill black -background white -bordercolor white -border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
        run_cmd("convert -fill black -background white -bordercolor white -border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
        run_cmd("convert -fill black -background white -bordercolor white -border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.png\""%(word, word))

    # change back to cwd
    os.chdir(cwd)  
    return


###############################################################################
def pair_im_vec_spacenet(rasterSrc, vecDir, new_schema=False):
    '''Pair image in rasterSrc with the appropriate geojson file'''
    
    # old example: 
    #   3band_013022223130_Public_img1.tif
    #   013022223130_Public_img1_Geo.geojson
    # get name root
    name_root0 = rasterSrc.split('/')[-1].split('.')[0]  
    #name_8band = im8Dir + '8' + name_root0[1:] + '.tif'
        
    # remove 3band or 8band prefix
    name_root = name_root0[6:]
           
    if not new_schema:
        vectorSrc = vecDir + name_root + '_Geo.geojson'
    else:
        # new naming schema:
        # 3band_AOI_1_RIO_img9.tif
        # Geo_AOI_1_RIO_img9.geojson
        vectorSrc = vecDir + 'Geo_' + name_root + '.geojson'
    
    return vectorSrc


###############################################################################
def pair_im_vec_spacenet_v2(rasterSrc, vecDir, new_schema=False):
    '''Pair image in rasterSrc with the appropriate geojson file
    buildings_AOI_2_Vegas_img1.geojson
    RGB-PanSharpen_AOI_2_Vegas_8bit_img1.tif
    '''
    
    name_root0 = rasterSrc.split('/')[-1].split('.')[0]  
    #name_8band = im8Dir + '8' + name_root0[1:] + '.tif'
        
    # remove 3band or 8band prefix
    name_root_l = name_root0.split('_')
    name_root = '_'.join([name_root_l[1], name_root_l[2], name_root_l[3], name_root_l[-1]])
           
    vectorSrc = vecDir + 'buildings_' + name_root + '.geojson'
    
    return vectorSrc


###############################################################################    
def geojson_to_pixel_arr(raster_file, geojson_file, acceptable_categories=[],
                         pixel_ints=True,
                         verbose=False):
    '''
    adapted from https://github.com/avanetten/spacenet_buildings_exploration
    specifically for poi data
    Tranform geojson file into array of points in pixel (and latlon) coords
    pixel_ints = 1 sets pixel coords as integers
    '''
    
    # load geojson file
    with open(geojson_file) as f:
        geojson_data = json.load(f)
    
    if len(acceptable_categories) > 0:
        acc_cat_set = set(acceptable_categories)
    else:
        acc_cat_set = set([])

    # load raster file and get geo transforms
    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())
        
    geom_transform = src_raster.GetGeoTransform()
    if verbose:
        print ("geom_transform:", geom_transform )
    
    # get latlon coords
    latlons = []
    poly_types = []
    categories = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        poly_type_tmp = feature['geometry']['type']
        cat_tmp = feature['properties']['spaceNetFeature']
        if verbose: 
            print ("features:", feature.keys() )
            print ("geometry:features:", feature['geometry'].keys() )
            #print "feature['geometry']['coordinates'][0]", z
        # save only desired categories
        if acc_cat_set:
            if cat_tmp in acc_cat_set:
                latlons.append(coords_tmp)
                poly_types.append(poly_type_tmp)
                categories.append(cat_tmp)
        # else save all categories
        else:
            latlons.append(coords_tmp)
            poly_types.append(poly_type_tmp)
            categories.append(cat_tmp)        
    
    # convert latlons to pixel coords
    pixel_coords = []
    latlon_coords = []
    for i, (cat, poly_type, poly0) in enumerate(zip(categories, poly_types, latlons)):
        
        if poly_type.upper() == 'POLYGON':
            poly=np.array(poly0)
            if verbose:
                print ("poly.shape:", poly.shape )
                
            # account for nested arrays
            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]
                
            poly_list_pix = []
            poly_list_latlon = []
            if verbose: 
                print ("poly", poly )
            for coord in poly:
                if verbose: 
                    print ("coord:", coord )
                lon, lat, z = coord 
                #px, py = gT.latlon2pixel(lat, lon, input_raster=src_raster, 
                px, py = latlon2pixel(lat, lon, input_raster=src_raster, 
                                     targetsr=targetsr, 
                                     geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                if verbose:
                    print ("px, py", px, py )
                poly_list_latlon.append([lat, lon])
            
            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)
            
        elif poly_type.upper() == 'POINT':
            print ("Skipping shape type: POINT in geojson_to_pixel_arr()" )
            continue
        
        else:
            print ("Unknown shape type:", poly_type, " in geojson_to_pixel_arr()" )
            return
            
    return categories, pixel_coords, latlon_coords


###############################################################################
def get_yolt_coords_spacenet(rasterSrc, vecDir, new_schema=False,
                             pixel_ints=True, dl=0.8, verbose=False):
    
    '''
    Take raster image as input, along with location of labels
    return:
        rasterSrc
        label file
        pixel coords of buildings
        latlon coords of buildings
        building coords converted to yolt coords
        building coords in pixel coords for plotting   
    
    dl =  fraction of size for bounding box
    '''
    
    vectorSrc = pair_im_vec_spacenet_v2(rasterSrc, vecDir, 
                                     new_schema=new_schema)  
    
    #if len(maskDir) > 0:
    #    name_root = rasterSrc.split('/')[-1]
    #    maskSrc = maskDir + name_root
    #else:
    #    maskSrc = ''

        
    
    # get size
    h,w,bands = cv2.imread(rasterSrc,1).shape

    if verbose:
        print ( "\nrasterSrc:", rasterSrc )
        print ("  vectorSrc:", vectorSrc )
        print ("  rasterSrc.shape:", (h,w,bands))

    pixel_coords, latlon_coords = geojson_to_pixel_arr(rasterSrc, 
                                                       vectorSrc, 
                                                       pixel_ints=pixel_ints,
                                                       verbose=verbose)
    
    yolt_coords, cont_plot_box = pixel_coords_to_yolt(pixel_coords, w, h, dl=dl)

#    # Get yolt coords
#    cont_plot_box = []
#    yolt_coords = []
#    # get extent of building footprint
#            
#    
#    for c in pixel_coords:
#        carr = np.array(c)
#        xs, ys = carr[:,0], carr[:,1]
#        minx, maxx = np.min(xs), np.max(xs)
#        miny, maxy = np.min(ys), np.max(ys)
#        
#        ## take histogram of coordinate counts and use that to estimate
#        ## best bounding box (this doesn't work all that well if the 
#        ## point are not uniformly distrbuted about the polygon)
#        ##xmid, ymid = np.mean(xs), np.mean(ys)
#        #x0 = np.percentile(xs, 15)
#        #x1 = np.percentile(xs, 85)
#        #y0 = np.percentile(ys, 15)
#        #y1 = np.percentile(ys, 85)
#        
#        # midpoint 
#        xmid, ymid = np.mean([minx,maxx]), np.mean([miny,maxy])
#        dx = dl*(maxx - minx) / 2
#        dy = dl*(maxy - miny) / 2
#        x0 = xmid - dx
#        x1 = xmid + dx
#        y0 = ymid - dy
#        y1 = ymid + dy
#        yolt_row = convert.convert((w,h), [x0,x1,y0,y1])
#        yolt_coords.append(yolt_row)
#        
#        row = [[x0, y0],
#               [x0, y1],
#               [x1, y1],
#               [x1, y0]]
#        cont_plot_box.append(np.rint(row).astype(int))

    return rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, \
                cont_plot_box
                

###############################################################################
def get_yolt_coords_spacenet_v0(rasterList, imDir, vecDir, maskDir='', 
                             new_schema=False, outDir='', use8band=False):
    '''get building coords
    if outDir != '', replace imDir with outDir in returned array'''
    
    out_list = []
    rlen = len(rasterList)
    for i,rasterSrc in enumerate(rasterList):
            
        vectorSrc = pair_im_vec_spacenet(rasterSrc, vecDir, 
                                         new_schema=new_schema)            
#        # old example: 
#        #   3band_013022223130_Public_img1.tif
#        #   013022223130_Public_img1_Geo.geojson
#        # get name root
#        name_root0 = rasterSrc.split('/')[-1].split('.')[0]  
#        #name_8band = im8Dir + '8' + name_root0[1:] + '.tif'
#        if len(maskDir) > 0:
#            maskSrc = maskDir + name_root0 + '.tif'
#        else:
#            maskSrc = ''
#            
#        # remove 3band or 8band prefix
#        name_root = name_root0[6:]
#           
#        if (i % 20) == 0:
#            print i, "/", rlen, name_root       
#        
#        if not new_schema:
#            vectorSrc = vecDir + name_root + '_Geo.geojson'
#        else:
#            # new naming schema:
#            # 3band_AOI_1_RIO_img9.tif
#            # Geo_AOI_1_RIO_img9.geojson
#            vectorSrc = vecDir + 'Geo_' + name_root + '.geojson'
            
        # get coords
        # could use geoTools function, but output is in a gdal polygon,
        # which is still requires significant parsing to be useful
                #buildinglist.append({'ImageId': image_id,
                #             'BuildingId': building_id,
                #             'polyGeo': ogr.CreateGeometryFromWkt(geom.ExportToWkt()),
                #             'polyPix': ogr.CreateGeometryFromWkt('POLYGON((0 0, 0 0, 0 0, 0 0))')
                #             })

        #buildinglist = gT.convert_wgs84geojson_to_pixgeojson(vectorSrc, 
        #                                                      rasterSrc)
        
        pixel_coords, latlon_coords = geojson_to_pixel_arr(rasterSrc, 
                                                         vectorSrc, 
                                                     pixel_ints=True,
                                                     verbose=False)

        if len(outDir) > 0:
            rasterSrc_out = rasterSrc.replace(imDir, outDir)
        else:
            rasterSrc_out = rasterSrc
        if use8band:
            rasterSrc_out = rasterSrc_out.replace('3band', '8band')
            name_root0 = name_root0.replace('3band', '8band')

        
        # append to list
        row = [name_root0, rasterSrc_out, vectorSrc, maskSrc, pixel_coords, latlon_coords]
        out_list.append(row)
            
    return out_list

###############################################################################        
def spacenet_yolt_setup(building_list, classes_dic, im_input_dir,
                        labels_outdir, images_outdir,
                        train_images_list_file,
                        deploy_dir, 
                        imtype='3band', band_delim='#', maskDir='',
                        sample_mask_vis_dir=''):
    
    '''    
    Set up yolt training data, take output of get_yolt_coords_spacenet()
    as input.  
    # output of get_yolt_coords_spacenet is:
    #   rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, 
    #       cont_plot_box    
    dl =  fraction of size for bounding box
    '''

    category = 'building'
    if imtype in ['iband', 'iband3']:
        out_ext = '.png'
    else:
        out_ext = '.tif'     
    list_file = open(train_images_list_file, 'wb')
    nplots = 50
    
    for p in [labels_outdir, images_outdir]:
        if not os.path.exists(p):
            os.mkdir(p)
    
    t0 = time.time()
    yolt_list = []
    count = 0
    for i,row in enumerate(building_list):

        [rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, 
           cont_plot_box] = row
        
        if (i % 50) == 0:
            print ( i, "/", len(building_list), rasterSrc   )         
            
        # skip empty scenes
        if len(pixel_coords) == 0:
            continue

        count += 1
        ext = rasterSrc.split('.')[1]
        name_root_tmp3 = rasterSrc.split('/')[-1].split('.')[0]  # e.g. '3band_013022223131_Public_img1014'
        name_root = name_root_tmp3[1:]  # e.g. 'band_013022223131_Public_img1014'

        # copy file(s) of name_root from im_input_dir to images_outdir
        # if not using iband, copy file
        if imtype == '3band':
            imloc = im_input_dir + name_root_tmp3 + '.' + out_ext
            if images_outdir != im_input_dir:
                shutil.copy(imloc, images_outdir)
            name_rootf = name_root_tmp3
        # else, copy all files in 'iband' directory with appropriate root
        elif imtype in ['iband', 'iband3']:
            im_list = glob.glob(os.path.join(im_input_dir, '*' + name_root + band_delim + '*'))
            for imloctmp in im_list:
                shutil.copy(imloctmp, images_outdir)
            # set name_rootf as file ending in '#1' + ext
            name_rootf = 'i' + name_root + band_delim + '1'
        else:
            print ("Unsupported image type (imtype) in spacenet_yolt_setup()")
            return


        # make plots
        if count < nplots and len(sample_mask_vis_dir) > 0:
            name_root = rasterSrc.split('/')[-1]
            plot_name = sample_mask_vis_dir + name_root + '_mask.png'
            maskSrc = maskDir + name_root
            mask_tmp = cv2.imread(maskSrc, 0)
            im_tmp = cv2.imread(rasterSrc, 1)     
            print ("rasterSrc:", rasterSrc)
            print ("maksSrc:", maskSrc )
            plot_contours_yolt(im_tmp, mask_tmp,
                   cont_plot=cont_plot_box, figsize=(8,8), 
                           plot_name=plot_name,
                           add_title=False)

                
        txt_outpath = labels_outdir + name_rootf + '.txt'
        txt_outfile = open(txt_outpath, "w")
        for bb in yolt_coords:
            cls_id = classes_dic[category]
            outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            #print "outstring:", outstring
            txt_outfile.write(outstring)
        txt_outfile.close()               

        # create list of training image locations
        list_file.write('%s/%s%s\n'%(deploy_dir, name_rootf, out_ext))

    list_file.close()        
    print ("building list has length:", len(building_list), "though only", \
            count, "images were processed, the remainder are empty" )
    print ("Time to setup training data for:", images_outdir, "of length:", \
            count, time.time() - t0, "seconds" )
 

################################################################################
## import functions
#sys.path.append(os.path.join(root_dir, 'sivnet/src/')) #'/Users/avanetten/Documents/cosmiq/sivnet/src/')
#import sivnet_data_prep
#from sivnet_data_prep import get_contours, plot_contours
################################################################################        
#def spacenet_yolt_setup_v0(building_list, classes_dic, labels_dir, images_dir,
#                        sample_label_vis_dir, train_images_list_file_loc,
#                        deploy_dir,  dl=0.8):
#    
#    list_file = open(train_images_list_file_loc, 'wb')
#                
#    '''
#    Set up yolt training data, take output of get_yolt_coords_spacenet()
#    as input.  
#    dl =  fraction of size for bounding box
#    '''
#    
#    t0 = time.time()
#    yolt_list = []
#    count = 0
#    for i,row in enumerate(building_list):
#        
#        if (i % 50) == 0:
#            print (i, "/", len(building_list))
#            
#        [name_root0, rasterSrc, vectorSrc, maskSrc, pixel_coords, latlon_coords] = row
#        # get size
#        h,w = cv2.imread(rasterSrc,0).shape
#    
#        cont_plot_box = []
#        yolt_coords = []
#        # get extent of building footprint
#        
#        # skip empty scenes
#        if len(pixel_coords) == 0:
#            yolt_list.append([name_root0, rasterSrc, vectorSrc, maskSrc, \
#                          [], []])
#            continue
#
#        count += 1
#        for c in pixel_coords:
#            carr = np.array(c)
#            xs, ys = carr[:,0], carr[:,1]
#            minx, maxx = np.min(xs), np.max(xs)
#            miny, maxy = np.min(ys), np.max(ys)
#            
##            # take histogram of coordinate counts and use that to estimate
##            # best bounding box (this doesn't work all that well if the 
##            # point are not uniformly distrbuted about the polygon)
##            #xmid, ymid = np.mean(xs), np.mean(ys)
##            x0 = np.percentile(xs, 15)
##            x1 = np.percentile(xs, 85)
##            y0 = np.percentile(ys, 15)
##            y1 = np.percentile(ys, 85)
#            
#            # midpoint 
#            xmid, ymid = np.mean([minx,maxx]), np.mean([miny,maxy])
#            dx = dl*(maxx - minx) / 2
#            dy = dl*(maxy - miny) / 2
#            x0 = xmid - dx
#            x1 = xmid + dx
#            y0 = ymid - dy
#            y1 = ymid + dy
#            
#            row = [[x0, y0],
#                   [x0, y1],
#                   [x1, y1],
#                   [x1, y0]]
#            cont_plot_box.append(np.rint(row).astype(int))
#
#                    
#            yolt_row = convert((w,h), [x0,x1,y0,y1])
#            yolt_coords.append(yolt_row)
#    
#        # write to files
#        # copy input image to correct dir
#        shutil.copy(rasterSrc, images_dir)
#        # write yolt label to file
#
#        txt_outpath = labels_dir + name_root0 + '.txt'
#        txt_outfile = open(txt_outpath, "w")
#        for bb in yolt_coords:
#            category = 'building'
#            cls_id = classes_dic[category]
#            outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
#            #print "outstring:", outstring
#            txt_outfile.write(outstring)
#        txt_outfile.close()               
#            
#        # make maskplots
#        if i < 10 and len(maskSrc) > 0:
#            plot_name = sample_mask_vis_dir + name_root0 + '_mask.png'
#            vis = cv2.imread(maskSrc, 0)
#            im_tmp = cv2.imread(rasterSrc, 1)
#            plot_contours(im_tmp, vis, vis,    
#                      contours=[], cont_plot=cont_plot_box, figsize=(8,8), 
#                      plot_name=plot_name)
#            
#            
#        yolt_list.append([name_root0, rasterSrc, vectorSrc, maskSrc,
#                          cont_plot_box, yolt_coords])
#        
#        # create list of training image locations
#        list_file.write('%s/%s.tif\n'%(deploy_dir, name_root0))
#
#       
#    list_file.close()        
#    print ("building list has length:", len(building_list), "though only", \
#            count, "images were processed, the remainder are empty" )
#    print ("Time to setup training data for:", images_dir, "of length:", \
#            len(yolt_list), time.time() - t0, "seconds" )
#        
#    return yolt_list


###############################################################################
def plot_contours_yolt(input_image, mask_true, cont_plot=[], figsize=(8,8), 
                       plot_name='', add_title=False):
    
    #fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(2*figsize[0], 2*figsize[1]))
    #fig, ((ax0, ax1, ax2)) = plt.subplots(1, 3, figsize=(3*figsize[0], 1*figsize[1]))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))
    
    if add_title:
        suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')
    
    print ("input_image", input_image )
    # ax0: raw image
    ax0.imshow(input_image)
    # ground truth
    # set zeros to nan
    palette = plt.cm.gray
    palette.set_over('orange', 1.0)
    z = mask_true.astype(float)
    z[z==0] = np.nan
    ax0.imshow(z, cmap=palette, alpha=0.25, 
               norm=matplotlib.colors.Normalize(vmin=0.6, vmax=0.9, clip=False))
    ax0.set_title('Input Image + Ground Truth Labels (Yellow)') 
    ax0.axis('off')

    # truth mask
    #ax1.imshow(mask_true, cmap='bwr')
    #ax1.set_title('Ground Truth Building Mask')
    
    # ax1: input image overlaid predicted contours
    vis1 = input_image.copy()
    ax1.imshow(vis1)
    # overlay building contours
    coll = PolyCollection(cont_plot, facecolors='red', edgecolors='white', alpha=0.4)
    ax1.add_collection(coll, autolim=True)
    ## add roads
    #ax1.imshow(threshr, cmap='bwr', alpha=0.5)#'Blues')
    ax1.set_title('Input Image + YOLT Labels (Red)')
    ax1.axis('off')
    
    plt.axis('off')
    plt.tight_layout()
    if add_title:
        #suptitle.set_y(0.95)
        fig.subplots_adjust(top=0.96)
    plt.show()
 
    if len(plot_name) > 0:
        plt.savefig(plot_name)
    
    return


###############################################################################        
def yolt_labels_to_bbox(label_loc, w, h):
    '''coords from yolt labels
    height, width = image.shape[:2]
    shape = (width, height)
    return list of [ [cat_int, [xmin, xmax, ymin, ymax]] , ...]
    '''
    shape = (w, h)
    z = pd.read_csv(label_loc, sep = ' ', names=['cat', 'x', 'y', 'w', 'h'])
    #print "z", z.values
    box_list, cat_list = [], []
    for yolt_box in z.values:
        cat_int = int(yolt_box[0])
        yb = yolt_box[1:]
        box0 = convert_reverse(shape, yb)
        # convert to int
        box1 = [int(round(b,2)) for b in box0]
        [xmin, xmax, ymin, ymax] = box1
        box_list.append(box1)
        cat_list.append(cat_int)
        #outputs.append([cat_int, box1])
    return cat_list, box_list
    

###############################################################################        
def plot_training_bboxes(label_folder, image_folder, ignore_augment=True,
                         figsize=(10,10), color=(0,0,255), thickness=2, 
                         max_plots=100, sample_label_vis_dir=None, ext='.png',
                         verbose=False, show_plot=False, specific_labels=[],
                         label_dic=[], output_width=60000, shuffle=True):
    '''Plot bounding boxes for yolt
    specific_labels allows user to pass in labels of interest'''
    
    out_suff = ''#'_vis'
    
    if sample_label_vis_dir and not os.path.exists(sample_label_vis_dir):
        os.mkdir(sample_label_vis_dir)
        
    # boats, boats_harbor, airplanes, airports (blue, green, red, orange)
    # remember opencv uses bgr, not rgb
    colors = [(255,0,0), (0,255,0), (0,0,255), (0,140,255), (0,255,125),
              (125,125,125)]  
              
    cv2.destroyAllWindows()
    i = 0
    
    if len(specific_labels) == 0:
        label_list = os.listdir(label_folder)
        # shuffle?
        if shuffle:
            random.shuffle(label_list)

    else:
        label_list = specific_labels
    
    for label_file in label_list:
                            
        if ignore_augment:
            if (label_file == '.DS_Store') or (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt'))):
                continue
        #else:
        #     if (label_file == '.DS_Store'):
        #         continue
             
        if i >= max_plots:
            #print "i, max_plots:", i, max_plots
            return
        else:
            i += 1

        print (i, "/", max_plots  )
        if verbose:
            print  ("  label_file:", label_file )
            
        # get image
        #root = label_file.split('.')[0]
        root = label_file[:-4]
        im_loc = os.path.join(image_folder,  root + ext)
        label_loc = os.path.join(label_folder, label_file)
        if verbose:
            print (" root:", root )
            print ("  label loc:", label_loc )
            print ("  image loc:", im_loc )
        image0 = cv2.imread(im_loc, 1)
        height, width = image0.shape[:2]
        # resize output file
        if output_width < width:
            height_mult = 1.*height / width
            output_height = int(height_mult * output_width)
            outshape = (output_width, output_height)
            image = cv2.resize(image0, outshape)
        else:
            image = image0
        
        height, width = image.shape[:2]
        shape = (width, height)
        if verbose:        
            print ("im.shape:", image.shape)
        
        
        ## start plot (mpl)
        #fig, ax = plt.subplots(figsize=figsize)
        #img_mpl = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #ax.imshow(img_mpl)
        # just opencv
        img_mpl = image
        
        # get and plot labels
        #z = pd.read_csv(label_folder + label_file, sep = ' ', names=['cat', 'x', 'y', 'w', 'h'])
        z = pd.read_csv(label_loc, sep = ' ', names=['cat', 'x', 'y', 'w', 'h'])
        #print "z", z.values
        for yolt_box in z.values:
            cat_int = int(yolt_box[0])
            color = colors[cat_int]
            yb = yolt_box[1:]
            box0 = convert_reverse(shape, yb)
            # convert to int
            box1 = [int(round(b,2)) for b in box0]
            [xmin, xmax, ymin, ymax] = box1
            # plot
            cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)    
            
        # add border
        if label_dic:

            # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
            font = cv2.FONT_HERSHEY_TRIPLEX#FONT_HERSHEY_SIMPLEX #_SIMPLEX _TRIPLEX
            font_size = 0.25
            label_font_width = 1
            #text_offset = [3, 10]          
            ydiff = 35
                    
            # add border
            # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
            # top, bottom, left, right - border width in number of pixels in corresponding directions
            border = (0, 0, 0, 200) 
            border_color = (255,255,255)
            label_font_width = 1
            img_mpl = cv2.copyMakeBorder(img_mpl, border[0], border[1], border[2], border[3],
                                         cv2.BORDER_CONSTANT,value=border_color)
            # add legend
            xpos = img_mpl.shape[1] - border[3] + 15
            #for itmp, k in enumerate(sorted(label_dic.keys())):
            for itmp, (k, value) in  enumerate(sorted(label_dic.items(), key=operator.itemgetter(1))):
                labelt = label_dic[k]     
                colort = colors[k]                        
                #labelt, colort = label_dic[k]                             
                text = '- ' + labelt #str(k) + ': ' + labelt
                ypos = ydiff + (itmp) * ydiff
                #cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 1.5*font_size, colort, label_font_width, cv2.CV_AA)#, cv2.LINE_AA)
                cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 1.5*font_size, colort, label_font_width, cv2.CV_AA)#cv2.LINE_AA)
        
            # legend box
            cv2.rectangle(img_mpl, (xpos-5, 2*border[0]), (img_mpl.shape[1]-10, ypos+int(0.75*ydiff)), (0,0,0), label_font_width)   
                                              
            ## title                                  
            #title = figname.split('/')[-1].split('_')[0] + ':  Plot Threshold = ' + str(plot_thresh) # + ': thresh=' + str(plot_thresh)
            #title_pos = (border[0], int(border[0]*0.66))
            ##cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), label_font_width, cv2.CV_AA)#, cv2.LINE_AA)
            #cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), label_font_width,  cv2.CV_AA)#cv2.LINE_AA)
                    
                    
        if show_plot:
            cv2.imshow(root, img_mpl)
            cv2.waitKey(0)
        
        if sample_label_vis_dir:
            fout = os.path.join(sample_label_vis_dir,  root + out_suff + ext)
            cv2.imwrite(fout, img_mpl)

    return

###############################################################################
def np_to_geotiff(array, outfile, im_for_geom=''):
    '''inDs is the image that provides the projection and geometry'''
    #https://borealperspectives.wordpress.com/2014/01/16/data-type-mapping-when-using-pythongdal-to-write-numpy-arrays-to-geotiff/
 
    src_dataset = gdal.Open(im_for_geom)
     
    # get parameters
    geotransform = src_dataset.GetGeoTransform()
    spatialreference = src_dataset.GetProjection()
    ncol = src_dataset.RasterXSize
    nrow = src_dataset.RasterYSize
    #nrows,ncols,nbands = np.shape(array)
    nband = array.shape[-1]
     
    # create dataset for output
    fmt = 'GTiff'
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(outfile, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    
    for b in range(1, 1+nband):
        idx = b-1
        outData = array[:,:,idx]
        outBand = dst_dataset.GetRasterBand(b).WriteArray(outData)
        # flush data to disk, set the NoData value and calculate stats
        #outBand.FlushCache()
        #outBand.SetNoDataValue(-99)
    dst_dataset = None
    
    return

###############################################################################
# from blupr_data_prep.py
def comb_3band_8band(band3_file, band8_file, w0, h0, 
                     verbose=False, show_cv2=False, rgb_first=True):
    '''
    combine 3 band and 8 band images
    assume 8 band file is ordered by wavelength:
        file:///Users/avanetten/Documents/cosmiq/papers/dg_worldview2_ds_prod-1.pdf
        coastal, blue, green, yellow, red, red_edge, nearIR1, nearIR2
    assume 3 band file is ordered as RGB
    can't just use the output of ReadAsArray() since the array has different 
    indices than standard np images, and we need to resize in cv2
    if rgb_first, put rgb as first three bands, then the remaining 5 bands
    '''
    
    # creat dictionary of 8 band index to 3 band index (1 indexed, not 0 indexed)
    band_dic_8to3 = {2: 3, # blue
                     3: 2, # green
                     5: 1  # red
                     }
                     
    im8_raw = gdal.Open(band8_file)
    im8 = im8_raw.ReadAsArray()
    im3_raw = gdal.Open(band3_file)
    im3 = im3_raw.ReadAsArray()    
    if verbose:
        print ("input im8.shape:", im8.shape )
        print ("[RASTER BAND COUNT]: ", im8_raw.RasterCount )
        print ("input im3.shape:", im3.shape )
        print ("[RASTER BAND COUNT]: ", im3_raw.RasterCount    )
    bandlist = []
               
    # if desired, ingest 3 band first and put rgb as first three bands
    # Band 1 Block=438x6 Type=Byte, ColorInterp=Red
    # Band 2 Block=438x6 Type=Byte, ColorInterp=Green
    # Band 3 Block=438x6 Type=Byte, ColorInterp=Blue
    if rgb_first:
        for band in range(1, im3_raw.RasterCount+1):
            srcband = im3_raw.GetRasterBand(band)
            band_arr = srcband.ReadAsArray()
            image_rs = cv2.resize(band_arr, (w0, h0))
            bandlist.append(image_rs)
            
    # ingest 8 band arrays
    cv2.destroyAllWindows()
    for band in range(1, im8_raw.RasterCount+1):
        if verbose:
            print ("[ GETTING BAND ]: ", band)
        if band in band_dic_8to3.keys():
            # skip if we put rgb first in the array
            if rgb_first:
                continue
            else:
                srcband = im3_raw.GetRasterBand( band_dic_8to3[band] )
                if verbose:
                    print ("band_dic_8to3[band]:",  band_dic_8to3[band] )
        else:
            srcband = im8_raw.GetRasterBand(band)
        band_arr = srcband.ReadAsArray()
        # now resize to im.shape (cv2 used (width, height))
        image_rs = cv2.resize(band_arr, (w0, h0))
        if verbose:
            print ("image_rs.dtype:", image_rs.dtype )
            print ("image_rs.shape", image_rs.shape )
            print ("image min, max", np.min(image_rs), np.max(image_rs) )
        if show_cv2:
            cv2.imshow('band' + str(band), image_rs.astype('uint8'))
            cv2.waitKey(0)
        bandlist.append(image_rs)
    
    band_arr = np.stack(bandlist, axis=2)
    if verbose:
        print ("outarr.shape:", band_arr.shape )
        print ("outarr.dtype:", band_arr.dtype )
        

    return band_arr

###############################################################################
def split_8band(band8_file, outdir, band_delim = '#', out_ext='.png'):
    # save individual bands in groupds of 3
    
    fname = band8_file.split('/')[-1]
    mroot, out_ext_tmp = fname.split('.')
    if len(out_ext) == 0:
        out_ext = '.' + out_ext_tmp
    
    # ingest im8
    im8_raw = gdal.Open(band8_file)
    bandlist = []
    for band in range(1, im8_raw.RasterCount+1):
        srcband = im8_raw.GetRasterBand(band)
        band_arr_tmp = srcband.ReadAsArray()
        bandlist.append(band_arr_tmp)
    arr_rescale = np.stack(bandlist, axis=2)
    

    nbandtmp = arr_rescale.shape[-1]
    #outfile = outdir + 'i' + mroot.split('.')[0][1:] + out_ext
    nout = 1                            
    for k in range(0, nbandtmp, 3):
        #print "k", k
        #outfile_tmp = outfile.split('.')[0] + band_delim + str(nout) + out_ext
        outfile_tmp = outdir + mroot + band_delim + str(nout) + out_ext

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
        #print "outfile:", outfile_tmp
        cv2.imwrite(outfile_tmp, out_im)
        
    return


###############################################################################
def rescale_intensity(band, nstd=3, method='std', out_range='uint8',
                      verbose=False):
    '''assume band is 2-d array
    method = ['std', 'hist', 'uint16', other]
    if other, return original band
    stretch pixel intensities to vary from 0 to 255'''
    
    mean = np.mean(band)
    std = np.std(band)
    minv, maxv = mean - nstd*std, mean + nstd*std
    
    if method == 'uint16':
        band_rescale = exposure.rescale_intensity(band, 
                                in_range=('uint16'), 
                                out_range=('uint8'))
    elif method == 'hist':
        band_rescale = 255 * exposure.equalize_hist(band)
    
    elif method == 'std':
        mean = np.mean(band)
        std = np.std(band)
        minv, maxv = max(0, mean - nstd*std), mean + nstd*std
        if verbose:
            print ("mean, std, minv, maxv:", mean, std, minv, maxv )
        band_rescale = exposure.rescale_intensity(band, 
                            in_range=(minv, maxv), 
                            out_range=('uint8'))
    else:
        if verbose:
            print ("Unknown method for rescale_intensity" )
        band_rescale = band
    
    return band_rescale
    
###############################################################################
#http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy   


###############################################################################        
def augment_training_data(label_folder, image_folder, hsv_range=[0.5,1.5],
                          skip_hsv_transform=True, ext='.jpg'):
    '''Rotate data to augment training sizeo 
    darknet c functions already to HSV transform, and left-right swap, so
    skip those transforms
    Image augmentation occurs in data.c load_data_detection()'''
    
    hsv_diff = hsv_range[1] - hsv_range[0]
    im_l_out = []
    for label_file in os.listdir(label_folder):
        
        # don't augment the already agumented data
        if (label_file == '.DS_Store') or \
            (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt','_rot90.txt','_rot180.txt','_rot270.txt'))):
            continue
        
        # get image
        print ("image loc:", label_file )
        root = label_file.split('.')[0]
        im_loc = os.path.join(image_folder, root + ext)
        
        #image = skimage.io.imread(f, as_grey=True)
        image = cv2.imread(im_loc, 1)
        
        # randoly scale in hsv space, create a list of images
        if skip_hsv_transform:
            img_hsv = image
        else:
            try:
                img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            except:
                continue
        
        img_out_l = []
        np.random.seed(42)

        # three mirrorings        
        if skip_hsv_transform:
            img_out_l = 6*[image]

        else:
            for i in range(6):
                im_tmp = img_hsv.copy()
                ## alter values for each of 2 bands (hue and saturation)
                #for j in range(2):                
                #    rand = hsv_range[0] + hsv_diff*np.random.random()  # between 0,5 and 1.5
                #    z0 = (im_tmp[:,:,j]*rand).astype(int)
                #    im_tmp[:,:,j] = z0
               # alter values for each of 3 bands (hue and saturation, value
                for j in range(3):         
                    # set 'value' range somewhat smaller
                    if j == 2:
                        rand = 0.7 + 0.6*np.random.random()
                    else:
                        rand = hsv_range[0] + hsv_diff*np.random.random()  # between 0,5 and 1.5
                    z0 = (im_tmp[:,:,j]*rand).astype(int)
                    z0[z0 > 255]  = 255
                    im_tmp[:,:,j] = z0
  
                # convert back to bgr and add to list of ims
                img_out_l.append(cv2.cvtColor(im_tmp, cv2.COLOR_HSV2BGR))
            
        #print "image.shape", image.shape
        # reflect or flip image left to right (skip since yolo.c does this?)
        image_lr = np.fliplr(img_out_l[0])#(image)
        image_ud = np.flipud(img_out_l[1])#(image) 
        image_lrud = np.fliplr(np.flipud(img_out_l[2]))#(image_ud)
        
        #cv2.imshow("in", image)
        #cv2.imshow("lr", image_lr)
        #cv2.imshow("ud", image_ud)
        #cv2.imshow("udlr", image_udlr)
        
        image_rot90 = np.rot90(img_out_l[3])
        image_rot180 = np.rot90(np.rot90(img_out_l[4]))
        image_rot270 = np.rot90(np.rot90(np.rot90(img_out_l[5])))
        
        # flip coords of bounding boxes too...
        # boxes have format: (x,y,w,h)
        z = pd.read_csv(os.path.join(label_folder, label_file), sep = ' ', names=['x', 'y', 'w', 'h'])
        
        # left right flip
        lr_out = z.copy()
        lr_out['x'] = 1. - z['x']
        
        # left right flip
        ud_out = z.copy()
        ud_out['y'] = 1. - z['y']        
        
        # left right, up down, flip
        lrud_out = z.copy()
        lrud_out['x'] = 1. - z['x']        
        lrud_out['y'] = 1. - z['y']        

        ##################
        # rotate bounding boxes X degrees
        origin = [0.5, 0.5]
        point = [z['x'], z['y']]

        # 90 degrees
        angle = -1*np.pi/2
        xn, yn = rotate(origin, point, angle)
        rot_out90 = z.copy()
        rot_out90['x'] = xn
        rot_out90['y'] = yn
        rot_out90['h'] = z['w']
        rot_out90['w'] = z['h']

        # 180 degrees (same as lrud)
        angle = -1*np.pi
        xn, yn = rotate(origin, point, angle)
        rot_out180 = z.copy()
        rot_out180['x'] = xn
        rot_out180['y'] = yn
        
        # 270 degrees
        angle = -3*np.pi/2
        xn, yn = rotate(origin, point, angle)
        rot_out270 = z.copy()
        rot_out270['x'] = xn
        rot_out270['y'] = yn
        rot_out270['h'] = z['w']
        rot_out270['w'] = z['h']        
        ##################

        # print to files, add to list
        im_l_out.append(im_loc)
        
#        # reflect or flip image left to right (skip since yolo.c does this?)
#        imout_lr = image_folder + root + '_lr.jpg'
#        labout_lr = label_folder + root + '_lr.txt'
#        cv2.imwrite(imout_lr, image_lr)
#        lr_out.to_csv(labout_lr, sep=' ', header=False)
#        #im_l_out.append(imout_lr)
        
        # flip vertically or rotate 180 randomly
        if bool(random.getrandbits(1)):
            # flip vertically 
            imout_ud = os.path.join(image_folder, root + '_ud' + ext)
            labout_ud = os.path.join(label_folder, root + '_ud.txt')
            cv2.imwrite(imout_ud, image_ud)
            ud_out.to_csv(labout_ud, sep=' ', header=False)
            im_l_out.append(imout_ud)
        else:
            im180_path = os.path.join(image_folder, root + '_rot180' + ext)
            cv2.imwrite(os.path.join(im180_path), image_rot180)
            rot_out180.to_csv(os.path.join(label_folder, root + '_rot180.txt'), sep=' ', header=False)
            im_l_out.append(im180_path)

        
#        # lrud flip, same as rot180
#        #  skip lrud flip because yolo does this sometimes
#        imout_lrud = image_folder + root + '_lrud.jpg'
#        labout_lrud = label_folder + root + '_lrud.txt'
#        cv2.imwrite(imout_lrud, image_lrud)
#        lrud_out.to_csv(labout_lrud, sep=' ', header=False)
#        #im_l_out.append(imout_lrud)

        # same as _lrud
        #im180 = image_folder + root + '_rot180.jpg'
        #cv2.imwrite(image_folder + root + '_rot180.jpg', image_rot180)
        #rot_out180.to_csv(label_folder + root + '_rot180.txt', sep=' ', header=False)
        #im_l_out.append(im180)

        # rotate 90 degrees or 270 randomly
        if bool(random.getrandbits(1)):
            im90_path = os.path.join(image_folder, root + '_rot90' + ext)
            #lab90 = label_folder + root + '_rot90.txt'
            cv2.imwrite(im90_path, image_rot90)
            rot_out90.to_csv(os.path.join(label_folder, root + '_rot90.txt'), sep=' ', header=False)
            im_l_out.append(im90_path)
            
        else:
            # rotate 270 degrees ()
            im270_path = os.path.join(image_folder, root + '_rot270' + ext)
            cv2.imwrite(im270_path, image_rot270)
            rot_out270.to_csv(os.path.join(label_folder, root + '_rot270.txt'), sep=' ', header=False)
            im_l_out.append(im270_path)
    
    return im_l_out
    
#cv2.destroyAllWindows()
#label_folder = labels_dir
#image_folder = images_dir


###############################################################################        
def rm_augment_training_data(label_folder, image_folder, tmp_dir):
    '''Remove previusly created augmented data since it's done in yolt.c and 
    need not be doubly augmented'''
    

    # mv augmented labels
    for label_file in os.listdir(label_folder):
        
        if (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt','_rot90.txt','_rot180.txt','_rot270.txt'))):
               
            try: os.mkdir(tmp_dir)
            except: print (""  )  
            
            # mv files to tmp_dir
            print ("label_file", label_file)
            #shutil.move(label_file, tmp_dir)
            # overwrite:
            shutil.move(os.path.join(label_folder, label_file), os.path.join(tmp_dir, label_file))   
    
            # just run images separately below to make sure we get errthing
            ## get image
            #print "image loc:", label_file
            #root = label_file.split('.')[0]
            #im_loc = image_folder + root + '.jpg'
            # mv files to tmp_dir
            #shutil.move(im_loc, tmp_dir)

    # mv augmented images
    for image_file in os.listdir(image_folder):
        
        if (image_file.endswith(('_lr.jpg', '_ud.jpg', '_lrud.jpg','_rot90.jpg','_rot180.jpg','_rot270.jpg'))):
                
            try: os.mkdir(tmp_dir)
            except: print ("")                  
                
            # mv files to tmp_dir
            #shutil.move(image_file, tmp_dir)
            # overwrite
            shutil.move(os.path.join(image_folder, image_file), os.path.join(tmp_dir, image_file))   

            
    return

###############################################################################
def convertTo8Bit(rasterImageName, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff'):
    '''https://github.com/SpaceNetChallenge/utilities/blob/master/python/spaceNetUtilities/labelTools.py'''
    

    srcRaster = gdal.Open(rasterImageName)

    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-co', '"PHOTOMETRIC=rgb"']
    scaleList = []
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band=srcRaster.GetRasterBand(bandId)
        min = band.GetMinimum()
        max = band.GetMaximum()

        # if not exist minimum and maximum values
        if min is None or max is None:
            (min, max) = band.ComputeRasterMinMax(1)
        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(max))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(rasterImageName)

    #if outputFormat == 'JPEG':
    #    outputRaster = xmlFileName.replace('.xml', '.jpg')
    #else:
    #    outputRaster = xmlFileName.replace('.xml', '.tif')
    #
    #outputRaster = outputRaster.replace('_img', '_8bit_img')
    
    cmd.append(outputRaster)
    print(cmd)
    subprocess.call(cmd)
    
    return

###############################################################################
          


###############################################################################
def get_yolt_coords_poi(rasterSrc, vecDir, new_schema=False,
                             pixel_ints=True, dl=0.8, verbose=False):
    
    '''
    Take raster image as input, along with location of labels
    return:
        rasterSrc
        label file
        pixel coords of buildings
        latlon coords of buildings
        building coords converted to yolt coords
        building coords in pixel coords for plotting   
    
    dl =  fraction of size for bounding box
    '''
    
    vectorSrc = pair_im_vec_spacenet_v2(rasterSrc, vecDir, 
                                     new_schema=new_schema)  
    
    #if len(maskDir) > 0:
    #    name_root = rasterSrc.split('/')[-1]
    #    maskSrc = maskDir + name_root
    #else:
    #    maskSrc = ''

        
    
    # get size
    h,w,bands = cv2.imread(rasterSrc,1).shape

    if verbose:
        print ( "\nrasterSrc:", rasterSrc )
        print ("  vectorSrc:", vectorSrc )
        print ("  rasterSrc.shape:", (h,w,bands) )

    pixel_coords, latlon_coords = spacenet_utilities.geojson_to_pixel_arr(rasterSrc, 
                                                       vectorSrc, 
                                                       pixel_ints=pixel_ints,
                                                       verbose=verbose)
    
    yolt_coords, cont_plot_box = pixel_coords_to_yolt(pixel_coords, w, h)

    # Get yolt coords
    cont_plot_box = []
    yolt_coords = []
    # get extent of building footprint
            
    for c in pixel_coords:
        carr = np.array(c)
        xs, ys = carr[:,0], carr[:,1]
        minx, maxx = np.min(xs), np.max(xs)
        miny, maxy = np.min(ys), np.max(ys)
        
        ## take histogram of coordinate counts and use that to estimate
        ## best bounding box (this doesn't work all that well if the 
        ## point are not uniformly distrbuted about the polygon)
        ##xmid, ymid = np.mean(xs), np.mean(ys)
        #x0 = np.percentile(xs, 15)
        #x1 = np.percentile(xs, 85)
        #y0 = np.percentile(ys, 15)
        #y1 = np.percentile(ys, 85)
        
        # midpoint 
        xmid, ymid = np.mean([minx,maxx]), np.mean([miny,maxy])
        dx = dl*(maxx - minx) / 2
        dy = dl*(maxy - miny) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        yolt_row = convert((w,h), [x0,x1,y0,y1])
        yolt_coords.append(yolt_row)
        
        row = [[x0, y0],
               [x0, y1],
               [x1, y1],
               [x1, y0]]
        cont_plot_box.append(np.rint(row).astype(int))

    return rasterSrc, vectorSrc, pixel_coords, latlon_coords, yolt_coords, \
                cont_plot_box
                

###############################################################################
def pixel_coords_to_yolt(pixel_coords, w, h, dl=0.8):
    '''
    convert pixel coords to yolt coords
    dl =  fraction of size for bounding box
    '''

    cont_plot_box = []
    yolt_coords = []
    # get extent of object footprint
    for c in pixel_coords:
        carr = np.array(c)
        xs, ys = carr[:,0], carr[:,1]
        minx, maxx = np.min(xs), np.max(xs)
        miny, maxy = np.min(ys), np.max(ys)
        
        ## take histogram of coordinate counts and use that to estimate
        ## best bounding box (this doesn't work all that well if the 
        ## point are not uniformly distrbuted about the polygon)
        ##xmid, ymid = np.mean(xs), np.mean(ys)
        #x0 = np.percentile(xs, 15)
        #x1 = np.percentile(xs, 85)
        #y0 = np.percentile(ys, 15)
        #y1 = np.percentile(ys, 85)
        
        # midpoint 
        xmid, ymid = np.mean([minx,maxx]), np.mean([miny,maxy])
        dx = dl*(maxx - minx) / 2
        dy = dl*(maxy - miny) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        yolt_row = convert((w,h), [x0,x1,y0,y1])
        yolt_coords.append(yolt_row)
        
        row = [[x0, y0],
               [x0, y1],
               [x1, y1],
               [x1, y0]]
        cont_plot_box.append(np.rint(row).astype(int))
        
    return yolt_coords, cont_plot_box

###############################################################################
def rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=False):
    '''
    take images in indir and rescale them to the appropriate GSD
    assume inputs are square
    if resize_orig, rescale downsampled image up to original image size
    '''
    
    t0 = time.time()
    print ("indir:", indir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #filelist = [f for f in os.listdir(indir) if f.endswith('.png')]
    filelist = [f for f in os.listdir(indir) if f.endswith('.tif')]
    lenf = len(filelist)
    for i,f in enumerate(filelist):
        if (i % 100) == 0:
            print (i, "/", lenf )
        ftot = indir + f
        # load image
        img_in = cv2.imread(ftot, 1)
            
        # set kernel, if kernel = 1 blur will be non-zero, so mult by 0.5 
        kernel = 0.5 * outGSD/inGSD #int(round(blur_meters/GSD_in))
        img_out = cv2.GaussianBlur(img_in, (0, 0), kernel, kernel, 0)
        
        # may want to rescale?
        # reshape, assume that the pixel density is double the point spread
        # function sigma value
        # use INTER_AREA interpolation function
        rescale_frac = inGSD / outGSD
        rescale_shape = int( np.rint(img_in.shape[0] * rescale_frac) ) # / kernel)# * 0.5)# * 2
        #print "rescale_shape:", rescale_shape
        #print "f", f, "kernel", kernel, "shape_in", img_in.shape[0], "shape_out", rescale_shape

        # resize to the appropriate number of pixels for the given GSD
        img_out = cv2.resize(img_out, (rescale_shape,rescale_shape), interpolation=cv2.INTER_AREA)

        if resize_orig:
            # scale back up to original size (useful for length calculations, but
            #   keep pixelization)
            img_out = cv2.resize(img_out, (img_in.shape[1], img_in.shape[0]), interpolation=cv2.INTER_LINEAR)#cv2.INTER_NEAREST)
        
        # write to file
        outf = outdir + f
        #print "outf:", outf
        #outf = fold_tot + f.split('.')[0] + '_blur' + str(blur_meters) + 'm.' + f.split('.')[1] 
        cv2.imwrite(outf, img_out)

    print ("Time to rescale", lenf, "images from", indir, inGSD, "GSD, to", \
            outdir, outGSD, "=", time.time() - t0, "seconds" )
    return
