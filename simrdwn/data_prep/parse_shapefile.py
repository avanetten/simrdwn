#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:04:00 2017

@author: avanetten
"""

import geopandas as gpd
import numpy as np
import sys
import os
import shapely
import rasterio
import affine
import gdal, osr, ogr
import cv2
import argparse
import shutil


###############################################################################
def geomGeo2geomPixel(geom, affineObject=[], input_raster='', gdal_geomTransform=[]):
    '''From SpaceNet Utilities'''

    # This function transforms a shapely geometry in geospatial coordinates into pixel coordinates
    # geom must be shapely geometry
    # affineObject = rasterio.open(input_raster).affine
    
    gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    # input_raster is path to raster to gather georectifcation information
    if not affineObject:
        if input_raster != '':
            affineObject = rasterio.open(input_raster).affine
        else:
            affineObject = affine.Affine.from_gdal(gdal_geomTransform)

    affineObjectInv = ~affineObject

    geomTransform = shapely.affinity.affine_transform(geom,
                                      [affineObjectInv.a,
                                       affineObjectInv.b,
                                       affineObjectInv.d,
                                       affineObjectInv.e,
                                       affineObjectInv.xoff,
                                       affineObjectInv.yoff]
                                      )

    return geomTransform


###############################################################################
def geomPixel2geomGeo(geom, affineObject=[], input_raster='', gdal_geomTransform=[]):
    '''
    From SpaceNet Utilities
    # This function transforms a shapely geometry in pixel coordinates into geospatial coordinates
    # geom must be shapely geometry
    # affineObject = rasterio.open(input_raster).affine
    # gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    # input_raster is path to raster to gather georectifcation information
    '''
    gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    if not affineObject:
        if input_raster != '':
            affineObject = rasterio.open(input_raster).affine
        else:
            affineObject = affine.Affine.from_gdal(gdal_geomTransform)
            

###############################################################################
def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    # type: (object, object, object, object, object) -> object
    '''from spacenet geotools'''

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
def transform_crs(input_raster):
    
    '''If crs of input_raster is not epsg:4326, transform it to be so.
    Copy original file to: os.path.join(dirname + '/orig', basename)
    '''

    truth_dir = os.path.dirname(input_raster)
    basename = os.path.basename(input_raster)

    #input_raster = os.path.join(truth_dir, input_raster_part)
    
    orig_dir = os.path.join(truth_dir, 'orig')
    if not os.path.exists(orig_dir): 
        os.mkdir(orig_dir)

    if input_raster.endswith('_new.tif'):
        return
    
    crs = rasterio.open(input_raster).crs 
    
    # check if the crs is not epsg:4326
    if crs.data['init'] == u'epsg:4326':
        return
    
    else:        
        # copy file to orig folder, if it doesn't already exist
        orig_dest = os.path.join(orig_dir, basename)
        if not os.path.exists(orig_dest):
            print ("input_raster:", input_raster)
            print ("  crs:", crs)
            print ("  copy", input_raster, "to", orig_dest)
            shutil.copy(input_raster, orig_dest)
            
        # transform                
        output_raster = input_raster.split('.')[0] + '_new.tif'
        cmd = 'gdalwarp -t_srs "EPSG:4326" ' +  orig_dest + ' ' + output_raster
        os.system(cmd)
        
        # move files around
        os.remove(input_raster)
        shutil.copy(output_raster, input_raster)
        return


        
###############################################################################
def get_gdf_pix_coords(shp_file, im_file, category='',
                       max_aspect_ratio=3, line_padding=0.1,
                       enforce_rects=False,
                       verbose=False):
    '''Get pixel coords of shape file
    If the labels are single linestrings, infer a bounding box
       max_aspect_ratio prevents a window of zero width or height
       line_padding adds a buffer to the boundong box if labels are lines'''
    
    df_shp = gpd.read_file(shp_file)
    init_proj = df_shp.crs
    
    if len(category) == 0:
        category = shp_file.split('.')[0].split('_')[-1]

    if verbose:
        print ("\nGet gdf for shape file:", shp_file)
        print ("init_proj:", init_proj)
        print ("category:", category)
    
    # get transform data (if using latlton2pixel)
    #src_raster = gdal.Open(im_file)
    #transform = src_raster.GetGeoTransform()
    #targetsr = osr.SpatialReference()
    #targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    
    pix_geom_list = []
    pix_geom_poly_list = []
    pix_coords = []
    bad_idxs = []
    x0_list, x1_list, y0_list, y1_list = [], [], [], []
    
    
    #for obj in df_shp['geometry'].values:
    for index, row in df_shp.iterrows():
        
        obj = row['geometry']
        
        # get coords
        #init_coords = list(obj.coords)
        
        ## with latlon2pixel
        #print "init coords:", init_coords
        #for i,c in enumerate(init_coords):
        #    print i, "init_coords:", c
        #    (lon, lat) = c
        #    x,y = latlon2pixel(lat, lon, input_raster='', targetsr=targetsr, 
        #                 geom_transform=transform)
        #    print "x,y:", x,y
            
        # all at once
        try:
            pix = geomGeo2geomPixel(obj, rasterio.open(im_file).affine, im_file)
        except:
            print (index, "bad row:", row)
            bad_idxs.append(index)
            #pix_geom_list.append([0])
            #pix_geom_poly_list.append([0])
            #pix_coords.append([0])
            continue
        
        if verbose:
            print ("pix.geom_type:", pix.geom_type)
            
        # if pix is a linestring with only 2 points,
        # lets make sure that it is transformed into a bounding box
        if len(np.array(pix)) == 2:
            if verbose:
                print ("Inferring bounding box from line...")
            coords = np.array(pix)
            
            # get x and y coords
            x, y = coords.T
            # get midpoints of line
            m_x, m_y = np.mean(x), np.mean(y)
            dx0 = np.abs(x[1] - x[0])
            dy0 = np.abs(y[1] - y[0])
            # check aspect ratio
            if dx0 > max_aspect_ratio * dy0:
                dx = dx0
                dy = dx0 / max_aspect_ratio
            elif dy0 > max_aspect_ratio * dx0:
                dy = dy0
                dx = dy0 / max_aspect_ratio 
            else:
                dx, dy = dx0, dy0
            # add padding
            dx *= (1 + line_padding)
            dy *= (1 + line_padding)
            # create bounding boxes
            x0, x1 = m_x - dx/2,  m_x + dx/2
            y0, y1 = m_y - dy/2,  m_y + dy/2
            out_coords = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
            points = [shapely.geometry.Point(coord) for coord in out_coords]
            pix_poly = shapely.geometry.Polygon([[p.x, p.y] for p in points])
            
        else:
            pix_poly = shapely.geometry.Polygon(pix)
            

        # convert to bounding box, if desired
        if enforce_rects:
            (x0_tmp, y0_tmp, x1_tmp, y1_tmp) = pix_poly.bounds
            pix_poly = shapely.geometry.box(x0_tmp, y0_tmp, x1_tmp, 
                                                 y1_tmp, ccw=True)
            
        minx, miny, maxx, maxy = pix_poly.bounds
        x0_list.append(minx)
        x1_list.append(maxx)
        y0_list.append(miny)
        y1_list.append(maxy)
        pix_geom_list.append(pix)
        pix_geom_poly_list.append(pix_poly)
        pix_coords.append(list(pix.coords))
        
    # drop bad indexs
    df_shp = df_shp.drop(df_shp.index[bad_idxs])
    
    if verbose:
        print ("len df_shp:", len(df_shp))
        print ("len pix_geom_list:", len(pix_geom_list))
        
    df_shp['geometry_pixel'] = pix_geom_list
    df_shp['geometry_poly_pixel'] = pix_geom_poly_list
    df_shp['xmin'] = x0_list
    df_shp['xmax'] = x1_list
    df_shp['ymin'] = y0_list
    df_shp['ymax'] = y1_list
    df_shp['shp_file'] = shp_file
    df_shp['Category'] = category
    df_shp['Image_Path'] = im_file
    df_shp['Image_Root'] = im_file.split('/')[-1]
    
    return df_shp, pix_coords


###############################################################################
def win_jitter(window_size, jitter_frac=0.1):
    '''get x and y jitter'''
    val = np.rint(jitter_frac * window_size)
    dx = np.random.randint(-val, val)
    dy = np.random.randint(-val, val)
    
    return dx, dy


###############################################################################
def get_window_geoms(df, window_size=416, jitter_frac=0.2, verbose=False):
    '''Iterate through dataframe and get the window cutouts centered on each
    object, modulu some jitter'''
    
    geom_windows = []
    for index, row in df.iterrows():
        print ("\n", index, row['Category'])
        # get coords
        geom_pix = row['geometry_poly_pixel']
        #pix_coords = list(geom_pix.coords)
        bounds = geom_pix.bounds
        area = geom_pix.area
        (minx, miny, maxx, maxy) = bounds
        dx, dy = maxx-minx, maxy-miny
        if verbose:
            print ("bounds:", bounds )
            print ("dx, dy:", dx, dy )
            print ("area:", area )
        
        # get centroid
        centroid = geom_pix.centroid
        #print "centroid:", centroid
        cx_tmp, cy_tmp = list(centroid.coords)[0]
        cx, cy = np.rint(cx_tmp), np.rint(cy_tmp)
        
        # get window coords, jitter, and shapely geometry for window
        jx, jy = win_jitter(window_size, jitter_frac=jitter_frac)
        x0 = cx - window_size/2 + jx
        y0 = cy - window_size/2 + jy
        x1 = x0 + window_size
        y1 = y0 + window_size
        win_p1 = shapely.geometry.Point(x0, y0)
        win_p2 = shapely.geometry.Point(x1, y0)
        win_p3 = shapely.geometry.Point(x1, y1)
        win_p4 = shapely.geometry.Point(x0, y1)
        pointList = [win_p1, win_p2, win_p3, win_p4, win_p1]
        geom_window = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
        if verbose:
            print ("geom_window.bounds", geom_window.bounds )
        geom_windows.append(geom_window)

    return geom_windows


###############################################################################
def get_objs_in_window(df_, geom_window, min_obj_frac=0.7, 
                       use_box_geom=True, verbose=False):    
    '''Find all objects in the window'''
    
    (minx_win, miny_win, maxx_win, maxy_win) = geom_window.bounds
    if verbose:
        print ("geom_window.bounds:", geom_window.bounds)

    obj_list = []
    for index_nest, row_nest in df_.iterrows():
        cat_nest = row_nest['Category']
        geom_pix_nest_tmp = row_nest['geometry_poly_pixel']
        
        # if use_box_geom, turn the shapefile object geom into a bounding box
        if use_box_geom:
            (x0, y0, x1, y1) = geom_pix_nest_tmp.bounds
            geom_pix_nest = shapely.geometry.box(x0, y0, x1, y1, ccw=True)
        else:
            geom_pix_nest = geom_pix_nest_tmp
        
        #pix_coords = list(geom_pix.coords)
        #bounds_nest = geom_pix_nest.bounds
        area_nest = geom_pix_nest.area
        # sometimes we get an invalid geometry, not sure why
        try:
            intersect_geom = geom_pix_nest.intersection(geom_window)
        except:
            # create a buffer around the exterior
            geom_pix_nest = geom_pix_nest.buffer(0)
            intersect_geom = geom_pix_nest.intersection(geom_window)
            print ("Had to update geom_pix_nest:", geom_pix_nest.bounds  )
            
        intersect_bounds = intersect_geom.bounds
        intersect_area = intersect_geom.area
        intersect_frac = intersect_area / area_nest
        
        
        # skip if object not in window, else add to window
        if intersect_frac < min_obj_frac:
            continue
        else:
            # get window coords
            (minx_nest, miny_nest, maxx_nest, maxy_nest) = intersect_bounds
            dx_nest, dy_nest = maxx_nest - minx_nest, maxy_nest - miny_nest
            x0_obj, y0_obj = minx_nest - minx_win, miny_nest - miny_win
            x1_obj, y1_obj = x0_obj + dx_nest, y0_obj + dy_nest
        
            x0_obj, y0_obj, x1_obj, y1_obj = np.rint(x0_obj), np.rint(y0_obj),\
                                             np.rint(x1_obj), np.rint(y1_obj)
            obj_list.append([index_nest, cat_nest, x0_obj, y0_obj, x1_obj, 
                             y1_obj])                                
            if verbose:
                print (" ", index_nest, "geom_obj.bounds:", geom_pix_nest.bounds )
                print ("  intesect area:", intersect_area )
                print ("  obj area:", area_nest )
                print ("  intersect_frac:", intersect_frac )
                print ("  intersect_bounds:", intersect_bounds )
                print ("  category:", cat_nest )
                
    return obj_list
    

###############################################################################
def get_image_window(im, window_geom):
    '''Get sub-window in image'''  
    
    bounds_int = [int(itmp) for itmp in window_geom.bounds]
    (minx_win, miny_win, maxx_win, maxy_win) = bounds_int
    window = im[miny_win:maxy_win, minx_win:maxx_win]
    return window


###############################################################################
def plot_obj_list(window, obj_list, color_dic, thickness=2,
                  show_plot=False, outfile=''):
    '''Plot the cutout, and the object bounds'''
        
    print ("window.shape:", window.shape )

    for row in obj_list:
        [index_nest, cat_nest, x0_obj, y0_obj, x1_obj, y1_obj] = row
        color = color_dic[cat_nest]
        
        cv2.rectangle(window, (int(x0_obj), int(y0_obj)), 
                      (int(x1_obj), int(y1_obj)), 
                      (color), thickness)    

        if show_plot:
            cv2.imshow(str(index_nest), window)
            cv2.waitKey(0)
        
    if outfile:
        cv2.imwrite(outfile, window)


###############################################################################
def plot_obj_list_v0(im, window_geom, obj_list, color_dic, thickness=2,
                  show_plot=False, outfile=''):
    '''Plot the cutout, and the object bounds'''
    
    bounds_int = [int(itmp) for itmp in window_geom.bounds]
    (minx_win, miny_win, maxx_win, maxy_win) = bounds_int
    
    
    window = im[miny_win:maxy_win, minx_win:maxx_win]
    #window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
    print ("window.shape:", window.shape )

    for row in obj_list:
        [index_nest, cat_nest, x0_obj, y0_obj, x1_obj, y1_obj] = row
        color = color_dic[cat_nest]
        
        cv2.rectangle(window, (int(x0_obj), int(y0_obj)), 
                      (int(x1_obj), int(y1_obj)), 
                      (color), thickness)    

        if show_plot:
            cv2.imshow(str(index_nest), window)
            cv2.waitKey(0)
        
    if outfile:
        cv2.imwrite(outfile, window)




###############################################################################
###############################################################################
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', default='/Users/avanetten/cosmiq/simrdwn', 
        type=str,
        help='Data directory')
    parser.add_argument('--outdir', 
        default='/Users/avanetten/cosmiq/simrdwn/sweet_sweet_nectar', type=str,
        help='Output folder for labels')
    parser.add_argument('--im_file', default='pic.tif', type=str,
        help='Image file to analyze')
    parser.add_argument('--shp_file', default='obj.shp', type=str,
        help='Shapefile containing object labels')
    parser.add_argument('--window_size', default=416, type=int,
        help='Size of window cutouts (assumed to be square)')
    parser.add_argument('--jitter_frac', default=0.2, type=float,
        help='Jiffer fraction')
    parser.add_argument('--min_obj_frac', default=0.7, type=float,
        help='Minimum portion of object to include in a subwindow label')
    parser.add_argument('--max_plots', default=20, type=int,
        help='Maximum number of illustrative plots to create')
    parser.add_argument('--max_redundancy', default=10, type=int,
        help='By default, we extract a window for each object.  For densely' \
        + ' packed objects this means many windows that are nearly identical.'\
        + ' max_reduncancy sets the maximum times a given obejct can be seen.'\
        + ' If any object has been seen more that this value, skip the window')
    parser.add_argument('--yolt_dir', default='/Users/avanetten/cosmiq/simrdwn', type=str,
        help='Location of yolt directory')
    parser.add_argument('--overwrite_output', default='True', type=str,
        help='If true, overwrite output directory')
    parser.add_argument('--augment', default='False', type=str,
        help='If true, augment data with rotations and hsv rescalings')
    parser.add_argument('--image_extension', default='.png', type=str,
        help='Extension for output images, set to "" to use input extension')
    
    args = parser.parse_args()
    print ("args:", args )
    
    sys.path.extend([args.yolt_dir])
    import convert
    import yolt_data_prep_funcs

    verbose = True  
    plot_thickness = 2
    im_root = args.im_file.split('.')[0]
    if len(args.image_extension) == 0:
        ext = '.'+ args.im_file.split('.')[-1]
    else:
        ext = args.image_extension
    color_dic =     {'airplane':        (0,   0,   255),
                     'airport:':        (0,   140, 255),
                     'boat':            (255, 0,   0),
                     'boat_harbor':     (0,   255, 0),
                     'car':             (125, 125, 0)}
    cat_idx_dic =   {'airplane':        1,
                     'airport:':        2,
                     'boat':            3,
                     'boat_harbor':     4,
                     'car':             5}
        
#    if args.overwrite_output.upper() == 'TRUE':
#        print ("Removing outdir:", args.outdir
#        if os.path.exists(args.outdir):
#            # lets make sure we don't remove something important on accident
#            if len(args.outdir) > len()
#            shutil.rmtree(args.outdir)
    
    outdir_tot = os.path.join(args.outdir, im_root)
    outdir_ims = os.path.join(outdir_tot, 'images')
    outdir_labels = os.path.join(outdir_tot, 'labels')
    outdir_plots = os.path.join(outdir_tot, 'example_label_plots')
    outdir_yolt_plots = os.path.join(outdir_tot, 'yolt_plot_bboxes')
    train_images_list_file = os.path.join(outdir_tot, 'train_images_list.txt')
    for f in [outdir_tot, outdir_ims, outdir_labels]:
        if not os.path.exists(f): 
            os.mkdir(f)
    if (args.max_plots > 0):
        for f in [outdir_plots, outdir_yolt_plots]:
            if not os.path.exists(f): 
                os.mkdir(f)
    
    # clean directories
    if args.overwrite_output.upper() == 'TRUE':
        dirs_clean = [outdir_labels, outdir_ims, outdir_plots, 
                      outdir_yolt_plots]
        print ("Cleansing dirs :", dirs_clean )
        for d in dirs_clean:
            shutil.rmtree(d)
            os.mkdir(d)
    
    #######################
    # ingest files
    #im_file = os.path.join(indir, im_root + '.tif')
    im_file = os.path.join(args.indir, args.im_file)
    
    print ("Reading file:", im_file, "..." )
    if not os.path.exists(im_file):
        print ("   file DNE..." )
    im = cv2.imread(im_file, 1)  # fails on some tiffs
    
    # sometimes opencv can't read weird tiffs.  If so, convert
    # gdal_translate -co "COMPRESS=LZW" -co PHOTOMETRIC=rgb in.tif out.tif
    # gdalinfo out.tif 
    
    # other ingest options?
    #im = tiff.imread(im_file)
    #im_pil = Image.open(im_file)
    #im_rgb = np.array(im_pil)
    #im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    
    if verbose:
        print ("im_file:", im_file )
        print ("  im.shape:", im.shape )
    
    # get shape files
    #shp_files  = glob.glob(os.path.join(indir, im_root + '*.shp'))
    shp_files  = [os.path.join(args.indir, args.shp_file)]
    # create combined dataframe
    df = []
    for i,shp_f in enumerate(shp_files):
        category = shp_f.split('_')[-1].split('.')[0]
        #category = args.category
        print ("shape_file:", shp_f )
        print (os.path.exists(shp_f ))
        print ("  category:", category )
        df_tmp, _ =  get_gdf_pix_coords(shp_f, im_file, category)
        if i == 0:
            df = df_tmp
        else:
            df = df.append(df_tmp)
    df.index = np.arange(len(df))
    
    if verbose:
        print ("df.columns:", df.columns )
    #######################   
    
    # get window cutouts centered at each object
    window_geoms = get_window_geoms(df, window_size=args.window_size, 
                                    jitter_frac=args.jitter_frac, 
                                    verbose=verbose)
    
    idx_count_dic = {}
    for idx_tmp in df.index:
        idx_count_dic[idx_tmp] = 0
    idx_count_tot_dic = {}
    for idx_tmp in df.index:
        idx_count_tot_dic[idx_tmp] = 0    
    # get objects in each window
    win_iter = 0
    for i,window_geom in enumerate(window_geoms):

        (minx_win, miny_win, maxx_win, maxy_win) = window_geom.bounds
        
        # get window
        window = get_image_window(im, window_geom)
        h, w = window.shape[:2]
        if (h==0) or (w==0):
            continue

        # get objects in window
        obj_list = get_objs_in_window(df, window_geom, 
                                      min_obj_frac=args.min_obj_frac,
                                      verbose=verbose)  
        if verbose:
            print ("\nWindow geom:", window_geom )
            print ("  window shape:", window.shape )
            print ("  obj_list:", obj_list )
    
        if len(obj_list) > 0 :
            
            # update idx_count_tot_dic
            idxs_list = [z[0] for z in obj_list]
            for idx_tmp in idxs_list:
                idx_count_tot_dic[idx_tmp] += 1
                
            # Check idx count dic.  If an object has appeared too frequently,
            #   skip the window
            excess = False
            for idx_tmp in idxs_list:
                if idx_count_dic[idx_tmp] >= args.max_redundancy:
                    print ("Index", idx_tmp, "seen too frequently, skipping..." )
                    excess = True
                    break
            if excess:
                continue
            
            # create yolt images and labels
            outroot =    im_root + '|x0_' + str(int(minx_win)) + '_y0_' \
                                          + str(int(miny_win)) + '_dxdy_' \
                                          + str(int(args.window_size))
                                          
            image_outfile = os.path.join(outdir_ims, outroot + ext)
            label_outfile = os.path.join(outdir_labels, outroot + '.txt')
            plot_outfile = os.path.join(outdir_plots,  outroot + ext)

            if verbose:
                print ("  Saving window to file..." )
                print ("    window.dtype:", window.dtype )
                print ("    window.shape:", window.shape )
            # save image
            cv2.imwrite(image_outfile, window)
                
            # get yolt labels
            #if verbose:
            #    print ("  Creating yolt labels..."
            yolt_coords = []
            for row in obj_list:
                [index_nest, cat_nest, x0, y0, x1, y1] = row
                cat_idx = cat_idx_dic[cat_nest]
                yolt_row = [cat_idx, cat_nest] + list(convert.convert((w,h), [x0,x1,y0,y1]))
                yolt_coords.append(yolt_row)
            if verbose:
                print ("   yolt_coords:", yolt_coords )
                
            # save labels
            txt_outfile = open(label_outfile, "w")
            for j, yolt_row in enumerate(yolt_coords):
                cls_id = yolt_row[0]
                bb = yolt_row[2:]
                outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                #print ("outstring:", outstring
                txt_outfile.write(outstring)
            txt_outfile.close()             

            # make plots
            if i <= args.max_plots:
                if verbose:
                    print ("obj_list:",obj_list )
                    print ("plot outfile:", plot_outfile )
                #im_copy = im.copy()
                #plot_obj_list(im_copy, window_geom, obj_list, color_dic, 
                plot_obj_list(window.copy(), obj_list, color_dic, 
                              thickness=plot_thickness, show_plot=False, 
                              outfile=plot_outfile) 
                
            # update idx_count_dic
            for idx_tmp in idxs_list:
                idx_count_dic[idx_tmp] += 1
            # update win_iter
            win_iter += 1


    # augment, if desired
    if args.augment.upper() == 'TRUE':
        yolt_data_prep_funcs.augment_training_data(outdir_labels, outdir_ims, 
                                                   hsv_range=[0.5,1.5],
                                                   ext=ext,
                                                   skip_hsv_transform=False)
                
    # make sure labels and images are created correctly
    print ("\nPlotting yolt training bounding boxes..." )
    print ("outdir_labels:", outdir_labels )
    print ("outdir_ims:", outdir_ims )
    yolt_data_prep_funcs.plot_training_bboxes(outdir_labels, outdir_ims, 
                                        ignore_augment=True,
                 figsize=(10,10), color=(0,0,255), thickness=2, 
                 max_plots=100, sample_label_vis_dir=outdir_yolt_plots,
                 ext=ext, verbose=verbose, show_plot=False, 
                 specific_labels=[], label_dic=[], output_width=500,
                 shuffle=True)
                
    # make training list
    list_file = open(train_images_list_file, 'wb')
    for f in os.listdir(outdir_ims):
        list_file.write('%s/%s\n'%(outdir_ims, f))
    list_file.close()  
    
    print ("\nArgs:", args )
    print ("\nidx_count_tot_dic:", idx_count_tot_dic )
    print ("\nidx_count_dic:", idx_count_dic )
    print ("\nArgs:", args )
    print ("\ndf.columns:", list(df.columns) )
    print ("Number of objects:", len(df) )
    print ("Number of original images used:", win_iter )
    print ("Number of orginal+augmented images used:", len(os.listdir(outdir_ims)) )
    cats = np.unique(df['Category']) 
    print ("Categories in image file:", im_file, ':', cats )
    for c in cats:
        print ("Number of items of category:", category, ":", len(df[df['Category'] == c]) )

    # print (to file
    lines = ['Args ' + str(args) + '\n',
             'Categories in image file: ' + im_file +  ' : ' + str(cats) + '\n',
             'Number of objects: ' + str(len(df)) + '\n',
             'Number of original images used: ' + str(win_iter) + '\n',
             'Number of orginal+augmented images used: ' + str(len(os.listdir(outdir_ims))) + '\n',
            ]

    log_file = os.path.join(outdir_tot, 'log.txt')
    f = open(log_file,'w')
    for l in lines:
        f.write(l)
    f.close()

    return

###############################################################################
if __name__ == "__main__":
    main()

