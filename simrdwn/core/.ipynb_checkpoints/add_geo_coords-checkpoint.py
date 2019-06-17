#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:14:08 2018

@author: avanetten

https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/geoTools.py

https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects

"""


import os
import gdal
import geopandas as gpd
import shapely
import shapely.geometry
import rasterio as rio
import pyproj
import affine as af
import pandas as pd
import numpy as np
import time
#import json

###############################################################################
def geomPixel2geomGeo(shapely_geom, input_raster='',  
                      affineObject=[], gdal_geomTransform=[] ):
    ''' 
    https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/geoTools.py
    This function transforms a shapely geometry in pixel coordinates 
    into geospatial coordinates
    # geom must be shapely geometry
    # affineObject = rasterio.open(input_raster).affine
    # gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    # input_raster is path to raster to gather georectifcation information
    '''
    if not affineObject:
        if input_raster != '':
            affineObject = rio.open(input_raster).transform
        else:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)

    if not gdal_geomTransform:
        gdal_geomTransform = gdal.Open(raster_loc).GetGeoTransform()


    geomTransform = shapely.affinity.affine_transform(shapely_geom,
                                                      [affineObject.a,
                                                       affineObject.b,
                                                       affineObject.d,
                                                       affineObject.e,
                                                       affineObject.xoff,
                                                       affineObject.yoff]
                                                      )

    return geomTransform


###############################################################################
def add_geo_coords_to_row(row, affineObject=[], gdal_geomTransform=[], 
                          verbose=False):
    '''Determine geo coords (latlon + wmp) of the bounding box'''

    # convert to wmp
    outProj = pyproj.Proj(init='epsg:3857')
    inProj = pyproj.Proj(init='epsg:4326')
    
    raster_loc = row['Image_Path']
    x0, y0 = row['Xmin_Glob'], row['Ymin_Glob'] 
    x1, y1 = row['Xmax_Glob'], row['Ymax_Glob']
    if verbose:
        print ("idx,  x0, y0, x1, y1:", row.values[0], x0, y0, x1, y1)
        
    poly = shapely.geometry.Polygon([ (x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    # transform
    t01 = time.time()
    poly_geo = geomPixel2geomGeo(poly, input_raster=raster_loc, 
                                 affineObject=affineObject, 
                                 gdal_geomTransform=gdal_geomTransform)
    #print ("geo_coords.coords:", list(poly_geo.exterior.coords))
    t02 = time.time()
    if verbose:
        print ("  Time to compute transform:", t02-t01, "seconds")
    
    # x-y bounding box is a (minx, miny, maxx, maxy) tuple.
    lon0, lat0, lon1, lat1 = poly_geo.bounds
    #wkt_latlon = poly_geo.wkt
    if verbose:
        print ("  lon0, lat0, lon1, lat1:", lon0, lat0, lon1, lat1)
        
    # convert to other coords: 
    #  https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
    #  https://openmaptiles.com/coordinate-systems/
    #  https://ocefpaf.github.io/python4oceanographers/blog/2013/12/16/utm/
    #    Web Mercator projection (EPSG:3857)
    # convert to wmp
    x0, y0 = pyproj.transform(inProj, outProj, lon0, lat0)
    x1, y1 = pyproj.transform(inProj, outProj, lon1, lat1)

    # create data array
    out_arr = [lon0, lat0, lon1, lat1, x0, y0, x1, y1]
    
    return out_arr, poly_geo

###############################################################################
def add_geo_coords_to_df_filt(df_filt, create_geojson=False, verbose=False):
    '''Assume df_filt has bounding boxes from only one image.
    Determine geo coords (latlon + wmp) of the bounding boxes'''

    crs_init = {'init': 'epsg:4326'}

    ## convert to wmp
    #outProj = pyproj.Proj(init='epsg:3857')
    #inProj = pyproj.Proj(init='epsg:4326')
    
    raster_loc = df_filt.iloc[0]['Image_Path']
    # get raster geo transform
    gdal_geomTransform = gdal.Open(raster_loc).GetGeoTransform()
    if verbose:
        print ("gdal_geomTransform:", gdal_geomTransform)
    # get affine object
    affineObject = rio.open(raster_loc).transform
    if verbose:
        print ("affineObject:", affineObject)
        
    # iterate through dataframe
    #columns = ['geometry']
    out_arr_json = []
    out_arr = []
    for idx, row in df_filt.iterrows():
        if verbose:
            x0, y0, x1, y1 = row['Xmin_Glob'], row['Ymin_Glob'], row['Xmax_Glob'], row['Ymax_Glob']
            print ("idx,  x0, y0, x1, y1:", idx, x0, y0, x1, y1)

        out_arr_row, poly_geo = add_geo_coords_to_row(row, affineObject=affineObject, 
                                              gdal_geomTransform=gdal_geomTransform, 
                                              verbose=verbose)
        out_arr.append(out_arr_row)
        if create_geojson:
            out_arr_json.append(poly_geo)
        
       
        #x0, y0, x1, y1 = row['Xmin_Glob'], row['Ymin_Glob'], row['Xmax_Glob'], row['Ymax_Glob']
        #if verbose:
        #    print ("idx,  x0, y0, x1, y1:", idx, x0, y0, x1, y1)
        #poly = shapely.geometry.Polygon([ (x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        ## transform
        #t01 = time.time()
        #poly_geo = geomPixel2geomGeo(poly, affineObject=affineObject, 
        #                  input_raster='', 
        #                  gdal_geomTransform=gdal_geomTransform)
        ##print ("geo_coords.coords:", list(poly_geo.exterior.coords))
        #t02 = time.time()
        #if verbose:
        #    print ("  Time to compute transform:", t02-t01, "seconds")
        #
        ## x-y bounding box is a (minx, miny, maxx, maxy) tuple.
        #lon0, lat0, lon1, lat1 = poly_geo.bounds
        ##wkt_latlon = poly_geo.wkt
        #if verbose:
        #    print ("  lon0, lat0, lon1, lat1:", lon0, lat0, lon1, lat1)
        #
        ## convert to other coords: 
        ##  https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
        ##  https://openmaptiles.com/coordinate-systems/
        ##  https://ocefpaf.github.io/python4oceanographers/blog/2013/12/16/utm/
        ##    Web Mercator projection (EPSG:3857)
        ## convert to wmp
        #x0, y0 = pyproj.transform(inProj, outProj, lon0, lat0)
        #x1, y1 = pyproj.transform(inProj, outProj, lon1, lat1)
        #
        ## add to data array
        #out_arr.append(lon0, lat0, lon1, lat1, x0, y0, x1, y1)
        #if create_geojson:
        #    out_arr_json.append(poly_geo)
        
    # update dataframe
    out_arr = np.array(out_arr)
    #df_filt[']
         
    # geodataframe
    #   https://gis.stackexchange.com/questions/174159/convert-a-pandas-dataframe-to-a-geodataframe
    if create_geojson:
        df_json = pd.DataFrame(out_arr_json, columns=['geometry'])
        gdf = gpd.GeoDataFrame(df_json, crs=crs_init, geometry=out_arr_json)
        json_out = gdf.to_json()
    else:
        json_out = []
    
    return out_arr, json_out

###############################################################################
# test
   
this_dir = '/Users/avanetten/Desktop/geos'

# Raster
raster_dir = '/raid/cosmiq/qgis_labels/data/qgis_validation/all'
raster_file = '054593918020_01_assembly_3_5_LondonCityAir.tif'
raster_loc = os.path.join(raster_dir, raster_file)

# dataframe
pred_dir = '/raid/cosmiq/simrdwn/results/valid_yolt_3class_qgis_416res_416slices_2018_03_08_19-03-38/'
pred_csv_loc = os.path.join(pred_dir, 'valid_predictions_refine_thresh=0.1.csv')

df_tot = pd.read_csv(pred_csv_loc)
df_filt = df_tot[df_tot['Image_Root'] == raster_file]
print ("df_filt.iloc[0]:", df_filt.iloc[0])

# raster geo transform
gdal_geomTransform = gdal.Open(raster_loc).GetGeoTransform()
print ("gdal_geomTransform:", gdal_geomTransform)
affineObject = rio.open(raster_loc).transform
print ("affineObject:", affineObject)

# iterate through dataframe
columns = ['geometry']
out_arr = []
for idx, row in df_filt.iterrows():
    x0, y0, x1, y1 = row['Xmin_Glob'], row['Ymin_Glob'], row['Xmax_Glob'], row['Ymax_Glob']
    print ("idx,  x0, y0, x1, y1:", idx, x0, y0, x1, y1)
    
    out_arr_row, poly_geo = add_geo_coords_to_row(row, affineObject=affineObject, 
                                              gdal_geomTransform=gdal_geomTransform, 
                                              verbose=True)
    #poly = shapely.geometry.Polygon([ (x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    ## transform
    #t01 = time.time()
    #poly_geo = geomPixel2geomGeo(poly, affineObject=affineObject, 
    #                  input_raster='', 
    #                  gdal_geomTransform=gdal_geomTransform)
    ##print ("geo_coords.coords:", list(poly_geo.exterior.coords))
    #t02 = time.time()
    #print ("  Time to compute transform:", t02-t01, "seconds")
    ## x-y bounding box is a (minx, miny, maxx, maxy) tuple.
    #lon0, lat0, lon1, lat1 = poly_geo.bounds
    #print ("  lon0, lat0, lon1, lat1:", lon0, lat0, lon1, lat1)
    #wkt = poly_geo.wkt
    
    out_arr.append(poly_geo)
    #out_arr.append([idx, wkt])
         
     
# geodataframe
#   https://gis.stackexchange.com/questions/174159/convert-a-pandas-dataframe-to-a-geodataframe
crs = {'init': 'epsg:4326'}
json_out_file = os.path.join(this_dir, raster_file.split('.')[0] + '.geojson')
df_out = pd.DataFrame(out_arr, columns=columns)
gdf = gpd.GeoDataFrame(df_out, crs=crs, geometry=out_arr)
json_out = gdf.to_json()
# save to file
gdf.to_file(json_out_file, driver="GeoJSON")
#with open(json_out_file, 'wb') as outfile:
#    json.dump(json_out, outfile)
