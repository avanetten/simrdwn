#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:14:08 2018

@author: avanetten

See:
https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/geoTools.py
https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
"""

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
# import json
# import os


###############################################################################
def geomPixel2geomGeo(shapely_geom, affineObject=[],
                      input_raster='', gdal_geomTransform=[]):

    """
    Transform a shapely geometry from pixel to geospatial coordinates

    Arguments
    ---------
    shapely_geom : shapely.geometry
        Input geometry in pixel coordinates
    affineObject : affine Object ?
        Object that is used to transform geometries (Preferred method).
        This argument supercedes input_raster. Defaults to ``[]``.
    input_raster : str
        Path to raster to gather georectification information. If string is
        empty, use gdal_geomTransform.  Defaults to ``''``.
    gdal_geomTransform : gdal transform?
        The geographic transform of the image. Defaults to ``[]``.

    Returns
    -------
    geomTransform : shapely transform?
        Geometric transform.
    """

    if not affineObject:
        if input_raster != '':
            affineObject = rio.open(input_raster).transform
        else:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)

    if not gdal_geomTransform:
        gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()

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
def get_row_geo_coords(row, affineObject=[], gdal_geomTransform=[],
                       inProj_str='epsg:4326', outProj_str='epsg:3857',
                       verbose=False):

    """
    Get geo coords for SIMRSWN dataframe row containing pixel coords.

    Arguments
    ---------
    row : pandas dataframe
        Row in the pandas dataframe
    affineObject : affine Object ?
        Object that is used to transform geometries. Defaults to ``[]``.
    gdal_geomTransform : gdal transform?
        The geographic transform of the image. Defaults to ``[]``.
    inProj_str : str
        Projection of input image.  Defaults to ``'epsg:4326'`` (WGS84)
    outProj_str : str
        Desired projection of bounding box coordinate.
        Defaults to ``'epsg:3857'`` (WGS84 Web Mercator for mapping).
    verbose : boolean
        Switch to print values to screen. Defaults to ``False``.

    Returns
    -------
    out_arr, poly_geo : tuple
        out_arr is a list of geo coordinates computd from pixels coords:
        [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]
        poly_geo is the shapely polygon in geo coords of the box
    """

    # convert latlon to wmp
    inProj = pyproj.Proj(init=inProj_str)    # (init='epsg:4326')
    outProj = pyproj.Proj(init=outProj_str)  # init='epsg:3857')

    raster_loc = row['Image_Path']
    x0, y0 = row['Xmin_Glob'], row['Ymin_Glob']
    x1, y1 = row['Xmax_Glob'], row['Ymax_Glob']
    if verbose:
        print("idx,  x0, y0, x1, y1:", row.values[0], x0, y0, x1, y1)

    poly = shapely.geometry.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    # transform
    t01 = time.time()
    poly_geo = geomPixel2geomGeo(poly, input_raster=raster_loc,
                                 affineObject=affineObject,
                                 gdal_geomTransform=gdal_geomTransform)
    # print ("geo_coords.coords:", list(poly_geo.exterior.coords))
    t02 = time.time()
    if verbose:
        print("  Time to compute transform:", t02-t01, "seconds")

    # x-y bounding box is a (minx, miny, maxx, maxy) tuple.
    lon0, lat0, lon1, lat1 = poly_geo.bounds
    # wkt_latlon = poly_geo.wkt
    if verbose:
        print("  lon0, lat0, lon1, lat1:", lon0, lat0, lon1, lat1)

    # convert to other coords?:
    #  https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
    #  https://openmaptiles.com/coordinate-systems/
    #  https://ocefpaf.github.io/python4oceanographers/blog/2013/12/16/utm/
    #    Web Mercator projection (EPSG:3857)
    # convert to wmp
    x0_wmp, y0_wmp = pyproj.transform(inProj, outProj, lon0, lat0)
    x1_wmp, y1_wmp = pyproj.transform(inProj, outProj, lon1, lat1)

    # create data array
    out_arr = [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]

    return out_arr, poly_geo


###############################################################################
def add_geo_coords_to_df(df_, inProj_str='epsg:4326', outProj_str='epsg:3857',
                         create_geojson=False, verbose=False):
    '''Determine geo coords (latlon + wmp) of the bounding boxes'''

    """
    Get geo coords for SIMRSWN dataframe containing pixel coords.

    Arguments
    ---------
    df_ : pandas dataframe
        Dataframe containing bounding box predictions
    inProj_str : str
        Projection of input image.  Defaults to ``'epsg:4326'`` (WGS84)
    outProj_str : str
        Desired projection of bounding box coordinate.
        Defaults to ``'epsg:3857'`` (WGS84 Web Mercator for mapping).
    create_geojson : boolean
        Switch to output a geojson.  If False, return [].
        Defaults to ``False``.
    verbose : boolean
        Switch to print values to screen. Defaults to ``False``.

    Returns
    -------
    df_, json : tuple
        df_ is the updated dataframe with geo coords.
        json is the optional GeoJSON corresponding to the boxes.
    """

    t0 = time.time()
    print("Adding geo coords...")

    # first, create a dictionary of geo transforms
    raster_locs = np.unique(df_['Image_Path'].values)
    geo_dict = {}
    for raster_loc in raster_locs:
        # raster geo transform
        gdal_geomTransform = gdal.Open(raster_loc).GetGeoTransform()
        affineObject = rio.open(raster_loc).transform
        if verbose:
            print("raster_loc:", raster_loc)
            print("gdal_geomTransform:", gdal_geomTransform)
            print("affineObject:", affineObject)
        geo_dict[raster_loc] = [gdal_geomTransform, affineObject]

    # iterate through dataframe
    # columns = ['geometry']
    out_arr_json = []
    out_arr = []
    for idx, row in df_.iterrows():
        # get transform
        [gdal_geomTransform, affineObject] = geo_dict[row['Image_Path']]
        if verbose:
            print(idx, " row['Xmin_Glob'], row['Ymin_Glob'],",
                  "row['Xmax_Glob'], row['Ymax_Glob']",
                  row['Xmin_Glob'], row['Ymin_Glob'],
                  row['Xmax_Glob'], row['Ymax_Glob'])

        out_arr_row, poly_geo = get_row_geo_coords(
            row, affineObject=affineObject,
            gdal_geomTransform=gdal_geomTransform,
            inProj_str=inProj_str, outProj_str=outProj_str,
            verbose=verbose)
        out_arr.append(out_arr_row)
        if create_geojson:
            out_arr_json.append(poly_geo)

    # update dataframe
    # [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]
    out_arr = np.array(out_arr)
    df_['lon0'] = out_arr[:, 0]
    df_['lat0'] = out_arr[:, 1]
    df_['lon1'] = out_arr[:, 2]
    df_['lat1'] = out_arr[:, 3]
    df_['x0_wmp'] = out_arr[:, 4]
    df_['y0_wmp'] = out_arr[:, 5]
    df_['x1_wmp'] = out_arr[:, 6]
    df_['y1_wmp'] = out_arr[:, 7]

    # geodataframe if desired
    #   https://gis.stackexchange.com/questions/174159/convert-a-pandas-dataframe-to-a-geodataframe
    if create_geojson:
        crs_init = {'init': inProj_str}
        df_json = pd.DataFrame(out_arr_json, columns=['geometry'])
        # add important columns to df_json
        df_json['category'] = df_['Category'].values
        df_json['prob'] = df_['Prob'].values
        gdf = gpd.GeoDataFrame(df_json, crs=crs_init, geometry=out_arr_json)
        json_out = gdf
        # json_out = gdf.to_json()
    else:
        json_out = []

    t1 = time.time()
    print("Time to add geo coords to df:", t1 - t0, "seconds")
    return df_, json_out
