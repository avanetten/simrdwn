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
def add_geo_coords_to_df(df_, create_geojson=False, 
                         inProj_str='epsg:4326', outProj_str='epsg:3857',
                         verbose=False):
    '''Determine geo coords (latlon + wmp) of the bounding boxes'''

    
    # Placeholder
    return df_, []
    
