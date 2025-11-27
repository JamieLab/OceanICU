#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 24 October 2025

# Functions to aid in generating masking variables
# @author: Daniel Ford

import numpy as np;
from netCDF4 import Dataset
import os
import geopandas as gpd
from shapely.geometry import Point
from math import cos, sin, asin, sqrt, radians
import construct_input_netcdf as cinp
import Data_Loading.data_utils as du

def calc_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees):
    from: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points/4913653#4913653
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) # convert decimal degrees to radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2  #haversine formula
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def calc_distance_to_coastline(longitude,latitude,coastline):
    target_coordinate=Point(longitude,latitude)
    return coastline.distance(target_coordinate).values[0]

def distance_degrees_to_kilometers(distance,coord=[0,0]):
    coord_plus=[c+distance for c in coord]
    coord_minus=[c-distance for c in coord]
    return (calc_distance(*coord,*coord_plus)+calc_distance(*coord,*coord_minus))*0.5

def calc_distance_to_coastline_km(longitude,latitude,coastline):
    target_coordinate=Point(longitude,latitude )

    return distance_degrees_to_kilometers(coastline.distance(target_coordinate).values[0],[longitude,latitude])

def mask_distance_to_coast(coastfile,output_file,lon,lat,start_yr,end_yr,mask_var='coast',req_dist =100):
    timesteps = (end_yr-start_yr+1) *12
    coastline_data = gpd.read_file(coastfile)
    coastline = gpd.GeoSeries(coastline_data.geometry.unary_union)

    coast = np.zeros((len(lon),len(lat)))
    for i in range(len(lon)):
        print(i)
        for j in range(len(lat)):

            coast_temp = calc_distance_to_coastline_km(lon[i],lat[j],coastline)
            if coast_temp <= req_dist:
                coast[i,j] = 1
    coast = np.repeat(coast[:, :, np.newaxis], timesteps, axis=2)
    direct = {}
    direct[mask_var] = coast

    if du.checkfileexist(output_file):
        cinp.append_netcdf(output_file,direct,lon,lat,timesteps)

def mask_generated_from_single(input_file,input_var,output_file,mask_var,mask_val,less = True, grea = False, lat_lim=None,lon_lim=None):
    c = Dataset(input_file,'r')
    lon = c.variables['longitude'][:]
    lat = c.variables['latitude'][:]
    var = c.variables[input_var][:]
    time = len(c.variables['time'][:])
    c.close()

    mask = np.zeros((var.shape))
    if less:
        mask[var < mask_val] = 1
    if grea:
        mask[var>mask_val] = 1

    if lat_lim != None:
        f = np.where((lat > lat_lim[1]) | (lat< lat_lim[0]))
        mask[:,f,:] = 0
    if lon_lim != None:
        f = np.where((lon < lon_lim[0]) | (lon> lon_lim[1]))
        mask[f,:,:] = 0

    direct = {}
    direct[mask_var] = mask
    if du.checkfileexist(output_file):
        cinp.append_netcdf(output_file,direct,lon,lat,time)
