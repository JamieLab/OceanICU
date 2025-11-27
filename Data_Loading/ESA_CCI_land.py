#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import cmocean
import geopandas as gpd
import matplotlib.transforms
from netCDF4 import Dataset
import datetime

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import sys
import os
sys.path.append(os.path.join('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU','Data_Loading'))
sys.path.append(os.path.join('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU'))
import data_utils as du

def land_proportion_calc(var,lo_grid,la_grid):
    out = np.empty((len(lo_grid),len(la_grid)))
    out[:] = np.nan
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            if (la_grid[j][0].size == 0) or (lo_grid[i][0].size == 0):
                out[i,j] = np.nan
            else:
                temp_lo,temp_la = np.meshgrid(lo_grid[i],la_grid[j])
                temp_lo = temp_lo.ravel(); temp_la = temp_la.ravel()
                v = var[temp_lo,temp_la]
                #print(v)
                p = np.where(v ==0)
                #print(len(p[0]))
                #print(v.size)
                out[i,j] = len(p[0]) / v.size
    return out

def generate_land_cci(file,file_tif,long,latg,outloc):
    if not du.checkfileexist(outloc):
        area = np.transpose(du.area_grid(long,latg,np.abs(long[0]-long[1])))
        c = Dataset(file,'r')
        lat = np.array(c['lat'][::1])
        lon = np.array(c['lon'][::1])
        c.close()



        with rasterio.open(file_tif) as src:
            data= np.transpose(src.read(1)[::1,::1])

        lo_grid,la_grid = du.determine_grid_average(lon,lat,long,latg)
        out = land_proportion_calc(data,lo_grid,la_grid)
        land = np.abs(1-out)
        du.netcdf_create_basic(outloc,out,'ocean_proportion',latg,long)
        du.netcdf_append_basic(outloc,land,'land_proportion')
        du.netcdf_append_basic(outloc,area,'area')
        c=Dataset(outloc,'a')
        c.input_netcdf = file
        c.input_tif = file_tif
        c.generation_code = 'ESA_CCI_land.generate_land_cci'
        c.close()
    else:
        print('File: "' + outloc + '" exists')
