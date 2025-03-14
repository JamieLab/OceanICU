#!/usr/bin/env python

import datetime
import os
import glob
import time
import data_utils as du
from netCDF4 import Dataset
import numpy as np
import calendar

def cci_sss_spatial_average(data='E:/Data/SSS-CCI/v4.41/%Y/ESACCI-SEASURFACESALINITY-L4-SSS-GLOBAL-MERGED_OI_Monthly_CENTRED_15Day_0.25deg-%Y%m15-fv4.41.nc',start_yr = 1981, end_yr=2023,out_loc='',log='',lag='',flip=False,area_wei=False):
    du.makefolder(out_loc)
    res = np.round(np.abs(log[0]-log[1]),2)
    ye = start_yr
    mon = 1

    t=0
    while ye <= end_yr:

        du.makefolder(os.path.join(out_loc,str(ye)))
        file = data.replace('%Y',str(ye)).replace('%m',du.numstr(mon))
        file_o = os.path.join(out_loc,str(ye),str(ye)+du.numstr(mon)+f'_ESA_SSS_MONTHLY_SSS_{res}_deg.nc')
        print(file)

        if (du.checkfileexist(file_o) == 0) & (du.checkfileexist(file) == 1):
            if t == 0:
                [lon,lat] = du.load_grid(file,latv='lat',lonv='lon')
                [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
                #print(lo_grid)
                #print(la_grid)
                t = 1
            c = Dataset(file,'r')
            sst = np.squeeze(c.variables['sss'][:]); sst[sst<0] = np.nan
            unc = np.squeeze(c.variables['sss_random_error'][:])*2 # We want 2 sigma/95% confidence uncertainties
            # lon = np.array(c.variables['lon'][:])
            # lat = np.array(c.variables['lat'][:])
            vers = c.product_version
            c.close()
            #print(sst.shape)

            if sst.shape[1]>sst.shape[0]:
                sst = np.transpose(sst)
                unc = np.transpose(unc)
            sst_o = du.grid_average(sst,lo_grid,la_grid,lon=lon,lat=lat,area_wei=True)
            unc_o = du.grid_average(unc,lo_grid,la_grid,lon=lon,lat=lat,area_wei=True)

            du.netcdf_create_basic(file_o,sst_o,'sss',lag,log,flip=flip,units='')
            du.netcdf_append_basic(file_o,unc_o,'sss_random_error',flip=flip,units = '')
            c = Dataset(file_o,'a')
            c.variables['sss_random_error'].uncertainty = 'These uncertainties are 2 sigma (95% confidence) equivalents!'
            if area_wei:
                c.variables['sss_random_error'].area_weighted_average = 'True'
                c.variables['sss'].area_weighted_average = 'True'

            c.monthly_file_loc = data
            c.output_loc = out_loc
            c.start_yr = start_yr
            c.end_yr = end_yr
            c.product_version = vers

            if flip:
                c.flip = 'True'
            else:
                c.flip = 'False'
            if area_wei:
                c.area_weighting = 'True'
            else:
                c.area_weighting = 'False'
            c.created_with = 'cci_sss.cci_sss_spatial_average'
            c.close()
        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1
