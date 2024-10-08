#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 05/2023

"""

import datetime
import data_utils as du
import os
from netCDF4 import Dataset
import numpy as np

def bicep_pp_log_average(loc,outloc,filename_struct='',start_yr=1990,end_yr=2023,res=1,netcdf_var = 'npp',make_month_folder = True,log='',lag=''):
    du.makefolder(outloc)
    # log,lag = du.reg_grid(lon=res,lat=res)
    res = np.abs(log[0] - log[1])
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        d = datetime.datetime(yr,mon,1)
        if mon == 1:
            if make_month_folder:
                du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,filename_struct.replace('%Y',d.strftime('%Y')).replace('%m',d.strftime('%m')))
        outfile = os.path.join(outloc,str(yr),filename_struct.replace('%Y',d.strftime('%Y')).replace('%m',d.strftime('%m')))
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file,latv='lat',lonv='lon')

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables[netcdf_var][:])))
            va_da[va_da <= 0.0] = np.nan
            va_da = np.log10(va_da)
            va_da[va_da > 10] = np.nan
            #print(va_da)
            c.close()
            #lon,va_da=du.grid_switch(lon,va_da)

            if t == 0:
                lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                t = 1
            va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            du.netcdf_create_basic(outfile,va_da_out,netcdf_var,lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def bicep_poc_log_average(loc,outloc,start_yr=1990,end_yr=2023,res=1):
    du.makefolder(outloc)
    log,lag = du.reg_grid(lon=res,lat=res)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),'BICEP_NCEO_POC_ESA-OC-L3S-MERGED-1M_MONTHLY_4km_GEO_PML_'+str(str(yr)) + du.numstr(mon)+'-fv5.0.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_BICEP_NCEO_POC_ESA-OC-L3S-MERGED-1M_MONTHLY_4km_GEO_PML-fv5.0_'+str(res)+'_deg.nc')
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file,latv='lat',lonv='lon')

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables['POC'][:])))
            va_da[va_da <= 0.0] = np.nan
            va_da = np.log10(va_da)
            va_da[va_da > 10] = np.nan
            #print(va_da)
            c.close()
            #lon,va_da=du.grid_switch(lon,va_da)

            if t == 0:
                lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                t = 1
            va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            du.netcdf_create_basic(outfile,va_da_out,'POC',lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def bicep_ep_log_average(loc,outloc,var= [],start_yr=1990,end_yr=2023,res=1):
    du.makefolder(outloc)
    log,lag = du.reg_grid(lon=res,lat=res)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),'BICEP_NCEO_ExportProduction_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_'+str(str(yr)) + du.numstr(mon)+'-fv4.2.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_BICEP_NCEO_ExportProduction_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_fv4.2_'+str(res)+'_deg_' + var +'.nc')
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file,latv='lat',lonv='lon')

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables[var][:])))
            va_da[va_da <= 0.0] = np.nan
            va_da = np.log10(va_da)
            va_da[va_da > 10] = np.nan
            #print(va_da)
            c.close()
            #lon,va_da=du.grid_switch(lon,va_da)

            if t == 0:
                lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                t = 1
            va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            du.netcdf_create_basic(outfile,va_da_out,var,lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
