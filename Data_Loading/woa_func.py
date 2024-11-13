#!/usr/bin/env python3
"""
Functions for working with World Ocean Atlas data
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 04/2024
"""
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import data_utils as du
import pandas as pd
from scipy.ndimage import generic_filter

def append_woa(file,delimiter,col_year,col_mon,col_lat,col_lon,woa_location,woa_var,latg,long,woa_save_loc,box=5):
    """
    This function appends the WOA monthly climatological value of a variable to the orginial file.

    file = is the text file containing the data points that need matching to WOA
    delimiter = the delimiter for the file containing the data points
    col_year = the column name for the year
    col_month = the column name for the month
    col_lat = the column name for the latitude
    col_lon = the column name for the longitude
    woa_location = the location of the WOA netcdf's
    woa_var = the variable within the WOA netcdf's to load
    """
    du.makefolder(woa_save_loc)
    data = pd.read_table(file,sep=delimiter)
    uni = np.unique(data[col_mon])
    out = np.zeros(len(data));out[:] = np.nan
    print(out.shape)
    for i in uni:
        print(i)
        print(os.path.join(woa_location,'*'+du.numstr(int(i))+'_01.nc'))
        woa_file = glob.glob(os.path.join(woa_location,'*'+du.numstr(int(i))+'_01.nc'))[0]

        c = Dataset(woa_file,'r')
        lat = np.array(c['lat'])
        lon = np.array(c['lon'])
        res = np.abs(lon[0]-lon[1])
        lon = np.concatenate((lon[0] -res,lon,lon[-1]+res),axis=None)

        woa_data = np.array(c[woa_var][0,0,:,:])
        woa_data[woa_data == c[woa_var]._FillValue] = np.nan
        c.close()

        woa_data_ex = np.zeros((woa_data.shape[0],woa_data.shape[1]+2))
        woa_data_ex[:,0] = woa_data[:,-1]
        woa_data_ex[:,1:-1] = woa_data
        woa_data_ex[:,-1] = woa_data[:,0]
        woa_data_ex2 = generic_filter(woa_data_ex,np.nanmean,[box,box])
        woa_data_ex[np.isnan(woa_data_ex) == 1] = woa_data_ex2[np.isnan(woa_data_ex) == 1]
        f = np.where(data[col_mon] == i)[0]
        a = du.point_interp(lon,lat,woa_data_ex,data[col_lon][f],data[col_lat][f],plot=False)
        grid_out = du.grid_interp(lon,lat,woa_data_ex,long,latg,plot=False)
        du.netcdf_create_basic(os.path.join(woa_save_loc,woa_var+'_'+du.numstr(int(i))+'.nc'),grid_out,woa_var,latg,long)

        out[f] = a
    data[woa_var] = out
    data.to_csv(file,sep=delimiter)
