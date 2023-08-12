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

def ccmp_average(loc,outloc,start_yr=1990,end_yr=2023,log='',lag='',orgi_res = 0.25):
    du.makefolder(outloc)
    res = np.round(np.abs(log[0]-log[1]),2)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,'y'+str(yr),'m'+du.numstr(mon),'CCMP_Wind_Analysis_'+str(yr) + du.numstr(mon)+'_V03.0_L4.5.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CCMP_Wind_Analysis__V03.0_L4.5_'+str(res)+'_deg.nc')
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file)
            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables['w'][:])))
            va_da[va_da < 0.0] = np.nan
            lon,va_da = du.grid_switch(lon,va_da)
            c.close()

            if res > orgi_res:
                # If we are averaging to a 1 deg grid for example then we use the grid averaging.
                # However if the new grid is higher resolution then we need to spatially interpolate.
                if t == 0:
                    lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                    t = 1
                va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            else:
                va_da_out = du.grid_interp(lon,lat,va_da,log,lag)
                t=1
            du.netcdf_create_basic(outfile,va_da_out,'w',lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
