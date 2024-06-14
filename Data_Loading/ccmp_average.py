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
import gebco_resample as geb
import glob
import matplotlib.pyplot as plt

def ccmp_average(loc,outloc,start_yr=1990,end_yr=2023,log='',lag='',orgi_res = 0.25,var='w',v=3,geb_file = False):
    du.makefolder(outloc)
    res = np.round(np.abs(log[0]-log[1]),2)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        if v == 3:
            file = os.path.join(loc,'y'+str(yr),'m'+du.numstr(mon),'CCMP_Wind_Analysis_'+str(yr) + du.numstr(mon)+'_V03.0_L4.5.nc')
            outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CCMP_Wind_Analysis__V03.0_L4.5_'+str(res)+'_deg_'+var+'.nc')
        elif v == 3.1:
            file = os.path.join(loc,str(yr),f'CCMP_{v}_{var}_{yr}' + du.numstr(mon)+'.nc')

            outfile = os.path.join(outloc,str(yr),f'CCMP_{v}_{var}_{yr}' + du.numstr(mon)+ f'_{res}_deg_{var}.nc')
        print(file)
        print(outfile)
        t2 = 0
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.squeeze(np.array(c.variables[var][:])); va_da[va_da < -100] = np.nan
            va_da2 = np.squeeze(np.array(c.variables[var+'^2'][:])); va_da2[va_da2 < -100] = np.nan
            lon2 = np.copy(lon)
            lon,va_da = du.grid_switch(lon,va_da)
            lon2,va_da2 = du.grid_switch(lon2,va_da2)

            c.close()
            if t2 == 0:
                gebfile = os.path.join(loc,'gebco_0.25_grid.nc')
                geb.gebco_resample(geb_file,lon,lat,save_loc = gebfile)
                c = Dataset(gebfile)
                ocean = np.array(c.variables['ocean_proportion'])
                c.close()

                #lon2 = np.copy(lon)
                #lon2,ocean = du.grid_switch(lon2,ocean)
                #ocean = np.flipud(ocean)
                t2=1
            if res > orgi_res:
                # If we are averaging to a 1 deg grid for example then we use the grid averaging.
                # However if the new grid is higher resolution then we need to spatially interpolate.
                if t == 0:
                    lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                    t = 1
                va_da[ocean == 0.0] = np.nan; va_da2[ocean == 0.0] = np.nan;
                va_da_out = du.grid_average(va_da,lo_grid,la_grid)
                va_da_out2 = du.grid_average(va_da2,lo_grid,la_grid)
            else:
                va_da_out = du.grid_interp(lon,lat,va_da,log,lag)
                va_da_out2 = du.grid_interp(lon,lat,va_da2,log,lag)
                t=1
            du.netcdf_create_basic(outfile,va_da_out,var,lag,log)
            du.netcdf_append_basic(outfile,va_da_out2,var+'^2')
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def ccmp_temporal_average(loc,start_yr=1990,end_yr=2023,var='ws',v=3):
    du.makefolder(os.path.join(loc,'monthly'))
    if start_yr <= 1993:
        ye = 1993
        mon = 1
    else:
        ye = start_yr
        mon = 1
    i = 0 # Variable for loading lat and lon grid once (set to 1 once loaded for the first time)

    while ye <= end_yr:
        outfile = os.path.join(loc,'monthly',str(ye),f'CCMP_{v}_{var}_'+str(ye)+du.numstr(mon)+'.nc')
        print(outfile)
        if du.checkfileexist(outfile) == 0:
            # Get the year and month folder path in the input directory, and then find all netCDF files, should
            # equal the number of days (maybe add check for this?)
            fold = os.path.join(loc,'Y'+str(ye),'M'+du.numstr(mon))
            files = glob.glob(fold+f'\*.nc')
            files = [x for x in files if "monthly_mean" not in x]
            #print(fold)
            print(files)
            # Load the lat and lon grid for the first time (set i to 1 as we only want to do this once)
            if i == 0:
                lon,lat = du.load_grid(os.path.join(fold,files[0]),latv='latitude',lonv='longitude')
                i = 1
            #
            wind = np.empty((lon.shape[0],lat.shape[0],len(files)*4))
            wind[:] = np.nan
            cou = 0
            for j in range(len(files)):
                print(files[j])
                c = Dataset(os.path.join(fold,files[j]),'r')
                wind_t = np.array(c.variables[var][:])
                wind_t[wind_t < 0 ] = np.nan
                #print(np.transpose(sst[0,:,:]).shape)
                if wind_t.shape[2] == 4:
                    # If wind speed is in dimensions (lat,lon, time)
                    wind_t = np.transpose(wind_t,[1,0,2])

                else:
                    # SOme files come as dimensions (time,lat,lon) so do this tranposition
                    wind_t = np.transpose(wind_t,[2,1,0])
                print(wind_t.shape)
                c.close()


                wind[:,:,cou:cou+4] = wind_t


                # plt.figure()
                # plt.pcolor(wind_t[:,:,0])
                # plt.figure()
                # plt.pcolor(wind[:,:,cou])
                # plt.show()
                cou = cou+4

            wind_o = np.nanmean(wind,axis=2)
            print(wind_o.shape)
            # plt.figure()
            # plt.pcolor(wind_o)
            # plt.show()
            print(wind_o)
            wind_o2 = np.nanmean(wind**2,axis=2)
            if mon == 1:
                du.makefolder(os.path.join(loc,'monthly',str(ye)))
            du.netcdf_create_basic(outfile,wind_o,var,lat,lon)
            du.netcdf_append_basic(outfile,wind_o2,var+'^2')
        mon = mon+1
        if mon == 13:
            mon = 1
            ye = ye+1
