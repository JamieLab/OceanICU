#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 09/2023

"""
import datetime
import os
import glob
import time
import data_utils as du
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def OSISAF_monthly_av(inp='D:/Data/OSISAF',start_yr = 1978,end_yr = 2023,time_cor = 5):
    import shutil
    du.makefolder(os.path.join(inp,'monthly'))
    for hemi in ['SH','NH']:
        if start_yr <= 1978:
            ye = 1978
            mon = 10
        else:
            ye = start_yr
            mon = 1
        i = 0 # Variable for loading lat and lon grid once (set to 1 once loaded for the first time)

        du.makefolder(os.path.join(inp,'monthly',str(ye)))
        while ye < end_yr:
            print(os.path.join(inp,'monthly',str(ye),'OSISAF_MONTHLY_'+hemi+'_'+str(ye)+du.numstr(mon)+'.nc'))
            outfile = os.path.join(inp,'monthly',str(ye),'OSISAF_MONTHLY_'+hemi+'_'+str(ye)+du.numstr(mon)+'.nc')
            if os.path.exists(outfile) == 0:
                # Get the year and month folder path in the input directory, and then find all netCDF files, should
                # equal the number of days (maybe add check for this?)
                fold = os.path.join(inp,str(ye),du.numstr(mon))
                files = glob.glob(fold+'\ice_conc_'+hemi+'*.nc')
                #print(fold)
                #print(files)
                if len(files)>0:
                    # Load the lat and lon grid for the first time (set i to 1 as we only want to do this once)
                    if i == 0:
                        lon,lat = du.load_grid(os.path.join(fold,files[0]),latv='lat',lonv='lon')
                        i = 1
                    #
                    sst = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    sst[:] = np.nan
                    # ice = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    # ice[:] = np.nan
                    unc = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    unc[:] = np.nan
                    for j in range(len(files)):
                        print(files[j])
                        c = Dataset(os.path.join(fold,files[j]),'r')
                        sst_t = np.array(c.variables['ice_conc'][:]); sst_t[sst_t < 0 ] = np.nan
                        #print(np.transpose(sst[0,:,:]).shape)
                        sst_t = sst_t[0,:,:]

                        # ice_t = np.array(c.variables['sea_ice_fraction'][:]); ice_t[ice_t < 0] = np.nan
                        # ice_t = np.transpose(ice_t[0,:,:])

                        unc_t = np.array(c.variables['total_standard_uncertainty'][:]); unc_t[unc_t < 0] = np.nan
                        unc_t = unc_t[0,:,:]
                        c.close()

                        sst[:,:,j] = sst_t
                        #ice[:,:,j] = ice_t
                        unc[:,:,j] = unc_t
                    sst_o = np.nanmean(sst,axis=2)
                    # ice_o = np.nanmean(ice,axis=2)
                    unc_o = np.nanmean(unc,axis=2)/np.sqrt(len(files)/time_cor); unc_o[unc_o < 0] = np.nan
                    if mon == 1:
                        du.makefolder(os.path.join(inp,'monthly',str(ye)))
                    if len(files)>0:
                        shutil.copy(os.path.join(fold,files[0]),outfile)
                        c = Dataset(outfile,'a')
                        c.variables['ice_conc'][:] = sst_o[np.newaxis,:,:]
                        c.variables['total_standard_uncertainty'][:] = unc_o[np.newaxis,:,:]
                        c.close()
                        # du.netcdf_create_basic(outfile,sst_o,'ice_conc',np.array(range(0,len(lat))),np.array(range(0,len(lon))))
                        # #du.netcdf_append_basic(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),ice_o,'sea_ice_fraction')
                        # du.netcdf_append_basic(outfile,unc_o,'total_standard_uncertainty')
                        # # du.netcdf_append_basic(outfile,lat,'latitude_g')
                        # # du.netcdf_append_basic(outfile,lon,'longitude_g')
            mon = mon+1
            if mon == 13:
                mon = 1
                ye = ye+1

def OSISAF_spatial_average(data='D:/Data/OSISAF/monthly',start_yr = 1981, end_yr=2023,out_loc='',log='',lag='',hemi = 'NH'):
    du.makefolder(out_loc)
    res = np.round(np.abs(log[0]-log[1]),2)
    if start_yr < 1981:
        ye = 1981
        mon= 9
    else:
        ye = start_yr
        mon=1
    st_ye = 1981
    st_mon = 9

    t=0
    while ye <= end_yr:
        du.makefolder(os.path.join(out_loc,str(ye)))

        file = os.path.join(data,str(ye),'OSISAF_MONTHLY_'+hemi+'_'+str(ye)+du.numstr(mon)+'.nc')
        #file = 'D:/Data/OSISAF/2022/01/ice_conc_nh_ease2-250_icdr-v3p0_202201011200.nc'
        file_o = os.path.join(out_loc,str(ye),str(ye)+du.numstr(mon)+f'_OSISAF_MONTHLY_{hemi}_{res}_deg.nc')
        print(file)
        if t == 0:
            #lon,lat = du.load_grid(file,latv = 'latitude_g',lonv = 'longitude_g')
            lon,lat = du.load_grid(file,latv = 'lat', lonv = 'lon')
            lon2 = lon; lat2 = lat
            lon = lon.ravel(); lat = lat.ravel();
            grid = du.determine_grid_average_nonreg(lon,lat,log,lag)
            # plt.figure()
            # plt.scatter(lon,lat)
            # plt.xlim(-180,180)
            # plt.ylim(-90,90)
            # plt.show()
            grid_t = np.zeros((len(grid)))
            #grid_t[:] = np.nan
            for l in range(len(grid)):
                grid_t[l] = grid[l].size
            grid_t[grid_t < 1] = np.nan
            grid_t = np.reshape(grid_t,(len(lag),len(log)))
            # plt.pcolor(log,lag,grid_t)
            # plt.colorbar()
            # plt.show()
            t = 1
        if du.checkfileexist(file_o) == 0:
            if du.checkfileexist(file) == 1:
                c = Dataset(file,'r')
                sst = np.array(c.variables['ice_conc'][:]).ravel(); sst[sst<0] = np.nan
                # plt.figure()
                # plt.pcolor(lat2,lon2,sst)
                # plt.colorbar()
                # plt.show()
                #ice = np.array(c.variables['sea_ice_fraction'][:])
                unc = np.array(c.variables['total_standard_uncertainty'][:]).ravel(); unc[unc<0] = np.nan
                c.close()
                #print(sst.shape)
                sst_o = du.grid_average_nonreg(sst,grid)
                #ice_o = du.grid_average(ice,lo_grid,la_grid)
                unc_o = du.grid_average_nonreg(unc,grid)
                sst_o = np.transpose(np.reshape(sst_o,(len(lag),len(log))))
                print(sst_o.shape)
                unc_o = np.transpose(np.reshape(unc_o,(len(lag),len(log)))) * 2 #We want 2 sigma uncertainties
                du.netcdf_create_basic(file_o,sst_o,'ice_conc',lag,log)
                #du.netcdf_append_basic(file_o,ice_o,'sea_ice_fraction')
                du.netcdf_append_basic(file_o,unc_o,'total_standard_uncertainty')
                du.netcdf_append_basic(file_o,np.transpose(grid_t),'grid_t')
                c = Dataset(file_o,'a')
                c.variables['total_standard_uncertainty'].uncertainty = 'These uncertainties are 2 sigma (95% confidence) equivalents!'
                c.close()

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1

def OSISAF_merge_hemisphere(loc,bath,start_yr = 1981, end_yr=2023,log='',lag=''):
    from scipy.interpolate import NearestNDInterpolator
    res = np.round(np.abs(log[0]-log[1]),2)
    c = Dataset(bath,'r')
    ocean = np.array(c.variables['ocean_proportion'])
    c.close()

    if start_yr < 1981:
        ye = 1981
        mon= 9
    else:
        ye = start_yr
        mon=1
    while ye <= end_yr:
        file = os.path.join(loc,str(ye),str(ye)+du.numstr(mon)+f'_OSISAF_MONTHLY_{res}_deg_COM.nc')
        files = glob.glob(os.path.join(loc,str(ye),str(ye)+du.numstr(mon)+f'_OSISAF_MONTHLY_*_{res}_deg.nc'))
        if du.checkfileexist(file) == 0:
            t = 0
            for i in range(len(files)):
                c = Dataset(files[i])
                sst = np.array(c.variables['ice_conc'][:])
                if t == 0:
                    ice_o = np.zeros((sst.shape[0],sst.shape[1]))
                    ice_o[:] = np.nan
                    unc_o = np.copy(ice_o)
                    t=1
                f = np.where(np.isnan(sst) == 0)
                ice_o[f[0],f[1]] = sst[f[0],f[1]]

                unc_t = np.array(c.variables['total_standard_uncertainty'][:])
                unc_o[f[0],f[1]] = unc_t[f[0],f[1]]
                # ice[:,:,i] = sst
                # unc[:,:,i] =
                c.close()
            #unc[unc<0] = -1
            #ice[ice<0] = -1
            f = np.where((lag <=30) & (lag>=-30))
            ice_o[:,f] = 0.0
            unc_o[:,f] = 0.0
            mask = np.where(~np.isnan(ice_o))
            interp = NearestNDInterpolator(np.transpose(mask), ice_o[mask])
            ice_o = interp(*np.indices(ice_o.shape))
            ice_o[ocean==0] = np.nan

            mask = np.where(~np.isnan(unc_o))
            interp = NearestNDInterpolator(np.transpose(mask), unc_o[mask])
            unc_o = interp(*np.indices(unc_o.shape))
            unc_o[ocean==0] = np.nan
            du.netcdf_create_basic(file,ice_o/100,'ice_conc',lag,log)
            #du.netcdf_append_basic(file_o,ice_o,'sea_ice_fraction')
            du.netcdf_append_basic(file,unc_o/100,'total_standard_uncertainty')
            c = Dataset(file,'a')
            c.variables['total_standard_uncertainty'].uncertainty = 'These uncertainties are 2 sigma (95% confidence) equivalents!'
            c.close()
        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1
