#!/usr/bin/env python3
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import data_utils as du

def oc_cci_average(loc,out_folder = '',start_yr = 1993,end_yr = 2020,log='',lag='',conv = False):
    if start_yr <= 1997:
        start_yr = 1997
        st_mon = 9 # Needs to be manually modified - OC-CCI starts in 09 / 1997
    else:
        st_mon = 1
    res = np.abs(log[0] - log[1])
    du.makefolder(os.path.join(out_folder,str(start_yr)))
    ye = start_yr
    mon = st_mon
    t = 0
    while ye <= end_yr:
        du.makefolder(os.path.join(out_folder,str(ye)))

        file = os.path.join(loc,str(ye),'ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-'+str(ye)+du.numstr(mon)+'-fv6.0.nc')
        file_o = os.path.join(out_folder,str(ye),str(ye)+'_'+du.numstr(mon)+f'_ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_-fv6.0_{res}_deg.nc')
        print(file)
        if t == 0:
            [lon,lat] = du.load_grid(file,latv = 'lat',lonv = 'lon')
            [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
            #print(lo_grid)
            #print(la_grid)
            t = 1
        if du.checkfileexist(file_o) == 0:
            c = Dataset(file,'r')
            chl = np.transpose(np.array(c.variables['chlor_a'][0,:,:]))
            chl_rmsd = np.transpose(np.array(c.variables['chlor_a_log10_rmsd'][0,:,:]))
            chl_bias = np.transpose(np.array(c.variables['chlor_a_log10_bias'][0,:,:]))
            chl[chl>2000] = np.nan; chl_rmsd[chl_rmsd>2000] = np.nan; chl_bias[chl_bias>2000] = np.nan;
            # Averaging done in log10 space due to log normal distribution of chl-a values.
            chl = np.log10(chl)
            chl_o = du.grid_average(chl,lo_grid,la_grid)
            chl_rmsd_o = du.grid_average(chl_rmsd,lo_grid,la_grid)
            chl_bias_o = du.grid_average(chl_bias,lo_grid,la_grid)
            # Convert back to normal unit space.
            if conv:
                chl_o = 10**chl_o
            du.netcdf_create_basic(file_o,chl_o,'chlor_a',lag,log)
            du.netcdf_append_basic(file_o,chl_rmsd_o,'chlor_a_log10_rmsd')
            du.netcdf_append_basic(file_o,chl_bias_o,'chlor_a_log10_bias')

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1

def oc_cci_average_day(loc,out_folder,log,lag,start_yr = 1997,end_yr=2022,conv=False):
    if start_yr<1997:
        start_yr = 1997
    res = np.abs(log[0] - log[1])
    du.makefolder(os.path.join(out_folder,str(start_yr)))
    d = datetime.datetime(start_yr,1,1)
    t = 0
    while d.year <= end_yr:
        ye = d.year
        du.makefolder(os.path.join(out_folder,str(d.year)))
        print(d)
        file = os.path.join(loc,str(ye),'ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_OCx-'+d.strftime("%Y%m%d")+'-fv6.0.nc')
        file_o = os.path.join(out_folder,str(ye),'ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_OCx-'+d.strftime("%Y%m%d")+'-fv6.0_'+str(res)+'.nc')
        if (du.checkfileexist(file_o) == 0) & (du.checkfileexist(file) == 1):
            if t == 0:
                [lon,lat] = du.load_grid(file,latv = 'lat',lonv = 'lon')
                [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
                #print(lo_grid)
                #print(la_grid)
                t = 1
            c = Dataset(file,'r')
            chl = np.transpose(np.array(c.variables['chlor_a'][0,:,:]))
            chl_rmsd = np.transpose(np.array(c.variables['chlor_a_log10_rmsd'][0,:,:]))
            chl_bias = np.transpose(np.array(c.variables['chlor_a_log10_bias'][0,:,:]))
            chl[chl>2000] = np.nan; chl_rmsd[chl_rmsd>2000] = np.nan; chl_bias[chl_bias>2000] = np.nan;
            # Averaging done in log10 space due to log normal distribution of chl-a values.
            chl = np.log10(chl)
            chl_o = du.grid_average(chl,lo_grid,la_grid)
            chl_rmsd_o = du.grid_average(chl_rmsd,lo_grid,la_grid)
            chl_bias_o = du.grid_average(chl_bias,lo_grid,la_grid)
            # Convert back to normal unit space.
            if conv:
                chl_o = 10**chl_o
            du.netcdf_create_basic(file_o,chl_o,'chlor_a',lag,log)
            du.netcdf_append_basic(file_o,chl_rmsd_o,'chlor_a_log10_rmsd')
            du.netcdf_append_basic(file_o,chl_bias_o,'chlor_a_log10_bias')

        d = d + datetime.timedelta(days=8)
        if ye != d.year:
            d =datetime.datetime(d.year,1,1)
