#!/usr/bin/env python

import datetime
import os
import glob
import time
import data_utils as du
from netCDF4 import Dataset
import numpy as np

def cci_retrieve(loc="D:/Data/SST-CCI/",start_yr = 1981,end_yr = 2023):
    import cdsapi
    import zipfile
    if start_yr < 1981:
        d = datetime.datetime(1981,9,1)
    else:
        d = datetime.datetime(start_yr,1,1)

    while d.year < end_year:
        ye = d.strftime("%Y")
        mon = d.strftime("%m")
        day = d.strftime("%d")
        if d.day == 1:
            p = loc+str(d.year)
            du.makefolder(p)
            p = p+'/'+mon
            du.makefolder(p)

        if du.checkfileexist(p+'/'+d.strftime("%Y%m%d")+'*.nc') == 0:
            cou = 0
            while True:
                try:
                    time.sleep(5)
                    if cou == 3:
                        break
                    c = cdsapi.Client()
                    c.retrieve(
                    'satellite-sea-surface-temperature',
                    {
                    'version': '2_1',
                    'variable': 'all',
                    'format': 'zip',
                    'processinglevel': 'level_4',
                    'sensor_on_satellite': 'combined_product',
                    'year': ye,
                    'month': mon,
                    'day': day,
                    },
                    loc+'download.zip')
                except:
                    cou = cou+1
                break
            if cou == 3:
                raise RetryError

            with zipfile.ZipFile(loc+'download.zip', 'r') as zip_ref:
                zip_ref.extractall(p)
        d = d+datetime.timedelta(days=1)

def cci_monthly_av(inp='D:/Data/SST-CCI',start_yr = 1981,end_yr = 2023):
    du.makefolder(os.path.join(inp,'monthly'))
    if start_yr < 1981:
        ye = 1981
        mon = 9
    else:
        ye = start_yr
        mon = 1
    i = 0 # Variable for loading lat and lon grid once (set to 1 once loaded for the first time)

    while ye <= end_yr:
        print(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'))
        if os.path.exists(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')) == 0:
            # Get the year and month folder path in the input directory, and then find all netCDF files, should
            # equal the number of days (maybe add check for this?)
            fold = os.path.join(inp,str(ye),du.numstr(mon))
            files = glob.glob(fold+'\*.nc')
            #print(fold)
            #print(files)
            # Load the lat and lon grid for the first time (set i to 1 as we only want to do this once)
            if i == 0:
                lon,lat = lat_lon(os.path.join(fold,files[0]))
                i = 1
            #
            sst = np.empty((lon.shape[0],lat.shape[0],len(files)))
            sst[:] = np.nan
            ice = np.empty((lon.shape[0],lat.shape[0],len(files)))
            ice[:] = np.nan
            for j in range(len(files)):
                print(files[j])
                c = Dataset(os.path.join(fold,files[j]),'r')
                sst_t = np.array(c.variables['analysed_sst'][:])
                sst_t = np.transpose(sst[0,:,:])
                ice_t = np.array(c.variables['sea_ice_fraction'][:])
                ice_t = np.transpose(ice[0,:,:])
                ice[ice < -0] = np.nan
                c.close()

                sst[:,:,j] = sst_t
                ice[:,:,j] = ice_t
            sst_o = np.nanmean(sst,axis=2)
            ice_o = np.nanmean(ice,axis=2)
            if mon == 1:
                du.makefolder(os.path.join(inp,'monthly',str(ye)))
            du.netcdf_create(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+numstr(mon)+'.nc'),sst_o,'analysed_sst',lat,lon)
            du.netcdf_append(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+numstr(mon)+'.nc'),ice_o,'sea_ice_fraction')
        mon = mon+1
        if mon == 13:
            mon = 1
            ye = ye+1

def cci_sst_spatial_average(data='D:/Data/SST-CCI/monthly',start_yr = 1981, end_yr=2023,out_loc='',log='',lag=''):
    du.makefolder(out_loc)
    res = np.abs(log[0] - log[1])
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

        file = os.path.join(data,str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')
        file_o = os.path.join(out_loc,str(ye),str(ye)+du.numstr(mon)+f'_ESA_CCI_MONTHLY_SST_{res}_deg.nc')
        print(file)
        if t == 0:
            [lon,lat] = du.load_grid(file)
            [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
            #print(lo_grid)
            #print(la_grid)
            t = 1
        if du.checkfileexist(file_o) == 0:
            c = Dataset(file,'r')
            sst = np.array(c.variables['analysed_sst'][:]); sst[sst<0] = np.nan
            ice = np.array(c.variables['sea_ice_fraction'][:])
            c.close()
            #print(sst.shape)
            sst_o = du.grid_average(sst,lo_grid,la_grid)
            ice_o = du.grid_average(ice,lo_grid,la_grid)
            du.netcdf_create_basic(file_o,sst_o,'analysed_sst',lag,log)
            du.netcdf_append_basic(file_o,ice_o,'sea_ice_fraction')

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1
