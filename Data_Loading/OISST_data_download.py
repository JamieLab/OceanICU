#!/usr/bin/env python
"""
"""
import datetime
import os
import data_utils as du
from netCDF4 import Dataset
import numpy as np
import urllib.request
import requests

def download_oisst_v21_daily(loc,start_yr=1990,end_yr=2023):
    du.makefolder(loc)
    htt = 'http://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr'
    d = datetime.datetime(start_yr,1,1)
    # t = 1
    while d.year < end_yr:
        if d.day == 1:
            du.makefolder(os.path.join(loc,str(d.year)))
            du.makefolder(os.path.join(loc,str(d.year),str(d.month)))

        file = os.path.join(loc,str(d.year),str(d.month),d.strftime('oisst-avhrr-v02r01.%Y%m%d.nc'))
        htt_file = htt+'/'+d.strftime('%Y%m')+'/'+d.strftime('oisst-avhrr-v02r01.%Y%m%d.nc')
        print(file)
        #print(htt_file)
        if not du.checkfileexist(file):
            print('Downloading... ' + file)
            urllib.request.urlretrieve(htt_file,file)
        #open(file).write(requests.get(htt_file))
        d = d + datetime.timedelta(days=1)
        # t = t+1
        # if t == 30:
        #     break

def OISST_monthly_split(file,loc,res=1):
    lon,lat = du.load_grid(os.path.join(loc,file),latv='lat',lonv='lon')
    lon = lon-180
    lon_g,lat_g = du.reg_grid()
    lo_grid,la_grid = du.determine_grid_average(lon,lat,lon_g,lat_g)
    c = Dataset(os.path.join(loc,file),'r')
    time = np.array(c.variables['time'])
    sst = np.array(c.variables['sst'])
    sst[sst<-5] = np.nan
    sst = sst+273.15
    d = datetime.datetime(1800,1,1)
    for i in range(0,len(time)):
        time_o = d + datetime.timedelta(days=int(time[i]))
        du.makefolder(os.path.join(loc,time_o.strftime('%Y')))
        l,ss = du.grid_switch(lon,np.transpose(np.squeeze(sst[i,:,:])))
        sst_o = du.grid_average(ss,lo_grid,la_grid)
        outfile = os.path.join(loc,time_o.strftime('%Y'),time_o.strftime('%Y%m_OISSTv2.nc'))
        du.netcdf_create_basic(outfile,sst_o,'sst',lat_g,lon_g)
