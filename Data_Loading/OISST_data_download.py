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
import glob
import time

def download_oisst_v21_daily(loc,start_yr=1990,end_yr=2023):
    du.makefolder(loc)
    htt = 'http://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr'
    if start_yr <= 1981:
        d = datetime.datetime(1981,9,1)
    else:
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

def oisst_monthly_av(inp='D:/Data/OISSTv2_1',start_yr = 1981,end_yr = 2023,time_cor = 5):
    du.makefolder(os.path.join(inp,'monthly'))
    if start_yr <= 1981:
        ye = 1981
        mon = 9
    else:
        ye = start_yr
        mon = 1
    i = 0 # Variable for loading lat and lon grid once (set to 1 once loaded for the first time)
    du.makefolder(os.path.join(inp,'monthly',str(ye)))
    while ye <= end_yr:
        print(os.path.join(inp,'monthly',str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'))
        if os.path.exists(os.path.join(inp,'monthly',str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')) == 0:
            # Get the year and month folder path in the input directory, and then find all netCDF files, should
            # equal the number of days (maybe add check for this?)
            fold = os.path.join(inp,str(ye),str(mon))
            files = glob.glob(fold+'\*.nc')
            #print(fold)
            #print(files)
            # Load the lat and lon grid for the first time (set i to 1 as we only want to do this once)
            if i == 0:
                lon,lat = du.load_grid(os.path.join(fold,files[0]),latv='lat',lonv='lon')
                i = 1
            #
            sst = np.empty((lon.shape[0],lat.shape[0],len(files)))
            sst[:] = np.nan
            ice = np.empty((lon.shape[0],lat.shape[0],len(files)))
            ice[:] = np.nan
            unc = np.empty((lon.shape[0],lat.shape[0],len(files)))
            unc[:] = np.nan
            for j in range(len(files)):
                print(files[j])
                c = Dataset(os.path.join(fold,files[j]),'r')
                sst_t = np.array(c.variables['sst'][:]); sst_t[sst_t < -10 ] = np.nan
                #print(np.transpose(sst[0,:,:]).shape)
                sst_t = np.transpose(sst_t[0,0,:,:])

                ice_t = np.array(c.variables['ice'][:]); ice_t[ice_t < -10] = np.nan
                ice_t = np.transpose(ice_t[0,0,:,:])

                unc_t = np.array(c.variables['err'][:]); unc_t[unc_t < -10] = np.nan
                unc_t = np.transpose(unc_t[0,0,:,:])
                c.close()

                sst[:,:,j] = sst_t
                ice[:,:,j] = ice_t
                unc[:,:,j] = unc_t
            sst_o = np.nanmean(sst,axis=2)
            ice_o = np.nanmean(ice,axis=2)
            unc_o = np.nanmean(unc,axis=2)/np.sqrt(len(files)/time_cor)
            if mon == 1:
                du.makefolder(os.path.join(inp,'monthly',str(ye)))
            du.netcdf_create_basic(os.path.join(inp,'monthly',str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),sst_o,'sst',lat,lon)
            du.netcdf_append_basic(os.path.join(inp,'monthly',str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),ice_o,'ice')
            du.netcdf_append_basic(os.path.join(inp,'monthly',str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),unc_o,'err')
        mon = mon+1
        if mon == 13:
            mon = 1
            ye = ye+1
def oisst_spatial_average(data='D:/Data/OISSTv2_1/monthly',start_yr = 1981, end_yr=2023,out_loc='',log='',lag=''):
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

        file = os.path.join(data,str(ye),'OISST_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')
        file_o = os.path.join(out_loc,str(ye),str(ye)+du.numstr(mon)+f'_OISST_MONTHLY_SST_{res}_deg.nc')
        print(file)
        if t == 0:
            [lon,lat] = du.load_grid(file)
            lon_t = np.copy(lon)
            lon = lon-180
            [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
            #print(lo_grid)
            #print(la_grid)
            t = 1
        if du.checkfileexist(file_o) == 0:
            c = Dataset(file,'r')
            l,sst = du.grid_switch(lon_t,np.array(c.variables['sst'][:])) ;
            sst = sst + 273.15
            sst[sst<-10] = np.nan
            l,ice = du.grid_switch(lon_t,np.array(c.variables['ice'][:]))
            l,unc = du.grid_switch(lon_t,np.array(c.variables['err'][:]))
            c.close()
            #print(sst.shape)
            sst_o = du.grid_average(sst,lo_grid,la_grid)
            ice_o = du.grid_average(ice,lo_grid,la_grid)
            unc_o = du.grid_average(unc,lo_grid,la_grid)
            du.netcdf_create_basic(file_o,sst_o,'sst',lag,log)
            du.netcdf_append_basic(file_o,ice_o,'ice')
            du.netcdf_append_basic(file_o,unc_o,'err')

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1

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

def oisst_socat_append(file,data_loc='D:/Data/OISSTv2_1',start_yr = 1981):
    import pandas as pd
    import calendar
    import glob
    import matplotlib.pyplot as plt
    import pickle
    data = pd.read_table(file,sep='\t')
    cci_sst = np.zeros((len(data)))
    cci_sst[:] = np.nan
    cci_sst_unc = np.copy(cci_sst)
    s = file.split('.')
    yr = [np.min(data['yr']),np.max(data['yr'])]
    t = 0
    if du.checkfileexist(s[0]+'.pkl'):
        print('Loading pickle file....')
        dbfile = open(s[0]+'.pkl', 'rb')
        temp = pickle.load(dbfile)
        dbfile.close()
        start_yr = temp[0]
        cci_sst = temp[1]
        cci_sst_unc = temp[2]
        print(start_yr)

    for yrs in range(start_yr,yr[1]+1):
        dbfile = open(s[0]+'.pkl', 'wb')
        pickle.dump([yrs,cci_sst,cci_sst_unc],dbfile)
        dbfile.close()
        for mon in range(1,13):

            days = calendar.monthrange(yrs,mon)[1]
            for day in range(1,days+1):
                f = np.where((data['yr'] == yrs) & (data['mon'] == mon) & (data['day'] == day))[0]
                print(f'Year: {yrs} Month: {mon} Day: {day}')
                #print(f)
                if len(f)>0:
                    sst_file = os.path.join(data_loc,str(yrs),str(mon),'*'+str(yrs)+du.numstr(mon)+du.numstr(day)+'.nc')
                    sst_file = glob.glob(sst_file)
                    if sst_file:
                        if t == 0:
                            [lon,lat] = du.load_grid(sst_file[0],latv = 'lat',lonv='lon')
                            res = np.abs(lon[0] - lon[1]) * 2
                            t = 1
                        for expo in set(data['Expocode'][f]):
                            print(expo)
                            g = np.where(data['Expocode'][f] == expo)[0]
                            #print(g)
                            lon_p = data['longitude [dec.deg.E]'][f[g]]
                            lon_p[lon_p<0] = lon_p[lon_p<0] + 360
                            lat_b = [np.min(data['latitude [dec.deg.N]'][f[g]]),np.max(data['latitude [dec.deg.N]'][f[g]])]
                            lon_b = [np.min(lon_p),np.max(lon_p)]
                            #print(lat_b)
                            #print(lon_b)
                            lat_b = np.where((lat < lat_b[1]+res) & (lat > lat_b[0]-res))[0]
                            lon_b = np.where((lon < lon_b[1]+res) & (lon > lon_b[0]-res))[0]
                            c = Dataset(sst_file[0],'r')
                            sst_data = np.squeeze(c['sst'][0,0,lat_b,lon_b])
                            sst_unc = np.squeeze(c['err'][0,0,lat_b,lon_b])
                            sst_data[sst_data<-250] = np.nan
                            sst_unc[sst_unc<-250] = np.nan
                            #sst_data = sst_data
                            c.close()
                            #print(data['longitude [dec.deg.E]'][f[g]])
                            #print(data['latitude [dec.deg.N]'][f[g]])
                            a = du.point_interp(lon[lon_b],lat[lat_b],sst_data,lon_p,data['latitude [dec.deg.N]'][f[g]],plot=False)
                            #print(a)
                            cci_sst[f[g]] = a
                            a = du.point_interp(lon[lon_b],lat[lat_b],sst_unc,lon_p,data['latitude [dec.deg.N]'][f[g]],plot=False)
                            cci_sst_unc[f[g]] = a

    data['oisst [C]'] = cci_sst
    data['oisst_unc [C]'] = cci_sst_unc
    st = file.split('.')
    data.to_csv(file,sep='\t',index=False)
