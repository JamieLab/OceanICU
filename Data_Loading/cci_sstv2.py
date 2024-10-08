#!/usr/bin/env python

import datetime
import os
import glob
import time
import data_utils as du
from netCDF4 import Dataset
import numpy as np
import calendar

def cci_retrieve_v2(loc="D:/Data/SST-CCI/",start_yr = 1981,end_yr = 2023):
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

def cci_sst_v3(loc,start_yr=1990,end_yr=2023):
    du.makefolder(loc)
    htt = 'https://data.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1/'
    d = datetime.datetime(start_yr,1,1)
    # t = 1
    while d.year < end_yr:
        if d.day == 1:
            du.makefolder(os.path.join(loc,str(d.year)))
            du.makefolder(os.path.join(loc,str(d.year),du.numstr(d.month)))

        file = os.path.join(loc,str(d.year),du.numstr(d.month),d.strftime('%Y%m%d120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc'))
        htt_file = htt+'/'+d.strftime('%Y/%m/%d')+'/'+d.strftime('%Y%m%d120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc')
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

def cci_monthly_av(inp='D:/Data/SST-CCI',start_yr = 1981,end_yr = 2023,time_cor = 5,v3 = False):
    du.makefolder(os.path.join(inp,'monthly'))
    if v3:
        ye = start_yr
        mon = 1
    else:
        if start_yr <= 1981:
            ye = 1981
            mon = 9
        else:
            ye = start_yr
            mon = 1

    i = 0 # Variable for loading lat and lon grid once (set to 1 once loaded for the first time)
    du.makefolder(os.path.join(inp,'monthly',str(ye)))
    while ye <= end_yr:
        print(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'))
        if os.path.exists(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')) == 0:
            # Get the year and month folder path in the input directory, and then find all netCDF files, should
            # equal the number of days (maybe add check for this?)
            fold = os.path.join(inp,str(ye),du.numstr(mon))
            if v3:
                files = []
                for f_loop in range(1,calendar.monthrange(int(ye),int(mon))[1]+1):
                    file_t = glob.glob(os.path.join(fold,du.numstr(f_loop),'*.nc'))
                    files.append(file_t[0])
                print(files)
            else:
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
                sst_t = np.array(c.variables['analysed_sst'][:]); sst_t[sst_t < 0 ] = np.nan
                #print(np.transpose(sst[0,:,:]).shape)
                sst_t = np.transpose(sst_t[0,:,:])

                ice_t = np.array(c.variables['sea_ice_fraction'][:]); ice_t[ice_t < 0] = np.nan
                ice_t = np.transpose(ice_t[0,:,:])

                unc_t = np.array(c.variables['analysed_sst_uncertainty'][:]); unc_t[unc_t < 0] = np.nan
                unc_t = np.transpose(unc_t[0,:,:])
                c.close()

                sst[:,:,j] = sst_t
                ice[:,:,j] = ice_t
                unc[:,:,j] = unc_t
            sst_o = np.nanmean(sst,axis=2)
            ice_o = np.nanmean(ice,axis=2)
            unc_o = np.nanmean(unc,axis=2)/np.sqrt(len(files)/time_cor)
            if mon == 1:
                du.makefolder(os.path.join(inp,'monthly',str(ye)))
            du.netcdf_create_basic(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),sst_o,'analysed_sst',lat,lon)
            du.netcdf_append_basic(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),ice_o,'sea_ice_fraction')
            du.netcdf_append_basic(os.path.join(inp,'monthly',str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc'),unc_o,'analysed_sst_uncertainty')

        mon = mon+1
        if mon == 13:
            mon = 1
            ye = ye+1

def cci_sst_8day(loc,out_folder,start_yr = 1993,end_yr=2022,time_cor = 5):

        du.makefolder(os.path.join(out_folder,str(start_yr)))
        d = datetime.datetime(start_yr,1,1)
        t = 0
        while d.year <= end_yr:
            ye = d.year
            du.makefolder(os.path.join(out_folder,str(d.year)))
            print(d)
            file_out = os.path.join(out_folder,str(d.year),'ESA_CCI_8DAY_SST_' + d.strftime("%Y%m%d")+'.nc')
            if (du.checkfileexist(file_out) == 0):
                files = []
                for i in range(0,8):
                    d2 = d + datetime.timedelta(days=int(i))
                    if d2.year == d.year: # A catch for the very last 8 day period of the year to match the OC-CCI as the last period doesn't always cover 8 days
                        g = glob.glob(os.path.join(loc,d2.strftime("%Y"),d2.strftime("%m"),d2.strftime("%d"),'*.nc'))
                        if len(g) > 0:
                            files.append(g[0])
                if len(files)>0:
                    if t == 0:
                        lon,lat = du.load_grid(files[0],latv='lat',lonv='lon')
                        t = 1
                    sst = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    sst[:] = np.nan
                    ice = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    ice[:] = np.nan
                    unc = np.empty((lon.shape[0],lat.shape[0],len(files)))
                    unc[:] = np.nan
                    for j in range(len(files)):
                        print(files[j])
                        c = Dataset(os.path.join(files[j]),'r')
                        sst_t = np.array(c.variables['analysed_sst'][:]); sst_t[sst_t < 0 ] = np.nan
                        #print(np.transpose(sst[0,:,:]).shape)
                        sst_t = np.transpose(sst_t[0,:,:])

                        ice_t = np.array(c.variables['sea_ice_fraction'][:]); ice_t[ice_t < 0] = np.nan
                        ice_t = np.transpose(ice_t[0,:,:])

                        unc_t = np.array(c.variables['analysed_sst_uncertainty'][:]); unc_t[unc_t < 0] = np.nan
                        unc_t = np.transpose(unc_t[0,:,:])
                        c.close()

                        sst[:,:,j] = sst_t
                        ice[:,:,j] = ice_t
                        unc[:,:,j] = unc_t
                    sst_o = np.nanmean(sst,axis=2)
                    ice_o = np.nanmean(ice,axis=2)
                    unc_o = np.nanmean(unc,axis=2)/np.sqrt(len(files)/time_cor)

                    du.netcdf_create_basic(file_out,sst_o,'analysed_sst',lat,lon)
                    du.netcdf_append_basic(file_out,ice_o,'sea_ice_fraction')
                    du.netcdf_append_basic(file_out,unc_o,'analysed_sst_uncertainty')


            d = d + datetime.timedelta(days=8)
            if ye != d.year:
                d = datetime.datetime(d.year,1,1)

def cci_sst_spatial_average(data='D:/Data/SST-CCI/monthly',start_yr = 1981, end_yr=2023,out_loc='',log='',lag='',v3=False,flip=False,bia=0,monthly = True):
    du.makefolder(out_loc)
    res = np.round(np.abs(log[0]-log[1]),2)
    if monthly:
        if v3:
            ye = start_yr
            mon = 1
        else:
            if start_yr <= 1981:
                ye = 1981
                mon = 9
            else:
                ye = start_yr
                mon = 1
    else:
        d = datetime.datetime(start_yr,1,1)
    ye = d.year
    t=0
    while ye <= end_yr:

        du.makefolder(os.path.join(out_loc,str(ye)))
        if monthly:
            file = os.path.join(data,str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+du.numstr(mon)+'.nc')
            file_o = os.path.join(out_loc,str(ye),str(ye)+du.numstr(mon)+f'_ESA_CCI_MONTHLY_SST_{res}_deg.nc')
        else:
            file = os.path.join(data,str(d.year),'ESA_CCI_8DAY_SST_'+d.strftime('%Y%m%d')+'.nc')
            file_o = os.path.join(out_loc,str(d.year),'ESA_CCI_8DAY_SST_'+d.strftime('%Y%m%d')+f'_{res}_deg.nc')
        print(file)

        if (du.checkfileexist(file_o) == 0) & (du.checkfileexist(file) == 1):
            if t == 0:
                [lon,lat] = du.load_grid(file)
                [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
                #print(lo_grid)
                #print(la_grid)
                t = 1
            c = Dataset(file,'r')
            sst = np.array(c.variables['analysed_sst'][:]); sst[sst<0] = np.nan
            ice = np.array(c.variables['sea_ice_fraction'][:])
            unc = np.array(c.variables['analysed_sst_uncertainty'][:])*2 # We want 2 sigma/95% confidence uncertainties
            c.close()
            #print(sst.shape)
            sst_o = du.grid_average(sst,lo_grid,la_grid)
            ice_o = du.grid_average(ice,lo_grid,la_grid)
            unc_o = du.grid_average(unc,lo_grid,la_grid)
            du.netcdf_create_basic(file_o,sst_o+bia,'analysed_sst',lag,log,flip=flip,units='Kelvin')
            du.netcdf_append_basic(file_o,ice_o,'sea_ice_fraction',flip=flip)
            du.netcdf_append_basic(file_o,unc_o,'analysed_sst_uncertainty',flip=flip,units = 'Kelvin')
            c = Dataset(file_o,'a')
            c.variables['analysed_sst_uncertainty'].uncertainty = 'These uncertainties are 2 sigma (95% confidence) equivalents!'
            c.variables['analysed_sst'].bias_correction = 'Bias correction of ' + str(bia) + ' applied!'
            c.close()
        if monthly:
            mon = mon+1
            if mon == 13:
                ye = ye+1
                mon = 1
        else:
            d = d + datetime.timedelta(days=8)
            if ye != d.year:
                d = datetime.datetime(d.year,1,1)
            ye = d.year

def cci_socat_append(file,data_loc='D:/Data/SST-CCI',start_yr = 1980,v3=False,plot=False):
    import pandas as pd
    import calendar
    import glob
    import matplotlib.pyplot as plt
    import pickle
    from scipy.ndimage import generic_filter
    data = pd.read_table(file,sep='\t')
    cci_sst = np.zeros((len(data)))
    cci_sst[:] = np.nan
    cci_sst_unc = np.copy(cci_sst)
    ice = np.copy(cci_sst)
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
        ice = temp[3]
        print(start_yr)

    for yrs in range(start_yr,yr[1]+1):
        print('Dumping file')
        dbfile = open(s[0]+'.pkl', 'wb')
        pickle.dump([yrs,cci_sst,cci_sst_unc,ice],dbfile)
        dbfile.close()
        for mon in range(1,13):

            days = calendar.monthrange(yrs,mon)[1]
            for day in range(1,days+1):
                f = np.where((data['yr'] == yrs) & (data['mon'] == mon) & (data['day'] == day))[0]
                print(f'Year: {yrs} Month: {mon} Day: {day}')
                #print(f)
                if len(f)>0:
                    if v3:
                        sst_file = os.path.join(data_loc,str(yrs),du.numstr(mon),du.numstr(day),str(yrs)+du.numstr(mon)+du.numstr(day)+'*.nc')
                    else:
                        sst_file = os.path.join(data_loc,str(yrs),du.numstr(mon),str(yrs)+du.numstr(mon)+du.numstr(day)+'*.nc')
                    sst_file = glob.glob(sst_file)
                    print(sst_file)
                    if len(sst_file)>0:
                        print('Loading file')
                        if t == 0:
                            [lon,lat] = du.load_grid(sst_file[0],latv = 'lat',lonv='lon')
                            res = np.abs(lon[0] - lon[1]) * 2
                            t = 1
                        for expo in set(data['Expocode'][f]):
                            print(expo)
                            g = np.where(data['Expocode'][f] == expo)[0]
                            #print(g)
                            lat_b = [np.min(data['latitude [dec.deg.N]'][f[g]]),np.max(data['latitude [dec.deg.N]'][f[g]])]
                            lon_b = [np.min(data['longitude [dec.deg.E]'][f[g]]),np.max(data['longitude [dec.deg.E]'][f[g]])]
                            #print(lat_b)
                            #print(lon_b)
                            lat_b = np.where((lat < lat_b[1]+res) & (lat > lat_b[0]-res))[0]
                            lon_b = np.where((lon < lon_b[1]+res) & (lon > lon_b[0]-res))[0]
                            c = Dataset(sst_file[0],'r')
                            sst_data = np.squeeze(c['analysed_sst'][0,lat_b,lon_b])
                            sst_unc = np.squeeze(c['analysed_sst_uncertainty'][0,lat_b,lon_b])
                            ice_data = np.squeeze(c['sea_ice_fraction'][0,lat_b,lon_b])

                            if (0 in list(lon_b)) or (len(lon)-1 in list(lon_b)):
                                #plot = True
                                print('Full lon grid loaded')
                                sst_data = np.column_stack((np.squeeze(c['analysed_sst'][0,lat_b,len(lon)-1])[:,None],np.squeeze(c['analysed_sst'][0,lat_b,:]),np.squeeze(c['analysed_sst'][0,lat_b,0])[:,None]))
                                sst_unc = np.column_stack((np.squeeze(c['analysed_sst_uncertainty'][0,lat_b,len(lon)-1]),np.squeeze(c['analysed_sst_uncertainty'][0,lat_b,:]),np.squeeze(c['analysed_sst_uncertainty'][0,lat_b,0])))
                                ice_data = np.column_stack((np.squeeze(c['sea_ice_fraction'][0,lat_b,len(lon)-1]),np.squeeze(c['sea_ice_fraction'][0,lat_b,:]),np.squeeze(c['sea_ice_fraction'][0,lat_b,0])))

                            if (len(lat)-1 in list(lat_b)):
                                #plot=True
                                sst_data = np.vstack((sst_data,sst_data[-1,:]))
                                sst_unc = np.vstack((sst_unc,sst_unc[-1,:]))
                                ice_data =np.vstack((ice_data,ice_data[-1,:]))

                            sst_data[sst_data<-250] = np.nan
                            sst_unc[sst_unc<-250] = np.nan
                            ice_data[ice_data<-250] = np.nan
                            #sst_data = sst_data
                            c.close()
                            #print(data['longitude [dec.deg.E]'][f[g]])
                            #print(data['latitude [dec.deg.N]'][f[g]])
                            if (0 in list(lon_b)) or (len(lon)-1 in list(lon_b)):
                                lon_g = np.hstack((lon[-1]-360,lon,lon[0]+360))
                            else:
                                lon_g = lon[lon_b]

                            if (len(lat)-1 in list(lat_b)):
                                lat_g = np.hstack((lat[lat_b],lat[-1]+(res/2)))
                            else:
                                lat_g = lat[lat_b]
                            a = du.point_interp(lon_g,lat_g,sst_data,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                            #print(a)
                            if np.sum(np.isnan(a)==1) > 0:
                                print('NaNs present in output - attempting second approach')
                                data_filt = generic_filter(sst_data,np.nanmean,[3,3])

                                ap = du.point_interp(lon_g,lat_g,data_filt,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                                a[np.isnan(a) == 1] = ap[np.isnan(a) == 1]
                                print('NaN left: '+str(np.sum(np.isnan(a)==1)))
                            cci_sst[f[g]] = a

                            a = du.point_interp(lon_g,lat_g,sst_unc,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                            if np.sum(np.isnan(a)) > 0:
                                print('NaNs present in output - attempting second approach')
                                data_filt = generic_filter(sst_unc,np.nanmean,[3,3])
                                ap = du.point_interp(lon_g,lat_g,data_filt,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                                a[np.isnan(a) == 1] = ap[np.isnan(a) == 1]
                                print('NaN left: '+str(np.sum(np.isnan(a)==1)))
                            cci_sst_unc[f[g]] = a

                            a = du.point_interp(lon_g,lat_g,ice_data,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                            if np.sum(np.isnan(a)) > 0:
                                print('NaNs present in output - attempting second approach')
                                data_filt = generic_filter(ice_data,np.nanmean,[3,3])
                                ap = du.point_interp(lon_g,lat_g,data_filt,data['longitude [dec.deg.E]'][f[g]],data['latitude [dec.deg.N]'][f[g]],plot=plot)
                                a[np.isnan(a) == 1] = ap[np.isnan(a) == 1]
                                print('NaN left: '+str(np.sum(np.isnan(a)==1)))
                            ice[f[g]] = a
                            plot = False

    data['cci_sst [C]'] = cci_sst - 273.15
    data['cci_sst_unc [C]'] = cci_sst_unc
    data['cci_ice_fraction'] = ice
    st = file.split('.')
    data.to_csv(file,sep='\t',index=False)
