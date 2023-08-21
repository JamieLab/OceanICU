#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script takes monthly regular gridded outputs from individual input parameters and merges these into
a single netcdf file. The script also produces climatology values to fill dates where no inputs are
avaiable (i.e before 1997 for ocean colour data).
"""
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import Data_Loading.data_utils as du

def driver(out_file,vars,start_yr=1990,end_yr=2020,lon=[],lat=[]):
    #lon,lat = du.reg_grid(lat=resolution,lon=resolution)
    direct = {}
    for a in vars:
        timesteps,direct[a[0]+'_'+a[1]],month_track = build_timeseries(a[2],a[1],start_yr,end_yr,lon,lat,a[0])
        if a[3] == 1:
            print('Producing anomaly...')
            direct[a[0] + '_' + a[1] + '_anom'] = produce_anomaly(direct[a[0]+'_'+a[1]],month_track)
    save_netcdf(out_file,direct,lon,lat,timesteps)

def build_timeseries(load_loc,variable,start_yr,end_yr,lon,lat,name):
    """
    Function to construct a monthly timeseries between start yr and end yr for the variable.
    Once the timeseries has been cycled through, missing data at the start (assumed to be) start of
    the timeseries is filled with climatology values from the first 10 years of avaiable data.
    """
    timesteps =((end_yr-start_yr)+1)*12
    out_data = np.empty((len(lon),len(lat),timesteps))
    out_data[:] = np.nan
    yr = start_yr
    mon = 1
    avai = []
    month_track = []
    t = 0
    while yr <= end_yr:
        """
        Here we set up the different filename structures for the different products.
        For each new variable a new elif needs to be added.
        """
        dat = datetime.datetime(yr,mon,1)
        file = load_loc.replace('%Y',dat.strftime('%Y')).replace('%m',dat.strftime('%m'))
        g = glob.glob(file)
        if g:
            file = g[0]
        print(file)

        if not du.checkfileexist(file):
            avai.append(0)
        else:
            avai.append(1)
            out_data[:,:,t] = du.load_netcdf_var(file,variable)

        month_track.append(mon)
        mon += 1
        t += 1
        if mon==13:
            yr += 1
            mon = 1
    out_data = fill_with_clim(out_data,np.array(avai),np.array(month_track))
    return timesteps,out_data,np.array(month_track)

def produce_anomaly(data,month_track):
    """
    Function to produce a monthly climatology over the whole timeseries, and use this to
    make a monthly anomaly map.
    """
    anom = np.zeros((data.shape))
    anom[:] = np.nan
    clim = construct_climatology(data,month_track)
    for i in range(0,12):
        f = np.squeeze(np.where(month_track == i+1))
        #print(data[:,:,f].shape)
        #print(clim[:,:,i].shape)
        for j in range(0,len(f)):
            anom[:,:,f[j]] = data[:,:,f[j]] - clim[:,:,i]
    return anom

def fill_with_clim(data,avai,month_track):
    """
    This function fills where we dont have data at the start and end of the timeseries due to
    satellites not being in orbit yet (i.e ocean colour satellite start in 1997), or
    data not being avaiable yet.
    These data are filled with a climatology from the first/last 10 years of avaiable data.
    """
    timesteps = 10 * 12 #10 years in months
    total = len(avai)
    f = np.squeeze(np.where((avai == 0)))
    print(f)
    print(np.diff(f))
    if f.size != 0:
        dif = np.squeeze(np.where((np.diff(f) > 1)))
        print(dif)
        if dif.size == 0:
            """
            Here we just need to fill either the start or the end of the timeseries.
            If f[-1] < timesteps/2 suggests we need to fill the start of the timeseries
            and f[0] > timesteps/2 suggests its the end of the timeseries.
            """
            #print(f[-1])
            #print(total/2)
            if f[-1] < total/2:
                # If there is not 10 years of data, then we make a climatology with the remaining data
                # after the last not avaiable data section.
                if total - f[-1] < timesteps:
                    #print('f < timesteps and total - f[-1] < timesteps')
                    clim = construct_climatology(data[:,:,f[-1]+1:total],month_track[f[-1]+1:total])

                    for i in range(0,len(f)):
                        data[:,:,f[i]] = clim[:,:,month_track[f[i]]]
                else:
                    #print('f < timesteps and total - f[-1] > timesteps')
                    # If we have more then 10 years of avaiable data, then we use the last 10 years from
                    # the point we have data.
                    clim = construct_climatology(data[:,:,f[-1]+1:f[-1]+timesteps],month_track[f[-1]+1:f[-1]+timesteps])
                    for i in range(0,len(f)):
                        data[:,:,f[i]] = clim[:,:,month_track[f[i]]-1]
            elif f[0] > len(month_track)/2:
                clim = construct_climatology(data[:,:,f[0]-timesteps:f[0]-1],month_track[f[0]-timesteps:f[0]-1])
                for i in range(0,len(f)):
                    data[:,:,f[i]] = clim[:,:,month_track[f[i]]-1]
        else:
            """
            Here the dif variable has a value so we have a long split in the times that have no
            data. This suggests we need to fill both ends of the timeseries so we split the f variable for a start and
            end values. Then compute the first end, and then the second.
            """
            #Filling the start of the timeseries
            fs = f[0:dif+1]
            print(fs)
            clim = construct_climatology(data[:,:,fs[-1]+1:fs[-1]+timesteps],month_track[fs[-1]+1:fs[-1]+timesteps])
            for i in range(0,len(fs)):
                data[:,:,fs[i]] = clim[:,:,month_track[fs[i]]-1]
            #Filling the end of the timeseries
            fs = f[dif+1:]
            print(fs)
            clim = construct_climatology(data[:,:,fs[0]-timesteps:fs[0]-1],month_track[fs[0]-timesteps:fs[0]-1])
            for i in range(0,len(fs)):
                data[:,:,fs[i]] = clim[:,:,month_track[fs[i]]-1]
    return data

def construct_climatology(data,month_track):
    """
    Function to construct a monthly climatology from the data provided. Data is a three
    dimensional array (axis =2 is time) and month_track is the month that data corresponds to.
    """
    shap = data.shape
    clim = np.empty((shap[0],shap[1],12))
    for i in range(0,12):
        f = np.squeeze(np.where((month_track == i+1)))
        # Edge case below in testing where only one month of data is avaiable to construct
        # timeseries - left in if process is run on a small number of year <2
        if f.size >1:
            clim[:,:,i] = np.nanmean(data[:,:,f],axis=2)
        else:
            clim[:,:,i] = np.squeeze(data[:,:,f])
    return clim

def save_netcdf(save_loc,direct,lon,lat,timesteps,flip=False):
    """
    Function to save the final netcdf output for use in the neural network training.
    For each variable in the direct dictionary a netcdf variable is generated - this
    brings multiple variables together onto the same monthly by resolution grid into one netcdf file
    to be import into matlab.
    """
    #copts={"zlib":True,"complevel":5}
    c = Dataset(save_loc,'w',format='NETCDF4_CLASSIC')
    #c.code_used = 'construct_input_netcdf.py'
    c.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    c.code_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'

    c.createDimension('longitude',lon.shape[0])
    c.createDimension('latitude',lat.shape[0])
    c.createDimension('time',timesteps)
    #c.createDimension('time_static',1)
    for var in list(direct.keys()):
        if flip:
            var_o = c.createVariable(var,'f4',('latitude','longitude','time'))#,**copts)
            var_o[:] = direct[var].transpose(1,0,2)
        else:
            var_o = c.createVariable(var,'f4',('longitude','latitude','time'))#,**copts)
            var_o[:] = direct[var]
        #print(direct[var].shape)


    lat_o = c.createVariable('latitude','f4',('latitude'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = c.createVariable('longitude','f4',('longitude'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    c.close()

def append_netcdf(save_loc,direct,lon,lat,timesteps,flip=False):
    c = Dataset(save_loc,'a',format='NETCDF4_CLASSIC')
    for var in list(direct.keys()):
        if flip:
            var_o = c.createVariable(var,'f4',('latitude','longitude','time'))#,**copts)
            var_o[:] = direct[var].transpose(1,0,2)
        else:
            var_o = c.createVariable(var,'f4',('longitude','latitude','time'))#,**copts)
            var_o[:] = direct[var]

def single_province(save_loc,var,lon,lat,start_yr,end_yr):
    timesteps =((end_yr-start_yr)+1)*12
    prov = np.ones((len(lon),len(lat),timesteps))
    direct = {}
    direct[var] = prov
    append_netcdf(save_loc,direct,lon,lat,timesteps)

def province_shape(save_loc,var,lon,lat,start_yr,end_yr,shp_lon,shp_lat):

    timesteps =((end_yr-start_yr)+1)*12

    long, latg = np.meshgrid(lon,lat)
    s = long.shape
    vals = du.inpoly2(np.column_stack((long.ravel(),latg.ravel())),np.column_stack((shp_lon,shp_lat)))[0]
    vals = np.array(vals).astype('float64')
    print(vals)
    vals = np.reshape(vals,(lat.shape[0],lon.shape[0]))
    vals = np.transpose(vals)
    vals[vals ==0.0] = np.nan

    prov = np.repeat(vals[:,:,np.newaxis],timesteps,axis=2)
    direct = {}
    direct[var] = prov
    plt.figure()
    plt.pcolor(direct[var][:,:,0])
    plt.show()
    append_netcdf(save_loc,direct,lon,lat,timesteps)
