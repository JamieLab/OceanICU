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

def driver(out_file,vars,start_yr=1990,end_yr=2020,lon=[],lat=[],time_ref_year = 1970,fill_clim=True):
    #lon,lat = du.reg_grid(lat=resolution,lon=resolution)
    direct = {}
    for a in vars:
        timesteps,direct[a[0]+'_'+a[1]],month_track,time_track_temp = build_timeseries(a[2],a[1],start_yr,end_yr,lon,lat,a[0],fill_clim=fill_clim)
        if a[3] == 1:
            print('Producing anomaly...')
            direct[a[0] + '_' + a[1] + '_anom'] = produce_anomaly(direct[a[0]+'_'+a[1]],month_track)
    time_track = []
    for i in range(len(time_track_temp)):
        time_track.append((time_track_temp[i] - datetime.datetime(time_ref_year,1,15)).days)
        #print(time_track)
    save_netcdf(out_file,direct,lon,lat,timesteps,time_track=time_track,ref_year = time_ref_year)

def build_timeseries(load_loc,variable,start_yr,end_yr,lon,lat,name,fill_clim=True):
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
    time_track = []
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
        time_track.append(datetime.datetime(yr,mon,15))
        #print(time_track)
        mon += 1
        t += 1
        if mon==13:
            yr += 1
            mon = 1
    if fill_clim:
        out_data = fill_with_clim(out_data,np.array(avai),np.array(month_track))
    return timesteps,out_data,np.array(month_track),time_track

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

def save_netcdf(save_loc,direct,lon,lat,timesteps,flip=False,time_track=False,ref_year = 1970,units=False,long_name=False):
    """
    Function to save the final netcdf output for use in the neural network training.
    For each variable in the direct dictionary a netcdf variable is generated - this
    brings multiple variables together onto the same monthly by resolution grid into one netcdf file
    to be import into matlab.
    """
    #copts={"zlib":True,"complevel":5}
    c = Dataset(save_loc,'w',format='NETCDF4_CLASSIC')
    #c.code_used = 'construct_input_netcdf.py'
    c.date_file_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.code_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    c.code_location = 'https://github.com/JamieLab/OceanICU'

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
        if units:
            var_o.units = units[var]
        if long_name:
            var_o.long_name = long_name[var]
        var_o.date_variable = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    lat_o = c.createVariable('latitude','f4',('latitude'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = c.createVariable('longitude','f4',('longitude'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    print(time_track)
    if time_track:
        time_o = c.createVariable('time','f4',('time'))
        time_o[:] = time_track
        time_o.units = f'Days since {ref_year}-01-15'
        time_o.standard_name = 'Time of observations'
    c.close()

def append_netcdf(save_loc,direct,lon,lat,timesteps,flip=False,units=False):
    c = Dataset(save_loc,'a',format='NETCDF4_CLASSIC')
    v = c.variables.keys()
    for var in list(direct.keys()):
        if flip:
            if var in v:
                c.variables[var][:] = direct[var]
            else:
                var_o = c.createVariable(var,'f4',('latitude','longitude','time'))#,**copts)
                var_o[:] = direct[var].transpose(1,0,2)
        else:
            if var in v:
                c.variables[var][:] = direct[var]
            else:
                var_o = c.createVariable(var,'f4',('longitude','latitude','time'))#,**copts)
                var_o[:] = direct[var]
        var_o = c.variables[var]
        if units:
            var_o.units = units[var]
        var_o.date_variable = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.close()

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
    # plt.figure()
    # plt.pcolor(direct[var][:,:,0])
    # plt.show()
    append_netcdf(save_loc,direct,lon,lat,timesteps)

def append_variable(file,o_file,var,new_var=None):
    direct = {}
    c = Dataset(file,'r')
    if new_var:
        direct[new_var] = np.array(c[var])
    else:
        direct[var] = np.array(c[var])
    c.close()

    append_netcdf(o_file,direct,1,1,1,flip=False)

def append_longhurst_prov(model_save_loc,longhurstfile,long_prov,prov_val):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    prov = np.array(c.variables['prov'][:])

    d = Dataset(longhurstfile,'r')
    long = np.array(d.variables['longhurst'])
    d.close()

    d = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    bath = np.array(d.variables['ocean_proportion'][:])
    d.close()
    for i in range(len(long_prov)):
        f = np.where(long == long_prov[i])
        print(f[0].shape)
        print(np.nanmax(prov))
        prov[f[0],f[1],:] = prov_val
    f = np.where(bath == 0)
    prov[f[0],f[1],:] = np.nan
    c.variables['prov'][:] = prov
    c.close()

def manual_prov(model_save_loc,lat_g,lon_g,fill=np.nan):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    prov = np.array(c.variables['prov'][:])
    lat = np.array(c.variables['latitude'][:])
    lon = np.array(c.variables['longitude'][:])

    f = np.where((lat>lat_g[0]) & (lat <lat_g[1]) )[0]
    g = np.where((lon>lon_g[0]) & (lon <lon_g[1]) )[0]
    [g,f] = np.meshgrid(g,f)
    for i in range(prov.shape[2]):
        prov[g,f,i] = fill
    c.variables['prov'][:] = prov
    c.close()

def fill_with_var(model_save_loc,var_o,var_f,log,lag,mod = None):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    var_o_d = np.array(c.variables[var_o][:])
    var_f_d = np.array(c.variables[var_f][:])
    if mod == 'power2':
        var_f_d = var_f_d **2

    for i in range(var_o_d.shape[2]):
        f = np.where(np.isnan(var_o_d[:,:,i]) == 1)
        var_o_d[f[0],f[1],i] = var_f_d[f[0],f[1],i]
    var = var_o + '_' + var_f
    v = c.variables.keys()
    if var in v:
        c.variables[var][:] = var_o_d
    else:
        var_o = c.createVariable(var,'f4',('longitude','latitude','time'))#,**copts)
        var_o[:] = var_o_d
    c.variables[var].comment = 'Missing data filled with: '+var_f
    c.close()

def land_clear(model_save_loc):
    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    ocean = c.variables['ocean_proportion'][:]
    #ocean = ocean[:,:,np.newaxis]
    c.close()

    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    time = c.variables['time'][:]
    ocean = np.repeat(ocean[:, :, np.newaxis], len(time), axis=2)
    v = list(c.variables.keys())
    v.remove('time')
    v.remove('latitude')
    v.remove('longitude')

    for var in v:
        print(var)
        data = c.variables[var][:]
        data[ocean == 0.0] = np.nan
        c.variables[var][:] = data
    c.close()

def replace_socat_with_model(input_data_file,start_yr,end_yr,socat_var = False,gcb_model=None,mod_ref_year = 1959,plot=False,mod_variable='model_fco2'):
    c = Dataset(gcb_model,'r')
    time = c.variables['time'][:]
    mod_fco2 = c.variables['sfco2'][0,:,:,:]
    mod_lon = c.variables['lon'][:]
    c.close()
    ref_time = datetime.datetime(mod_ref_year,1,1)
    time2 = []
    for i in range(len(time)):
        time2.append((ref_time + datetime.timedelta(days = int(time[i]))).year)
    time2 = np.array(time2)
    #print(time2)
    f = np.where((time2 >= start_yr) & (time2 <= end_yr))[0]
    #print(f)
    mod_fco2 = mod_fco2[f,:,:]

    c = Dataset(input_data_file,'a')
    inp_fco2 = c.variables[socat_var][:]
    lon = c.variables['longitude'][:]
    lat = c.variables['latitude'][:]

    mod_fco2 = du.lon_switch(mod_fco2)
    mod_fco2 = mod_fco2.transpose((2,1,0))

    if plot:
        plt.figure()
        plt.pcolor(lon,lat,np.transpose(inp_fco2[:,:,0]))
        plt.figure()
        plt.pcolor(lon,lat,np.transpose(mod_fco2[:,:,0]))
        plt.show()


    f = np.where(np.isnan(inp_fco2) == 0)
    print(f)
    #
    inp_fco2[f[0],f[1],f[2]] = mod_fco2[f[0],f[1],f[2]]
    #inp_fco2 = np.reshape(inp_fco2,s)
    c.variables[socat_var][:] = inp_fco2
    c.variables[socat_var].model_modify = 'This dataset has been replace with model output from: ' + gcb_model
    c.close()
    direct = {}
    direct[mod_variable] = mod_fco2
    append_netcdf(input_data_file,direct,1,1,1)
