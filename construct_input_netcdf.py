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
from dateutil.relativedelta import relativedelta
import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import Data_Loading.data_utils as du

def driver(out_file,vars,start_yr=1990,end_yr=2020,lon=[],lat=[],time_ref_year = 1970,fill_clim=True,append = False):
    #lon,lat = du.reg_grid(lat=resolution,lon=resolution)
    direct = {}
    clim = {}
    for a in vars:
        timesteps,direct[a[0]+'_'+a[1]],month_track,time_track_temp = build_timeseries(a[2],a[1],start_yr,end_yr,lon,lat,a[0],fill_clim=fill_clim)
        if a[3] == 1:
            print('Producing anomaly...')
            direct[a[0] + '_' + a[1] + '_anom'],clim[a[0] + '_' + a[1]+'_clim'] = produce_anomaly(direct[a[0]+'_'+a[1]],month_track)
    time_track = []
    for i in range(len(time_track_temp)):
        time_track.append((time_track_temp[i] - datetime.datetime(time_ref_year,1,15)).days)
        #print(time_track)
    if append:
        append_netcdf(out_file,direct,lon,lat,timesteps)
    else:
        save_netcdf(out_file,direct,lon,lat,timesteps,time_track=time_track,ref_year = time_ref_year)
        save_climatology(out_file,clim)

def driver8day(out_file,vars,start_yr=1990,end_yr =2022,lon=[],lat=[],time_ref_year = 1970,fill_clim=False,append=False):
    print('Testing')
    d = datetime.datetime(start_yr,1,1)
    time_track = []
    while d.year <=end_yr:
        ye = d.year
        time_track.append(d)
        d = d + datetime.timedelta(days=8)
        if ye != d.year:
            d = datetime.datetime(d.year,1,1)
    time_track_int = []
    for i in range(len(time_track)):
        time_track_int.append((time_track[i] - datetime.datetime(time_ref_year,1,15)).days)
    time_track_int = np.array(time_track_int)
    print(time_track)
    print(len(time_track))
    direct = {}
    for a in vars:
        out_data = np.empty((len(lon),len(lat),len(time_track)))
        out_data[:] = np.nan
        for i in range(len(time_track)):
            file = a[2].replace('%Y',time_track[i].strftime('%Y')).replace('%m',time_track[i].strftime('%m')).replace('%d',time_track[i].strftime('%d'))
            print(file)
            if du.checkfileexist(file):
                out_data[:,:,i] = du.load_netcdf_var(file,a[1])
        direct[a[0]+'_'+a[1]] = out_data

    save_netcdf(out_file,direct,lon,lat,len(time_track),time_track=time_track_int,ref_year=time_ref_year)

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
    return anom,clim

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
    #print(f)
    #print(np.diff(f))
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
            #print(fs)
            clim = construct_climatology(data[:,:,fs[-1]+1:fs[-1]+timesteps],month_track[fs[-1]+1:fs[-1]+timesteps])
            for i in range(0,len(fs)):
                data[:,:,fs[i]] = clim[:,:,month_track[fs[i]]-1]
            #Filling the end of the timeseries
            fs = f[dif+1:]
            #print(fs)
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

def save_netcdf(save_loc,direct,lon,lat,timesteps,flip=False,time_track=[],ref_year = 1970,units=False,long_name=False,comment=False):
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
        if comment:
            var_o.comment = comment[var]
        var_o.date_variable = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    lat_o = c.createVariable('latitude','f4',('latitude'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = c.createVariable('longitude','f4',('longitude'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    #print(time_track)
    if len(time_track)>0:
        time_o = c.createVariable('time','f4',('time'))
        time_o[:] = time_track
        time_o.units = f'Days since {ref_year}-01-15'
        time_o.standard_name = 'Time of observations'
    c.close()

def save_climatology(save_loc,direct,flip=False):
    c = Dataset(save_loc,'a')
    try:
        c.createDimension('clim_time',12)
    except:
        print('Dimension exists?')
    for var in list(direct.keys()):
        if flip:
            var_o = c.createVariable(var,'f4',('latitude','longitude','clim_time'))#,**copts)
            var_o[:] = direct[var].transpose(1,0,2)
        else:
            var_o = c.createVariable(var,'f4',('longitude','latitude','clim_time'))#,**copts)
            var_o[:] = direct[var]
        var_o.date_variable = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.close()

def append_netcdf(save_loc,direct,lon,lat,timesteps,flip=False,units=False,longname =False,comment=False):
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
        if longname:
            var_o.long_name = longname[var]
        if comment:
            var_o.comment = comment[var]
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

def append_longhurst_prov(model_save_loc,longhurstfile,long_prov,prov_val,prov_var):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    prov = np.array(c.variables[prov_var][:])

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
    c.variables[prov_var][:] = prov
    c.close()

def manual_prov(model_save_loc,lat_g,lon_g,prov_var,fill=np.nan):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    prov = np.array(c.variables[prov_var][:])
    lat = np.array(c.variables['latitude'][:])
    lon = np.array(c.variables['longitude'][:])

    f = np.where((lat>lat_g[0]) & (lat <lat_g[1]) )[0]
    g = np.where((lon>lon_g[0]) & (lon <lon_g[1]) )[0]
    [g,f] = np.meshgrid(g,f)
    for i in range(prov.shape[2]):
        prov[g,f,i] = fill
    c.variables[prov_var][:] = prov
    c.close()

def convert_prov(model_save_loc,prov_var,prov_num_replace,prov_num_replaced):
    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    prov = np.array(c.variables[prov_var][:])
    f = np.where(prov == prov_num_replace)
    prov[f] = prov_num_replaced
    c.variables[prov_var][:] = prov
    c.close()

def fill_with_var(input_file,var_o,var_f,log,lag,mod = None,calc_anom=False):
    c = Dataset(input_file,'a')
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

    if calc_anom:
        time = c.variables['time'][:]
        uni = datetime.datetime.strptime(c.variables['time'].units.split(' ')[-1],'%Y-%m-%d')
        c.close()
        time2 = []
        for i in range(len(time)):
            time2.append((uni + relativedelta(days=time[i])).month)
        time2=np.array(time2)
        anom = {}
        clim = {}
        anom[var+'_anom'],clim[var+'_clim'] = produce_anomaly(var_o_d,time2)
        append_netcdf(input_file,anom,1,1,1)
        save_climatology(input_file,clim)
    else:
        c.close()

def land_clear(model_save_loc):
    """
    Function to clear all data that is deemed on land by the ocean_proportion calculated from bathymetry data.
    """
    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    ocean = c.variables['ocean_proportion'][:]
    #ocean = ocean[:,:,np.newaxis]
    c.close()

    c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'a')
    time = c.variables['time'][:]

    v = list(c.variables.keys())
    v.remove('time')
    v.remove('latitude')
    v.remove('longitude')

    for var in v:
        print(var)
        data = c.variables[var][:]
        ocean_t = np.repeat(ocean[:, :, np.newaxis], data.shape[2], axis=2)
        data[ocean_t == 0.0] = np.nan
        c.variables[var][:] = data
    c.close()

def clear_on_prov(data_file,var_p,val):
    """
    Function to clear a data file based on a province area (i.e remove all the data thats not in the province)
    """

    c = Dataset(data_file,'a')
    time = c.variables['time'][:]
    var_p = c.variables[var_p][:]
    v = list(c.variables.keys())
    v.remove('time')
    v.remove('latitude')
    v.remove('longitude')

    for var in v:
        print(var)
        data = c.variables[var][:]
        data[var_p != val] = np.nan
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

def copy_netcdf_vars(file,vars,outfile):
    c = Dataset(file,'r')
    direct = {}
    units = {}
    longname={}
    comment = {}
    for v in vars:
        direct[v] = np.array(c[v])
        units[v] = c[v].units
        longname[v] = c[v].long_name
        try:
            comment[v] = c[v].comment
        except:
            comment[v] = ''
    c.close()
    append_netcdf(outfile,direct,1,1,1,units=units,longname=longname,comment=comment)

def netcdf_var_bias(file,var,bias, nvar=0):
    c = Dataset(file,'r')
    direct = {}

    for v in range(len(var)):
        if nvar != 0:
            direct[nvar[v]] = np.array(c[var[v]]) + bias
        else:
            direct[var[v]] = np.array(c[var[v]]) + bias
    c.close()
    append_netcdf(file,direct,1,1,1)

def extract_independent_test(output_file,sst_name,province_file,province_var,percent=0.05,seed=42):
    """
    """
    rng = np.random.RandomState(seed) # This allows us to generated the same data again from the same input data.
    c = Dataset(output_file,'r')
    fco2 = np.array(c.variables[sst_name+'_reanalysed_fCO2_sw'])
    fco2_std = np.array(c.variables[sst_name+'_reanalysed_fCO2_sw_std'])
    fco2_count = np.array(c.variables[sst_name+'_reanalysed_count_obs'])
    sst = np.array(c.variables[sst_name+'_reanalysed_sst'])
    c.close()

    c = Dataset(province_file,'r')
    provs = np.array(c.variables[province_var])
    uni = np.unique(provs[~np.isnan(provs)])

    print(uni)
    c.close()

    ind_fco2 = np.zeros((fco2.shape)); ind_fco2[:] = np.nan;
    ind_fco2_std = np.copy(ind_fco2)
    ind_fco2_count = np.copy(ind_fco2)
    ind_sst = np.copy(ind_fco2)

    for i in range(0,fco2.shape[2]):
        for j in uni:
            print(j)
            f = np.where((provs == j) & (np.isnan(fco2[:,:,i])==0)) # Find all places in the region that aren't nan
            if len(f[0]) != 0:
                print(f)
                inds = rng.choice(len(f[0]), int(np.ceil(len(f[0])*percent)), replace=False)

                ind_fco2[f[0][inds],f[1][inds],i] = fco2[f[0][inds],f[1][inds],i]; fco2[f[0][inds],f[1][inds],i] = np.nan;
                ind_fco2_std[f[0][inds],f[1][inds],i] = fco2_std[f[0][inds],f[1][inds],i]; fco2_std[f[0][inds],f[1][inds],i] = np.nan;
                ind_fco2_count[f[0][inds],f[1][inds],i] = fco2_count[f[0][inds],f[1][inds],i]; fco2_count[f[0][inds],f[1][inds],i] = np.nan;
                ind_sst[f[0][inds],f[1][inds],i] = sst[f[0][inds],f[1][inds],i]; sst[f[0][inds],f[1][inds],i] = np.nan;

    direct = {}
    direct[sst_name+'_reanalysed_fCO2_sw'] = fco2
    direct[sst_name+'_reanalysed_fCO2_sw_std'] = fco2_std
    direct[sst_name+'_reanalysed_count_obs'] = fco2_count
    direct[sst_name+'_reanalysed_sst'] = sst
    direct[sst_name+'_reanalysed_fCO2_sw_indpendent'] = ind_fco2
    direct[sst_name+'_reanalysed_fCO2_sw_std_indpendent'] = ind_fco2_std
    direct[sst_name+'_reanalysed_count_obs_indpendent'] = ind_fco2_count
    direct[sst_name+'_reanalysed_sst_indpendent'] = ind_sst

    append_netcdf(output_file,direct,1,1,1)

    c = Dataset(output_file,'a')
    c.variables[sst_name+'_reanalysed_fCO2_sw_indpendent'].random_seed = seed
    c.variables[sst_name+'_reanalysed_fCO2_sw_std_indpendent'].random_seed = seed
    c.variables[sst_name+'_reanalysed_count_obs_indpendent'].random_seed = seed
    c.variables[sst_name+'_reanalysed_sst_indpendent'].random_seed = seed
    c.close()
