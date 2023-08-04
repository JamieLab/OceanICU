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

def driver(out_file,vars,start_yr=1990,end_yr=2020,resolution=1):
    lon,lat = du.reg_grid(lat=resolution,lon=resolution)
    direct = {}
    for a in vars:
        timesteps,direct[a[0]+'_'+a[1]],month_track = build_timeseries(a[2],a[1],start_yr,end_yr,lon,lat,a[0])
        if a[3] == 1:
            print('Producing anomaly...')
            direct[a[0] + '_' + a[1] + '_anom'] = produce_anomaly(direct[a[0]+'_'+a[1]],month_track)
    save_netcdf(out_file,direct,lon,lat,timesteps)

"""
Functions for setting up filenames for the different variables, allowing for a scalable
number of variables to be added to netCDF for neural network training.
"""
def sst_cci_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y%m_ESA_CCI_MONTHLY_SST_1_deg.nc'))

def oisst_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y%m_OISSTv2.nc'))

def oc_cci_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_%Y%m-fv6.0_1_deg.nc'))

def noaa_ersl_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y_%m_NOAA_ERSL_xCO2.nc'))

def gebco_filename(load_loc):
    return load_loc+'/GEBCO_2022_sub_ice_topo_1_deg.nc'

def longhurst_filename(load_loc):
    return load_loc+'/Longhurst_1_deg.nc'

def era5_ws_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_si10_1_deg.nc'))

def era5_mslp_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_msl_1_deg.nc'))

def era5_blh_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_blh_1_deg.nc'))

def era5_d2m_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_d2m_1_deg.nc'))

def era5_t2m_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_t2m_1_deg.nc'))

def era5_msdwlwrf_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_msdwlwrf_1_deg.nc'))

def era5_msdwswrf_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_ERA5_msdwswrf_1_deg.nc'))

def cmems_sss_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_CMEMS_GLORYSV12_SSS_1_deg.nc'))

def cmems_mld_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_CMEMS_GLORYSV12_MLD_1_deg.nc'))

def taka_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('takahashi_%m_.nc'))

def bicep_pp_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_BICEP_NCEO_PP_ESA-OC-L3S-MERGED-1M_MONTHLY_1_deg.nc'))

def bicep_poc_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_BICEP_NCEO_POC_ESA-OC-L3S-MERGED-1M_MONTHLY_4km_GEO_PML-fv5.0_1_deg.nc'))

def bicep_ep_dun_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_BICEP_NCEO_ExportProduction_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_fv4.2_1_deg_EP_Dunne.nc'))

def bicep_ep_li_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_BICEP_NCEO_ExportProduction_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_fv4.2_1_deg_EP_Li.nc'))

def bicep_ep_henson_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_BICEP_NCEO_ExportProduction_ESA-OC-L3S-MERGED-1M_MONTHLY_9km_mapped_fv4.2_1_deg_EP_Henson.nc'))

def watson_som_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y%m_watson_som.nc'))

def ccmp_filename(load_loc,yr,mon):
    dat = datetime.datetime(yr,mon,1)
    return os.path.join(load_loc,dat.strftime('%Y'),dat.strftime('%Y_%m_CCMP_Wind_Analysis__V03.0_L4.5_1_deg.nc'))

"""
End of filename definition functions! -------------------------------------------
"""
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
        if (variable == 'analysed_sst') or (variable == 'sea_ice_fraction'):
            file = sst_cci_filename(load_loc,yr,mon)
        elif variable == 'xCO2':
            file = noaa_ersl_filename(load_loc,yr,mon)
        elif variable == 'chlor_a':
            file = oc_cci_filename(load_loc,yr,mon)
        elif variable == 'elevation':
            file = gebco_filename(load_loc)
        elif variable == 'longhurst':
            file = longhurst_filename(load_loc)
        elif variable == 'si10':
            file = era5_ws_filename(load_loc,yr,mon)
        elif variable == 'msl':
            file = era5_mslp_filename(load_loc,yr,mon)
        elif variable == 'so':
            file = cmems_sss_filename(load_loc,yr,mon)
        elif variable == 'mlotst':
            file = cmems_mld_filename(load_loc,yr,mon)
        elif variable == 'blh':
            file = era5_blh_filename(load_loc,yr,mon)
        elif variable == 'd2m':
            file = era5_d2m_filename(load_loc,yr,mon)
        elif variable == 't2m':
            file = era5_t2m_filename(load_loc,yr,mon)
        elif variable == 'msdwlwrf':
            file = era5_msdwlwrf_filename(load_loc,yr,mon)
        elif variable == 'msdwswrf':
            file = era5_msdwswrf_filename(load_loc,yr,mon)
        elif variable == 'taka':
            file = taka_filename(load_loc,yr,mon)
        elif variable == 'pp':
            file = bicep_pp_filename(load_loc,yr,mon)
        elif variable == 'POC':
            file = bicep_poc_filename(load_loc,yr,mon)
        elif variable == 'EP_Dunne':
            file = bicep_ep_dun_filename(load_loc,yr,mon)
        elif variable == 'EP_Henson':
            file = bicep_ep_henson_filename(load_loc,yr,mon)
        elif variable == 'EP_Li':
            file = bicep_ep_li_filename(load_loc,yr,mon)
        elif variable == 'biome':
            file = watson_som_filename(load_loc,yr,mon)
        elif variable == 'w':
            file = ccmp_filename(load_loc,yr,mon)
        elif variable == 'sst':
            file = oisst_filename(load_loc,yr,mon)
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
