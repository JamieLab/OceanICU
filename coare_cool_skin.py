#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 05/2023

"""
from neural_network_train import load_data
from construct_input_netcdf import save_netcdf
import numpy as np
import noaa_coare.coare35vn as co
import noaa_coare_36.coare36vn_zrf_et as co36
import Data_Loading.data_utils as du
from netCDF4 import Dataset
import os

def calc_coare(data_file,out_loc,ws = None,tair = None,dewair = None, sst = None, msl = None,rs = None,rl = None, zi = None,start_yr = 1985,end_yr = 2022):
    out_file = os.path.join(out_loc,ws + '_' + tair + '_' + dewair + '_' + sst + '_' + msl + '_' + rs + '_' + rl + '_' + zi + '_coare_' + str(start_yr) + '_' + str(end_yr) + '_3-5.nc')
    if du.checkfileexist(out_file):
        print('COARE3.5 file exists - loading cool skin...')
        c = Dataset(out_file)
        out = np.array(c.variables['dter'])
        c.close()
    else:
        print('Generating COARE3.5 cool skin through time....')
        du.makefolder(out_loc)
        vars = [ws,tair,sst,dewair,msl,rs,rl,zi]
        tabl,output_size,lon,lat,time = load_data(data_file,vars,outp=False)
        lat_g,lon_g,t = np.meshgrid(lat,lon,range(output_size[2]))
        tabl['lat'] = np.reshape(lat_g,(-1,1))
        tair_c = np.array(tabl[tair]) -273.15
        dewair_c = np.array(tabl[dewair]) - 273.15
        tabl['rh'] = 100* (np.exp((17.625*dewair_c) / (243.04+dewair_c)) / np.exp((17.625*tair_c) / (243.04+tair_c)))

        out = co.coare35vn(u = tabl[ws],t = tabl[tair]-273.15, rh = tabl['rh'],ts = tabl[sst]-273.15,P=tabl[msl]/100, Rs = tabl[rs], Rl = tabl[rl], zu = 10,
            zt = 2, zq = 2, lat = tabl['lat'], zi = tabl[zi],jcool = 1)
        print(out)
        direct = {}
        direct['dter'] = np.reshape(out,(output_size))
        save_netcdf(out_file,direct,lon,lat,output_size[2],flip=False)
        out = direct['dter']
    return out

def calc_coare_36(data_file,out_loc,ws = None,tair = None,dewair = None, sst = None, msl = None,rs = None,rl = None, zi = None,sal=None,start_yr = 1985,end_yr = 2022):
    out_file = os.path.join(out_loc,ws + '_' + tair + '_' + dewair + '_' + sst + '_' + msl + '_' + rs + '_' + rl + '_' + zi + '_' + sal + '_coare_' + str(start_yr) + '_' + str(end_yr) + '_3-6.nc')
    if du.checkfileexist(out_file):
        print('COARE3.6 file exists - loading cool skin...')
        c = Dataset(out_file)
        out = np.array(c.variables['dter'])
        c.close()
    else:
        print('Generating COARE3.6 cool skin through time....')
        du.makefolder(out_loc)
        vars = [ws,tair,sst,dewair,msl,rs,rl,zi,sal]
        timesteps = (end_yr - start_yr+1) *12
        print(timesteps)

        for i in range(timesteps):
            tabl,output_size,lon,lat,time = load_data(data_file,vars,outp=False,load_single=True,timestep=i)
            if i ==0:
                out_full = np.zeros(([output_size[0],output_size[1],timesteps])); out_full[:] = np.nan
            lat_g,lon_g = np.meshgrid(lat,lon)
            tabl['lat'] = np.reshape(lat_g,(-1,1))
            tabl['lon'] =  np.reshape(lon_g,(-1,1))
            tair_c = np.array(tabl[tair]) -273.15
            dewair_c = np.array(tabl[dewair]) - 273.15
            tabl['rh'] = 100* (np.exp((17.625*dewair_c) / (243.04+dewair_c)) / np.exp((17.625*tair_c) / (243.04+tair_c)))
            # tabl[np.isnan(tabl) == 1] = 0
            # out = co.coare35vn(u = tabl[ws],t = tabl[tair]-273.15, rh = tabl['rh'],ts = tabl[sst]-273.15,P=tabl[msl]/100, Rs = tabl[rs], Rl = tabl[rl], zu = 10,
            #     zt = 2, zq = 2, lat = tabl['lat'], zi = tabl[zi],jcool = 1)
            out = co36.coare36vn_zrf_et(u = np.array(tabl[ws]),zu=10,t=np.array(tabl[tair])-273.15,zt=2,rh=np.array(tabl['rh']),zq=2,P=np.array(tabl[msl])/100,ts=np.array(tabl[sst])-273.15,sw_dn=np.array(tabl[rs]),lw_dn=np.array(tabl[rl]),lat=np.array(tabl['lat']),lon=np.array(tabl['lon']),
                jd=1,zi=np.array(tabl[zi]),rain=0,Ss=np.array(tabl[sal]))
            out = out[:,17]
            out_full[:,:,i] = np.reshape(out,([output_size[0],output_size[1]]))
        # print(out)
        direct = {}
        direct['dter'] = out_full
        save_netcdf(out_file,direct,lon,lat,timesteps,flip=False)
        out = direct['dter']
    return out
