#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 04/2023

"""
from fluxengine.core import fe_setup_tools as fluxengine
from netCDF4 import Dataset
import datetime
import numpy as np
import os
from construct_input_netcdf import save_netcdf
import Data_Loading.data_utils as du
import pandas as pd
import matplotlib.pyplot as plt

def fluxengine_netcdf_create(model_save_loc,input_file=None,tsub=None,ws=None,seaice=None,sal=None,msl=None,xCO2=None,coolskin='Donlon02',start_yr=1990, end_yr = 2020,
    coare_out=None,tair=None,dewair=None,rs=None,rl=None,zi=None):
    """
    Function to create a netcdf file with all variables required for the Fluxengine calculations using
    standard names.
    Here we define the cool skin deviation and calculate Tskin from Tsubskin.
    """
    print('Generating fluxengine netcdf files...')
    lon,lat = du.load_grid(input_file)
    timesteps =((end_yr-start_yr)+1)*12
    vars = [tsub,ws,seaice,sal,msl,xCO2]
    vars_name = ['t_subskin','wind_speed','sea_ice_fraction','salinity','air_pressure','xCO2_atm']
    direct = {}
    for i in range(len(vars)):
        direct[vars_name[i]] = du.load_netcdf_var(input_file,vars[i])

    if coolskin == 'Donlon02':
        coolskin_dt = (-0.14 - 0.30 * np.exp(- (direct['wind_speed'] / 3.7)))
        direct['t_skin'] = direct['t_subskin'] + coolskin_dt
    elif coolskin == 'Donlon98':
        direct['t_skin'] = direct['t_subskin'] - 0.17
    elif coolskin == 'COARE3.5':
        import coare_cool_skin as ccs
        dt_coare = ccs.calc_coare(input_file,coare_out,ws = ws,tair = tair, dewair = dewair, sst = tsub, msl = msl,
            rs = rs, rl = rl, zi = zi,start_yr=start_yr,end_yr=end_yr)
        direct['t_skin'] = direct['t_subskin'] - dt_coare


    direct['wind_speed_2'] = direct['wind_speed'] ** 2
    direct['wind_speed_3'] = direct['wind_speed'] ** 3

    direct['fco2sw'] = du.load_netcdf_var(os.path.join(model_save_loc,'output.nc'),'fco2')
    #save_netcdf(os.path.join(model_save_loc,'fluxengine.nc'),direct,lon,lat,timesteps)
    fluxengine_individual_netcdf(model_save_loc,direct,lon,lat,start_yr,end_yr)

def fluxengine_individual_netcdf(model_save_loc,direct,lon,lat,start_yr = 1990,end_yr = 2020):
    """
    Function to split out the individual monthly data into seperate files for input into fluxengine.
    """
    out_loc = os.path.join(model_save_loc,'fluxengine_input')
    du.makefolder(out_loc)
    yr = start_yr
    mon = 1
    head = list(direct.keys())
    print(head)
    t = 0
    while yr <= end_yr:
        dat = datetime.datetime(yr,mon,1)
        out_file = os.path.join(out_loc,dat.strftime('%Y_%m.nc'))
        direct_temp = {}
        for a in head:
            direct_temp[a] = np.expand_dims(direct[a][:,:,t],axis=2)
        save_netcdf(out_file,direct_temp,lon,lat,1,flip=True)
        t = t+1
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon = 1

def fluxengine_run(model_save_loc,config_file = None,start_yr = 1990, end_yr = 2020):
    """
    Function to run fluxengine for a particular neural network.
    """
    return_path = os.getcwd()
    os.chdir(model_save_loc)
    returnCode, fe = fluxengine.run_fluxengine(config_file, start_yr, end_yr, singleRun=False,verbose=True)
    os.chdir(return_path)

def load_annual_flux(model_save_loc):
    data = pd.read_csv(os.path.join(model_save_loc,'_global.txt'))
    data_year = data[data['MONTH'] == 'ALL'].copy()
    return data_year

def calc_watson(file_loc = 'D:/Data/_DataSets/Watson/GCB-2022_dataprod_UoEX_WAT_20v2_1985_2021.nc',flux_dir = 'D:/ESA_CONTRACT/Watson/flux', out_dir = 'D:/ESA_CONTRACT/Watson/'):
    import Data_Loading.data_utils as du

    if not du.checkfileexist(out_dir+'_global.txt'):
        import custom_flux_av.ofluxghg_flux_budgets as bud
        import glob

        c = Dataset(file_loc)
        flux = np.array(c.variables['fgco2'][:])
        c.close()
        yr = 1985
        mon = 1
        t = 0
        while t < flux.shape[2]:
            files = glob.glob(os.path.join(flux_dir,str(yr),du.numstr(mon),'*.nc'))
            c = Dataset(files[0],'r+')
            f = flux[:,:,t]
            f = np.transpose(f,(1,0))
            f = np.flip(np.expand_dims(f,axis=0),axis=1)*12.01*60*60*24
            f[np.isnan(f) == 1] = -999
            c.variables['OF'][:] = f
            c.close()
            mon = mon+1
            t = t+1
            if mon == 13:
                yr = yr+1
                mon = 1
        bud.run_flux_budgets(indir = flux_dir,outroot=out_dir)
    data_year = load_annual_flux(out_dir)
    return data_year

def plot_annual_flux(filename,model_save_loc,lab,gcb2022=False):
    plt.figure(figsize=(10, 8))
    for i in range(len(model_save_loc)):
        data_year = load_annual_flux(model_save_loc[i])
        plt.plot(data_year['YEAR']+0.5,data_year['NET FLUX TgC']/1000,label = lab[i])
    if gcb2022:
        watson = calc_watson()
        #watson = pd.read_csv('C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/Python/watson_flux.csv')
        plt.plot(watson['YEAR']+0.5,-watson['NET FLUX TgC']/1000,'k',label='Watson et al. (2020) - GCB 2022')
        #plt.plot(time,-flux,label='Watson et al. (2020) - GCB')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Air-sea CO${_2}$ flux (Pg C yr$^{-1}$)')
    plt.savefig(os.path.join('plots',filename),dpi=300)

def GCB_remove_prov17(model_save_loc):
    bio_file = os.path.join(model_save_loc,'networks','biomes.nc')
    model_out = os.path.join(model_save_loc,'output.nc')
    c = Dataset(bio_file,'r')
    prov = np.array(c.variables['prov'])
    print(prov.shape)
    c.close()

    c = Dataset(model_out,'r')
    fco2 = np.array(c.variables['fco2'])
    print(fco2.shape)
    c.close()

    fco2[prov == 17.0] = np.nan
    c = Dataset(model_out,'a')
    c.variables['fco2'][:] = fco2
    c.close()
