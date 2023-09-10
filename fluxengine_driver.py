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
from matplotlib.gridspec import GridSpec
import cmocean

def fluxengine_netcdf_create(model_save_loc,input_file=None,tsub=None,ws=None,seaice=None,sal=None,msl=None,xCO2=None,coolskin='Donlon02',start_yr=1990, end_yr = 2020,
    coare_out=None,tair=None,dewair=None,rs=None,rl=None,zi=None):
    """
    Function to create a netcdf file with all variables required for the Fluxengine calculations using
    standard names.
    Here we define the cool skin deviation and calculate Tskin from Tsubskin.
    """
    from construct_input_netcdf import append_netcdf
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

    append_netcdf(os.path.join(model_save_loc,'input_values.nc'),direct,lon,lat,timesteps)
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

def flux_uncertainty_calc(model_save_loc,start_yr = 1990,end_yr = 2020, k_perunc=0.2,atm_unc = 1, fco2_tot_unc = -1):
    """
    Function to calculate the time and space varying air-sea CO2 flux uncertainties
    """
    print('Calculating air-sea CO2 flux uncertainty...')
    fluxloc = model_save_loc+'/flux'
    #k_perunc = 0.2 # k percentage uncertainty
    #atm_unc = 1 # Atmospheric fco2 unc (1 uatm)

    # Flux is k([CO2(atm)] - CO2[sw])
    # So we want to know the [CO2] uncertainity. For the moment I assume no uncertainity in the solubility, so the [CO2] uncertainity is the fractional uncertainty of the fCO2 (as we are multipling)
    # [CO2(atm)] - CO2(sw) is a combination in quadrature.
    # k*[CO2] is a fractional uncertainty.
    # So I need k, fCO2 fields, [CO2] fields from the fluxengine output + the fCO2 unc from the output file.

    #Start with the fCO2 fields as this will give us the time lengths required for load_flux var.
    c = Dataset(model_save_loc+'/output.nc','r')
    fco2_sw = np.array(c.variables['fco2'])
    if fco2_tot_unc == -1:
        fco2_tot_unc = np.array(c.variables['fco2_tot_unc'])
    else:
        fco2_tot_unc = np.ones((fco2_sw.shape)) * fco2_tot_unc
    print(fco2_tot_unc)

    c.close()
    # fco2_tot_unc = np.zeros((fco2_sw.shape))
    # fco2_tot_unc[:] = 10
    fco2sw_perunc = fco2_tot_unc / fco2_sw

    #Load the flux
    flux = load_flux_var(fluxloc,'OF',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    flux = np.transpose(flux,(1,0,2))
    #Load the fco2 atm field and convert to percentage unc
    fco2_atm = load_flux_var(fluxloc,'pgas_air',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    fco2_atm = np.transpose(fco2_atm,(1,0,2))
    fco2atm_perunc = atm_unc / fco2_atm
    # Load the subskin and skin concentrations
    subskin_conc = load_flux_var(fluxloc,'OSFC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    subskin_conc = np.transpose(subskin_conc,(1,0,2))
    skin_conc = load_flux_var(fluxloc,'OIC1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    skin_conc = np.transpose(skin_conc,(1,0,2))
    # Calculate the concentration uncertainty in units (not percentage)
    skin_conc_unc = skin_conc * fco2atm_perunc
    subskin_conc_unc = subskin_conc * fco2sw_perunc

    dconc = load_flux_var(fluxloc,'Dconc',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    dconc = np.transpose(dconc,(1,0,2))
    # Add the concentration unc in quadrature and then convert to a percentage.
    conc_unc = np.sqrt(skin_conc_unc**2 + subskin_conc_unc**2) / np.abs(dconc)
    # Flux uncertainity is the percentage unc added and then converted to absolute units (not percentage)
    flux_unc = np.sqrt(conc_unc**2 + k_perunc**2) * np.abs(flux)
    conc_unc = conc_unc* np.abs(flux)
    gas_unc = k_perunc* np.abs(flux)

    #Save to the output netcdf
    c = Dataset(model_save_loc+'/output.nc','a')
    keys = c.variables.keys()
    if 'flux' in keys:
        c.variables['flux'][:] = flux
    else:
        var_o = c.createVariable('flux','f4',('longitude','latitude','time'))
        var_o[:] = flux
    if 'flux_unc' in keys:
        c.variables['flux_unc'][:] = flux_unc
    else:
        var_o = c.createVariable('flux_unc','f4',('longitude','latitude','time'))
        var_o[:] = flux_unc

    if 'flux_unc_conc' in keys:
        c.variables['flux_unc_conc'][:] = conc_unc
    else:
        var_o = c.createVariable('flux_unc_conc','f4',('longitude','latitude','time'))
        var_o[:] = conc_unc

    if 'flux_unc_k' in keys:
        c.variables['flux_unc_k'][:] = gas_unc
    else:
        var_o = c.createVariable('flux_unc_k','f4',('longitude','latitude','time'))
        var_o[:] = gas_unc

    if 'fco2_perc_unc' in keys:
        c.variables['fco2_perc_unc'][:] = fco2sw_perunc
    else:
        var_o = c.createVariable('fco2_perc_unc','f4',('longitude','latitude','time'))
        var_o[:] = fco2sw_perunc

    if 'fco2atm_perc_unc' in keys:
        c.variables['fco2atm_perc_unc'][:] = fco2atm_perunc
    else:
        var_o = c.createVariable('fco2atm_perc_unc','f4',('longitude','latitude','time'))
        var_o[:] = fco2atm_perunc
    c.close()

def load_flux_var(loc,var,start_yr,end_yr,lonl,latl,timel):
    """
    Load variables out of the fluxengine monthly output files into a single variable.
    """
    import glob
    out = np.zeros((latl,lonl,timel))
    out[:] = np.nan
    yr = start_yr
    mon = 1
    t=0
    while yr <= end_yr:
        fil = os.path.join(loc,str(yr),du.numstr(mon),'*.nc')
        #print(fil)
        g = glob.glob(fil)
        #print(g)
        c = Dataset(g[0])
        op = np.squeeze(np.array(c[var]))
        if t == 0:
            lat = np.array(c['latitude'])
            if np.sign(lat[1] - lat[0]) == -1:
                flip = True
            else:
                flip = False
        if flip:
            op = np.flipud(op)
        out[:,:,t] = op
        mon = mon+1
        t = t+1
        if mon == 13:
            yr = yr+1
            mon = 1
    out[out<-998] = np.nan
    return out

def flux_split(flux,flux_unc,f,g):
    f = f[:,np.newaxis]
    g = g[np.newaxis,:]
    flux = flux[f,g,:]
    flux_unc = flux_unc[f,g,:]
    out = []
    out_unc = []
    for i in range(0,flux.shape[2],12):
        out.append(np.nansum(flux[:,:,i:i+12]))
        out_unc.append(np.nansum(flux_unc[:,:,i:i+12])/2)

    return np.array(out),np.array(out_unc)

def calc_annual_flux(model_save_loc,lon,lat,bath_cutoff = False):
    """
    OceanICU version of the fluxengine budgets tool that allows for the uncertainities to be propagated...
    """
    import matplotlib.transforms
    font = {'weight' : 'normal',
            'size'   : 19}
    matplotlib.rc('font', **font)
    res = np.abs(lon[0]-lon[1])
    #lon,lat = du.reg_grid()
    area = du.area_grid(lat = lat,lon = lon,res=res) * 1e6

    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    land = np.transpose(np.squeeze(np.array(c.variables['ocean_proportion'])))
    if bath_cutoff:
        elev=  np.transpose(np.squeeze(np.array(c.variables['elevation'])))
    c.close()

    c = Dataset(model_save_loc+'/output.nc','r')
    flux = np.transpose(np.array(c.variables['flux']),(1,0,2))
    print(flux.shape)
    flux_unc = np.transpose(np.array(c.variables['flux_unc']),(1,0,2))
    c.close()
    for i in range(0,flux.shape[2]):
        flux[:,:,i] = (flux[:,:,i] * area * land * 30.5) /1e15
        flux_unc[:,:,i] = (flux_unc[:,:,i] * area * land * 30.5) / 1e15
        if bath_cutoff:
            flu = flux[:,:,i] ; flu[elev<=bath_cutoff] = np.nan; flux[:,:,i] = flu
            flu_u = flux_unc[:,:,i]; flu_u[elev<=bath_cutoff] = np.nan; flux_unc[:,:,i] = flu_u

    year = list(range(1985,2022+1,1))
    out = []
    up = []
    down = []
    out_unc = []
    for i in range(0,flux.shape[2],12):
        flu = flux[:,:,i:i+12]
        out.append(np.nansum(flu))
        f = np.where(np.sign(flu) == 1)
        up.append(np.nansum(flu[f]))
        f = np.where(np.sign(flu) == -1)
        down.append(np.nansum(flu[f]))

        out_unc.append(np.nansum(flux_unc[:,:,i:i+12])/2)

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(1,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.15,right=0.98)
    ax = [fig.add_subplot(gs[0,0])]#,fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])]

    ax[0].errorbar(year,out,yerr = out_unc)
    #ax[0].errorbar(year,out,yerr = np.ones((len(out)))*0.6)

    # f = np.squeeze(np.where(lat >= 30)); g = np.arange(0,len(lon));
    # out,out_unc = flux_split(flux,flux_unc,f,g)
    # ax[1].errorbar(year,out,yerr = out_unc)
    # ax[1].plot(year,out_unc*-1)
    #
    # f = np.squeeze(np.where(lat <= -30)); g = np.arange(0,len(lon));
    # out,out_unc = flux_split(flux,flux_unc,f,g)
    # ax[2].errorbar(year,out,yerr = out_unc)
    # ax[2].plot(year,out_unc*-1)
    #
    # # f = np.squeeze(np.where((lat > -30) & (lat < 30))); g = np.arange(0,len(lon));
    # # out,out_unc = flux_split(flux,flux_unc,f,g)
    # # ax[3].errorbar(year,out,yerr = out_unc)
    # # ax[3].plot(year,out_unc*-1)
    #
    # f = np.squeeze(np.where((lat > 65))); g = np.arange(0,len(lon));
    # out,out_unc = flux_split(flux,flux_unc,f,g)
    # ax[4].errorbar(year,out,yerr = out_unc)
    # ax[4].plot(year,out_unc*-1)
    #
    # # f = np.squeeze(np.where((lat > 70) & (lat < 71))); g = np.squeeze(np.where((lon < 27) & (lon>26)));
    # # print(f)
    # # print(g)
    # # print(np.arange(1985,2023,1/12).shape)
    # # ax[5].plot(np.arange(1985,2023,1/12),flux[f,g,:])
    # # ax[5].plot(np.arange(1985,2023,1/12),flux_unc[f,g,:]*-1)
    #
    # ax[0].set_title('Global')
    # ax[1].set_title('30N - 90N')
    # ax[2].set_title('30S - 90S')
    # ax[3].set_title('30N - 30S')
    # ax[4].set_title('65N - 90N')
    # ax[5].set_title('70.5N 26.5E')
    for i in range(0,1):
        ax[i].set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    out_f = np.stack((np.array(year),np.array(out),np.array(out_unc),np.array(up),np.array(down)))
    np.savetxt(os.path.join(model_save_loc,'annual_flux.csv'),out_f,delimiter=',',fmt='%.5f')
    fig.savefig(os.path.join(model_save_loc,'plots','global_flux_unc.png'))
    #return flux,flux_u

def plot_example_flux(model_save_loc):

    c = Dataset(model_save_loc+'/output.nc','r')
    lat = np.array(c.variables['latitude'])
    lon = np.array(c.variables['longitude'])
    flux = np.transpose(np.array(c.variables['flux']),(1,0,2))
    print(flux.shape)
    flux_unc = np.transpose(np.array(c.variables['flux_unc']),(1,0,2))
    c.close()

    fig = plt.figure(figsize=(21,7))
    gs = GridSpec(1,2, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.05,right=0.95)
    ax1 = fig.add_subplot(gs[0,0]);
    pc = ax1.pcolor(lon,lat,np.nanmean(flux[:,:,-12:],axis=2),cmap=cmocean.cm.balance)
    cbar = plt.colorbar(pc,ax=ax1)
    cbar.set_label('Air-sea CO$_{2}$ flux (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([-0.2,0.2])

    ax2 = fig.add_subplot(gs[0,1]);
    pc = ax2.pcolor(lon,lat,np.nanmean(flux_unc[:,:,-12:],axis=2),cmap=cmocean.cm.thermal)
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([0,0.2])
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_flux_example.png'),format='png',dpi=300)

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
