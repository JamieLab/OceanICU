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
import matplotlib.transforms
font = {'weight' : 'normal',
        'size'   : 19}
matplotlib.rc('font', **font)

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

def solubility_wannink2014(sst,sal):
    sol = -58.0931 + ( 90.5069*(100.0 / sst) ) + (22.2940 * (np.log(sst/100.0))) + (sal * (0.027766 + ( (-0.025888)*(sst/100.0)) + (0.0050578*( (sst/100.0)*(sst/100.0) ) ) ) );
    sol = np.exp(sol)
    return sol

def flux_uncertainty_calc(model_save_loc,start_yr = 1990,end_yr = 2020, k_perunc=0.2,atm_unc = 1, fco2_tot_unc = -1,sst_unc = 0.27,wind_unc=0.901,sal_unc =0.1):
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
        fco2_tot_unc[fco2_tot_unc>1000] = np.nan


    else:
        fco2_tot_unc = np.ones((fco2_sw.shape)) * fco2_tot_unc
    #print(fco2_tot_unc)
    fco2_net_unc = np.array(c.variables['fco2_net_unc'])
    fco2_net_unc[fco2_net_unc>1000] = np.nan
    fco2_para_unc = np.array(c.variables['fco2_para_unc'])
    fco2_para_unc[fco2_para_unc>1000] = np.nan
    fco2_val_unc = np.array(c.variables['fco2_val_unc'])
    fco2_val_unc[fco2_val_unc>1000] = np.nan

    fco2sw_perunc = fco2_tot_unc / fco2_sw
    fco2sw_val_unc = fco2_val_unc / fco2_sw
    fco2sw_para_unc = fco2_para_unc / fco2_sw
    fco2sw_net_unc = fco2_net_unc / fco2_sw
    c.close()
    # Here we setup some montecarlo arrays - some equations are more complicated...
    sal_skin = load_flux_var(fluxloc,'salinity_skin',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    sal_skin = np.transpose(sal_skin,(1,0,2))
    sal_subskin = load_flux_var(fluxloc,'OKS1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    sal_subskin = np.transpose(sal_subskin,(1,0,2))
    sst_skin = load_flux_var(fluxloc,'ST1_Kelvin_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    sst_skin = np.transpose(sst_skin,(1,0,2))
    sst_subskin = load_flux_var(fluxloc,'FT1_Kelvin_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    sst_subskin = np.transpose(sst_subskin,(1,0,2))

    sal_skin_monte = sal_skin[:,:,:,np.newaxis]
    sal_subskin = sal_subskin[:,:,:,np.newaxis]
    sst_skin_monte = sst_skin[:,:,:,np.newaxis]
    sst_subskin = sst_subskin[:,:,:,np.newaxis]
    norm_sst = np.random.normal(0,sst_unc,20)
    norm_sst = norm_sst[np.newaxis,np.newaxis,np.newaxis,:]
    norm_sss = np.random.normal(0,sal_unc,20)
    norm_sss = norm_sss[np.newaxis,np.newaxis,np.newaxis,:]

    sal_skin_monte = sal_skin_monte + norm_sss
    sst_skin_monte = sst_skin_monte + norm_sst
    sst_skin_monte_c = sst_skin_monte - 273.15
    sst_subskin = sst_subskin+norm_sst
    sal_subskin = sal_subskin+norm_sss

    # Here we load the concentrations
    subskin_conc = load_flux_var(fluxloc,'OSFC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    subskin_conc = np.transpose(subskin_conc,(1,0,2))
    skin_conc = load_flux_var(fluxloc,'OIC1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    skin_conc = np.transpose(skin_conc,(1,0,2))
    dconc = load_flux_var(fluxloc,'Dconc',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    dconc = np.transpose(dconc,(1,0,2))

    #Load the flux
    flux = load_flux_var(fluxloc,'OF',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    flux = np.transpose(flux,(1,0,2))

    sst_skin_perc = sst_unc / sst_skin

    #Schmidt number - #Here we do a MonteCarlo as the rules are confusing to implement...
    print('Calculating schmidt uncertainty...')
    schmidt = load_flux_var(fluxloc,'SC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    schmidt = np.transpose(schmidt,(1,0,2))
    schmidt_unc = np.std(2116.8 + (-136.25 * sst_skin_monte_c) + (4.7353 * sst_skin_monte_c**2) + (-0.092307 * sst_skin_monte_c**3) + (0.0007555 * sst_skin_monte_c**4),axis=3)/schmidt
    #Wanninkhof 2014 suggest a 5% uncertainty on the polynomial fit for the schmidt number.
    # We then convert from just schmidt number to (Sc/600)^(-0.5) term in air-sea CO2 flux uncertainty
    schmidt_unc = np.sqrt(schmidt_unc**2 + 0.05**2) / 2

    # Wind_unc
    print('Calculating wind uncertainty...')
    w_2 = load_flux_var(fluxloc,'windu10_moment2',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    w_2 = np.transpose(w_2,(1,0,2))
    w_1 = load_flux_var(fluxloc,'WS1_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    w_1 = np.transpose(w_1,(1,0,2))
    gas_trans = load_flux_var(fluxloc,'OK3',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    gas_trans= np.transpose(gas_trans,(1,0,2))
    w_1 = w_1[:,:,:,np.newaxis]
    w_2 = w_2[:,:,:,np.newaxis]
    norm_w1 = np.random.normal(0,wind_unc,20)
    norm_w1 = norm_w1[np.newaxis,np.newaxis,np.newaxis,:]
    w1 = w_1 + norm_w1
    w2 = w1**2
    wind_unc = (0.333 * w1) + (0.222 * w2)
    for j in range(20):
        wind_unc[:,:,:,j] = wind_unc[:,:,:,j]*(np.sqrt(600.0/schmidt))
    wind_unc = np.std(wind_unc ,axis=3)/gas_trans

    #pH20 uncertainty - #Here we do a MonteCarlo as the rules are confusing to implement...
    print('Calculation pH20 correction uncertainty...')
    vco2_atm = load_flux_var(fluxloc,'V_gas',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    vco2_atm = np.transpose(vco2_atm,(1,0,2))
    ph2O_t = load_flux_var(fluxloc,'PH2O',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    ph2O_t = np.transpose(ph2O_t,(1,0,2))
    pressure = load_flux_var(fluxloc,'air_pressure',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    pressure = np.transpose(pressure,(1,0,2))

    fco2_atm = load_flux_var(fluxloc,'pgas_air',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    fco2_atm = np.transpose(fco2_atm,(1,0,2))

    ph20 = np.zeros((sst_skin_monte.shape))
    for i in range(20):
        ph20[:,:,:,i] = vco2_atm[:,:,:] * (pressure[:,:,:] - 1013.25 * np.exp(24.4543 - (67.4509 * (100.0/sst_skin_monte[:,:,:,i])) - (4.8489 * np.log(sst_skin_monte[:,:,:,i]/100.0)) - 0.000544 * sal_skin_monte[:,:,:,i]))/1013.25
    ph20 = np.nanstd(ph20,axis=3)/fco2_atm
    #Weiss and Price 1970 uncertainty on pH20 correction is 0.015%
    ph20 = np.sqrt(ph20**2 + 0.00015**2)
    ph20 = (ph20 * skin_conc)/np.abs(dconc)
    #Concentration unc
    #Load the fco2 atm field and convert to percentage unc

    print('Calculating xCO2atm uncertainty...')
    vco2_atm_unc = (atm_unc / vco2_atm)*skin_conc/np.abs(dconc)

    #Solubility calculations
    print('Calculating subskin solubility uncertainty...')
    subskin_sol = load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    subskin_sol = np.transpose(subskin_sol,(1,0,2))
    sol_subskin_unc = solubility_wannink2014(sst_subskin,sal_subskin)
    sol_subskin_unc = np.std(sol_subskin_unc,axis=3)/subskin_sol
    #Weiss 1974 suggest a 0.2 % uncertainity on the solubility fit
    sol_subskin_unc = np.sqrt(sol_subskin_unc **2 + 0.002**2)
    sol_subskin_unc = (sol_subskin_unc*subskin_conc)/np.abs(dconc)
    print('Calculating skin solubility uncertainty...')
    skin_sol = load_flux_var(fluxloc,'skin_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    skin_sol = np.transpose(skin_sol,(1,0,2))
    sol_skin_unc = solubility_wannink2014(sst_skin_monte,sal_skin_monte)
    sol_skin_unc = np.std(sol_skin_unc,axis=3) /skin_sol
    # Weiss 1974 suggest a 0.2 % uncertainity on the solubility fit
    sol_skin_unc = np.sqrt(sol_skin_unc **2 + 0.002**2)
    sol_skin_unc = (sol_skin_unc*skin_conc)/np.abs(dconc)
    # Load the subskin and skin concentrations


    # Calculate the concentration uncertainty in units (not percentage)
    #skin_conc_unc = skin_conc * np.sqrt(fco2atm_perunc ** 2 + sol_skin_unc **2)
    print('Calculating fCO2(sw) uncertainty...')
    subskin_conc_unc = (fco2sw_perunc * subskin_conc) / np.abs(dconc)
    subskin_conc_unc_net = (fco2sw_net_unc * subskin_conc) / np.abs(dconc)
    subskin_conc_unc_para = (fco2sw_para_unc * subskin_conc) / np.abs(dconc)
    subskin_conc_unc_val = (fco2sw_val_unc * subskin_conc) / np.abs(dconc)
    # Add the concentration unc in quadrature and then convert to a percentage.
    #conc_unc = np.sqrt(skin_conc_unc**2 + subskin_conc_unc**2) / np.abs(dconc)
    # Flux uncertainity is the percentage unc added and then converted to absolute units (not percentage)
    #flux_unc = np.sqrt(conc_unc**2 + k_perunc**2) * np.abs(flux)
    #conc_unc = conc_unc
    print('Calculating total flux uncertainty...')
    flux_unc = np.sqrt(k_perunc**2 + wind_unc **2 + schmidt_unc**2 + vco2_atm_unc**2 + ph20**2 + sol_skin_unc**2 + subskin_conc_unc**2 + sol_subskin_unc**2)
    #Save to the output netcdf
    print('Saving data...')
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

    if 'flux_unc_fco2sw' in keys:
        c.variables['flux_unc_fco2sw'][:] = subskin_conc_unc
    else:
        var_o = c.createVariable('flux_unc_fco2sw','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc

    if 'flux_unc_fco2sw_net' in keys:
        c.variables['flux_unc_fco2sw_net'][:] = subskin_conc_unc_net
    else:
        var_o = c.createVariable('flux_unc_fco2sw_net','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_net

    if 'flux_unc_fco2sw_para' in keys:
        c.variables['flux_unc_fco2sw_para'][:] = subskin_conc_unc_para
    else:
        var_o = c.createVariable('flux_unc_fco2sw_para','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_para

    if 'flux_unc_fco2sw_val' in keys:
        c.variables['flux_unc_fco2sw_val'][:] = subskin_conc_unc_val
    else:
        var_o = c.createVariable('flux_unc_fco2sw_val','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_val

    if 'flux_unc_k' in keys:
        c.variables['flux_unc_k'][:] = k_perunc
    else:
        var_o = c.createVariable('flux_unc_k','f4',('longitude','latitude','time'))
        var_o[:] = k_perunc

    if 'flux_unc_wind' in keys:
        c.variables['flux_unc_wind'][:] = wind_unc
    else:
        var_o = c.createVariable('flux_unc_wind','f4',('longitude','latitude','time'))
        var_o[:] = wind_unc

    if 'flux_unc_schmidt' in keys:
        c.variables['flux_unc_schmidt'][:] = schmidt_unc
    else:
        var_o = c.createVariable('flux_unc_schmidt','f4',('longitude','latitude','time'))
        var_o[:] = schmidt_unc

    if 'flux_unc_ph2o' in keys:
        c.variables['flux_unc_ph2o'][:] = ph20
    else:
        var_o = c.createVariable('flux_unc_ph2o','f4',('longitude','latitude','time'))
        var_o[:] = ph20

    if 'flux_unc_xco2atm' in keys:
        c.variables['flux_unc_xco2atm'][:] = vco2_atm_unc
    else:
        var_o = c.createVariable('flux_unc_xco2atm','f4',('longitude','latitude','time'))
        var_o[:] = vco2_atm_unc


    if 'flux_unc_solsubskin_unc' in keys:
        c.variables['flux_unc_solsubskin_unc'][:] = sol_subskin_unc
    else:
        var_o = c.createVariable('flux_unc_solsubskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_subskin_unc

    if 'flux_unc_solskin_unc' in keys:
        c.variables['flux_unc_solskin_unc'][:] = sol_skin_unc
    else:
        var_o = c.createVariable('flux_unc_solskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_skin_unc

    c.close()
    print('Done uncertainty calculations!')

def load_flux_var(loc,var,start_yr,end_yr,lonl,latl,timel):
    """
    Load variables out of the fluxengine monthly output files into a single variable.
    """
    import glob
    print('Loading fluxengine variable: ' + var + '...')
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
    flux_unc_components = ['fco2sw','k','wind','schmidt','ph2o','xco2atm','solsubskin_unc','solskin_unc','fco2sw_net','fco2sw_para','fco2sw_val']
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
    flux_unc[flux_unc > 2000] = np.nan
    flux_unc = flux_unc * np.abs(flux)
    flux_components = {}
    for i in flux_unc_components:
        flux_components[i] = np.transpose(np.array(c.variables['flux_unc_'+i]),(1,0,2)) * np.abs(flux)
        flux_components[i][flux_components[i]>2000] = np.nan
    c.close()
    for i in range(0,flux.shape[2]):
        flux[:,:,i] = (flux[:,:,i] * area * land * 30.5) /1e15
        flux_unc[:,:,i] = (flux_unc[:,:,i] * area * land * 30.5) / 1e15
        for j in flux_unc_components:
            flux_components[j][:,:,i] = (flux_components[j][:,:,i] * area * land * 30.5) /1e15
        if bath_cutoff:
            flu = flux[:,:,i] ; flu[elev<=bath_cutoff] = np.nan; flux[:,:,i] = flu
            flu_u = flux_unc[:,:,i]; flu_u[elev<=bath_cutoff] = np.nan; flux_unc[:,:,i] = flu_u

    year = list(range(1985,2022+1,1))
    out = []
    up = []
    down = []
    out_unc = []
    are = []
    comp = {}
    for j in flux_unc_components:
        comp[j] = []
    area_rep = np.repeat(area[:, :, np.newaxis], 12, axis=2).reshape((-1,1))
    for i in range(0,flux.shape[2],12):
        flu = flux[:,:,i:i+12]
        out.append(np.nansum(flu))
        f = np.where(np.sign(flu) == 1)
        up.append(np.nansum(flu[f]))
        f = np.where(np.sign(flu) == -1)
        down.append(np.nansum(flu[f]))

        flu = flu.reshape((-1,1))
        f = np.where(np.isnan(flu) == 0)[0]
        are.append(np.sum(area_rep[f])/12)

        out_unc.append(np.nansum(flux_unc[:,:,i:i+12])/2)
        for j in flux_unc_components:
            comp[j].append(np.nansum(flux_components[j][:,:,i:i+12])/2)

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(1,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.15,right=0.98)
    ax = [fig.add_subplot(gs[0,0])]#,fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])]

    ax[0].errorbar(year,out,yerr = out_unc)

    for i in range(0,1):
        ax[i].set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')

    mol_c = (np.array(out) * 10**15) / np.array(are) / 12
    head = 'Year, Net air-sea CO2 flux (Pg C yr-1),Net air-sea CO2 flux uncertainty (Pg C yr-1),Upward air-sea CO2 flux (Pg C yr-1),Downward air-sea CO2 flux (Pg C yr-1),Area of net air-sea CO2 flux (m-2),Mean air-sea CO2 flux rate (mol C m-2 yr-1)'
    out_f = np.transpose(np.stack((np.array(year),np.array(out),np.array(out_unc),np.array(up),np.array(down),np.array(are),np.array(mol_c))))
    for j in flux_unc_components:

        t2 = np.array(comp[j])
        t2 = t2[:,np.newaxis]
        out_f = np.concatenate((out_f,t2),axis=1)
        print(out_f.shape)
        print(np.array(comp[j]).shape)
        head = head+',flux_unc_'+j+' (Pg C yr-1)'
    np.savetxt(os.path.join(model_save_loc,'annual_flux.csv'),out_f,delimiter=',',fmt='%.5f',header=head)
    fig.savefig(os.path.join(model_save_loc,'plots','global_flux_unc.png'))
    #return flux,flux_u

def plot_example_flux(model_save_loc):
    import geopandas as gpd
    worldmap = gpd.read_file(gpd.datasets.get_path("ne_50m_land"))
    c = Dataset(model_save_loc+'/output.nc','r')
    lat = np.array(c.variables['latitude'])
    lon = np.array(c.variables['longitude'])
    flux = np.transpose(np.array(c.variables['flux']),(1,0,2))
    print(flux.shape)
    flux_unc = np.transpose(np.array(c.variables['flux_unc']),(1,0,2))
    c.close()

    fig = plt.figure(figsize=(14,14))
    gs = GridSpec(2,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.05,right=0.95)


    ax1 = fig.add_subplot(gs[0,0]);
    worldmap.plot(color="lightgrey", ax=ax1)
    ax1.text(0.03,1.07,f'(a)',transform=ax1.transAxes,va='top',fontweight='bold',fontsize = 26)
    pc = ax1.pcolor(lon,lat,np.nanmean(flux[:,:,:],axis=2),cmap=cmocean.cm.balance)
    cbar = plt.colorbar(pc,ax=ax1)
    cbar.set_label('Air-sea CO$_{2}$ flux (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([-0.2,0.2])

    ax2 = fig.add_subplot(gs[1,0]);
    worldmap.plot(color="lightgrey", ax=ax2)
    ax1.text(0.03,1.07,f'(b)',transform=ax2.transAxes,va='top',fontweight='bold',fontsize = 26)
    pc = ax2.pcolor(lon,lat,np.nanmean(flux_unc[:,:,:]*np.abs(flux),axis=2),cmap='Blues')
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([0,0.1])
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

def plot_relative_contribution(model_save_loc):
    label=['fCO$_{2 (sw)}$ Total','Gas Transfer','Wind','Schmidt','pH$_{2}$O','xCO$_{2 (atm)}$','Solubility subskin','Solubility skin']
    data = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',')
    print(data)
    totals = []
    gross = []
    for i in range(data.shape[0]):
        totals.append(np.sum(data[i,7:-3]))
        gross.append(np.sum(np.abs(data[i,3:5])))
    print(totals)
    print(gross)
    data2 = data[:,data.shape[1]-3:]
    year = data[:,0]
    data = data[:,7:-3]
    fig = plt.figure(figsize=(13,15))
    gs = GridSpec(2,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.1,right=0.6)
    ax = fig.add_subplot(gs[0,0])
    ax2 = ax.twinx()
    cols = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#91bfdb','#e5c494','#b3b3b3']
    for i in range(data.shape[0]):
        bottom = 0
        for j in range(data.shape[1]):
            if i == 1:
                p = ax.bar(year[i],data[i,j]/totals[i],bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax.bar(year[i],data[i,j]/totals[i],bottom=bottom,color=cols[j])
            bottom = bottom + (data[i,j]/totals[i])
    ax2.plot(year,gross,'k-')
    ax.legend(bbox_to_anchor=(1.14, 0.8))
    ax.set_ylabel('Relative contribution to uncertainty')
    ax.set_xlabel('Year')
    ax.set_ylim([0,1])
    ax2.set_ylabel('Gross air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')

    ax = fig.add_subplot(gs[1,0])
    label = ['fCO$_{2 (sw)}$ Network','fCO$_{2 (sw)}$ Parameter','fCO$_{2 (sw)}$ Validation']
    totals = []
    for i in range(data2.shape[0]):
        totals.append(np.sum(data2[i,:]))
    for i in range(data2.shape[0]):
        bottom = 0
        for j in range(data2.shape[1]):
            if i == 1:
                p = ax.bar(year[i],data2[i,j]/totals[i],bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax.bar(year[i],data2[i,j]/totals[i],bottom=bottom,color=cols[j])
            bottom = bottom + (data2[i,j]/totals[i])
    ax.legend(bbox_to_anchor=(1.14, 0.8))
    ax.set_ylabel('Relative contribution to uncertainty')
    ax.set_xlabel('Year')
    ax.set_ylim([0,1])
    fig.savefig(os.path.join(model_save_loc,'plots','relative_uncertainty_contribution.png'),format='png',dpi=300)
