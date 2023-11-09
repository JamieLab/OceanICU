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

def flux_uncertainty_calc(model_save_loc,start_yr = 1990,end_yr = 2020, k_perunc=0.2,atm_unc = 1, fco2_tot_unc = -1,sst_unc = 0.27,wind_unc=0.901,sal_unc =0.1,ens=100):
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

    norm_sst = np.random.normal(0,sst_unc,ens)
    norm_sss = np.random.normal(0,sal_unc,ens)
    norm_wind = np.random.normal(0,wind_unc,ens)


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

    #sst_skin_perc = sst_unc / sst_skin

    #Schmidt number - #Here we do a MonteCarlo as the rules are confusing to implement...
    print('Calculating schmidt uncertainty...')
    schmidt = load_flux_var(fluxloc,'SC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    schmidt = np.transpose(schmidt,(1,0,2))
    schmidt_unc = np.zeros((fco2_sw.shape))
    schmidt_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            sst_t = sst_skin[:,:,j] + norm_sst[i] - 273.15
            tem_u[:,:,i] = 2116.8 + (-136.25 * sst_t) + (4.7353 * sst_t**2) + (-0.092307 * sst_t**3) + (0.0007555 * sst_t**4)
        schmidt_unc[:,:,j] = np.nanstd(tem_u,axis=2) / schmidt[:,:,j]

    #Wanninkhof 2014 suggest a 5% uncertainty on the polynomial fit for the schmidt number.
    # We then convert from just schmidt number to (Sc/600)^(-0.5) term in air-sea CO2 flux uncertainty
    schmidt_unc = np.sqrt(schmidt_unc**2) / 2
    schmidt_fixed = 0.05 / 2
    # Wind_unc
    print('Calculating wind uncertainty...')
    # w_2 = load_flux_var(fluxloc,'windu10_moment2',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # w_2 = np.transpose(w_2,(1,0,2))
    w_1 = load_flux_var(fluxloc,'WS1_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    w_1 = np.transpose(w_1,(1,0,2))
    gas_trans = load_flux_var(fluxloc,'OK3',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    gas_trans= np.transpose(gas_trans,(1,0,2))


    wind_unc = np.zeros((fco2_sw.shape))
    wind_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            w1 = w_1[:,:,j] + norm_wind[i]
            w2 = w1**2
            tem_u[:,:,i] = ((0.333 * w1) + (0.222 * w2))*(np.sqrt(600.0/schmidt[:,:,j]))
        wind_unc[:,:,j] = np.std(tem_u,axis=2)/gas_trans[:,:,j]

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

    ph20 = np.zeros((fco2_sw.shape))
    ph20[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            sst_t = sst_skin[:,:,j] + norm_sst[i]
            sal_t = sal_skin[:,:,j] + norm_sss[i]
            tem_u[:,:,i] = vco2_atm[:,:,j] * (pressure[:,:,j] - 1013.25 * np.exp(24.4543 - (67.4509 * (100.0/sst_t)) - (4.8489 * np.log(sst_t/100.0)) - 0.000544 * sal_t))/1013.25
        ph20[:,:,j] = np.nanstd(tem_u,axis=2) / fco2_atm[:,:,j]
    #Weiss and Price 1970 uncertainty on pH20 correction is 0.015%
    #ph20 = np.sqrt(ph20**2 + 0.00015**2)
    ph20 = (ph20 * skin_conc)/np.abs(dconc)
    ph20_fixed = (0.00015 * skin_conc)/np.abs(dconc)
    #Concentration unc
    #Load the fco2 atm field and convert to percentage unc

    print('Calculating xCO2atm uncertainty...')
    vco2_atm_unc = (atm_unc / vco2_atm)*skin_conc/np.abs(dconc)

    #Solubility calculations
    print('Calculating subskin solubility uncertainty...')
    subskin_sol = load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    subskin_sol = np.transpose(subskin_sol,(1,0,2))
    sol_subskin_unc = np.zeros((fco2_sw.shape))
    sol_subskin_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            sst_t = sst_subskin[:,:,j] + norm_sst[i]
            sal_t = sal_subskin[:,:,j] + norm_sss[i]
            tem_u[:,:,i] = solubility_wannink2014(sst_t,sal_t)
        sol_subskin_unc[:,:,j] = np.std(tem_u,axis=2)/subskin_sol[:,:,j]

    #Weiss 1974 suggest a 0.2 % uncertainity on the solubility fit
    #sol_subskin_unc = np.sqrt(sol_subskin_unc **2 + 0.002**2)
    sol_subskin_unc = (sol_subskin_unc*subskin_conc)/np.abs(dconc)
    sol_subskin_fixed = (0.002*subskin_conc)/np.abs(dconc)

    print('Calculating skin solubility uncertainty...')
    skin_sol = load_flux_var(fluxloc,'skin_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    skin_sol = np.transpose(skin_sol,(1,0,2))
    sol_skin_unc = np.zeros((fco2_sw.shape))
    sol_skin_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            sst_t = sst_skin[:,:,j] + norm_sst[i]
            sal_t = sal_skin[:,:,j] + norm_sss[i]
            tem_u[:,:,i] = solubility_wannink2014(sst_t,sal_t)
        sol_skin_unc[:,:,j] = np.std(tem_u,axis=2)/skin_sol[:,:,j]

    # Weiss 1974 suggest a 0.2 % uncertainity on the solubility fit
    #sol_skin_unc = np.sqrt(sol_skin_unc **2 + 0.002**2)
    sol_skin_unc = (sol_skin_unc*skin_conc)/np.abs(dconc)
    sol_skin_fixed = (0.002 * skin_conc)/np.abs(dconc)
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

    """
    fCO2(sw) components - Fixed algorithm component and variable wind unc driven component
    """

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

    """
    K parameterisation components - Fixed algorithm component and variable wind unc driven component
    """
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

    """
    Schmidt number components - Fixed algorithm component and variable SST unc driven component
    """

    if 'flux_unc_schmidt' in keys:
        c.variables['flux_unc_schmidt'][:] = schmidt_unc
    else:
        var_o = c.createVariable('flux_unc_schmidt','f4',('longitude','latitude','time'))
        var_o[:] = schmidt_unc

    if 'flux_unc_schmidt_fixed' in keys:
        c.variables['flux_unc_schmidt_fixed'][:] = schmidt_fixed
    else:
        var_o = c.createVariable('flux_unc_schmidt_fixed','f4',('longitude','latitude','time'))
        var_o[:] = schmidt_fixed
    """
    pH20 components - Fixed component and variable SST/SSS driven component
    """

    if 'flux_unc_ph2o' in keys:
        c.variables['flux_unc_ph2o'][:] = ph20
    else:
        var_o = c.createVariable('flux_unc_ph2o','f4',('longitude','latitude','time'))
        var_o[:] = ph20

    if 'flux_unc_ph2o_fixed' in keys:
        c.variables['flux_unc_ph2o_fixed'][:] = ph20_fixed
    else:
        var_o = c.createVariable('flux_unc_ph2o_fixed','f4',('longitude','latitude','time'))
        var_o[:] = ph20_fixed

    """
    xCO2 component - xCO2atm fixed component
    """
    if 'flux_unc_xco2atm' in keys:
        c.variables['flux_unc_xco2atm'][:] = vco2_atm_unc
    else:
        var_o = c.createVariable('flux_unc_xco2atm','f4',('longitude','latitude','time'))
        var_o[:] = vco2_atm_unc
    """
    Solubility subskin components
    """
    if 'flux_unc_solsubskin_unc' in keys:
        c.variables['flux_unc_solsubskin_unc'][:] = sol_subskin_unc
    else:
        var_o = c.createVariable('flux_unc_solsubskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_subskin_unc

    if 'flux_unc_solsubskin_unc_fixed' in keys:
        c.variables['flux_unc_solsubskin_unc_fixed'][:] = sol_subskin_fixed
    else:
        var_o = c.createVariable('flux_unc_solsubskin_unc_fixed','f4',('longitude','latitude','time'))
        var_o[:] = sol_subskin_fixed
    """
    Solubility skin components
    """
    if 'flux_unc_solskin_unc' in keys:
        c.variables['flux_unc_solskin_unc'][:] = sol_skin_unc
    else:
        var_o = c.createVariable('flux_unc_solskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_skin_unc

    if 'flux_unc_solskin_unc_fixed' in keys:
        c.variables['flux_unc_solskin_unc_fixed'][:] = sol_skin_fixed
    else:
        var_o = c.createVariable('flux_unc_solskin_unc_fixed','f4',('longitude','latitude','time'))
        var_o[:] = sol_skin_fixed

    if 'flux_unc_fco2atm' in keys:
        c.variables['flux_unc_fco2atm'][:] = np.sqrt(ph20**2 + vco2_atm_unc**2)
    else:
        var_o = c.createVariable('flux_unc_fco2atm','f4',('longitude','latitude','time'))
        var_o[:] = np.sqrt(ph20**2 + vco2_atm_unc**2)

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
    flux_unc_components = ['fco2sw','k','wind','schmidt','ph2o','xco2atm','solsubskin_unc','solskin_unc','fco2atm','fco2sw_net','fco2sw_para','fco2sw_val']
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
    ax[0].plot(year,out,'k-',linewidth=2)
    ax[0].fill_between(year,np.array(out)-np.array(out_unc),np.array(out)+ np.array(out_unc),alpha=0.4,label='Total uncertainty')
    ax[0].fill_between(year,np.array(out)-np.array(comp['k']),np.array(out)+ np.array(comp['k']),alpha=0.4,label='Gas Transfer')
    ax[0].legend()

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
    #worldmap = gpd.read_file(gpd.datasets.get_path("ne_50m_land"))
    c = Dataset(model_save_loc+'/output.nc','r')
    lat = np.array(c.variables['latitude'])
    lon = np.array(c.variables['longitude'])
    flux = np.transpose(np.array(c.variables['flux']),(1,0,2))
    print(flux.shape)
    flux_unc = np.transpose(np.array(c.variables['flux_unc']),(1,0,2))
    c.close()
    dat = 300
    fig = plt.figure(figsize=(14,14))
    gs = GridSpec(2,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.05,right=0.95)


    ax1 = fig.add_subplot(gs[0,0]);
    #worldmap.plot(color="lightgrey", ax=ax1)
    ax1.text(0.03,1.07,f'(a)',transform=ax1.transAxes,va='top',fontweight='bold',fontsize = 26)
    pc = ax1.pcolor(lon,lat,flux[:,:,dat],cmap=cmocean.cm.balance)
    cbar = plt.colorbar(pc,ax=ax1)
    cbar.set_label('Air-sea CO$_{2}$ flux (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([-0.2,0.2])

    ax2 = fig.add_subplot(gs[1,0]);
    #worldmap.plot(color="lightgrey", ax=ax2)
    ax1.text(0.03,1.07,f'(b)',transform=ax2.transAxes,va='top',fontweight='bold',fontsize = 26)
    pc = ax2.pcolor(lon,lat,flux_unc[:,:,dat]*np.abs(flux[:,:,dat]),cmap='Blues')
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([0,0.1])
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_flux_example.png'),format='png',dpi=300)


    row = 2
    col = 2
    fig = plt.figure(figsize=(int(14*col),int(7*row)))
    gs = GridSpec(row,col, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.05,right=0.95)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]).ravel()
    vars = ['flux_unc_fco2sw','flux_unc_k','flux_unc_wind','flux_unc_solsubskin_unc']
    label = ['fCO$_{2 (sw)}$','Gas Transfer', 'Wind', 'Subskin solubility']
    cmax = [0.2,0.05,0.05,0.05]
    c = Dataset(model_save_loc+'/output.nc','r')

    flu = np.abs(np.array(c.variables['flux'][:,:,dat]))
    for i in range(len(vars)):
        print(i)
        data = np.transpose(np.array(c.variables[vars[i]][:,:,dat])*flu)
        pc = axs[i].pcolor(lon,lat,data,cmap='Blues',vmin = 0,vmax=cmax[i])
        cbar = plt.colorbar(pc,ax=axs[i])
        cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
        axs[i].set_title(label[i])
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_flux_components_example.png'),format='png',dpi=300)


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
    label=['fCO$_{2 (sw)}$ Total','Gas Transfer','Wind','Schmidt','Solubility subskin','Solubility skin','fCO$_{2 (atm)}$']
    label2 = ['pH$_{2}$O','xCO$_{2 (atm)}$']
    data = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',')
    print(data)
    totals = []
    gross = []
    for i in range(data.shape[0]):
        totals.append(np.sum(data[i,[7,8,9,10,13,14,15]]))
        gross.append(data[i,3:5])
    print(totals)
    gross = np.array(gross)
    print(gross)
    data2 = data[:,data.shape[1]-3:]
    year = data[:,0]
    #data = data[:,7:-3]
    data_atm = data[:,[11,12]]
    data = data[:,[7,8,9,10,13,14,15]]

    print(data.shape)
    fig = plt.figure(figsize=(26,15))
    gs = GridSpec(2,2, figure=fig, wspace=0.9,hspace=0.2,bottom=0.07,top=0.95,left=0.075,right=0.9)
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
    ax2.plot(year,-np.sum(np.abs(gross),axis=1),'k-',label = 'Gross Flux')
    ax2.plot(year,np.sum(gross,axis=1),'k--', label = 'Net Flux')
    ax2.legend(bbox_to_anchor=(1.6, 0.9))

    ax.legend(bbox_to_anchor=(1.17, 0.73))
    ax.set_ylabel('Relative contribution to uncertainty')
    ax.set_xlabel('Year')
    ax.set_ylim([0,1])
    ax2.set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')

    ax = fig.add_subplot(gs[1,0])
    label = ['fCO$_{2 (sw)}$ Network','fCO$_{2 (sw)}$ Parameter','fCO$_{2 (sw)}$ Evaluation']
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
    ax.set_title('fCO$_{2 (sw)}$ total uncertainty contributions')

    ax = fig.add_subplot(gs[0,1])
    label = ['pH$_{2}$O','xCO$_{2 (atm)}$']
    totals = []
    for i in range(data_atm.shape[0]):
        totals.append(np.sum(data_atm[i,:]))
    for i in range(data_atm.shape[0]):
        bottom = 0
        for j in range(data_atm.shape[1]):
            if i == 1:
                p = ax.bar(year[i],data_atm[i,j]/totals[i],bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax.bar(year[i],data_atm[i,j]/totals[i],bottom=bottom,color=cols[j])
            bottom = bottom + (data_atm[i,j]/totals[i])
    ax.legend(bbox_to_anchor=(1, 0.8))
    ax.set_ylabel('Relative contribution to uncertainty')
    ax.set_xlabel('Year')
    ax.set_ylim([0,1])
    ax.set_title('fCO$_{2 (atm)}$ uncertainty contributions')

    fig.savefig(os.path.join(model_save_loc,'plots','relative_uncertainty_contribution.png'),format='png',dpi=300)

def montecarlo_flux_testing(model_save_loc,start_yr = 1985,end_yr = 2022):
    import scipy.interpolate as interp
    #fluxloc = model_save_loc+'/flux'
    c = Dataset(model_save_loc+'/output.nc','r')
    c_flux = np.array(c.variables['flux'])
    c_flux_unc = np.array(c.variables['flux_unc_solskin_unc']) * np.abs(c_flux)
    #fco2_tot_unc[:] = 12
    lon = np.array(c.variables['longitude'])
    lat = np.array(c.variables['latitude'])
    time = np.array(c.variables['time'])
    # fco2_tot_unc[fco2_tot_unc>1000] = np.nan
    c.close()
    # Calculating avaiable in situ data to constrain neural network.
    # c = Dataset(model_save_loc+'/input_values.nc','r')
    # fco2_insitu = np.array(c.variables['CCI_SST_reanalysed_fCO2_sw'])
    #
    # c.close()
    time_2 = np.zeros((len(time)))
    for i in range(0,len(time)):
        time_2[i] = ((datetime.datetime(1970,1,15)+datetime.timedelta(days=int(time[i]))).year)



    # subskin_sol = load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # subskin_sol = np.transpose(subskin_sol,(1,0,2))
    # subskin_sol = subskin_sol * 12.0108/1000
    # skin_conc = load_flux_var(fluxloc,'OIC1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # skin_conc = np.transpose(skin_conc,(1,0,2))
    # skin_conc = skin_conc
    # gas_trans = load_flux_var(fluxloc,'OK3',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # gas_trans= np.transpose(gas_trans,(1,0,2))
    # gas_trans = gas_trans * 24.0 / 100.0
    # schmidt = load_flux_var(fluxloc,'SC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # schmidt = np.transpose(schmidt,(1,0,2))
    # schmidt = np.sqrt(schmidt/600)

    res = np.abs(lon[0]-lon[1])
    #lon,lat = du.reg_grid()
    area = du.area_grid(lat = lat,lon = lon,res=res) * 1e6
    print(area.shape)
    area = np.transpose(area)


    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    land = np.squeeze(np.array(c.variables['ocean_proportion']))
    print(land.shape)
    c.close()
    #fco2_tot_unc =  fco2_tot_unc[:,:,:,np.newaxis]
    a = list(range(start_yr,end_yr+1))
    # data_totals = np.zeros((len(a)))
    # for i in range(len(a)):
    #     f = np.where(time_2 == a[i])
    #     data_totals[i] = (np.sum(np.isnan(fco2_insitu[:,:,f]) == 0))
    #     print(data_totals[i])
    # data_plotting = np.copy(data_totals)
    # range_decorr = [800,3000]
    # data_totals = np.abs(data_totals - np.max(data_totals))
    # data_totals = (range_decorr[1] - range_decorr[0]) * ((data_totals - 0) / (np.max(data_totals) - 0)) + range_decorr[0]
    # print(data_totals)
    # decors = np.loadtxt(os.path.join(model_save_loc,'decorrelation.csv'),delimiter=',')
    # data_totals = decors[:,1]
    decors = np.zeros((len(a),2))
    decors[:,0] = decors[:,0] + 400
    decors[:,1] = decors[:,0] * 0.59
    ens = 200
    out = np.zeros((len(a),ens))
    #pad = 40 # 8 = 400km, 13 = 650km, 14 = 700 km, 28 = 1400km, 40 = 2000km
    print(f'Lat 1: {lat[0]} Lat2: {lat[-1]}')
    print(f'Lon 1: {lon[0]} Lon2: {lon[-1]}')

    lon_ns,lat_ns = np.meshgrid(lon,lat)
    for j in range(0,ens):
        print(j)
        t = 0
        t_c = 1
        unc = np.zeros((c_flux.shape))
        for i in range(c_flux.shape[2]):
            if t_c ==1:
                pad = -1
                while (pad < 1) | (pad>70):
                    pad = ((decors[t,0] + (decors[t,1]*np.random.normal(0,0.33,1))) / 110)*2
                    print(pad)
                lon_s = np.linspace(lon[0],lon[-1],int((lon[-1] - lon[0])/pad))
                lat_s = np.linspace(lat[0],lat[-1],int((lat[-1]-lat[0])/pad))
                lon_ss,lat_ss = np.meshgrid(lon_s,lat_s)
            unc_o = np.random.normal(0,1,(len(lon_s),len(lat_s)))

            points = np.stack([lon_ss.ravel(),lat_ss.ravel()],-1)
            u = unc_o[:,:].ravel()
            un_int = interp.griddata(points,u,(np.stack([lon_ns.ravel(),lat_ns.ravel()],-1)),method = 'cubic')
            un_int = np.transpose(un_int.reshape((len(lat),len(lon))))
            #un_int = un_int[:,:,np.newaxis]
            #print(un_int.shape)
            unc[:,:,i] = un_int
            t_c = t_c + 1
            if t_c == 13:
                t=t+1
                t_c = 1
            if (j ==0) & (i == 0):
                fig = plt.figure()
                plt.pcolor(lon,lat,np.transpose(np.squeeze(unc[:,:,i])))
                plt.colorbar()
                fig.savefig(os.path.join(model_save_loc,'plots','example_pco2_perturbation.png'),format='png',dpi=300)
                plt.show()


        #print(unc)
        #unc = np.random.normal(0,2,[fco2_tot_unc.shape[0],fco2_tot_unc.shape[1],fco2_tot_unc.shape[2],1])
        # fco2_unc = (unc*fco2_tot_unc) + fco2_sw
        e_flux = c_flux + (unc*c_flux_unc)
        flux = np.zeros((c_flux.shape))
        for i in range(0,flux.shape[2]):
            #flux[:,:,i] =
            flux[:,:,i] = (e_flux[:,:,i] * area * land * 30.5) /1e15

        t = 0
        for i in range(0,flux.shape[2],12):
            flu = flux[:,:,i:i+12]
            out[t,j] = np.nansum(flu)
            t = t+1

    fig = plt.figure(figsize=(14,7))
    gs = GridSpec(1,2, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
    ax = fig.add_subplot(gs[0,0])
    # ax.bar(a,data_plotting)
    ax.set_xlabel('Year')
    # ax.set_ylabel('Frequency of 1 deg observations in SOCAT')
    # ax2 = ax.twinx()
    ax.plot(a,decors[:,0],'k-')
    ax.fill_between(a,decors[:,0] - (decors[:,1]/2), decors[:,0]+ (decors[:,1]/2),alpha=0.4,color='k')
    ax.set_ylabel('Decorrelation length (km)')

    ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
    ax = fig.add_subplot(gs[0,1])
    ax.plot(a,ann[:,1],'k-',linewidth=3,zorder=6)
    st = np.std(out,axis=1)
    ax.plot(a,out,zorder=2)
    ax.fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5)
    ax.fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
    ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    print(st)
    fig.savefig(os.path.join(model_save_loc,'plots','pco2_uncertainty_contribution_revised.png'),format='png',dpi=300)
    np.savetxt(os.path.join(model_save_loc,'unc_monte_revised.csv'),np.stack((np.array(a),st)),delimiter=',',fmt='%.5f')
    #plt.show()

def plot_net_flux_unc(model_save_loc):
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(1,1, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.20,right=0.98)
    ax = fig.add_subplot(gs[0,0])

    ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
    sta = np.loadtxt(os.path.join(model_save_loc,'unc_monte.csv'),delimiter=',')
    st = sta[1,:]
    a = sta[0,:]
    ax.plot(a,ann[:,1],'k-',linewidth=3,zorder=6)

    #ax.plot(a,out,zorder=2)
    ax.fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5)
    ax.fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
    ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    ax.set_xlabel('Year')
    fig.savefig(os.path.join(model_save_loc,'plots','pco2_uncertainty_contribution_single.png'),format='png',dpi=300)

def variogram_evaluation(model_save_loc):
    import skgstat as skg
    import random
    import scipy
    fluxloc = model_save_loc+'/flux'
    c = Dataset(model_save_loc+'/output.nc','r')
    fco2_sw = np.array(c.variables['fco2'])
    #fco2_tot_unc[:] = 12
    lon = np.array(c.variables['longitude'])
    lat = np.array(c.variables['latitude'])
    #fco2_tot_unc[fco2_tot_unc>1000] = np.nan
    c.close()
    # Calculating avaiable in situ data to constrain neural network.
    c = Dataset(model_save_loc+'/input_values.nc','r')
    fco2_insitu = np.array(c.variables['CCI_SST_reanalysed_fCO2_sw'])
    time = np.array(c.variables['time'])
    c.close()
    time_2 = np.zeros((len(time)))
    for i in range(0,len(time)):
        time_2[i] = ((datetime.datetime(1970,1,15)+datetime.timedelta(days=int(time[i]))).year)

    # coordinates, values = skg.data.pancake(N=300).get('sample')
    # print(coordinates)
    # V = skg.Variogram(coordinates=coordinates, values=values)
    # print(V)
    lon_g,lat_g = np.meshgrid(lon,lat); lon_g = np.transpose(lon_g); lat_g = np.transpose(lat_g);
    a = []
    out = np.zeros((2022-1985+1,5))
    out[:] = np.nan
    yr =1985
    t = 0
    hei = int(np.ceil((2022-1985+1)/5))

    fig = plt.figure(figsize=(35,7*hei))
    gs = GridSpec(hei,5, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(hei)]).ravel()
    print(axs)
    for i in range(fco2_sw.shape[2]):
        if yr != time_2[i]:
            out[t,0] = yr
            axs[t].hist(a,50)
            axs[t].set_title(out[t,0]); axs[t].set_xlabel('Decorrelation (km)'); axs[t].set_ylabel('Frequecy')
            yr = time_2[i]
            out[t,1] = np.median(a); out[t,2] = scipy.stats.iqr(a); out[t,3] = np.mean(a); out[t,4] = np.std(a)

            a = []
            t = t+1
        f = np.argwhere(((np.isnan(fco2_insitu[:,:,i].ravel()) == 0) & (np.isnan(fco2_sw[:,:,i].ravel()) == 0)))
        coords = np.transpose(np.squeeze(np.stack((lon_g.ravel()[f],lat_g.ravel()[f]))))
        #print(coords)
        #print(coords.shape)
        values = fco2_insitu[:,:,i].ravel() - fco2_sw[:,:,i].ravel()

        values = np.squeeze(values[f])
        print(values.shape)
        #print(values)


        for j in range(200):
            if len(values) < 50:
                ran = random.sample(range(len(values)), int(len(values)/2))
            elif len(values) < 300:
                ran = random.sample(range(len(values)), int(len(values)/5))
            else:
                ran = random.sample(range(len(values)), int(len(values)/10))
            try:
                V = skg.Variogram(coordinates=coords[ran], values=values[ran],dist_func=getDistanceByHaversine,maxlag=4000,fit_method='lm',estimator='dowd')
                V.n_lags = 40
                des = V.describe()
                #print(des['effective_range'])
                if (des['effective_range'] > 100) & (des['effective_range'] < 5000):
                    a.append(des['effective_range'])
            except:
                print('Exception')
    out[t,0] = yr
    axs[t].hist(a,50)
    axs[t].set_title(out[t,0]); axs[t].set_xlabel('Decorrelation (km)'); axs[t].set_ylabel('Frequecy')
    yr = time_2[i]
    out[t,1] = np.median(a); out[t,2] = scipy.stats.iqr(a); out[t,3] = np.mean(a); out[t,4] = np.std(a)

    fig.savefig(os.path.join(model_save_loc,'plots','yearly_decorrelation.png'),format='png',dpi=300)
    vals = scipy.stats.iqr(a)
    print(f'Median: {np.median(a)}')
    print(f'IQR: {vals}')
    np.savetxt(os.path.join(model_save_loc,'decorrelation.csv'),out,delimiter=',',fmt='%.5f')
    # plt.plot(out[:,0],out[:,1],'k-',linewidth=3)
    # iq = out[:,2]/2
    # plt.fill_between(out[:,0],out[:,1]-iq,out[:,1]+iq)
    # plt.show()
    # # plt.figure()
    # _, bins, _ = plt.hist(a,50,density=True)
    # mu, sigma = scipy.stats.norm.fit(a)
    # print(mu)
    # print(sigma)
    # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    # plt.plot(bins, best_fit_line)
    # plt.show()

def plot_decorrelation(model_save_loc,start_yr = 1985,end_yr = 2022):
    c = Dataset(model_save_loc+'/input_values.nc','r')
    fco2_insitu = np.array(c.variables['CCI_SST_reanalysed_fCO2_sw'])
    time = np.array(c.variables['time'])
    c.close()
    time_2 = np.zeros((len(time)))
    for i in range(0,len(time)):
        time_2[i] = ((datetime.datetime(1970,1,15)+datetime.timedelta(days=int(time[i]))).year)
    out = np.loadtxt(os.path.join(model_save_loc,'decorrelation.csv'),delimiter=',')

    a = list(range(start_yr,end_yr+1))
    data_totals = np.zeros((len(a)))
    for i in range(len(a)):
        f = np.where(time_2 == a[i])
        data_totals[i] = (np.sum(np.isnan(fco2_insitu[:,:,f]) == 0))
        print(data_totals[i])
    data_plotting = np.copy(data_totals)
    range_decorr = [800,3000]
    data_totals = np.abs(data_totals - np.max(data_totals))
    data_totals = (range_decorr[1] - range_decorr[0]) * ((data_totals - 0) / (np.max(data_totals) - 0)) + range_decorr[0]
    print(data_totals)
    fig = plt.figure(figsize=(14,7))
    gs = GridSpec(1,1, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
    ax = fig.add_subplot(gs[0,0])
    plt.plot(a,data_totals,'k-',label = 'Scaling')
    plt.fill_between(a,data_totals - (data_totals*0.59), data_totals+ (data_totals*0.59),alpha=0.4,color='k')
    plt.plot(out[:,0],out[:,1],'b-',linewidth=3,label = 'Variogram Preliminary')
    iq = out[:,2]/2
    plt.fill_between(out[:,0],out[:,1]-iq,out[:,1]+iq,alpha = 0.4)
    plt.legend()
    # plt.plot(out[:,0],out[:,3],'r-',linewidth=3)
    # plt.fill_between(out[:,0],out[:,3]-out[:,4],out[:,3]+out[:,4],alpha = 0.4,color='r')
    plt.ylabel('Decorrelation Length (km)')
    plt.xlabel('Year')
    fig.savefig(os.path.join(model_save_loc,'plots','decorrelation_comparision.png'),format='png',dpi=300)

def plot_gcb(model_save_loc):
    data = np.loadtxt(os.path.join(model_save_loc,'unc_monte.csv'),delimiter=',')
    data2 = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',')

    combine = np.sqrt(data[1,:]**2 + data2[:,8]**2)
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(1,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.2,right=0.98)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(data2[:,0],data2[:,1],'k-',linewidth=2)
    ax.fill_between(data2[:,0],data2[:,1]-combine,data2[:,1]+combine)
    ax.set_xlabel('Year')
    ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    fig.savefig(os.path.join(model_save_loc,'plots','gcb_flux_workshop.png'),format='png',dpi=300)

    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(1,1, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.2,right=0.98)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(data2[:,0],data2[:,1],'k-',linewidth=2)
    combine2 = data[1,:] / (data[1,:] + data2[:,8])
    ax.fill_between(data2[:,0],data2[:,1]-combine,data2[:,1]+combine,label='Gas Transfer')
    ax.fill_between(data2[:,0],data2[:,1]-(combine*combine2),data2[:,1]+(combine*combine2),label = 'fCO$_{2 (sw)}$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    ax.legend()
    fig.savefig(os.path.join(model_save_loc,'plots','gcb_flux_workshop_split.png'),format='png',dpi=300)

EARTHRADIUS = 6371.0
def getDistanceByHaversine(loc1, loc2):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_denter link description hereecimal,lon_decimal) pairs'''
    """
    Code from: https://stackoverflow.com/questions/22081503/distance-matrix-creation-using-nparray-with-pdist-and-squareform
    """
    #
    # "unpack" our numpy array, this extracts column wise arrays
    lat1 = loc1[1]
    lon1 = loc1[0]
    lat2 = loc2[1]
    lon2 = loc2[0]
    #
    # convert to radians ##### Completely identical
    lon1 = lon1 * np.pi / 180.0
    lon2 = lon2 * np.pi / 180.0
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0
    #
    # haversine formula #### Same, but atan2 named arctan2 in numpy
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    km = EARTHRADIUS * c
    #print(km)
    return km
