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
let = ['a','b','c','d','e','f','g','h']

def fluxengine_netcdf_create(model_save_loc,input_file=None,tsub=None,ws=None,ws2=None,ws3=None,seaice=None,sal=None,msl=None,xCO2=None,coolskin='Donlon02',start_yr=1990, end_yr = 2020,
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
    if ws2 != None:
        vars.append(ws2)
        vars_name.append('wind_speed_2')
    if ws3 != None:
        vars.append(ws3)
        vars_name.append('wind_speed_3')
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
        #direct['t_skin'][direct['t_skin'] < 271.36] = 271.36 #Here we make sure the skin temperature isn't below the freezing point of seawater...
    elif coolskin == 'None':
        # No coolskin effect applied (so Tskin == Tsubskin)
        direct['t_skin'] = direct['t_subskin']

    direct['t_skin'][direct['t_skin'] < 271.36] = 271.36 #Here we make sure the skin temperature isn't below the freezing point of seawater...

    #append_netcdf(input_file,direct,lon,lat,timesteps)
    if ws2 == None:
        direct['wind_speed_2'] = direct['wind_speed'] ** 2
    if ws3 == None:
        direct['wind_speed_3'] = direct['wind_speed'] ** 3
    #direct['sea_ice_fraction'] = direct['sea_ice_fraction']# * 100
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

def fluxengine_run(model_save_loc,config_file = None,start_yr = 1990, end_yr = 2020,output_ov = False):
    """
    Function to run fluxengine for a particular neural network.
    """
    return_path = os.getcwd()
    os.chdir(model_save_loc)

    if output_ov:
        out_dir = os.path.join(model_save_loc,output_ov)
        du.makefolder(out_dir)
        returnCode, fe = fluxengine.run_fluxengine(config_file, start_yr, end_yr, singleRun=False,verbose=True,outputDirOverride=out_dir)
    else:
        returnCode, fe = fluxengine.run_fluxengine(config_file, start_yr, end_yr, singleRun=False,verbose=True)
    os.chdir(return_path)

def solubility_wannink2014(sst,sal):
    sol = -58.0931 + ( 90.5069*(100.0 / sst) ) + (22.2940 * (np.log(sst/100.0))) + (sal * (0.027766 + ( (-0.025888)*(sst/100.0)) + (0.0050578*( (sst/100.0)*(sst/100.0) ) ) ) );
    sol = np.exp(sol)
    return sol

def flux_uncertainty_calc(model_save_loc,start_yr = 1990,end_yr = 2020, k_perunc=0.2,atm_unc = 1, fco2_tot_unc = -1,sst_unc = 0.3,wind_unc=1.8,sal_unc =0.2,ens=100,unc_input_file=None,single_run = False,flux_loc = False,override_output=False):
    """
    Function to calculate the time and space varying air-sea CO2 flux uncertainties
    """
    print('Calculating air-sea CO2 flux uncertainty...')
    if not flux_loc:
        fluxloc = os.path.join(model_save_loc,'flux')
    else:
        fluxloc = os.path.join(model_save_loc,flux_loc)
    print(fluxloc)
    #k_perunc = 0.2 # k percentage uncertainty
    #atm_unc = 1 # Atmospheric fco2 unc (1 uatm)

    # Flux is k([CO2(atm)] - CO2[sw])
    # So we want to know the [CO2] uncertainity. For the moment I assume no uncertainity in the solubility, so the [CO2] uncertainity is the fractional uncertainty of the fCO2 (as we are multipling)
    # [CO2(atm)] - CO2(sw) is a combination in quadrature.
    # k*[CO2] is a fractional uncertainty.
    # So I need k, fCO2 fields, [CO2] fields from the fluxengine output + the fCO2 unc from the output file.

    #Start with the fCO2 fields as this will give us the time lengths required for load_flux var.
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
    lat = np.array(c.variables['latitude'])
    lon = np.array(c.variables['longitude'])
    time = np.array(c.variables['time'])
    fco2_sw = np.array(c.variables['fco2'])
    if fco2_tot_unc == -1:
        fco2_tot_unc = np.array(c.variables['fco2_tot_unc'])
        fco2_tot_unc[fco2_tot_unc>1000] = np.nan
    else:
        fco2_tot_unc = np.ones((fco2_sw.shape)) * fco2_tot_unc
    #print(fco2_tot_unc.shape)
    fco2_net_unc = np.array(c.variables['fco2_net_unc'])
    fco2_net_unc[fco2_net_unc>1000] = np.nan
    fco2_para_unc = np.array(c.variables['fco2_para_unc'])
    fco2_para_unc[fco2_para_unc>1000] = np.nan
    fco2_val_unc = np.array(c.variables['fco2_val_unc'])
    fco2_val_unc[fco2_val_unc>1000] = np.nan

    fco2sw_perunc = fco2_tot_unc / fco2_sw
    #print(fco2sw_perunc.shape)
    fco2sw_val_unc = fco2_val_unc / fco2_sw
    fco2sw_para_unc = fco2_para_unc / fco2_sw
    fco2sw_net_unc = fco2_net_unc / fco2_sw
    c.close()
    # Here we setup some montecarlo arrays - some equations are more complicated...
    sal_skin = load_flux_var(fluxloc,'salinity_skin',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    sal_skin = np.transpose(sal_skin,(1,0,2))
    sal_subskin = load_flux_var(fluxloc,'OKS1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    sal_subskin = np.transpose(sal_subskin,(1,0,2))
    sst_skin = load_flux_var(fluxloc,'ST1_Kelvin_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    sst_skin = np.transpose(sst_skin,(1,0,2))
    sst_subskin = load_flux_var(fluxloc,'FT1_Kelvin_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    sst_subskin = np.transpose(sst_subskin,(1,0,2))
    if type(sst_unc) == str:
        c = Dataset(unc_input_file,'r')
        sst_unc_d = np.array(c.variables[sst_unc])
        norm_sst = np.random.normal(0,1,ens)
        c.close()
    else:
        #Assume it's a number!
        norm_sst = np.random.normal(0,sst_unc,ens)
    norm_sss = np.random.normal(0,sal_unc,ens)
    norm_wind = np.random.normal(0,wind_unc,ens)
    #print(sst_unc_d)

    # Here we load the concentrations
    subskin_conc = load_flux_var(fluxloc,'OSFC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    subskin_conc = np.transpose(subskin_conc,(1,0,2))
    skin_conc = load_flux_var(fluxloc,'OIC1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    skin_conc = np.transpose(skin_conc,(1,0,2))
    dconc = load_flux_var(fluxloc,'Dconc',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    dconc = np.transpose(dconc,(1,0,2))
    # Load the ice data
    ice = load_flux_var(fluxloc,'P1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    ice = np.transpose(ice,(1,0,2))
    #Load the flux
    flux = load_flux_var(fluxloc,'OF',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    """
    Here we correct the flux for the ice coverage...
    """
    flux = np.transpose(flux,(1,0,2)) * (1-ice)

    #sst_skin_perc = sst_unc / sst_skin

    #Schmidt number - #Here we do a MonteCarlo as the rules are confusing to implement...
    print('Calculating schmidt uncertainty...')
    schmidt = load_flux_var(fluxloc,'SC',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    schmidt = np.transpose(schmidt,(1,0,2))
    schmidt_unc = np.zeros((fco2_sw.shape))
    schmidt_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            if type(sst_unc) == str:
                sst_t = sst_skin[:,:,j] + (norm_sst[i] * sst_unc_d[:,:,j]) - 273.15
            else:
                sst_t = sst_skin[:,:,j] + norm_sst[i] - 273.15
            tem_u[:,:,i] = 2116.8 + (-136.25 * sst_t) + (4.7353 * sst_t**2) + (-0.092307 * sst_t**3) + (0.0007555 * sst_t**4)
        schmidt_unc[:,:,j] = np.nanstd(tem_u,axis=2) / schmidt[:,:,j]

    #Wanninkhof 2014 suggest a 5% uncertainty on the polynomial fit for the schmidt number.
    # We then convert from just schmidt number to (Sc/600)^(-0.5) term in air-sea CO2 flux uncertainty
    # Dividing by two as the schmidt number is square rooted in the calculation.
    schmidt_unc = np.sqrt(schmidt_unc**2) / 2
    schmidt_fixed = 0.05 / 2
    # Wind_unc
    print('Calculating wind uncertainty...')
    # w_2 = load_flux_var(fluxloc,'windu10_moment2',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2])
    # w_2 = np.transpose(w_2,(1,0,2))
    w_1 = load_flux_var(fluxloc,'WS1_mean',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    w_1 = np.transpose(w_1,(1,0,2))
    gas_trans = load_flux_var(fluxloc,'OK3',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    gas_trans= np.transpose(gas_trans,(1,0,2))


    wind_unc = np.zeros((fco2_sw.shape))
    wind_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            w1 = w_1[:,:,j] + norm_wind[i]
            w1[w1 < 0] = 0.0
            w2 = w1**2
            tem_u[:,:,i] = ((0.333 * w1) + (0.222 * w2))*(np.sqrt(600.0/schmidt[:,:,j]))
        wind_unc[:,:,j] = np.std(tem_u,axis=2)/gas_trans[:,:,j]

    #pH20 uncertainty - #Here we do a MonteCarlo as the rules are confusing to implement...
    print('Calculation pH20 correction uncertainty...')
    vco2_atm = load_flux_var(fluxloc,'V_gas',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    vco2_atm = np.transpose(vco2_atm,(1,0,2))
    ph2O_t = load_flux_var(fluxloc,'PH2O',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    ph2O_t = np.transpose(ph2O_t,(1,0,2))
    pressure = load_flux_var(fluxloc,'air_pressure',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    pressure = np.transpose(pressure,(1,0,2))
    fco2_atm = load_flux_var(fluxloc,'OAPC1',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    fco2_atm = np.transpose(fco2_atm,(1,0,2))

    ph20 = np.zeros((fco2_sw.shape))
    ph20[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            if type(sst_unc) == str:
                sst_t = sst_skin[:,:,j] + (norm_sst[i] * sst_unc_d[:,:,j])
            else:
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
    subskin_sol = load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    subskin_sol = np.transpose(subskin_sol,(1,0,2))
    sol_subskin_unc = np.zeros((fco2_sw.shape))
    sol_subskin_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            if type(sst_unc) == str:
                sst_t = sst_subskin[:,:,j] + (norm_sst[i] * sst_unc_d[:,:,j])
            else:
                sst_t = sst_subskin[:,:,j] + norm_sst[i]
            sal_t = sal_subskin[:,:,j] + norm_sss[i]
            tem_u[:,:,i] = solubility_wannink2014(sst_t,sal_t)
        sol_subskin_unc[:,:,j] = np.std(tem_u,axis=2)/subskin_sol[:,:,j]

    #Weiss 1974 suggest a 0.2 % uncertainity on the solubility fit
    #sol_subskin_unc = np.sqrt(sol_subskin_unc **2 + 0.002**2)
    sol_subskin_unc = (sol_subskin_unc*subskin_conc)/np.abs(dconc)
    sol_subskin_fixed = (0.002*subskin_conc)/np.abs(dconc)

    print('Calculating skin solubility uncertainty...')
    skin_sol = load_flux_var(fluxloc,'skin_solubility',start_yr,end_yr,fco2_sw.shape[0],fco2_sw.shape[1],fco2_sw.shape[2],single_run=single_run)
    skin_sol = np.transpose(skin_sol,(1,0,2))
    sol_skin_unc = np.zeros((fco2_sw.shape))
    sol_skin_unc[:] = np.nan
    for j in range(fco2_sw.shape[2]):
        tem_u = np.zeros((fco2_sw.shape[0],fco2_sw.shape[1],ens))
        tem_u[:] = np.nan
        for i in range(ens):
            if type(sst_unc) == str:
                sst_t = sst_skin[:,:,j] + (norm_sst[i] * sst_unc_d[:,:,j])
            else:
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
    #print(subskin_conc_unc)
    # Add the concentration unc in quadrature and then convert to a percentage.
    #conc_unc = np.sqrt(skin_conc_unc**2 + subskin_conc_unc**2) / np.abs(dconc)
    # Flux uncertainity is the percentage unc added and then converted to absolute units (not percentage)
    #flux_unc = np.sqrt(conc_unc**2 + k_perunc**2) * np.abs(flux)
    #conc_unc = conc_unc
    print('Calculating total flux uncertainty...')
    flux_unc = np.sqrt(k_perunc**2 + wind_unc **2 + schmidt_unc**2 + vco2_atm_unc**2 + ph20**2 + sol_skin_unc**2 + subskin_conc_unc**2 + sol_subskin_unc**2
        + schmidt_fixed**2 + ph20_fixed**2 + sol_skin_fixed**2 + sol_subskin_fixed**2)
    #Save to the output netcdf
    print('Saving data...')
    if override_output:
        if du.checkfileexist(os.path.join(model_save_loc,override_output)):
            c = Dataset(os.path.join(model_save_loc,override_output),'a')
        else:
            c = Dataset(os.path.join(model_save_loc,override_output),'w')
            c.date_file_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
            c.code_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
            c.code_location = 'https://github.com/JamieLab/OceanICU'

            c.createDimension('longitude',lon.shape[0])
            c.createDimension('latitude',lat.shape[0])
            c.createDimension('time',time.shape[0])
            lat_o = c.createVariable('latitude','f4',('latitude'))
            lat_o[:] = lat
            lat_o.units = 'Degrees'
            lat_o.standard_name = 'Latitude'
            lon_o = c.createVariable('longitude','f4',('longitude'))
            lon_o.units = 'Degrees'
            lon_o.standard_name = 'Longitude'
            lon_o[:] = lon
            #print(time_track)
            time_o = c.createVariable('time','f4',('time'))
            time_o[:] = time
            time_o.units = f'Days since 1970-01-15'
            time_o.standard_name = 'Time of observations'
    else:
        c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')

    keys = c.variables.keys()
    if 'flux' in keys:
        c.variables['flux'][:] = flux
    else:
        var_o = c.createVariable('flux','f4',('longitude','latitude','time'))
        var_o[:] = flux
    c.variables['flux'].units = 'g C m-2 d-1'
    c.variables['flux'].calculations = 'Flux calculations completed using FluxEngine'
    c.variables['flux'].long_name = 'Air-sea CO2 flux'
    c.variables['flux'].comment = 'Negative flux indicates atmosphere to ocean exchange'
    c.variables['flux'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    if 'flux_unc' in keys:
        c.variables['flux_unc'][:] = flux_unc
    else:
        var_o = c.createVariable('flux_unc','f4',('longitude','latitude','time'))
        var_o[:] = flux_unc
    c.variables['flux_unc'].units = 'Relative to flux'
    c.variables['flux_unc'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc'].long_name = 'Air-sea CO2 flux total uncertainty'
    c.variables['flux_unc'].seaice = 'Sea ice uncertainty not included due to asymmetric nature of this flux uncertainty component'
    c.variables['flux_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    fCO2(sw) components - Fixed algorithm component and variable wind unc driven component
    """

    if 'flux_unc_fco2sw' in keys:
        c.variables['flux_unc_fco2sw'][:] = subskin_conc_unc
    else:
        var_o = c.createVariable('flux_unc_fco2sw','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc
    c.variables['flux_unc_fco2sw'].units = 'Relative to flux'
    c.variables['flux_unc_fco2sw'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_fco2sw'].long_name = 'Air-sea CO2 flux total fCO2sw uncertainty'
    c.variables['flux_unc_fco2sw'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_fco2sw'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    if 'flux_unc_fco2sw_net' in keys:
        c.variables['flux_unc_fco2sw_net'][:] = subskin_conc_unc_net
    else:
        var_o = c.createVariable('flux_unc_fco2sw_net','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_net
    c.variables['flux_unc_fco2sw_net'].units = 'Relative to flux'
    c.variables['flux_unc_fco2sw_net'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_fco2sw_net'].long_name = 'Air-sea CO2 flux fCO2sw network uncertainty'
    c.variables['flux_unc_fco2sw_net'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_fco2sw_net'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_fco2sw_para' in keys:
        c.variables['flux_unc_fco2sw_para'][:] = subskin_conc_unc_para
    else:
        var_o = c.createVariable('flux_unc_fco2sw_para','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_para
    c.variables['flux_unc_fco2sw_para'].units = 'Relative to flux'
    c.variables['flux_unc_fco2sw_para'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_fco2sw_para'].long_name = 'Air-sea CO2 flux fCO2sw parameter uncertainty'
    c.variables['flux_unc_fco2sw_para'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_fco2sw_para'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_fco2sw_val' in keys:
        c.variables['flux_unc_fco2sw_val'][:] = subskin_conc_unc_val
    else:
        var_o = c.createVariable('flux_unc_fco2sw_val','f4',('longitude','latitude','time'))
        var_o[:] = subskin_conc_unc_val

    c.variables['flux_unc_fco2sw_val'].units = 'Relative to flux'
    c.variables['flux_unc_fco2sw_val'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_fco2sw_val'].long_name = 'Air-sea CO2 flux fCO2sw evaluation uncertainty'
    c.variables['flux_unc_fco2sw_val'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_fco2sw_val'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    """
    K parameterisation components - Fixed algorithm component and variable wind unc driven component
    """
    if 'flux_unc_k' in keys:
        c.variables['flux_unc_k'][:] = k_perunc
    else:
        var_o = c.createVariable('flux_unc_k','f4',('longitude','latitude','time'))
        var_o[:] = k_perunc
    c.variables['flux_unc_k'].units = 'Relative to flux'
    c.variables['flux_unc_k'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_k'].long_name = 'Air-sea CO2 flux gas transfer algorithm uncertainty'
    c.variables['flux_unc_k'].fixed_value = f'Algorithm uncertainty set at {k_perunc*100}%'
    c.variables['flux_unc_k'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_k'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_wind' in keys:
        c.variables['flux_unc_wind'][:] = wind_unc
    else:
        var_o = c.createVariable('flux_unc_wind','f4',('longitude','latitude','time'))
        var_o[:] = wind_unc
    c.variables['flux_unc_wind'].units = 'Relative to flux'
    c.variables['flux_unc_wind'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_wind'].long_name = 'Air-sea CO2 flux gas transfer uncertainty due to wind speed uncertainty'
    c.variables['flux_unc_wind'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_wind'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    Schmidt number components - Fixed algorithm component and variable SST unc driven component
    """

    if 'flux_unc_schmidt' in keys:
        c.variables['flux_unc_schmidt'][:] = schmidt_unc
    else:
        var_o = c.createVariable('flux_unc_schmidt','f4',('longitude','latitude','time'))
        var_o[:] = schmidt_unc
    c.variables['flux_unc_schmidt'].units = 'Relative to flux'
    c.variables['flux_unc_schmidt'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_schmidt'].long_name = 'Air-sea CO2 flux Schmidt number uncertainty due to SST uncertainty'
    c.variables['flux_unc_schmidt'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_schmidt'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_schmidt_fixed' in keys:
        c.variables['flux_unc_schmidt_fixed'][:] = schmidt_fixed
    else:
        var_o = c.createVariable('flux_unc_schmidt_fixed','f4',('longitude','latitude','time'))
        var_o[:] = schmidt_fixed
    c.variables['flux_unc_schmidt_fixed'].units = 'Relative to flux'
    c.variables['flux_unc_schmidt_fixed'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_schmidt_fixed'].long_name = 'Air-sea CO2 flux Schmidt number uncertainty due to algorithm uncertainty'
    c.variables['flux_unc_schmidt_fixed'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_schmidt_fixed'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    pH20 components - Fixed component and variable SST/SSS driven component
    """

    if 'flux_unc_ph2o' in keys:
        c.variables['flux_unc_ph2o'][:] = ph20
    else:
        var_o = c.createVariable('flux_unc_ph2o','f4',('longitude','latitude','time'))
        var_o[:] = ph20
    c.variables['flux_unc_ph2o'].units = 'Relative to flux'
    c.variables['flux_unc_ph2o'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_ph2o'].long_name = 'Air-sea CO2 flux pH2O correction uncertainty due to SST uncertainty'
    c.variables['flux_unc_ph2o'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_ph2o'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_ph2o_fixed' in keys:
        c.variables['flux_unc_ph2o_fixed'][:] = ph20_fixed
    else:
        var_o = c.createVariable('flux_unc_ph2o_fixed','f4',('longitude','latitude','time'))
        var_o[:] = ph20_fixed

    c.variables['flux_unc_ph2o_fixed'].units = 'Relative to flux'
    c.variables['flux_unc_ph2o_fixed'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_ph2o_fixed'].long_name = 'Air-sea CO2 flux pH2O correction uncertainty due to algorithm uncertainty'
    c.variables['flux_unc_ph2o_fixed'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_ph2o_fixed'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    xCO2 component - xCO2atm fixed component
    """
    if 'flux_unc_xco2atm' in keys:
        c.variables['flux_unc_xco2atm'][:] = vco2_atm_unc
    else:
        var_o = c.createVariable('flux_unc_xco2atm','f4',('longitude','latitude','time'))
        var_o[:] = vco2_atm_unc
    c.variables['flux_unc_xco2atm'].units = 'Relative to flux'
    c.variables['flux_unc_xco2atm'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_xco2atm'].long_name = 'Air-sea CO2 flux xCO2atm uncertainty'
    c.variables['flux_unc_xco2atm'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_xco2atm'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    Solubility subskin components
    """
    if 'flux_unc_solsubskin_unc' in keys:
        c.variables['flux_unc_solsubskin_unc'][:] = sol_subskin_unc
    else:
        var_o = c.createVariable('flux_unc_solsubskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_subskin_unc

    c.variables['flux_unc_solsubskin_unc'].units = 'Relative to flux'
    c.variables['flux_unc_solsubskin_unc'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_solsubskin_unc'].long_name = 'Air-sea CO2 flux subskin solubility uncertainty due to SST and SSS uncertainties'
    c.variables['flux_unc_solsubskin_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_solsubskin_unc'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_solsubskin_unc_fixed' in keys:
        c.variables['flux_unc_solsubskin_unc_fixed'][:] = sol_subskin_fixed
    else:
        var_o = c.createVariable('flux_unc_solsubskin_unc_fixed','f4',('longitude','latitude','time'))
        var_o[:] = sol_subskin_fixed

    c.variables['flux_unc_solsubskin_unc_fixed'].units = 'Relative to flux'
    c.variables['flux_unc_solsubskin_unc_fixed'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_solsubskin_unc_fixed'].long_name = 'Air-sea CO2 flux subskin solubility uncertainty due to algorithm uncertaintity'
    c.variables['flux_unc_solsubskin_unc_fixed'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_solsubskin_unc_fixed'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    """
    Solubility skin components
    """
    if 'flux_unc_solskin_unc' in keys:
        c.variables['flux_unc_solskin_unc'][:] = sol_skin_unc
    else:
        var_o = c.createVariable('flux_unc_solskin_unc','f4',('longitude','latitude','time'))
        var_o[:] = sol_skin_unc

    c.variables['flux_unc_solskin_unc'].units = 'Relative to flux'
    c.variables['flux_unc_solskin_unc'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_solskin_unc'].long_name = 'Air-sea CO2 flux skin solubility uncertainty due to SST and SSS uncertainties'
    c.variables['flux_unc_solskin_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_solskin_unc'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    if 'flux_unc_solskin_unc_fixed' in keys:
        c.variables['flux_unc_solskin_unc_fixed'][:] = sol_skin_fixed
    else:
        var_o = c.createVariable('flux_unc_solskin_unc_fixed','f4',('longitude','latitude','time'))
        var_o[:] = sol_skin_fixed

    c.variables['flux_unc_solskin_unc_fixed'].units = 'Relative to flux'
    c.variables['flux_unc_solskin_unc_fixed'].comment = 'Multiple by absolute flux to get uncertainty in g C m-2 d-1'
    c.variables['flux_unc_solskin_unc_fixed'].long_name = 'Air-sea CO2 flux skin solubility uncertainty due to algorithm uncertaintity'
    c.variables['flux_unc_solskin_unc_fixed'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables['flux_unc_solskin_unc_fixed'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'

    """
    Adding ice to output flux file...
    """
    if 'ice' in keys:
        c.variables['ice'][:] = ice
    else:
        var_o = c.createVariable('ice','f4',('longitude','latitude','time'))
        var_o[:] = ice

    c.variables['ice'].units = 'Proportion'
    c.variables['ice'].long_name = 'Proportion of ice cover'
    c.variables['ice'].comment = 'See the OceanICU framework config file for the ice dataset used in these calculations'
    c.variables['ice'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    #Adding the temperature and salinty for the skin and subskin.

    if 'subskin_temp' in keys:
        c.variables['subskin_temp'][:] = sst_subskin
    else:
        var_o = c.createVariable('subskin_temp','f4',('longitude','latitude','time'))
        var_o[:] = sst_subskin

    c.variables['subskin_temp'].units = 'Kelvin'
    c.variables['subskin_temp'].long_name = 'Subskin temperature'
    c.variables['subskin_temp'].comment = 'See the OceanICU framework config file for the SST subskin dataset used in these calculations'
    c.variables['subskin_temp'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    if 'skin_temp' in keys:
        c.variables['skin_temp'][:] = sst_skin
    else:
        var_o = c.createVariable('skin_temp','f4',('longitude','latitude','time'))
        var_o[:] = sst_skin

    c.variables['skin_temp'].units = 'Kelvin'
    c.variables['skin_temp'].long_name = 'Skin temperature'
    c.variables['skin_temp'].comment = 'See the OceanICU framework config file for the SST skin dataset used in these calculations'
    c.variables['skin_temp'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    if 'skin_salinity' in keys:
        c.variables['skin_salinity'][:] = sal_skin
    else:
        var_o = c.createVariable('skin_salinity','f4',('longitude','latitude','time'))
        var_o[:] = sal_skin

    c.variables['skin_salinity'].units = 'psu'
    c.variables['skin_salinity'].long_name = 'Skin salinity'
    c.variables['skin_salinity'].comment = 'See the OceanICU framework config file for the SSS skin dataset used in these calculations'
    c.variables['skin_salinity'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))

    if 'subskin_salinity' in keys:
        c.variables['subskin_salinity'][:] = sal_subskin
    else:
        var_o = c.createVariable('subskin_salinity','f4',('longitude','latitude','time'))
        var_o[:] = sal_subskin

    c.variables['subskin_salinity'].units = 'psu'
    c.variables['subskin_salinity'].long_name = 'Subskin salinity'
    c.variables['subskin_salinity'].comment = 'See the OceanICU framework config file for the SSS subskin dataset used in these calculations'
    c.variables['subskin_salinity'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.close()
    print('Done uncertainty calculations!')

def load_flux_var(loc,var,start_yr,end_yr,lonl,latl,timel,flip=False,single_run=False):
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
        print(fil)
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
        if single_run:
            yr = end_yr+1
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

def calc_annual_flux(model_save_loc,lon,lat,start_yr,end_yr,bath_cutoff = False,bath_file=False,flux_file=False,save_file=False,mask_file=False,flux_var = 'flux',ice_var = 'ice',bath_var = 'ocean_proportion',mask_var='mask',
    area_file = False,area_var = 'area',gcbformat=False):
    """
    OceanICU version of the fluxengine budgets tool that allows for the uncertainities to be propagated...
    """
    #flux_unc_components = ['fco2sw','k','wind','schmidt','ph2o','xco2atm','solsubskin_unc','solskin_unc','fco2atm','fco2sw_net','fco2sw_para','fco2sw_val']
    import matplotlib.transforms
    font = {'weight' : 'normal',
            'size'   : 19}
    matplotlib.rc('font', **font)
    from calendar import monthrange
    res = np.abs(lon[0]-lon[1])
    if area_file:
        print('Loading area')
        c = Dataset(area_file,'r')
        area= np.transpose(np.squeeze(np.array(c.variables[area_var])))
        c.close()
    else:
        area = du.area_grid(lat = lat,lon = lon,res=res) * 1e6
    if bath_file:
        c = Dataset(bath_file,'r')
    else:
        c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    land = np.transpose(np.squeeze(np.array(c.variables[bath_var])))
    if bath_cutoff:
        elev=  np.transpose(np.squeeze(np.array(c.variables['elevation'])))
    c.close()

    if flux_file:
        c = Dataset(flux_file,'r')
    else:
        c = Dataset(model_save_loc+'/output.nc','r')
    flux = np.transpose(np.array(c.variables[flux_var]),(1,0,2))
    ice = np.transpose(np.array(c.variables[ice_var]),(1,0,2))
    time = np.array(c.variables['time'])
    ref_time=datetime.datetime(1970,1,15)
    time_f = np.zeros((len(time),2))
    for t in range(len(time)):
        temp = (ref_time + datetime.timedelta(days = int(time[t])))
        time_f[t,0] = temp.year
        time_f[t,1] = temp.month
    print(flux.shape)
    # Corrects area for land (the flux calculation was already correctly done but the area calc wasnt...)
    area = area * land

    c.close()

    if mask_file:
        c = Dataset(mask_file,'r')
        mask = np.transpose(np.array(c.variables[mask_var]),(1,0,2))
        print(mask.shape)
        time_mask = np.array(c.variables['time'])
        time_mask_f = np.zeros((len(time_mask),2))

        for t in range(len(time_mask)):
            temp = (ref_time + datetime.timedelta(days = int(time_mask[t])))
            time_mask_f[t,0] = temp.year
            time_mask_f[t,1] = temp.month
        c.close()

    if gcbformat:
        flux = np.transpose(flux,(2,0,1))
        ice = np.transpose(ice,(2,0,1))
        #area = np.transpose(area)
        if mask_file:
            mask = np.transpose(mask,(2,0,1))

    for i in range(0,flux.shape[2]):
        mon = monthrange(int(time_f[i,0]),int(time_f[i,1]))
        flux[:,:,i] = (flux[:,:,i] * area *mon[1]) /1e15 # Modified as area is now calculated correctly with land.

        if bath_cutoff:
            flu = flux[:,:,i] ; flu[elev<=bath_cutoff] = np.nan; flux[:,:,i] = flu

    """
    Calculating the monthly air-sea CO2 fluxes for the globe/region
    """
    year = list(range(start_yr,end_yr+1,1))
    out_month = np.zeros((len(year)*12,6))

    count = 0
    ye = start_yr
    mon = 1
    while ye <=end_yr:
        g = np.where((time_f[:,0] == ye) & (time_f[:,1] == mon))
        flu = np.squeeze(flux[:,:,g])
        ice_r = np.squeeze(ice[:,:,g])
        are_temp = np.copy(area)
        if mask_file:
            p = np.where((time_mask_f[:,0] == ye) & (time_mask_f[:,1] == mon))
            mask_t = np.squeeze(mask[:,:,p])

            f = np.where(mask_t == 0)
            flu[f] = np.nan
            ice_r[f] = np.nan
        out_month[count,0] = ye + ((mon-1)/12)
        if len(np.where(np.isnan(flu) == 0)[0]) == 0:
            out_month[count,1] = np.nan
        else:
            out_month[count,1] = np.nansum(flu)
        f = np.where(np.sign(flu) == 1)
        if len(f[0]) == 0:
            out_month[count,2] = np.nan
        else:
            out_month[count,2] = np.nansum(flu[f])

        f = np.where(np.sign(flu) == -1)
        if len(f[0]) == 0:
            out_month[count,3] = np.nan
        else:
            out_month[count,3] = np.nansum(flu[f])

        f = np.where(np.isnan(flu) == 1)
        #print(f)
        ice_r[f] = np.nan
        are_temp[f] = np.nan
        if len(np.where(np.isnan(are_temp) == 0)[0]) == 0:
            out_month[count,4] = np.nan
        else:
            out_month[count,4] = np.nansum(are_temp)
        if len(np.where(np.isnan(are_temp) == 0)[0]) == 0:
            out_month[count,5] = np.nan
        else:
            out_month[count,5] =  np.nansum(are_temp*(1-ice_r))

        count = count+1
        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1

    if save_file:
        fil = save_file.split('.')
        file = fil[0] + '_monthly.csv'
    else:
        file = os.path.join(model_save_loc,'monthly_flux.csv')

    np.savetxt(file,out_month,delimiter=',',fmt='%.8f',header='Year,Net air-sea CO2 flux (Pg C mon-1),Upward air-sea CO2 flux (Pg C mon-1),Downward air-sea CO2 flux (Pg C mon-1),Area (m-2),Ice-Free area (m-2)')
    """
    Turning monthly fluxes into yearly fluxes
    """
    month_year = np.floor(out_month[:,0])
    out_year = np.zeros((len(year),6))
    count = 0
    for i in year:
        f = np.where(month_year == i)[0]
        out_year[count,0] = i
        out_year[count,1] = np.sum(out_month[f,1])
        out_year[count,2] = np.sum(out_month[f,2])
        out_year[count,3] = np.sum(out_month[f,3])
        out_year[count,4] = np.mean(out_month[f,4])
        out_year[count,5] = np.mean(out_month[f,5])
        count=count+1
    if save_file:
        file = save_file
    else:
        file = os.path.join(model_save_loc,'annual_flux.csv')
    np.savetxt(file,out_year,delimiter=',',fmt='%.8f',header='Year,Net air-sea CO2 flux (Pg C yr-1),Upward air-sea CO2 flux (Pg C yr-1),Downward air-sea CO2 flux (Pg C yr-1),Area (m-2),Ice-Free area (m-2)')

    # out = []
    # up = []
    # down = []
    # # out_unc = []
    # are = []
    # ice_are = []
    # comp = {}
    # # for j in flux_unc_components:
    # #     comp[j] = []
    # area_rep = np.repeat(area[:, :, np.newaxis], 12, axis=2).reshape((-1,1))
    # for i in range(len(year)):
    #     g = np.where(time == year[i])
    #     flu = flux[:,:,g]
    #     ice_r = ice[:,:,g]
    #     if mask_file:
    #         p = np.where(time_mask == year[i])
    #         mas = mask[:,:,p]
    #         l = np.where(mas == 0)
    #         flu[l[0],l[1],l[2]] = np.nan
    #         ice_r[l[0],l[1],l[2]] = np.nan
    #     out.append(np.nansum(flu))
    #     f = np.where(np.sign(flu) == 1)
    #     up.append(np.nansum(flu[f]))
    #     f = np.where(np.sign(flu) == -1)
    #     down.append(np.nansum(flu[f]))
    #
    #     flu = flu.reshape((-1,1))
    #     ice_r = ice_r.reshape((-1,1))
    #     f = np.where(np.isnan(flu) == 0)[0]
    #     are.append(np.sum(area_rep[f])/12)
    #     ice_are.append(np.sum(area_rep[f]*(1-ice_r[f]))/12)


    # mol_c = (np.array(out) * 10**15) / np.array(are) / 12.011
    # head = 'Year, Net air-sea CO2 flux (Pg C yr-1),Upward air-sea CO2 flux (Pg C yr-1),Downward air-sea CO2 flux (Pg C yr-1),Mean area of net air-sea CO2 flux (m-2),Mean ice-free area of net air-sea CO2 flux (m-2),Mean air-sea CO2 flux rate (mol C m-2 yr-1)'
    # out_f = np.transpose(np.stack((np.array(year),np.array(out),np.array(up),np.array(down),np.array(are),np.array(ice_are),np.array(mol_c))))
    # #out_f[out_f[:,5]==0,1:] = np.nan
    # if save_file:
    #     file = save_file
    # else:
    #     file = os.path.join(model_save_loc,'annual_flux.csv')
    # np.savetxt(file,out_f,delimiter=',',fmt='%.5f',header=head)

def fixed_uncertainty_append(model_save_loc,lon,lat,bath_cutoff = False,output_file = 'annual_flux.csv'):
    fix = ['k','ph2o_fixed','schmidt_fixed','solskin_unc_fixed','solsubskin_unc_fixed']
    res = np.abs(lon[0]-lon[1])
    area = du.area_grid(lat = lat,lon = lon,res=res) * 1e6
    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    land = np.transpose(np.squeeze(np.array(c.variables['ocean_proportion'])))
    if bath_cutoff:
        elev=  np.transpose(np.squeeze(np.array(c.variables['elevation'])))
    c.close()

    c = Dataset(model_save_loc+'/output.nc','r')
    flux = np.transpose(np.array(c.variables['flux']),(1,0,2))
    print(flux.shape)
    flux_components = {}
    for i in fix:
        flux_components[i] = np.transpose(np.array(c.variables['flux_unc_'+i]),(1,0,2)) * np.abs(flux)
        flux_components[i][flux_components[i]>1000] = np.nan
    for i in range(0,flux.shape[2]):
        for j in fix:
            flux_components[j][:,:,i] = (flux_components[j][:,:,i] * area * land * 30.5) /1e15
            if bath_cutoff:
                flu = flux_components[j][:,:,i]  ; flu[elev<=bath_cutoff] = np.nan; flux_components[j][:,:,i] = flu
    comp = {}
    for j in fix:
        comp[j] = []
    for i in range(0,flux.shape[2],12):
        for j in fix:
            comp[j].append(np.nansum(flux_components[j][:,:,i:i+12])/2)
    data = pd.read_table(os.path.join(model_save_loc,output_file),delimiter=',')
    for j in fix:
        data['flux_unc_'+j+' (Pg C yr-1)'] = comp[j]
    data.to_csv(os.path.join(model_save_loc,output_file),index=False)

def plot_example_flux(model_save_loc):
    import geopandas as gpd
    #worldmap = gpd.read_file(gpd.datasets.get_path("ne_50m_land"))
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
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
    ax1.set_facecolor('gray')

    ax2 = fig.add_subplot(gs[1,0]);
    #worldmap.plot(color="lightgrey", ax=ax2)
    ax1.text(0.03,1.07,f'(b)',transform=ax2.transAxes,va='top',fontweight='bold',fontsize = 26)
    pc = ax2.pcolor(lon,lat,flux_unc[:,:,dat]*np.abs(flux[:,:,dat]),cmap='Blues')
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
    pc.set_clim([0,0.1])
    ax2.set_facecolor('gray')
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_flux_example.png'),format='png',dpi=300)
    fig.close()

    row = 2
    col = 2
    fig = plt.figure(figsize=(int(14*col),int(7*row)))
    gs = GridSpec(row,col, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.05,right=0.95)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]).ravel()
    vars = ['flux_unc_fco2sw','flux_unc_k','flux_unc_wind','flux_unc_solsubskin_unc']
    label = ['fCO$_{2 (sw)}$','Gas Transfer', 'Wind', 'Subskin solubility']
    cmax = [0.2,0.05,0.05,0.05]
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')

    flu = np.abs(np.array(c.variables['flux'][:,:,dat]))
    for i in range(len(vars)):
        print(i)
        data = np.transpose(np.array(c.variables[vars[i]][:,:,dat])*flu)
        pc = axs[i].pcolor(lon,lat,data,cmap='Blues',vmin = 0,vmax=cmax[i])
        cbar = plt.colorbar(pc,ax=axs[i])
        cbar.set_label('Air-sea CO$_{2}$ flux uncertainty (g C m$^{-2}$ d$^{-1}$)');
        axs[i].set_title(label[i])
        axs[i].set_facecolor('gray')
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_flux_components_example.png'),format='png',dpi=300)
    fig.close()


def plot_relative_contribution(model_save_loc,model_plot=False,model_plot_label=False):
    font = {'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)
    ye = '# Year'
    gro = ['Upward air-sea CO2 flux (Pg C yr-1)','Downward air-sea CO2 flux (Pg C yr-1)']
    uncs = ['k','wind','xco2atm','seaice','fco2sw_net','fco2sw_para','fco2sw_val']
    uncs_comp= ['ph2o','schmidt','solskin_unc','solsubskin_unc']
    uncs_fco2 = ['fco2sw_val','fco2sw_para','fco2sw_net']
    label=['Gas Transfer','Wind','Sea Ice','Schmidt','Solubility skin','Solubility subskin','fCO$_{2 (atm)}$','fCO$_{2 (sw)}$']
    label2 = ['pH$_{2}$O','xCO$_{2 (atm)}$']

    data = pd.read_table(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',')

    """
    Start by getting all the data out of the annual_flux.csv file. These could be run in any order so we put into a consitent format
    We then combine the fixed and non-fixed components of uncs_comp to produce a single estimate.
    """
    combined = np.zeros((len(data[ye]),len(uncs_comp)+len(uncs)+1))
    for i in range(len(uncs)):
        try:
            combined[:,i] = data['flux_unc_'+uncs[i]+' (Pg C yr-1)']
        except:
            combined[:,i] = 0
    for i in range(len(uncs_comp)):
        combined[:,i+len(uncs)] = np.sqrt(data['flux_unc_'+uncs_comp[i]+' (Pg C yr-1)']**2 + data['flux_unc_'+uncs_comp[i]+'_fixed (Pg C yr-1)']**2)
    combine_head = uncs+uncs_comp
    #print(combined)
    #print(combine_head)

    # #data = data[:,7:-3]
    data_atm = combined[:,[2,7]]
    data_fco2 = combined[:,[4,5,6]]
    combined = combined[:,[0,1,3,8,9,10]]
    print(combined.shape)
    atm = np.sqrt(np.sum(data_atm**2,axis=1))
    atm = atm[:,np.newaxis]
    combined = np.append(combined,atm,axis=1)
    sw = np.sqrt(np.sum(data_fco2**2,axis=1))
    sw = sw[:,np.newaxis]
    combined = np.append(combined,sw,axis=1)
    totals = []
    gross = []
    for i in range(len(data[ye])):
        totals.append(np.sum(combined[i,:]))
    print(totals)
    gross = np.array(data[gro])
    year = data[ye]
    #label = np.array(combine_head)[[0,1,3,4,6,7,8]]; label = np.append(label,'fCO2_atm')#
    #print(label)
    # data = data[:,[7,8,9,10,13,14,15]]
    #
    # print(data.shape)
    fig = plt.figure(figsize=(15,18))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.07,top=0.95,left=0.075,right=0.95)
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    #ax2 = ax.twinx()
    # cols = ['#6929c4','#1192e8','#005d5d','#e78ac3','#fa4d56','#570408','#e5c494','#198038','#002d9c']
    # cols = ["#fd7f6f", "#7eb0d5",  "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7","#b2e061"]
    # cols = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#44AA99',  '#882255','#999933', '#AA4499']
    cols = ['#332288','#44AA99','#882255','#DDCC77', '#117733', '#88CCEE','#999933','#CC6677']
    for i in range(combined.shape[0]):
        bottom = 0
        for j in range(combined.shape[1]):
            if i == 1:
                p = ax[0].bar(year[i],(combined[i,j]/totals[i])*100,bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax[0].bar(year[i],(combined[i,j]/totals[i])*100,bottom=bottom,color=cols[j])
            bottom = bottom + (combined[i,j]/totals[i])*100
    # ax2.plot(year,-np.sum(np.abs(gross),axis=1),'k-',label = 'Gross Flux',linewidth=3)
    # ax2.plot(year,np.sum(gross,axis=1),'k--', label = 'Net Flux',linewidth=3)
    # ax2.legend(bbox_to_anchor=(1.6, 0.9))
    #

    ax[0].set_ylabel('Relative contribution to uncertainty (%)')
    ax[0].set_xlabel('Year')
    ax[0].set_ylim([0,100])
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1],bbox_to_anchor=(0.49, 0.75))
    #ax2.set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    #
    label = ['fCO$_{2 (sw)}$ Network','fCO$_{2 (sw)}$ Parameter','fCO$_{2 (sw)}$ Evaluation']
    totals = []
    for i in range(data_fco2.shape[0]):
        totals.append(np.sum(data_fco2[i,:]))
    for i in range(data_fco2.shape[0]):
        bottom = 0
        for j in range(data_fco2.shape[1]):
            if i == 1:
                p = ax[1].bar(year[i],(data_fco2[i,j]/totals[i])*100,bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax[1].bar(year[i],(data_fco2[i,j]/totals[i])*100,bottom=bottom,color=cols[j])
            bottom = bottom + (data_fco2[i,j]/totals[i])*100
    #bbox_to_anchor=(1.14, 0.8))
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[::-1], labels[::-1],loc=2)
    ax[1].set_ylabel('Relative contribution to uncertainty (%)')
    ax[1].set_xlabel('Year')
    ax[1].set_ylim([0,100])
    #ax.set_title('fCO$_{2 (sw)}$ total uncertainty contributions')
    #
    label = ['xCO$_{2 (atm)}$','pH$_{2}$O']
    totals = []
    for i in range(data_atm.shape[0]):
        totals.append(np.sum(data_atm[i,:]))
    for i in range(data_atm.shape[0]):
        bottom = 0
        for j in range(data_atm.shape[1]):
            if i == 1:
                p = ax[2].bar(year[i],(data_atm[i,j]/totals[i])*100,bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax[2].bar(year[i],(data_atm[i,j]/totals[i])*100,bottom=bottom,color=cols[j])
            bottom = bottom + (data_atm[i,j]/totals[i])*100
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles[::-1], labels[::-1],loc=2)
    ax[2].set_ylabel('Relative contribution to uncertainty (%)')
    ax[2].set_xlabel('Year')
    ax[2].set_ylim([0,100])
    #ax.set_title('fCO$_{2 (atm)}$ uncertainty contributions')

    ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
    #sta = np.loadtxt(os.path.join(model_save_loc,'unc_monte.csv'),delimiter=',')
    st = np.sqrt(np.sum(ann[:,6:]**2,axis=1))
    a = ann[:,0]
    ax[3].plot(a,ann[:,1],'k-',linewidth=3,zorder=6,label='Net Flux')

    #ax.plot(a,out,zorder=2)
    ax[3].fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5)
    ax[3].fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
    ax[3].plot(year,-np.sum(np.abs(gross),axis=1),'b--',label = 'Absolute Flux',linewidth=3)

    if model_plot:
        a = np.loadtxt(model_plot,delimiter=',',skiprows=1)
        ax[3].plot(a[:,0],a[:,1],'b-',label = model_plot_label,linewidth=3,zorder=6)

    ax[3].set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    ax[3].set_xlabel('Year')
    ax[3].legend(loc = 3)
    for i in range(len(ax)):
        #worldmap.plot(color="lightgrey", ax=ax[i])
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig(os.path.join(model_save_loc,'plots','relative_uncertainty_contribution.png'),format='png',dpi=300)
    fig.close()
    # fig = plt.figure(figsize=(10,10))
    # gs = GridSpec(1,2, figure=fig, wspace=0.9,hspace=0.2,bottom=0.07,top=0.95,left=0.075,right=0.9)
    # ax = fig.add_subplot(gs[0,0])
    # ax2.plot(year,np.sum(gross,axis=1),'k-',linewidth=3)

def plot_absolute_contribution(model_save_loc,model_plot=False,model_plot_label=False):
    font = {'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)
    ye = '# Year'
    gro = ['Upward air-sea CO2 flux (Pg C yr-1)','Downward air-sea CO2 flux (Pg C yr-1)']
    uncs = ['k','wind','xco2atm','seaice','fco2sw_net','fco2sw_para','fco2sw_val']
    uncs_comp= ['ph2o','schmidt','solskin_unc','solsubskin_unc']
    uncs_fco2 = ['fco2sw_val','fco2sw_para','fco2sw_net']
    label=['Gas Transfer','Wind','Sea Ice','Schmidt','Solubility skin','Solubility subskin','fCO$_{2 (atm)}$','fCO$_{2 (sw)}$']
    label2 = ['pH$_{2}$O','xCO$_{2 (atm)}$']

    data = pd.read_table(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',')

    """
    Start by getting all the data out of the annual_flux.csv file. These could be run in any order so we put into a consitent format
    We then combine the fixed and non-fixed components of uncs_comp to produce a single estimate.
    """
    combined = np.zeros((len(data[ye]),len(uncs_comp)+len(uncs)+1))
    for i in range(len(uncs)):
        try:
            combined[:,i] = data['flux_unc_'+uncs[i]+' (Pg C yr-1)']
        except:
            combined[:,i] = 0
    for i in range(len(uncs_comp)):
        combined[:,i+len(uncs)] = np.sqrt(data['flux_unc_'+uncs_comp[i]+' (Pg C yr-1)']**2 + data['flux_unc_'+uncs_comp[i]+'_fixed (Pg C yr-1)']**2)
    combine_head = uncs+uncs_comp
    #print(combined)
    #print(combine_head)

    # #data = data[:,7:-3]
    data_atm = combined[:,[2,7]]
    data_fco2 = combined[:,[4,5,6]]
    combined = combined[:,[0,1,3,8,9,10]]
    print(combined.shape)
    atm = np.sqrt(np.sum(data_atm**2,axis=1))
    atm = atm[:,np.newaxis]
    combined = np.append(combined,atm,axis=1)
    sw = np.sqrt(np.sum(data_fco2**2,axis=1))
    sw = sw[:,np.newaxis]
    combined = np.append(combined,sw,axis=1)
    totals = []
    gross = []
    for i in range(len(data[ye])):
        totals.append(np.sum(combined[i,:]))
    print(totals)
    gross = np.array(data[gro])
    year = data[ye]
    #label = np.array(combine_head)[[0,1,3,4,6,7,8]]; label = np.append(label,'fCO2_atm')#
    #print(label)
    # data = data[:,[7,8,9,10,13,14,15]]
    #
    # print(data.shape)
    fig = plt.figure(figsize=(15,18))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.07,top=0.95,left=0.075,right=0.95)
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    #ax2 = ax.twinx()
    # cols = ['#6929c4','#1192e8','#005d5d','#e78ac3','#fa4d56','#570408','#e5c494','#198038','#002d9c']
    # cols = ["#fd7f6f", "#7eb0d5",  "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7","#b2e061"]
    # cols = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#44AA99',  '#882255','#999933', '#AA4499']
    cols = ['#332288','#44AA99','#882255','#DDCC77', '#117733', '#88CCEE','#999933','#CC6677']
    line_style = ['-','--','-.','-','--','-.','-','--']
    for i in range(combined.shape[1]):
        bottom = 0
        ax[0].plot(year,combined[:,i],color=cols[i],label=label[i],linestyle=line_style[i])


    ax[0].set_ylabel('Absolute contribution to uncertainty (Pg C yr$^{-1}$)')
    ax[0].set_xlabel('Year')
    ax[0].set_ylim([0,1])
    ax[0].grid()
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1],loc=2)
    #ax2.set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    #
    label = ['fCO$_{2 (sw)}$ Network','fCO$_{2 (sw)}$ Parameter','fCO$_{2 (sw)}$ Evaluation']
    totals = []

    for i in range(data_fco2.shape[1]):
        ax[1].plot(year,data_fco2[:,i],color=cols[i],label=label[i],linestyle=line_style[i])

    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[::-1], labels[::-1],loc=2)
    ax[1].set_ylabel('Absolute contribution to uncertainty (Pg C yr$^{-1}$)')
    ax[1].set_xlabel('Year')
    ax[1].set_ylim([0,0.7])
    ax[1].grid()
    #ax.set_title('fCO$_{2 (sw)}$ total uncertainty contributions')
    #
    label = ['xCO$_{2 (atm)}$','pH$_{2}$O']
    totals = []

    for i in range(data_atm.shape[1]):
        ax[2].plot(year,data_atm[:,i],color=cols[i],label=label[i],linestyle=line_style[i])

    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles[::-1], labels[::-1],loc=2)
    ax[2].set_ylabel('Absolute contribution to uncertainty (Pg C yr$^{-1}$)')
    ax[2].set_xlabel('Year')
    ax[2].set_ylim([0,0.01])
    ax[2].grid()
    #ax.set_title('fCO$_{2 (atm)}$ uncertainty contributions')

    ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
    #sta = np.loadtxt(os.path.join(model_save_loc,'unc_monte.csv'),delimiter=',')
    st = np.sqrt(np.sum(ann[:,6:]**2,axis=1))
    a = ann[:,0]
    ax[3].plot(a,ann[:,1],'k-',linewidth=3,zorder=6,label='Net Flux')

    #ax.plot(a,out,zorder=2)
    ax[3].fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5)
    ax[3].fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
    ax[3].plot(year,-np.sum(np.abs(gross),axis=1),'b--',label = 'Absolute Flux',linewidth=3)

    if model_plot:
        a = np.loadtxt(model_plot,delimiter=',',skiprows=1)
        ax[3].plot(a[:,0],a[:,1],'b-',label = model_plot_label,linewidth=3,zorder=6)

    ax[3].set_ylabel('Air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    ax[3].set_xlabel('Year')
    ax[3].legend(loc = 3)
    for i in range(len(ax)):
        #worldmap.plot(color="lightgrey", ax=ax[i])
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig(os.path.join(model_save_loc,'plots','absolute_uncertainty_contribution.jpg'),format='jpg',dpi=300)
    fig.close()
    # fig = plt.figure(figsize=(10,10))
    # gs = GridSpec(1,2, figure=fig, wspace=0.9,hspace=0.2,bottom=0.07,top=0.95,left=0.075,right=0.9)
    # ax = fig.add_subplot(gs[0,0])
    # ax2.plot(year,np.sum(gross,axis=1),'k-',linewidth=3)

def montecarlo_flux_testing(model_save_loc,start_yr = 1985,end_yr = 2022,decor=[2000,200],flux_var = '',flux_variable='flux',seaice = False,seaice_var='',
    inp_file=False,single_output=False,ens=100,bath_cutoff=False,output_file = 'annual_flux.csv'):
    """
    Code to evaluate the effect of uncertainties that decorrelate over a specified length scale.
    The pre-calculated flux uncertainties are loaded from the framework output, and then a random grid
    of random numbers is constructed over the region defined within the decorrelation length (i.e random
    tie points spread ~evenly over the globe so that each point is 2xdecorrelation length apart). This
    tie point grid is then interpolated over the domain, so that a systematic regional random number is
    assign. Flux uncertainites are multiplied by the random number, and added to the orginial flux. Globally
    intergrated values are then calculated annually for an ensemble of this approach (200 ensembles). The
    standard deviation is extract for the annual CO2 flux timeseries from the 200 ensembles giving the
    integrated flux uncertainty whilst accounting for decorrelation lengths.

    For sea ice, we treat the approach differently. The uncertainties for most sea ice cases are asymmetric
    and cannot be pre-computed. Therefore we take the flux output, and orginial sea ice concentration
    for the flux calculation, and calcualte the flux without sea ice. Then the flux is recomputed based on the sea
    ice concentration from the uncertainty montecarlo. Slightly different approach.
    """
    import scipy.interpolate as interp
    import random
    fluxloc = os.path.join(model_save_loc,'flux')
    # We have our default location for the output file (i.e where all the output data from the Nerual network and flux uncertainty computation per pixel)
    # but we can specify here if this file is different
    if inp_file:
        c = Dataset(inp_file,'r')
    else:
        c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')

    # If were not dealing with the seaice component we can load the flux (with the sea ice accounted for) from
    # the flux variable (this is normally 'flux')
    # The uncertainties are percentages of the flux, so we convert to g C m-2 d-1 by multipling the uncertainty
    # by the absolute flux.
    if not seaice:
        c_flux = np.array(c.variables[flux_variable])
        print(c_flux.shape)
        c_flux_unc = np.array(c.variables[flux_var]) * np.abs(c_flux)

    #Laoding the latitude/longitude/time grids from the input data file.
    lon = np.array(c.variables['longitude'])
    lat = np.array(c.variables['latitude'])
    time = np.array(c.variables['time'])
    # fig = plt.figure(figsize=(14,7))
    # gs = GridSpec(1,1, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
    # ax = fig.add_subplot(gs[0,0])
    # ax.pcolor(lon,lat,np.transpose(c_flux[:,:,0]))
    # plt.show()
    c.close()
    # Here we convert the times from 'days since' to a datetime array, and then extract the year (as this is the bit we need
    # to paritition the fluxes into different years)
    time_2 = np.zeros((len(time)))
    for i in range(0,len(time)):
        time_2[i] = ((datetime.datetime(1970,1,15)+datetime.timedelta(days=int(time[i]))).year)

    # If were dealing with the sea ice componenent we load the flux direct from the Fluxengine files (as these don't have the sea ice accounted for
    # with the flux * (1-ice)). We then load the sea ice used in the Fluxengine calculations, and then load the sea ice uncertainty from the
    # input data array
    if seaice:
        c_flux = load_flux_var(fluxloc,'OF',start_yr,end_yr,len(lon),len(lat),len(time))
        c_flux = np.transpose(c_flux,(1,0,2))
        ice = load_flux_var(fluxloc,'P1',start_yr,end_yr,len(lon),len(lat),len(time))
        ice = np.transpose(ice,(1,0,2))
        c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'r')
        ice_unc = np.array(c.variables[seaice_var][:])
        c.close()
    # Find the resolution of the data we are working with
    res = np.abs(lon[0]-lon[1])
    # Calcualte the area of each pixel and convert from km-2 to m-2 (this is used in the intergrated flux calculations)
    area = du.area_grid(lat = lat,lon = lon,res=res) * 1e6
    # Transpose to get onto the array shape as the other data (not sure why the du.area_grid function doesn't output the right way to start with)
    area = np.transpose(area)
    print(area.shape)

    # Here we load the ocean proportion data from the GEBCO bathymetry file for use in the intergrated flux calculations
    # We also load the elevation (i.e depth) if the bath_cutoff is specified (so we can trim the data to only the water depth required)
    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    land = np.squeeze(np.array(c.variables['ocean_proportion']))
    if bath_cutoff:
        elev=  np.squeeze(np.array(c.variables['elevation']))
    print(land.shape)
    c.close()

    # Here we cycle through the time dimension (third dimension) and set the areas where the water is deeper than the bathymetry cutoff to nan
    # then save back to the flux array.
    # Only applied if bath_cutoff is specified
    if bath_cutoff:
        for i in range(c_flux.shape[2]):
            flu = c_flux[:,:,i]  ; flu[elev<=bath_cutoff] = np.nan; c_flux[:,:,i] = flu
    #fco2_tot_unc =  fco2_tot_unc[:,:,:,np.newaxis]
    # Get a list of the years we are computing the fluxes for
    a = list(range(start_yr,end_yr+1))

    """
    This all seems like legacy code (leaving here for now....)
    """
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

    # data_totals = decors[:,1]

    #Here we load the decorrelation lengths from the semi-variogram analysis. Where within decor_loaded the first column is year, second
    # is the median decorelation lenght, thrid is the IQR, fourth is the mean, fifth is the standard deviation
    decors = np.zeros((len(a),2))
    #If decor is a string (we assume its a file name, or an absolute path to a file)
    if isinstance(decor, str):
        print('Loading Decorrelation')
        try:
            decor_loaded = np.loadtxt(os.path.join(model_save_loc,'decorrelation',decor),delimiter=',') # Seems ive hardcoded the location (where we'd expect the file)
        except:
            print('Bad file... trying a second attempt')
            decor_loaded = np.loadtxt(decor,delimiter=',') # And then if a actual path is specified the first call will fail and then load it from the absolute path supplied
        decors[:,0] = decor_loaded[:,1] # Median decorrelation length loaded

        decors[:,1] = decor_loaded[:,2] # IQR is left as is - we want 2 sigma equivalent, so I'd divide by 2 to get the IQR as a +-, then times by 2 to get 2 sigma equivalent.

        # If we have nans then the decorrelation length analysis failed for this year (likely due to no data) so we set the decorrelation length to the maximum of all
        # avaiable years
        f = np.where(np.isnan(decors[:,0]) == 1)[0]
        if len(f) > 0:
            print('NaN values present!')
            decors[f,0] = np.nanmax(decors[:,0])
            decors[f,1] = np.nanmax(decors[:,1])
        print(decors)
    else: # We assume its a number (decorrealtion and a uncertainty) which is fixed for all years/
        decors[:,0] = decors[:,0] + decor[0]
        decors[:,1] = decor[1]
        print(decors)

    #ens = 100
    """
    Into the monte carlo propagation of the uncertainties now...
    """
    out = np.zeros((len(a),ens)) # Setup an array to save the final fluxes (dimensions are (number of years, number of ensembles))
    out2 = np.zeros((len(a))) # Setup array to save the ensemble output before saveing to the above final output array
    #pad = 40 # 8 = 400km, 13 = 650km, 14 = 700 km, 28 = 1400km, 40 = 2000km
    #Printing the latitude/longitude bounds we are dealing with? I think
    print(f'Lat 1: {lat[0]} Lat2: {lat[-1]}')
    print(f'Lon 1: {lon[0]} Lon2: {lon[-1]}')

    lon_ns,lat_ns = np.meshgrid(lon,lat) # Create a 2d array of latitude and longitude values
    for j in range(0,ens): # Start montecarlo ensembles
        print(j)
        t = 0
        t_c = 1
        unc = np.zeros((c_flux.shape)) # Setup the uncertianty perturbation grid that has same dimensions as the flux (lon, lat, time)
        for i in range(c_flux.shape[2]):# Cycle through the time dimesnsion
            if t_c ==1: # This works out the decorrelation length for this run (so a random value within the uncertainties of the decorrlation in km)
                pad = -1 # Set pad to -1 (so if this is less than 1 degree run again)
                while (pad < 1):# | (pad>70):
                    ran = np.random.normal(0,0.5,1) # Select a single random value that will be between -1, 1.
                    de_len = (decors[t,0] + (decors[t,1]*ran)) # Calculate the randome decorrelation for this run (so median + perturbation)
                    pad = (de_len / 110.574)*2 # We need the tie point in the next step to be 2 times the decorrelation length apart, so we convert out decorrelation lenght to km (where latitude km -> deg is fixedish)
                lat_s = np.linspace(lat[0],lat[-1],int((lat[-1]-lat[0])/pad)) # Setup the tie point grid for the latitudes in degrees (so this gives out latitude tie points at spacings 2 times decorrelation length)
                lat_ss = []
                lon_ss = []
                for l in range(len(lat_s)): # We now need to calculate the longitudes for our tie points on each latitude band in lat_s (i.e how many longitude ties do we need)
                    lat_km = 111.320*np.cos(np.deg2rad(lat_s[l])) # Longirude convesions from km -> degrees varies with laittude so for our latitude we calculate the distance that each degree is in km
                    padl = (de_len/lat_km)*2# We then use this value to work out the number of degrees our decorrelation length is for at this particular latitude

                    in_padl = int((lon[-1] - lon[0])/padl) # Then we work out how many our degree decorrealtion length fits into the longitude grid
                    #print(in_padl)
                    if in_padl == 0: # If it turns out to be 0, then we have a single tie point on the grid that we set in the middle of the latitude/longitude grid
                        lon_s = [(lon[-1] - lon[0])/2]
                    else:
                        lon_s = np.linspace(lon[0],lon[-1],in_padl) # If we can fit more then these are linearly spaces along the longitude grid at that latitude
                    for p in range(len(lon_s)): # Now we cycle through the lons and latitudes to build our tie point grid so we have the coordinates for each point (latitude in lat_ss and lon in lon_ss)
                        lat_ss.append(lat_s[l])
                        lon_ss.append(lon_s[p])
                # To do the inteprolation of the uncertainty grid we need to make sure the whole grid is covered. The poles or edges of the grid become an issue so we add a pertirbation value for the 4 corners of our grid.
                lat_ss.append(lat[0]); lat_ss.append(lat[0]);lat_ss.append(lat[-1]);lat_ss.append(lat[-1]);
                lon_ss.append(lon[0]); lon_ss.append(lon[-1]); lon_ss.append(lon[0]); lon_ss.append(lon[-1]);

            unc_o = np.random.normal(0,0.5,(len(lon_ss))) # Here we select a random peturbation value for each of our tie points (lat_ss, lon_ss)
            #So here we set the 4 corners of the grid to the same perturbation value (im not quite sure why, but likely to do with having a single perturbation region at the poles.)
            v = np.random.normal(0,0.5)
            unc_o[-3:] = v

            #So we stack our tie points together into a list (tie_point_number by 2 columns)
            points = np.stack([np.array(lon_ss).ravel(),np.array(lat_ss).ravel()],-1)
            u = unc_o.ravel() # And put our perturbation values into a column (I think they are probably already in a column but just to be sure)
            un_int = interp.griddata(points,u,(np.stack([lon_ns.ravel(),lat_ns.ravel()],-1)),method = 'cubic') # Now we interpolate the values onto our full latitude and longitude grid (using cubic, as fully linear lead to harsh boundaries, where as cubic is smoother)
            un_int = np.transpose(un_int.reshape((len(lat),len(lon)))) # These come out as a column, so we reshape to the grid, and then transpose as it seems I got the dimensions all wrong, but this outputs them to the right orientation
            unc[:,:,i] = un_int # Then put these values into the final uncertainty perturbation grid at the time step
            t_c = t_c + 1 # Add one to our timestep counter so we don't recalculate tje decorrelation lengths and tie point again for this year (the tie points are fixed locations for each year but the perturbation value can now change)
            if t_c == 13:# When we get to 13 we are starting a new year, so we recalcuate the tie points for a new perturbation of the decorrelation length
                t=t+1
                t_c = 1
            # Here we plot the perturbation grid (first grid in time for checking/debugging)
            if (j ==0) & (i == 0):
                fig = plt.figure()
                plt.pcolor(lon,lat,np.transpose(np.squeeze(unc[:,:,i])),vmin=-1,vmax=1)
                plt.colorbar()
                fig.savefig(os.path.join(model_save_loc,'plots','example_pco2_perturbation.png'),format='png',dpi=300)
                #plt.show()
        # SO we now have our perturbation grid through time we can now apply it to the fluxes

        #If were dealing with seaice, the perturbation vlaues affect the ice and ice_unc.
        if seaice:
            ice_t = ice+(ice_unc*unc) # Calcualte our ice with the perturbation
            ice_t[ice_t > 1] = 1; ice_t[ice_t <0] = 0 # Can't have ice less than 0 or greater than 1 (or 100% ice cover)
            e_flux = c_flux * (1-ice_t) # Apply to flux
        else:# Were dealing with a precomputed flux uncertainty, so we multiple by the perturbation vlaue and add to the flux
            e_flux = c_flux + (unc*c_flux_unc)
        # Setup flux variables for computing the annual values
        flux = np.zeros((c_flux.shape))
        c_flux2 = np.zeros((c_flux.shape))
        # Cycle through each time step, and mutliple the flux (g C m-2 d-1) by the area (m-2), the ocean proportion (0-1), the mean number of days in a month (d-1) and convert to Pg C (1e15)
        # So units our Pg C mon-1
        for i in range(0,flux.shape[2]):
            flux[:,:,i] = (e_flux[:,:,i] * area * land * 30.5) /1e15 #This is our perturbed values
            c_flux2[:,:,i] = (c_flux[:,:,i] * area * land * 30.5) /1e15 # This is our unperturbed (raw flux)

        t = 0
        for i in range(0,flux.shape[2],12):# Cycle through the flux array, in annual increments and sum the flux thats in Pg C mon-1 into a Pg C yr-1
            flu = flux[:,:,i:i+12] #Extract the years values
            print(np.nansum(flu))
            c_flu = c_flux2[:,:,i:i+12] # Extract the unperturbed values - I think this only applies when we are doing a single output... so we have the actual calulated flux and the uncertainty
            out[t,j] = np.nansum(flu)
            out2[t] = np.nansum(c_flu)

            t = t+1# Keep a count so we can put the annual values in the right output row
    if single_output: #If a single value output i.e in its own table (e.g SOCOMv2 geospatial analysis only does one uncertainty component)
        data = pd.DataFrame(a,columns=['Year']) # So we save the year
        st = np.std(out,axis=1) # Calcualte the standard devaiiton of the ensembles
        data['std'] = st# Save these into the table
        data['Net air-sea CO2 flux (Pg C yr-1)'] = out2 # This is our calcualted ocena carbon sink with no perturbations (so flux as is calculated)

        #Then we plot the results (i.e a ocean sink with errorbars)
        fig = plt.figure(figsize=(7,7))
        gs = GridSpec(1,1, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.15,right=0.98)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(a,out,zorder=2)
        ax.plot(a,out2,'b',zorder=6,linewidth = 3)
        ax.fill_between(a,out2 - st,out2 + st,alpha = 0.6,color='k',zorder=5)
        ax.fill_between(a,out2 - (2*st),out2 + (2*st),alpha=0.4,color='k',zorder=4)
        ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
        ax.set_xlabel('Year')
        # ax.plot(a,out)
        # ax.plot(a,out2,'k--',linewidth=3)
        fig.savefig(single_output+'.png',format='png',dpi=300) #Save the figure
        fig.close()
        data.to_csv(single_output+'.csv',index=False) # Save the year, flux, standard deviation of the ensemble.
    else:
        # If multiple uncertainty components will be calculated then we do this
        # So we plot some debug information
        fig = plt.figure(figsize=(14,7))
        gs = GridSpec(1,2, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
        ax = fig.add_subplot(gs[0,0]) # Plotting the used decorrelation lenghts
        ax.set_xlabel('Year')
        ax.plot(a,decors[:,0],'k-')
        ax.fill_between(a,decors[:,0] - (decors[:,1]/2), decors[:,0]+ (decors[:,1]/2),alpha=0.4,color='k')
        ax.set_ylabel('Decorrelation length (km)')
        # We load the precomputed annual fluxes from the fixed_uncertainty_append function
        ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
        ax = fig.add_subplot(gs[0,1])
        ax.plot(a,ann[:,1],'k-',linewidth=3,zorder=6)# We plot this flux
        st = np.std(out,axis=1)#Calculate the standard deviation of the ensemble
        ax.plot(a,out,zorder=2)# We plot the individual ensembles
        ax.fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5) # Then plot the unperturbed flux with errorbars
        ax.fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
        ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
        print(st)
        fig.savefig(os.path.join(model_save_loc,'plots','pco2_uncertainty_contribution_revised.png'),format='png',dpi=300) # Save the debug figure
        fig.close()
        data = pd.read_table(os.path.join(model_save_loc,output_file),delimiter=',')# Read our annual flux table with the already computed flux uncertainties
        #Then we save the uncertainity on the fluxes into the table before resaving with the appended values
        if seaice:#For seaice we manual assign the column name and values
            data['flux_unc_seaice (Pg C yr-1)'] = st
        else: # If were using a precomputed value, then we name it as such.
            data[flux_var+' (Pg C yr-1)'] = st
        data.to_csv(os.path.join(model_save_loc,output_file),index=False) # save the data back on to the file.
        #np.savetxt(os.path.join(model_save_loc,'unc_monte_revised.csv'),np.stack((np.array(a),st)),delimiter=',',fmt='%.5f')
        #plt.show()

def plot_net_flux_unc(model_save_loc):
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(1,1, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.20,right=0.98)
    ax = fig.add_subplot(gs[0,0])

    ann = np.loadtxt(os.path.join(model_save_loc,'annual_flux.csv'),delimiter=',',skiprows=1)
    #sta = np.loadtxt(os.path.join(model_save_loc,'unc_monte.csv'),delimiter=',')
    st = np.sqrt(np.sum(ann[:,6:]**2,axis=1))
    a = ann[:,0]
    ax.plot(a,ann[:,1],'k-',linewidth=3,zorder=6)

    #ax.plot(a,out,zorder=2)
    ax.fill_between(a,ann[:,1] - st,ann[:,1] + st,alpha = 0.6,color='k',zorder=5)
    ax.fill_between(a,ann[:,1] - (2*st),ann[:,1] + (2*st),alpha=0.4,color='k',zorder=4)
    ax.set_ylabel('Net air-sea CO$_{2}$ flux (Pg C yr$^{-1}$)')
    ax.set_xlabel('Year')
    fig.savefig(os.path.join(model_save_loc,'plots','net_flux_uncertainty.png'),format='png',dpi=300)
    fig.close()

def variogram_evaluation(model_save_loc,output_file = 'decorrelation',input_array = False,input_datafile = False,ens=100,hemisphere=False,
    start_yr=1985,end_yr=2022,estimator='dowd'):
    def variogram_run(a,values,coords,ens,estimator):
        for j in range(ens):
            if len(values.shape) == 0:
                print('Empty')
            else:
                if len(values) < 50:
                    ran = random.sample(range(len(values)), int(len(values)/2))
                elif len(values) < 300:
                    ran = random.sample(range(len(values)), int(len(values)/5))
                else:
                    ran = random.sample(range(len(values)), 200)
                try:
                    V = skg.Variogram(coordinates=coords[ran], values=values[ran],dist_func=getDistanceByHaversine,maxlag=10000,fit_method='lm',estimator=estimator)
                    #V.n_lags = 80
                    V.model = 'exponential'
                    V.bin_func = 'scott'
                    des = V.describe()
                    # V.plot(show=True)
                    # print(des['effective_range'])
                    # wait = input("Press Enter to continue.")

                    ra = des['effective_range']
                    print(des['effective_range'])
                    if (ra > 100) & (ra < 12000) & (np.isnan(ra) == 0):
                        a.append(ra)
                    del V
                except:
                    print('Exception')

        return a

    import skgstat as skg
    import random
    import scipy
    if not input_array:
        c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
        fco2_sw = np.array(c.variables['fco2'])
        #fco2_tot_unc[:] = 12
        lon = np.array(c.variables['longitude'])
        lat = np.array(c.variables['latitude'])
        #fco2_tot_unc[fco2_tot_unc>1000] = np.nan
        c.close()
        # Calculating avaiable in situ data to constrain neural network.
        c = Dataset(os.path.join(model_save_loc,'inputs','neural_network_input.nc'),'r')
        fco2_insitu = np.array(c.variables['CCI_SST_reanalysed_fCO2_sw'])
        time = np.array(c.variables['time'])
        c.close()
    elif type(input_array) is list:
        print(input_datafile[0])
        c = Dataset(input_datafile[0],'r')
        lon = np.array(c.variables['longitude'])
        lat = np.array(c.variables['latitude'])
        time = np.array(c.variables['time'])

        data1 = np.array(c.variables[input_array[0]][:])
        data1[data1<=0.0] = np.nan
        data1[data1>=100000000] = np.nan
        c.close()
        c = Dataset(input_datafile[1],'r')
        data2 = np.array(c.variables[input_array[1]][:])
        data2[data2<=0.0] = np.nan
        data2[data2>=100000000] = np.nan
        c.close()

    else:
        c = Dataset(input_datafile,'r')
        data = np.array(c.variables[input_array][:])
        data[data<=0.0] = np.nan
        data[data>=10000000000] = np.nan
        time = np.array(c.variables['time'])
        lon = np.array(c.variables['longitude'])
        lat = np.array(c.variables['latitude'])
        c.close()
    time_2 = np.zeros((len(time)))
    for i in range(0,len(time)):
        time_2[i] = ((datetime.datetime(1970,1,15)+datetime.timedelta(days=int(time[i]))).year)

    lon_g,lat_g = np.meshgrid(lon,lat); lon_g = np.transpose(lon_g); lat_g = np.transpose(lat_g);
    a = []
    out = np.zeros((end_yr-start_yr+1,5))
    out[:] = np.nan
    yr =start_yr
    t = 0
    hei = int(np.ceil((end_yr-start_yr+1)/5))

    fig = plt.figure(figsize=(35,7*hei))
    gs = GridSpec(hei,5, figure=fig, wspace=0.5,hspace=0.2,bottom=0.1,top=0.95,left=0.10,right=0.98)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(hei)]).ravel()
    print(axs)
    for i in range(len(time)):
        print(f'Current step = {i} out of {len(time)}')
        if yr != time_2[i]:
            out[t,0] = yr
            axs[t].hist(a,50)
            axs[t].set_title(out[t,0]); axs[t].set_xlabel('Decorrelation (km)'); axs[t].set_ylabel('Frequency')
            axs[t].set_xlim([0,8000])
            yr = time_2[i]
            out[t,1] = np.median(a); out[t,2] = scipy.stats.iqr(a); out[t,3] = np.mean(a); out[t,4] = np.std(a)

            a = []
            t = t+1
        if not input_array:
            f = np.argwhere(((np.isnan(fco2_insitu[:,:,i].ravel()) == 0) & (np.isnan(fco2_sw[:,:,i].ravel()) == 0)))
            values = fco2_insitu[:,:,i].ravel() - fco2_sw[:,:,i].ravel()
        elif type(input_array) is list:
            f = np.argwhere(((np.isnan(data1[:,:,i].ravel()) == 0) & (np.isnan(data2[:,:,i].ravel()) == 0)))
            values = data1[:,:,i].ravel() - data2[:,:,i].ravel()
        else:
            f = np.argwhere((np.isnan(data[:,:,i].ravel()) == 0))
            values = data[:,:,i].ravel()
            print(values)
        coords = np.transpose(np.squeeze(np.stack((lon_g.ravel()[f],lat_g.ravel()[f]))))
        values = np.squeeze(values[f])
        print(values.shape)
        #print(values)
        if hemisphere:
            #print(coords)
            f = np.where(coords[:,1] <= 0)[0]
            coords2 = coords[f,:]
            #print(coords2)
            values2 = values[f]
            a = variogram_run(a,values2,coords2,int(ens/2),estimator)

            f = np.where(coords[:,1] > 0)[0]
            coords2 = coords[f,:]
            values2 = values[f]
            a = variogram_run(a,values2,coords2,int(ens/2),estimator)
        else:
            a = variogram_run(a,values,coords,ens,estimator)


    out[t,0] = yr
    axs[t].hist(a,50)
    axs[t].set_title(out[t,0]); axs[t].set_xlabel('Decorrelation (km)'); axs[t].set_ylabel('Frequency')
    axs[t].set_xlim([0,12000])
    yr = time_2[i]
    out[t,1] = np.median(a); out[t,2] = scipy.stats.iqr(a); out[t,3] = np.mean(a); out[t,4] = np.std(a)

    fig.savefig(os.path.join(model_save_loc,'plots',output_file+'.png'),format='png',dpi=300)
    fig.close()
    vals = scipy.stats.iqr(a)
    print(f'Median: {np.median(a)}')
    print(f'IQR: {vals}')
    np.savetxt(os.path.join(model_save_loc,'decorrelation',output_file+'.csv'),out,delimiter=',',fmt='%.5f')

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
    fig.close()

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
    fig.close()

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
    fig.close()

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
