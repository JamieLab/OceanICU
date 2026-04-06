#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 06/2023

Script to package neural network output into the global carbon budget format

"""
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import datetime
import sys
import os
import glob
import Data_Loading.data_utils as du
import fluxengine_driver as fl

def lon_switch(var,axis=1):
    temp = np.roll(var,int(var.shape[axis]/2),axis=axis)
    return temp



def PlantaryHealthCheckPackage(model_save_loc,fluxloc,output_file,start_yr,end_yr,lon,lat,version = 'v0-1',copts={}):

    # Loading the fco2_sw_subskin output from the neural network
    # Secondary function to get time variable length
    c = Dataset(model_save_loc+'/output.nc','r')
    time = np.array(c.variables['time'])
    units = c.variables['time'].units
    c.close()


    outs = Dataset(output_file,'w')
    outs.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outs.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    outs.created_from = 'Data created from ' + model_save_loc
    outs.packed_with = 'Data packed with PlanetaryPackage.PlantaryHealthCheckPackage'

    print('Check the netCDF global variables are correct!')
    outs.flux_calculations = 'Flux calculations completed with FluxEngine v4.1. See details in Shutler et al. (2016; https://doi.org/10.1175/JTECH-D-14-00204.1) and Holding et al. (2019; https://doi.org/10.5194/os-15-1707-2019)'

    outs.sst_citation = 'Embury, O., Merchant, C.J., Good, S.A. et al. Satellite-based time-series of sea-surface temperature since 1980 for climate applications. Sci Data 11, 326 (2024). https://doi.org/10.1038/s41597-024-03147-w'
    outs.sst_data_location = 'https://dx.doi.org/10.5285/4a9654136a7148e39b7feb56f8bb02d2'
    outs.sst_bias_corrected = 'CCI-SST data has been bias corrected to surface drifters with a consistent 0.04K increase (as identified by Embury et al. 2024; which follows the recommendations in Dong et al. 2022)'

    outs.sss_citation = 'Jean-Michel, L., Eric, G., Romain, B.-B., Gilles, G., Angélique, M., Marie, D., et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. Frontiers in Earth Science 9, 585. doi:10.3389/feart.2021.698876. '
    outs.sss2_ciation = 'Boutin, J.; Vergely, J.-L.; Reul, N.; Catany, R.; Jouanno, J.; Martin, A.; Rouffi, F.; Bertino, L.; Bonjean, F.; Corato, G.; Gévaudan, M.; Guimbard, S.; Khvorostyanov, D.; Kolodziejczyk, N.; Matthews, M.; Olivier, L.; Raj, R.; Rémy, E.; Reverdin, G.; Supply, A.; Thouvenin-Masson, C.; Vialard, J.; Sabia, R.; Mecklenburg, S. (2025): ESA Sea Surface Salinity Climate Change Initiative (Sea_Surface_Salinity_cci): Monthly sea surface salinity product on a 0.25 degree global grid, v5.5, for 2010 to 2023. NERC EDS Centre for Environmental Data Analysis, date of citation. https://catalogue.ceda.ac.uk/uuid/3339dec1fbd94599802aba7f1c665679'
    outs.sss_data_location = 'https://doi.org/10.48670/moi-00021'
    outs.sss2_data_location = 'https://catalogue.ceda.ac.uk/uuid/3339dec1fbd94599802aba7f1c665679'


    outs.wind_citation = 'Mears, C.; Lee, T.; Ricciardulli, L.; Wang, X.; Wentz, F. Improving the Accuracy of the Cross-Calibrated Multi-Platform (CCMP) Ocean Vector Winds. Remote Sens. 2022, 14, 4230. https://doi.org/10.3390/rs14174230'
    outs.wind_data_location = 'https://data.remss.com/ccmp/v03.1/'

    outs.mld_citation = 'Jean-Michel, L., Eric, G., Romain, B.-B., Gilles, G., Angélique, M., Marie, D., et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. Frontiers in Earth Science 9, 585. doi:10.3389/feart.2021.698876. '
    outs.mld_data_location = 'https://doi.org/10.48670/moi-00021'

    outs.chl_citation = 'Ford, D. J., Kulk, G., Sathyendranath, S., & Shutler, J. D. (2025). Decadal and spatially complete global surface chlorophyll-a data record from satellite and BGC-Argo observations. https://doi.org/10.5194/essd-2025-389'
    outs.chl_data_location = 'https://doi.org/10.5281/zenodo.15689006'

    outs.version = version

    outs.interpolation_method = 'SOM-FNN uses CCI-SST, CMEMS SSS, CMEMS MLD, OC-CCI chla and Takahashi Climatology - FNN uses CCI-SST, CMEMS SSS, CMEMS MLD, ERSL xCO2atm, OC-CCI chla and anomalies of all 5 products (8 inputs total)'
    outs.interpolation_method_ref = 'Landschützer, P., Gruber, N., E. Bakker, D. C., and Schuster, U. (2014), Recent variability of the global ocean carbon sink, Global Biogeochem. Cycles, 28, 927– 949, doi:10.1002/2014GB004853.'
    outs.method_citation = 'Watson, A.J., Schuster, U., Shutler, J.D. et al. Revised estimates of ocean-atmosphere CO2 flux are consistent with ocean carbon inventory. Nat Commun 11, 4422 (2020). https://doi.org/10.1038/s41467-020-18203-3'
    outs.method_citation_updates = 'Ford, D. J., Blannin, J., Watts, J., Watson, A. J., Landschützer, P., Jersild, A., & Shutler, J. D. (2024). A comprehensive analysis of air-sea CO2 flux uncertainties constructed from surface ocean data products. Global Biogeochemical Cycles, 38, e2024GB008188. https://doi.org/10.1029/2024GB008188'

    outs.socat_version = 'v2025; https://doi.org/10.25921/r7xa-bt92'
    outs.reanalysis_data = 'Ford, D. J., Shutler, J. D., Ashton, I., Sims, R. P., & Holding, T. (2025). Recalculated (depth and temperature consistent) surface ocean CO₂ atlas (SOCAT) version 2025 (v0-1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15656803'

    outs.coolskin = 'Cool Skin calculated using NOAA COARE 3.6, with ERA5 monthly inputs - to get cool skin deviation, compute tos_skin - tos'
    outs.era5_data_location = 'https://doi.org/10.24381/cds.f17050d7'
    outs.noaa_coare_36_code = 'https://github.com/NOAA-PSL/COARE-algorithm/tree/master/Python/COARE3.6'
    outs.noaa_ersl_marine_boundary_layer_citation = 'Dlugokencky, E.J., K.W. Thoning, X. Lan, and P.P. Tans (2021), NOAA Greenhouse Gas Reference from Atmospheric Carbon Dioxide Dry Air Mole Fractions from the NOAA GML Carbon Cycle Cooperative Global Air Sampling Network. Data Path: ftp://aftp.cmdl.noaa.gov/data/trace_gases/co2/flask/surface/'

    outs.createDimension('lon',lon.shape[0])
    outs.createDimension('lat',lat.shape[0])
    outs.createDimension('time',time.shape[0])


    var = outs.createVariable('lat','f4',('lat'))
    var[:] = lat
    var.Long_name = 'Latitude'
    var.units = 'Degrees North'

    var = outs.createVariable('lon','f4',('lon'))
    var[:] = lon #Convert form -180 to 180, 0 to 360
    var.Long_name = 'Longitude'
    var.units = 'Degrees East'

    var = outs.createVariable('time','f4',('time'))
    var[:] = time
    var.Long_name = 'Time'
    var.units = units

    ##################
    ##################
    ##################


    c = Dataset(model_save_loc+'/output.nc','r')
    keys = list(c.variables.keys())
    keys.remove('latitude')
    keys.remove('longitude')
    keys.remove('time')
    for i in keys:
        var = outs.createVariable(i,'f4',('lon','lat','time'),fill_value=np.nan,**copts)
        data = np.array(c.variables[i])
        data[data>200000] = np.nan
        var[:] = data
        l = c[i].__dict__;
        if '_FillValue' in list(l.keys()):
            l.pop('_FillValue')
        var.setncatts(l)
        #var._FillValue = np.nan
    c.close()

    ou = fl.load_flux_var(fluxloc,'OAPC1',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    print(ou.shape)
    var = outs.createVariable('fco2atm','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2))
    var.Long_name = 'Atmospheric fCO2'
    var.units = 'uatm'
    var.description = 'Atmospheric fCO2 evaluated at the skin temperature; skin_temp'


    ou = fl.load_flux_var(fluxloc,'OK3',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    var = outs.createVariable('kw','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2))
    var.Long_name = 'Gas Transfer Velocity'
    var.Units = 'cm hr^-1'
    var.description = 'Gas transfer velocity estimated from CCMPv3.1 wind speed (filled with ERA5) and Nightingale et al. 2000 gas transfer parameterisation'

    ou = fl.load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    var = outs.createVariable('alpha','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2))/1000
    var.long_name = 'Solubilty of CO2 in seawater'
    var.units = 'mol m^-3 uatm^-1'
    var.description = 'Solubility of CO2 in seawater evaluated at the subskin; subskin_temp + subskin_salinity'


    ou = fl.load_flux_var(fluxloc,'skin_solubility',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    var = outs.createVariable('alpha_skin','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2))/1000 # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
    var.long_name = 'Skin solubilty of CO2 in seawater'
    var.units = 'mol m^-3 uatm^-1'
    var.description = 'Solubility of CO2 in seawater evaluated at the skin temperature (skin_temp) and a salty skin (+0.1 psu to subskin salinity)'

    ou = fl.load_flux_var(fluxloc,'WS1_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    var = outs.createVariable('wind_speed','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2))# 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
    var.long_name = 'Wind speed'
    var.units = 'm s^-1'
    var.description = 'Wind speed from the Cross Calibrated Wind Speed (v3.1) filled with ERA5'


    ou = fl.load_flux_var(fluxloc,'windu10_moment2',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0])
    ou[ou == -999] = np.nan
    var = outs.createVariable('wind_speed_second_moment','f4',('lon','lat','time'),fill_value=np.nan,**copts)
    var[:] = np.transpose(ou,(1,0,2)) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
    var.long_name = 'Wind speed second moment'
    var.units = '(m s^-1)^2'
    var.description = 'Second moment of wind speed from the Cross Calibrated Wind Speed (v3.1) filled with ERA5'


    #############
    #############
    #############

    c = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'))
    land = np.array(c['ocean_proportion'])
    c.close()
    var = outs.createVariable('mask_sfc','f4',('lon','lat'),fill_value=np.nan,**copts)
    land_mask = land
    var[:] = land_mask
    var.long_name = 'Fractional coverage of ocean in each grid tile'
    var.units = ''
    var.description = 'This is the ocean proportion mask calculated from ESA-Land CCI data.'

    area = du.area_grid(lat=lat,lon=lon,res=1) * 1e6
    var = outs.createVariable('area','f4',('lon','lat'),fill_value=np.nan,**copts)
    var[:] = np.transpose(area)
    var.long_name = 'Total surface area of each grid cell'
    var.units = 'm^2'
    var.description = 'Calculated assuming the Earth is a oblate sphere with major and minor radius of 6378.137 km and 6356.7523 km respectively'
    var.comment = 'Multiply "area" and "mask_sfc" to get true area used in flux calculations.'

    # Loading annual flux and adding into array

    outs.close()
    # variable_int_dict = {
    #     'Net air-sea CO2 flux (Pg C yr-1)': ['fgco2_reg','Regionally integrated air-sea CO2 flux','Pg C yr^-1'],
    #     'Area (m-2)': ['area_reg','Regionally integrated area','m^-2'],
    #     'Ice-Free area (m-2)': ['area_reg_icefree','Regionally integrated icefree area','m^-2'],
    #     'flux_unc_k (Pg C yr-1)': ['fgco2_reg_k_unc','Regionally integrated air-sea CO2 flux gas transfer algorithm uncertainty','Pg C yr^-1'],
    #     'flux_unc_ph2o_fixed (Pg C yr-1)': ['fgco2_reg_ph20_fixed_unc','Regionally integrated air-sea CO2 flux pH2O correction uncertainty due to algorithm uncertainty','Pg C yr^-1'],
    #     'flux_unc_ph2o (Pg C yr-1)': ['fgco2_reg_ph20_unc','Regionally integrated air-sea CO2 flux pH2O correction uncertainty due to SST uncertainty','Pg C yr^-1'],
    #     'flux_unc_schmidt_fixed (Pg C yr-1)': ['fgco2_reg_schmidt_fixed_unc','Regionally integrated air-sea CO2 flux Schmidt number uncertainty due to algorithm uncertainty','Pg C yr^-1'],
    #     'flux_unc_schmidt (Pg C yr-1)': ['fgco2_reg_schmidt_unc','Regionally integrated air-sea CO2 flux Schmidt number uncertainty due to SST uncertainty','Pg C yr^-1'],
    #     'flux_unc_fco2sw_net (Pg C yr-1)': ['fgco2_reg_sfco2_net_unc','Regionally integrated air-sea CO2 flux fCO2sw network uncertainty','Pg C yr^-1'],
    #     'flux_unc_fco2sw_para (Pg C yr-1)': ['fgco2_reg_sfco2_para_unc','Regionally integrated air-sea CO2 flux fCO2sw parameter uncertainty','Pg C yr^-1'],
    #     'flux_unc_fco2sw_val (Pg C yr-1)': ['fgco2_reg_sfco2_val_unc','Regionally integrated air-sea CO2 flux fCO2sw evaluation uncertainty','Pg C yr^-1'],
    #     'flux_unc_solskin_unc_fixed (Pg C yr-1)': ['fgco2_reg_solskin_fixed_unc','Regionally integrated air-sea CO2 flux skin solubility uncertainty due to algorithm uncertaintity','Pg C yr^-1'],
    #     'flux_unc_solskin_unc (Pg C yr-1)': ['fgco2_reg_solskin_unc','Regionally integrated air-sea CO2 flux skin solubility uncertainty due to SST and SSS uncertainties','Pg C yr^-1'],
    #     'flux_unc_solsubskin_unc_fixed (Pg C yr-1)': ['fgco2_reg_solsubskin_fixed_unc','Regionally integrated air-sea CO2 flux subskin solubility uncertainty due to algorithm uncertaintity','Pg C yr^-1'],
    #     'flux_unc_solsubskin_unc (Pg C yr-1)': ['fgco2_reg_solsubskin_unc','Regionally integrated air-sea CO2 flux subskin solubility uncertainty due to SST and SSS uncertainties','Pg C yr^-1'],
    #     'flux_unc_wind (Pg C yr-1)': ['fgco2_reg_wind_unc','Regionally integrated air-sea CO2 flux gas transfer uncertainty due to wind speed uncertainty','Pg C yr^-1'],
    #     'flux_unc_xco2atm (Pg C yr-1)': ['fgco2_reg_xco2atm_unc','Regionally integrated air-sea CO2 flux gas transfer uncertainty due to xCO2atm uncertainty','Pg C yr^-1']
    # }
    # outs = Dataset(output_file,'a')
    # outs.createDimension('region',1)
    # outs.createDimension('time_annual',(end_yr-start_yr)+1)
    #
    # var = outs.createVariable('time_annual','f4',('time_annual'))
    # var[:] = range(start_yr,end_yr+1)
    # var.Long_name = 'Annual time'
    # var.units = 'Year'
    #
    # data = pd.read_table(os.path.join(model_save_loc,'annual_flux.csv'),sep=',')
    # keys = list(variable_int_dict.keys())
    #
    # for i in keys:
    #     t = variable_int_dict[i]
    #     var = outs.createVariable(t[0],'f4',('region','time_annual'),fill_value=np.nan)
    #     if i == 'Net air-sea CO2 flux (Pg C yr-1)':
    #         var[:] = -data[i]
    #     else:
    #         var[:] = data[i]
    #     var.long_name = t[1]
    #     var.units = t[2]
    #     if i not in ['Net air-sea CO2 flux (Pg C yr-1)', 'Area (m-2)', 'Ice-Free area (m-2)']:
    #         var.uncertainties = "Uncertainties considered ~67% confidence (1 sigma)"
    # data = np.array(data)
    # tot = np.sqrt(np.sum(data[:,6:]**2,axis=1))
    #
    # var = outs.createVariable('fgco2_reg_tot_unc','f4',('region','time_annual'),fill_value=np.nan)
    # var[:] = tot
    # var.long_name = 'Regionally intergrated air-sea CO2 flux total uncertainty'
    # var.units = 'Pg C yr^-1'
    # var.uncertainties = "Uncertainties considered ~67% confidence (1 sigma)"
    # print(tot)

    # outs.close()
