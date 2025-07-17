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

#Location of OceanICU neural network framework
oceanicu = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/OceanICU'
sys.path.append(os.path.join(oceanicu,'Data_Loading'))
sys.path.append(oceanicu)
import Data_Loading.data_utils as du
import fluxengine_driver as fl

def lon_switch(var,axis=1):
    temp = np.roll(var,int(var.shape[axis]/2),axis=axis)
    return temp

# def lon_switch_2d(var):
#     temp = np.roll(var,int(var.shape[axis]/2),axis=axis)
#     return temp

model_save_loc_cor = 'F:/OceanCarbon4Climate/GCB2025'
model_save_loc_uncor = 'F:/OceanCarbon4Climate/GCB2025_nonreanalysed'
fluxloc = model_save_loc_cor+'/flux'
fluxloc2 = model_save_loc_uncor+'/flux'
gcb_file = 'F:/OceanCarbon4Climate/GCB_output.nc'
lon,lat = du.reg_grid()
start_yr = 1980
end_yr = 2024


# Loading the fco2_sw_subskin output from the neural network
# Secondary function to get time variable length
c = Dataset(model_save_loc_cor+'/output.nc','r')
fco2_sw_cor = np.array(c['fco2'])
flux_cor = np.array(c['flux'])
time = np.array(c['time'])
t_units = np.array(c['time'].units)
c.close()

c=Dataset(model_save_loc_uncor+'/output.nc','r')
fco2_sw_uncor = np.array(c['fco2'])
flux_uncor = np.array(c['flux'])
c.close()


outs = Dataset(gcb_file,'w')
outs.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
outs.cited_author = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
outs.recommendations = 'fgco2_corrected'

outs.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk), Jamie D. Shutler (j.d.shutler@exeter.ac.uk) and Andrew Watson (Andrew.Watson@exeter.ac.uk)'
outs.created_from_uncorrected = 'Data created from ' + model_save_loc_uncor
outs.created_from_corrected = 'Data created from ' + model_save_loc_cor
outs.packed_with = 'Data packed with global_carbon_budget_package_dual.py'
outs.flux_calculations = 'Flux calculations completed with FluxEngine v4.0.9.1 - See details in Shutler et al. (2016; https://doi.org/10.1175/JTECH-D-14-00204.1) and Holding et al. (2019; https://doi.org/10.5194/os-15-1707-2019)'
outs.method_citation = 'Watson, A.J., Schuster, U., Shutler, J.D. et al. Revised estimates of ocean-atmosphere CO2 flux are consistent with ocean carbon inventory. Nat Commun 11, 4422 (2020). https://doi.org/10.1038/s41467-020-18203-3'
outs.method_citation_updates = 'Ford, D. J., Blannin, J., Watts, J., Watson, A. J., Landschützer, P., Jersild, A., & Shutler, J. D. (2024). A comprehensive analysis of air-sea CO2 flux uncertainties constructed from surface ocean data products. Global Biogeochemical Cycles, 38, e2024GB008188. https://doi.org/10.1029/2024GB008188'

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

outs.version = 'v25'
outs.interpolation_method = 'SOM-FNN uses CCI-SST, CCI-SSS + CMEMS SSS, CMEMS MLD and Takahashi Climatology - FNN uses CCI-SST, CMEMS SSS, CMEMS MLD, ERSL xCO2atm and anomalies of all 4 products (8 inputs total)'
outs.interpolation_method_ref = 'Landschützer, P., Gruber, N., E. Bakker, D. C., and Schuster, U. (2014), Recent variability of the global ocean carbon sink, Global Biogeochem. Cycles, 28, 927– 949, doi:10.1002/2014GB004853.'
outs.socat_version = 'v2025; https://doi.org/10.25921/r7xa-bt92'
outs.reanalysis_data = 'Ford, D. J., Shutler, J. D., Ashton, I., Sims, R. P., & Holding, T. (2025). Recalculated (depth and temperature consistent) surface ocean CO₂ atlas (SOCAT) version 2025 (v0-1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15656803'
outs.coolskin = 'Cool Skin calculated using NOAA COARE 3.5, with ERA5 monthly inputs - to get cool skin deviation, compute tos_skin - tos'
outs.era5_data_location = 'https://doi.org/10.24381/cds.f17050d7'
outs.noaa_coare_35_code = 'https://doi.org/10.5281/zenodo.5110991'
outs.noaa_ersl_marine_boundary_layer_citation = 'Dlugokencky, E.J., K.W. Thoning, X. Lan, and P.P. Tans (2021), NOAA Greenhouse Gas Reference from Atmospheric Carbon Dioxide Dry Air Mole Fractions from the NOAA GML Carbon Cycle Cooperative Global Air Sampling Network. Data Path: ftp://aftp.cmdl.noaa.gov/data/trace_gases/co2/flask/surface/'
outs.createDimension('lon',lon.shape[0])
outs.createDimension('lat',lat.shape[0])
outs.createDimension('time',fco2_sw_cor.shape[2])
outs.createDimension('region',4)

var = outs.createVariable('lat','f4',('lat'))
var[:] = lat
var.Long_name = 'Latitude'
var.Units = 'Degrees North'

var = outs.createVariable('lon','f4',('lon'))
var[:] = lon + 180 #Convert form -180 to 180, 0 to 360
var.Long_name = 'Longitude'
var.Units = 'Degrees East'

var = outs.createVariable('time','f4',('time'))
var[:] = time
var.Long_name = 'Time'
var.Units = t_units

##################
##################
##################
var = outs.createVariable('sfco2_corrected','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(fco2_sw_cor,(2,1,0)),axis=2)
var.Long_name = 'Surface ocean fCO2 corrected'
var.description = 'Surface ocean fCO2 corrected at the subskin temperature; fCO2(sw,subskin)'
var.subskin_temp_dataset = 'Subskin temperature dataset used = SST-CCI v3.0 (https://dx.doi.org/10.5285/4a9654136a7148e39b7feb56f8bb02d2)'
var.Units = 'uatm; micro atmospheres'

var = outs.createVariable('sfco2','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(fco2_sw_uncor,(2,1,0)),axis=2)
var.Long_name = 'Surface ocean fCO2'
var.description = 'Surface ocean fCO2 at the subskin temperature; fCO2(sw,subskin)'
var.subskin_temp_dataset = 'Subskin temperature dataset used = SST-CCI v3.0 (https://dx.doi.org/10.5285/4a9654136a7148e39b7feb56f8bb02d2)'
var.Units = 'uatm; micro atmospheres'

ou = fl.load_flux_var(fluxloc,'OAPC1',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
print(ou.shape)
var = outs.createVariable('fco2atm_corrected','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)),axis=2)
var.Long_name = 'Atmospheric fCO2 at skin temperature'
var.Units = 'uatm; microatmospheres'
var.description = 'Atmospheric fCO2 evaluated at the skin temperature; tos_skin'

ou = fl.load_flux_var(fluxloc2,'OAPC1',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
print(ou.shape)
var = outs.createVariable('fco2atm','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)),axis=2)
var.Long_name = 'Atmospheric fCO2 at subskin temperature'
var.Units = 'uatm; microatmospheres'
var.description = 'Atmospheric fCO2 evaluated at the subskin temperature; tos'

ou = fl.load_flux_var(fluxloc,'OK3',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('kw_corrected','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)),axis=2)
var.Long_name = 'Gas Transfer Velocity'
var.Units = 'cm hr-1'
var.description = 'Gas transfer velocity estimated from CCMPv3.1 wind speed and Nightingale et al. 2000 gas transfer parameterisation at skin temperature'

ou = fl.load_flux_var(fluxloc2,'OK3',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('kw','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)),axis=2)
var.Long_name = 'Gas Transfer Velocity'
var.Units = 'cm hr-1'
var.description = 'Gas transfer velocity estimated from CCMPv3.1 wind speed and Nightingale et al. 2000 gas transfer parameterisation at subskin temperature'

ou = fl.load_flux_var(fluxloc,'FT1_Kelvin_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('tos','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)) - 273.15,axis=2) #Kelvin to degC
var.Long_name = 'Surface ocean temperature'
var.Units = 'degC'
var.description = 'Surface ocean subskin temperature'

ou = fl.load_flux_var(fluxloc,'ST1_Kelvin_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('tos_skin','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)) - 273.15,axis=2) #Kelvin to degC
var.Long_name = 'Surface ocean skin temperature'
var.Units = 'degC'
var.description = 'Surface ocean skin temperature'

ou = fl.load_flux_var(fluxloc,'fnd_solubility',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('alpha','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1))/1000,axis=2) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
var.Long_name = 'Solubilty of CO2 in seawater'
var.Units = 'mol m-3 uatm-1'
var.description = 'Solubility of CO2 in seawater evaluated at the subskin; tos temperature'

ou = fl.load_flux_var(fluxloc,'skin_solubility',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('alpha_skin','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1))/1000,axis=2) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
var.Long_name = 'Skin solubilty of CO2 in seawater'
var.Units = 'mol m-3 uatm-1'
var.description = 'Solubility of CO2 in seawater evaluated at the skin temperature (tos_skin) and a salty skin (+0.1 psu to subskin salinity)'

ou = fl.load_flux_var(fluxloc,'P1',start_yr,end_yr,lon.shape[0],lat.shape[0],fco2_sw_cor.shape[2])
ou[ou == -999] = np.nan
var = outs.createVariable('fice','f4',('time','lat','lon'))
var[:] = lon_switch(np.transpose(ou,(2,0,1)),axis=2) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
var.Long_name = 'Fraction of sea ice cover'
var.Units = ''
var.description = 'Percentage of sea ice cover corresponding to the tos dataset'

#############
#############
#############
# ou = load_flux_var(fluxloc,'OF',1985,2022,lon.shape[0],lat.shape[0],fco2_sw.shape[2])
# ou[ou == -999] = np.nan
var = outs.createVariable('fgco2_corrected','f4',('time','lat','lon'))
flux_cor = lon_switch(np.transpose(flux_cor,(2,1,0)) / (12.011) / (24*3600) * -1,axis=2) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
var[:] = flux_cor
var.Long_name = 'Air-sea CO2 flux corrected'
var.Units = 'mol m-2 s-1; +ve into ocean'
var.description = 'Air-sea CO2 flux evaluated with respect to vertical temperature gradients using Nightingale et al. 2000 gas transfer parameterisation'
var.Note = 'Positive flux indicates flux into the ocean!'
var.skin_effects = 'Includes a cool salty skin'

var = outs.createVariable('fgco2','f4',('time','lat','lon'))
flux_uncor = lon_switch(np.transpose(flux_uncor,(2,1,0)) / (12.011) / (24*3600) * -1,axis=2) # 12 to convert grams to moles; 24*3600 to convert days to seconds; -1 to make positive flux into the ocean
var[:] = flux_uncor
var.Long_name = 'Air-sea CO2 flux'
var.Units = 'mol m-2 s-1; +ve into ocean'
var.description = 'Air-sea CO2 flux using Nightingale et al. 2000 gas transfer parameterisation'
var.Note = 'Positive flux indicates flux into the ocean!'
var.skin_effects = 'None'

c = Dataset(os.path.join(model_save_loc_cor,'inputs','bath.nc'))
land = np.array(c['ocean_proportion'])
c.close()
var = outs.createVariable('mask_sfc','f4',('lat','lon'))
land_mask = lon_switch(np.transpose(land,(1,0)),axis=1)
var[:] = land_mask
var.Long_name = 'Fractional coverage of ocean in each grid tile'
var.Units = ''
var.description = 'This is the ocean proportion mask calculated from GEBCO2023 data. '

area = du.area_grid(lat=lat,lon=lon,res=1) * 1e6
var = outs.createVariable('area','f4',('lat','lon'))
var[:] = lon_switch(area,axis=1)
var.Long_name = 'Total surface area of each grid cell'
var.Units = 'm2'
var.description = 'Calculated assuming the Earth is a oblate sphere with major and minor radius of 6378.137 km and 6356.7523 km respectively'
var.comment = 'Multiply "area" and "mask_sfc" to get true area used in flux calculations.'


#### Here we caluclate the reg area values for the flux...

final_area = np.zeros((4)); final_area[:] = np.nan
final_flux = np.zeros((4,fco2_sw_cor.shape[2])); final_flux[:] = np.nan
#Convert flux from mol m-2 s-1 to g m-2 yr-1 (first term mol -> g) (second term seconds -> year)
flux_cor = flux_cor * (12.011) * (365.25*24*3600)
# Multiplying our fluxes by the area and land proportions
print(flux_cor.shape)
print(land_mask.shape)
print(area.shape)
for j in range(fco2_sw_cor.shape[2]):
    flux_cor[j,:,:] = flux_cor[j,:,:] * area * land_mask

#Masking out area array where we don't calculate any fluxes...
f = np.sum(np.isnan(flux_cor),axis=0)
area2 = np.copy(area*land_mask)
area2[f == fco2_sw_cor.shape[2]] = np.nan

## Global fluxes
#Global area similar to Andy's work (lower as we have a higher resolution land proportion mask)
final_area[0] = np.nansum(area2)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_cor[j,:,:]) / 1e15) # gC to Pg C
final_flux[0,:] = fl

###30N
area_n = np.copy(area2)
flux_n = np.copy(flux_cor)
f = np.where(lat < 30)
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[1] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[1,:] = fl

###30N - 30S
area_n = np.copy(area2)
flux_n = np.copy(flux_cor)
f = np.argwhere((lat >= 30) | (lat <= -30)) #
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[2] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[2,:] = fl

###30S
area_n = np.copy(area2)
flux_n = np.copy(flux_cor)
f = np.where(lat > -30) #
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[3] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[3,:] = fl

var = outs.createVariable('fgco2_reg_corrected','f4',('region','time'))
var[:] = final_flux
var.Long_name = 'Regionally intergrated air-sea CO2 flux corrected'
var.Regions = '1 = Global; 2 = North of 30N; 3 = 30N to 30S; 4 = South of 30S'
var.Units = 'Pg C yr-1; +ve into ocean'
var.description = 'Regionally intergrated air-sea CO2 flux evaluated with respect to vertical temperature gradients using Nightingale et al. 2000 gas transfer parameterisation'
var.Notes2 = 'Areas are calculated from "area" and "mask_sfc" so area account for land proportions within grid cells'
var.Notes = 'Positive flux indicates flux into the ocean!'

var = outs.createVariable('area_reg_corrected','f4',('region'))
var[:] = final_area
var.Long_name = 'Regionally intergrated area'
var.Regions = '1 = Global; 2 = North of 30N; 3 = 30N to 30S; 4 = South of 30S'
var.Units = 'm-2'
var.description = 'Regionally intergrated area for regionally intergrated air-sea CO2 fluxes (fgco2_reg_corrected)'
var.Notes = 'Calculated from "area" and "mask_sfc" so areas account for land proportions within grid cells'

########################################
########################################
final_area = np.zeros((4)); final_area[:] = np.nan
final_flux = np.zeros((4,fco2_sw_cor.shape[2])); final_flux[:] = np.nan
#Convert flux from mol m-2 s-1 to g m-2 yr-1 (first term mol -> g) (second term seconds -> year)
flux_uncor = flux_uncor * (12.011) * (365.25*24*3600)
# Multiplying our fluxes by the area and land proportions

for j in range(fco2_sw_cor.shape[2]):
    flux_uncor[j,:,:] = flux_uncor[j,:,:] * area * land_mask

#Masking out area array where we don't calculate any fluxes...
f = np.sum(np.isnan(flux_uncor),axis=0)
area2 = np.copy(area*land_mask)
area2[f == fco2_sw_cor.shape[2]] = np.nan

## Global fluxes
#Global area similar to Andy's work (lower as we have a higher resolution land proportion mask)
final_area[0] = np.nansum(area2)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_uncor[j,:,:]) / 1e15) # gC to Pg C
final_flux[0,:] = fl

###30N
area_n = np.copy(area2)
flux_n = np.copy(flux_uncor)
f = np.where(lat < 30)
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[1] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[1,:] = fl

###30N - 30S
area_n = np.copy(area2)
flux_n = np.copy(flux_uncor)
f = np.argwhere((lat >= 30) | (lat <= -30)) #
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[2] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[2,:] = fl

###30S
area_n = np.copy(area2)
flux_n = np.copy(flux_uncor)
f = np.where(lat > -30) #
area_n[f,:] = np.nan; flux_n[:,f,:] = np.nan
final_area[3] = np.nansum(area_n)
fl = []
for j in range(fco2_sw_cor.shape[2]):
    fl.append(np.nansum(flux_n[j,:,:]) / 1e15) # gC to Pg C
final_flux[3,:] = fl

var = outs.createVariable('fgco2_reg','f4',('region','time'))
var[:] = final_flux
var.Long_name = 'Regionally intergrated air-sea CO2 flux'
var.Regions = '1 = Global; 2 = North of 30N; 3 = 30N to 30S; 4 = South of 30S'
var.Units = 'Pg C yr-1; +ve into ocean'
var.description = 'Regionally intergrated air-sea CO2 flux using Nightingale et al. 2000 gas transfer parameterisation'
var.Notes2 = 'Areas are calculated from "area" and "mask_sfc" so area account for land proportions within grid cells'
var.Notes = 'Positive flux indicates flux into the ocean!'

var = outs.createVariable('area_reg','f4',('region'))
var[:] = final_area
var.Long_name = 'Regionally intergrated area'
var.Regions = '1 = Global; 2 = North of 30N; 3 = 30N to 30S; 4 = South of 30S'
var.Units = 'm-2'
var.description = 'Regionally intergrated area for regionally intergrated air-sea CO2 fluxes (fgco2_reg)'
var.Notes = 'Calculated from "area" and "mask_sfc" so areas account for land proportions within grid cells'

outs.close()
