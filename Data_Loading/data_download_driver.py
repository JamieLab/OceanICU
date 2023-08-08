#!/usr/bin/env python
"""
"""
import data_utils as du
start_yr = 1985
end_yr = 2022

noaa = False
era5 = False
cmems = False
bicep = False
ccmp = False
oisst = False
cci = False
gebco = True

if gebco:
    import gebco_resample as ge
    lon,lat = du.reg_grid()
    ge.gebco_resample('D:/Data/Bathymetry/GEBCO_2023.nc',lon,lat)

if noaa:
    import interpolate_noaa_ersl as noa
    noa.interpolate_noaa('D:/Data/NOAA_ERSL/2023_download.txt',end_yr = end_yr)

if era5:
    import ERA5_data_download as er
    #Download script for downloading monthly ERA5 air pressure and wind speed.
    # from the Copernicus Climate Data Store (CCDS)
    er.download_era5("D:/Data/ERA5/MONTHLY/DATA",start_yr=start_yr,end_yr=end_yr)
    #Averaging from 0.25 to 1 deg
    #mean sea level pressure
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='msl')
    #wind speed at 10 m
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='si10')
    #wind speed at 10 m
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='blh')
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='d2m')
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='msdwlwrf')
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='msdwswrf')
    er.era5_average("D:/Data/ERA5/MONTHLY/DATA","D:/Data/ERA5/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1,var='t2m')

if cmems:
    import cmems_glorysv12_download as cm
    #cm.load_glorysv12_monthly('D:\Data\CMEMS\SSS\MONTHLY_TEST',end_yr = end_yr,variable = 'zos')
    lon,lat = du.reg_grid(lon=0.25,lat=0.25,latm=[10,40],lonm=[-80,0])
    cm.cmems_average('D:/Data/CMEMS/SSS/MONTHLY_TEST','D:/Data/CMEMS/SSS/MONTHLY_TEST/025DEG_test',log=lon,lag=lat,variable='zos',log_av=False)
    # cm.cmems_sss_load('D:\Data\CMEMS\SSS\MONTHLY',end_yr = end_yr)
    # cm.cmems_average_sss("D:/Data/CMEMS/SSS/MONTHLY/","D:/Data/CMEMS/SSS/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1)
    # cm.cmems_mld_load('D:\Data\CMEMS\MLD\MONTHLY',end_yr = end_yr)
    # cm.cmems_average_mld("D:/Data/CMEMS/MLD/MONTHLY/","D:/Data/CMEMS/MLD/MONTHLY/1DEG",start_yr=start_yr,end_yr=end_yr,res=1)

if bicep:
    #Retrieve the BICEP data from the CEDA archive and then use these to average.
    import bicep_average as bicep
    bicep.bicep_pp_log_average('D:/Data/BICEP/marine_primary_production/v4.2/monthly','D:/Data/BICEP/marine_primary_production/v4.2/monthly/1DEG',start_yr=start_yr,end_yr=end_yr,res=1)
    bicep.bicep_poc_log_average('D:/Data/BICEP/particulate_organic_carbon/v5.0/monthly/GEO','D:/Data/BICEP/particulate_organic_carbon/v5.0/monthly/GEO/1DEG',start_yr=start_yr,end_yr=end_yr,res=1)
    bicep.bicep_ep_log_average('D:/Data/BICEP/oceanic_export_production/v1.0/monthly','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',var = 'EP_Dunne',start_yr=start_yr,end_yr=end_yr,res=1)
    bicep.bicep_ep_log_average('D:/Data/BICEP/oceanic_export_production/v1.0/monthly','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',var = 'EP_Henson',start_yr=start_yr,end_yr=end_yr,res=1)
    bicep.bicep_ep_log_average('D:/Data/BICEP/oceanic_export_production/v1.0/monthly','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',var = 'EP_Li',start_yr=start_yr,end_yr=end_yr,res=1)

if ccmp:
    import ccmp_average as ccmp
    ccmp.ccmp_average('D:/Data/CCMP/v3.0/monthly','D:/Data/CCMP/v3.0/monthly/1DEG',start_yr=start_yr,end_yr=end_yr,res=1)

if oisst:
    import OISST_data_download as OI
    #OI.download_oisst_v21_daily('D:/Data/OISSTv2_1',start_yr=start_yr,end_yr=end_yr)
    OI.OISST_monthly_split('sst.mon.mean.nc','D:/Data/OISSTv2_1/monthly')

if cci:
    import cci_sst_retrieve_v2 as cc
    cc.cci_sst_v3_trailblaze('D:/Data/SST-CCI/v3',start_yr=start_yr,end_yr=end_yr)
