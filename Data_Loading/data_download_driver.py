#!/usr/bin/env python
"""
"""
import data_utils as du
start_yr = 1980
end_yr = 2024

noaa = False
era5 = False
era5_daily = False
cmems = False
cmems_daily=True
bicep = False
ccmp = False
oisst = False
cci = False
gebco = False
osisaf = False
lon,lat = du.reg_grid()
if gebco:
    import gebco_resample as ge
    ge.gebco_resample('D:/Data/Bathymetry/GEBCO_2023.nc',lon,lat)

if era5:
    import ERA5_data_download as er
    #Download script for downloading monthly ERA5 air pressure and wind speed.
    # from the Copernicus Climate Data Store (CCDS)
    er.download_era5("F:/Data/ERA5/MONTHLY/DATA",start_yr=start_yr,end_yr=end_yr)
    #Averaging from 0.25 to 1 deg
    #mean sea level pressure


if cmems_daily:
    import cmems_glorysv12_download as cm
    cm.load_glorysv12_daily('F:\Data\CMEMS\SSS\DAILY',end_yr = end_yr,variable = 'so')
    # cm.load_glorysv12_daily('F:\Data\CMEMS\SSS\DAILY_5m',end_yr = end_yr,variable = 'so',depth=5.078224182128906)
    # cm.load_glorysv12_daily('F:\Data\CMEMS\MLD\DAILY',end_yr = end_yr,variable = 'mlotst')

if cmems:
    import cmems_glorysv12_download as cm
    cm.load_glorysv12_monthly('F:\Data\CMEMS\SSS\MONTHLY',end_yr = end_yr,variable = 'so')
    cm.load_glorysv12_monthly('F:\Data\CMEMS\MLD\MONTHLY',end_yr = end_yr,variable = 'mlotst')
    cm.load_glorysv12_monthly('F:\Data\CMEMS\SST\MONTHLY',end_yr = end_yr,variable = 'thetao')


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
    ccmp.ccmp_temporal_average('D:/Data/CCMP/v3.1',v=3.1,var='ws')
    #ccmp.ccmp_average('D:/Data/CCMP/v3.0/monthly','D:/Data/CCMP/v3.0/monthly/1DEG',start_yr=start_yr,end_yr=end_yr,res=1)

if oisst:
    import OISST_data_download as OI
    OI.download_oisst_v21_daily('D:/Data/OISSTv2_1',start_yr=start_yr,end_yr=end_yr)
    OI.oisst_monthly_av('D:/Data/OISSTv2_1',start_yr=start_yr,end_yr=end_yr)

if cci:

    import cci_sstv2 as cc2
    cc2.cci_sst_v3(loc='F:/Data/SST-CCI/v301',start_yr=start_yr,end_yr=end_yr)
    cc2.cci_monthly_av('F:/Data/SST-CCI/v301',start_yr=start_yr,end_yr=end_yr,v3 = True)
    #cc2.cci_sst_spatial_average(data='D:/Data/SST-CCI/v301/monthly',start_yr = start_yr,end_yr = end_yr,out_loc = 'D:/Data/SST-CCI/v301/monthly/1DEG',log = lon,lag = lat,v3=True,flip=True)
    # cc2.cci_sst_spatial_average(data='E:/Data/SST-CCI/v301/monthly',start_yr = start_yr,end_yr = end_yr,out_loc = 'E:/Data/SST-CCI/v301/monthly/1DEG',log = lon,lag = lat,v3=True,flip=True,bia = 0.00,monthly=True,area_wei=True)
    # cc2.cci_sst_spatial_average(data='E:/Data/SST-CCI/v301/monthly',start_yr = start_yr,end_yr = end_yr,out_loc = 'E:/Data/SST-CCI/v301/monthly/1DEG_biascorrected',log = lon,lag = lat,v3=True,flip=True,bia = 0.04,monthly=True,area_wei=True)

if osisaf:
    import OSISAF_download as OSI
    OSI.OSISAF_monthly_av('F:/Data/OSISAF',start_yr=start_yr,end_yr=end_yr)

if era5_daily:
    from ERA5_data_download import era5_average,era5_daily,era5_wind_time_average
    era5_daily(loc = "F:/Data/ERA5/DAILY",start_yr=start_yr,end_yr=end_yr)
    era5_wind_time_average(loc = "F:/Data/ERA5/DAILY",outloc = "F:/Data/ERA5/DAILY/monthly",start_yr=start_yr,end_yr=end_yr)
