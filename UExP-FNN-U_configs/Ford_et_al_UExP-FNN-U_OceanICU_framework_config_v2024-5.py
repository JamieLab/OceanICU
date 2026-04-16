#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""
#This is needed or the code crashes with the reanalysis step...
if __name__ == '__main__':
    import os
    import sys
    import shutil
    import numpy as np
    os.chdir('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU')

    print(os.getcwd())
    print(os.path.join(os.getcwd(),'Data_Loading'))

    sys.path.append(os.path.join(os.getcwd(),'Data_Loading'))
    sys.path.append(os.path.join(os.getcwd()))
    import data_utils as du
    create_inp =False
    run_neural =False
    run_flux = False
    add_land = True
    geopan=True
    fluxengine_config = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/OceanICU/fluxengine_config/fluxengine_config_night.conf'

    model_save_loc = 'E:/SCOPE/NN/Ford_et_al_SOM'
    inps = os.path.join(model_save_loc,'inputs')
    data_file = os.path.join(inps,'neural_network_input.nc')
    start_yr = 1985
    end_yr = 2023
    log,lag = du.reg_grid(lat=1,lon=1)

    if create_inp:
        from neural_network_train import make_save_tree
        make_save_tree(model_save_loc)
        #
        # from Data_Loading.cmems_glorysv12_download import cmems_average
        #
        # cmems_average('D:/Data/CMEMS/SSS/MONTHLY',os.path.join(inps,'SSS'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='so')
        # cmems_average('D:/Data/CMEMS/MLD/MONTHLY',os.path.join(inps,'MLD'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='mlotst',log_av = True)
        # #
        # from Data_Loading.cci_sstv2 import cci_sst_spatial_average
        # cci_sst_spatial_average(data='D:/Data/SST-CCI/V301/monthly',out_loc=os.path.join(inps,'SST'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,bia=0.05)
        # # #
        # from Data_Loading.interpolate_noaa_ersl import interpolate_noaa
        # interpolate_noaa('D:/Data/NOAA_ERSL/2024_download.txt',grid_lon = log,grid_lat = lag,out_dir = os.path.join(inps,'xco2atm'),start_yr=start_yr,end_yr = end_yr)
        # #
        # from Data_Loading.ERA5_data_download import era5_average
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msl'),log=log,lag=lag,var='msl',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'blh'),log=log,lag=lag,var='blh',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'d2m'),log=log,lag=lag,var='d2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'t2m'),log=log,lag=lag,var='t2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msdwlwrf'),log=log,lag=lag,var='msdwlwrf',start_yr = start_yr,end_yr =end_yr)
        #
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'si10'),log=log,lag=lag,var='si10',start_yr = start_yr,end_yr =end_yr)
        # #
        # # import Data_Loading.gebco_resample as ge
        # # ge.gebco_resample('D:/Data/Bathymetry/GEBCO_2023.nc',log,lag,save_loc = os.path.join(inps,'bath.nc'),save_loc_fluxengine = os.path.join(inps,'fluxengine_bath.nc'))
        # # #
        # from Data_Loading.OSISAF_download import OSISAF_spatial_average
        # from Data_Loading.OSISAF_download import OSISAF_merge_hemisphere
        # OSISAF_spatial_average(data='D:/Data/OSISAF/monthly',out_loc=os.path.join(inps,'OSISAF'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,hemi = 'NH')
        # OSISAF_spatial_average(data='D:/Data/OSISAF/monthly',out_loc=os.path.join(inps,'OSISAF'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,hemi = 'SH')
        # OSISAF_merge_hemisphere(os.path.join(inps,'OSISAF'),os.path.join(inps,'bath.nc'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag)
        #
        # import Data_Loading.ccmp_average as cc
        # cc.ccmp_average('D:/Data/CCMP/v3.1/monthly',outloc=os.path.join(inps,'ccmpv3.1'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,v =3.1,geb_file='D:/Data/Bathymetry/GEBCO_2023.nc',var='ws')

        # import Data_Loading.cci_sss as ccisss
        # ccisss.cci_sss_spatial_average(data='E:/Data/SSS-CCI/v4.41/%Y/ESACCI-SEASURFACESALINITY-L4-SSS-GLOBAL-MERGED_OI_Monthly_CENTRED_15Day_0.25deg-%Y%m15-fv4.41.nc',start_yr=start_yr,end_yr=end_yr,out_loc = os.path.join(inps,'CCISSS'),log=log,lag=lag,area_wei=True)

        import construct_input_netcdf as cinp
        # #Vars should have each entry as [Extra_Name, netcdf_variable_name,data_location,produce_anomaly]
        vars = [['CCI_SST','analysed_sst',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        ,['CCI_SST','sea_ice_fraction',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        ,['CCI_SST','analysed_sst_uncertainty',os.path.join(inps,'SST','%Y','%Y%m*.nc'),0]
        ,['NOAA_ERSL','xCO2',os.path.join(inps,'xco2atm','%Y','%Y_%m*.nc'),1]
        ,['ERA5','blh',os.path.join(inps,'blh','%Y','%Y_%m*.nc'),0]
        ,['ERA5','d2m',os.path.join(inps,'d2m','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msdwlwrf',os.path.join(inps,'msdwlwrf','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msdwswrf',os.path.join(inps,'msdwswrf','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msl',os.path.join(inps,'msl','%Y','%Y_%m*.nc'),0]
        ,['ERA5','t2m',os.path.join(inps,'t2m','%Y','%Y_%m*.nc'),0]
        ,['ERA5','si10',os.path.join(inps,'si10','%Y','%Y_%m*.nc'),0]
        ,['CMEMS','so',os.path.join(inps,'SSS','%Y','%Y_%m*.nc'),1]
        ,['CMEMS','mlotst',os.path.join(inps,'MLD','%Y','%Y_%m*.nc'),1]
        ,['CCMP','ws',os.path.join(inps,'ccmpv3.1','%Y','CCMP_3.1_ws_%Y%m*.nc'),0]
        ,['CCMP','ws^2',os.path.join(inps,'ccmpv3.1','%Y','CCMP_3.1_ws_%Y%m*.nc'),0]
        ,['OSISAF','total_standard_uncertainty',os.path.join(inps,'OSISAF','%Y','%Y%m_*_COM.nc'),0]
        ,['OSISAF','ice_conc',os.path.join(inps,'OSISAF','%Y','%Y%m_*_COM.nc'),0]
        ,['Takahashi','taka','F:/Data/Takahashi_Clim/monthly/takahashi_%m_.nc',0]
        ]
        cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag)
        vars = [['CCI_SSS','sss',os.path.join(inps,'CCISSS','%Y','%Y%m*.nc'),0]]
        cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag,append=True,fill_clim=False)
        cinp.fill_with_var(data_file,'CCI_SSS_sss','CMEMS_so',log=log,lag=lag,calc_anom=True)

        import run_reanalysis as rean
        socat_file = 'E:/Data/_DataSets/SOCAT/v2024/SOCATv2024_reanalysed_v1/SOCATv2024_ESACCIv3_biascorrected.nc'
        rean.load_prereanalysed(socat_file,data_file,start_yr = start_yr,end_yr=end_yr,name = 'CCI_SST')
        #
        #
        cinp.fill_with_var(data_file,'CCMP_ws','ERA5_si10',log=log,lag=lag)
        cinp.fill_with_var(data_file,'CCMP_ws^2','ERA5_si10',log=log,lag=lag,mod ='power2')
        cinp.land_clear(model_save_loc)
        import self_organising_map as som
        som.som_feed_forward(model_save_loc,data_file,['Takahashi_taka','CCI_SST_analysed_sst','CCI_SSS_sss_CMEMS_so','CMEMS_mlotst'])
        cinp.append_longhurst_prov(model_save_loc,'F:/Data/Longhurst/Longhurst_1_deg.nc',[1],17,'prov_smoothed')
        cinp.append_longhurst_prov(model_save_loc,'F:/Data/Longhurst/Longhurst_1_deg.nc',[16,25],16,'prov_smoothed')
        cinp.manual_prov(model_save_loc,[35,50],[44,60],'prov_smoothed')
        cinp.manual_prov(model_save_loc,[40,48],[27,43],'prov_smoothed')

    if run_neural:
        import neural_network_train as nnt
        # nnt.driver(data_file,fco2_sst = 'CCI_SST', prov = 'prov_smoothed',var = ['CCI_SST_analysed_sst','NOAA_ERSL_xCO2','CCI_SSS_sss_CMEMS_so','CMEMS_mlotst','CCI_SST_analysed_sst_anom','NOAA_ERSL_xCO2_anom','CCI_SSS_sss_CMEMS_so_anom','CMEMS_mlotst_anom'],
        #    model_save_loc = model_save_loc +'/',unc =[0.35,0.4,0.2,0.1,0.35,0.4,0.2,0.1],bath = 'GEBCO_elevation',bath_cutoff = None,fco2_cutoff_low = 50,fco2_cutoff_high = 750,sea_ice = None,
        #    tot_lut_val=20000)
        nnt.plot_total_validation_unc(fco2_sst = 'CCI_SST',model_save_loc = model_save_loc,ice = None,prov='prov_smoothed')
        # nnt.plot_mapped(model_save_loc)
    if run_flux:
        import fluxengine_driver as fl
        # print('Running flux calculations....')
        # fl.fluxengine_netcdf_create(model_save_loc,input_file = data_file,tsub='CCI_SST_analysed_sst',ws = 'CCMP_ws_ERA5_si10',ws2 = 'CCMP_ws^2_ERA5_si10',seaice = 'OSISAF_ice_conc',
        #      sal='CMEMS_so',msl = 'ERA5_msl',xCO2 = 'NOAA_ERSL_xCO2',start_yr=start_yr,end_yr=end_yr, coare_out = os.path.join(inps,'coare'), tair = 'ERA5_t2m', dewair = 'ERA5_d2m',
        #      rs = 'ERA5_msdwswrf', rl = 'ERA5_msdwlwrf', zi = 'ERA5_blh',coolskin = 'COARE3.5')
        # fl.fluxengine_run(model_save_loc,fluxengine_config,start_yr,end_yr)
        # fl.flux_uncertainty_calc(model_save_loc,start_yr = start_yr,end_yr=end_yr,fco2_tot_unc = -1,k_perunc=0.2,unc_input_file=data_file,sst_unc='CCI_SST_analysed_sst_uncertainty',atm_unc=0.4)
        # fl.calc_annual_flux(model_save_loc,lon=log,lat=lag,start_yr=start_yr,end_yr=end_yr)
        # fl.fixed_uncertainty_append(model_save_loc,lon=log,lat=lag)
        # #fl.variogram_evaluation(model_save_loc,output_file='sst_decorrelation',input_array='CCI_SST_analysed_sst_uncertainty',input_datafile=data_file)
        # #fl.variogram_evaluation(model_save_loc,output_file='ice_decorrelation',input_array='OSISAF_total_standard_uncertainty',hemisphere=True,input_datafile=data_file)
        # #fl.variogram_evaluation(model_save_loc,output_file='fco2_net_decorrelation',input_datafile =os.path.join(model_save_loc,'output.nc'),input_array='fco2_net_unc',start_yr =start_yr,end_yr=end_yr)
        # # fl.variogram_evaluation(model_save_loc,output_file='fco2_decorrelation',start_yr =start_yr,end_yr=end_yr)
        # # #fl.variogram_evaluation(model_save_loc,output_file='wind_decorrelation',input_array=['CCMP_w','ERA5_si10'],input_datafile=[data_file,data_file])
        # #
        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_net_decorrelation.csv',flux_var = 'flux_unc_fco2sw_net',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_para',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_val',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='ice_decorrelation.csv',seaice=True,seaice_var='OSISAF_total_standard_uncertainty',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_ph2o',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_schmidt',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solskin_unc',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solsubskin_unc',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='wind_decorrelation.csv',flux_var = 'flux_unc_wind',start_yr =start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor=[2000,1500],flux_var = 'flux_unc_xco2atm',start_yr =start_yr,end_yr=end_yr)
        # fl.plot_relative_contribution(model_save_loc,model_plot='C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/Watsonetal2023.csv',model_plot_label='UoEx-Watson')


        # fl.generate_mask_reccap(data_file,'F:/Data/RECCAP/RECCAP2_region_masks_all_v20221025_DJF_SOCATpaper.nc','reccap',5,'reccap_southern_ocean')
        # fl.calc_annual_flux(model_save_loc,lon=log,lat=lag,start_yr=start_yr,end_yr=end_yr,mask_file=data_file,mask_var='reccap_southern_ocean',save_file=os.path.join(model_save_loc,'southern_ocean.csv'))
        # fl.fixed_uncertainty_append(model_save_loc,lon=log,lat=lag,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')

        fl.montecarlo_flux_testing(model_save_loc,decor='fco2_net_decorrelation.csv',flux_var = 'flux_unc_fco2sw_net',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_para',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_val',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='ice_decorrelation.csv',seaice=True,seaice_var='OSISAF_total_standard_uncertainty',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_ph2o',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_schmidt',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solskin_unc',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solsubskin_unc',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor='wind_decorrelation.csv',flux_var = 'flux_unc_wind',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')
        fl.montecarlo_flux_testing(model_save_loc,decor=[2000,1500],flux_var = 'flux_unc_xco2atm',start_yr =start_yr,end_yr=end_yr,output_file='southern_ocean.csv',mask_file=data_file,mask_var='reccap_southern_ocean')

"""
Total Alkalinity
"""

model_save_loc2 = 'E:/SCOPE/NN/Ford_et_al_SOM/ALKALINITY'
GLODAP_file = 'F:/Data/_DataSets/GLODAP/GLODAPv2.2023_Merged_Master_File_SNAPOCO2appended_sharkwebappended_global_surface.csv'
glodap_file = os.path.join(model_save_loc2,'inputs','glodap.csv')
from neural_network_train import make_save_tree
make_save_tree(model_save_loc2)

create_inp = False
run_neural = False
run_pyco2 = False
if create_inp:
    import construct_input_netcdf as cinp
    import convex as cox
    import pyCO2sys_func as co2sys
    co2sys.load_DIVA_glodap(os.path.join(model_save_loc2,'inputs','window_data_from_glodap.txt'),delimiter='\t',out_file = os.path.join(model_save_loc2,'inputs','ta_interpolated.nc'),log=log,lag=lag,var_name='ta')
    vars = [
        ['insitu','ta',os.path.join(model_save_loc2,'inputs','ta_interpolated.nc'),0]
        ]
    cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag,append=True)

    import self_organising_map as som
    som.som_feed_forward(model_save_loc,data_file,['insitu_ta','CCI_SSS_sss_CMEMS_so','CCI_SST_analysed_sst'],normalise=True,o_var='TA_prov',m=4,box = [5,3])

    cox.append_som_prov(GLODAP_file,data_file,sep = ',',lon=log,lat=lag,prov_var = 'TA_prov_smoothed',out_file = os.path.join(model_save_loc2,'inputs','glodap.csv'),
        lon_var='G2longitude',lat_var='G2latitude',mon_var='G2month')
    import Data_Loading.woa_func as woa
    woa.append_woa(os.path.join(model_save_loc2,'inputs','glodap.csv'),',','G2year','G2month','G2latitude','G2longitude','F:/Data/_DataSets/WOA/2023/PHOSPATE','p_an',
        lag,log,os.path.join(inps,'woa'))
    woa.append_woa(os.path.join(model_save_loc2,'inputs','glodap.csv'),',','G2year','G2month','G2latitude','G2longitude','F:/Data/_DataSets/WOA/2023/NITRATE','n_an',
        lag,log,os.path.join(inps,'woa'))
    woa.append_woa(os.path.join(model_save_loc2,'inputs','glodap.csv'),',','G2year','G2month','G2latitude','G2longitude','F:/Data/_DataSets/WOA/2023/SILICATE','i_an',
        lag,log,os.path.join(inps,'woa'))

    import construct_input_netcdf as cinp
    vars = [['WOA','n_an',os.path.join(inps,'woa','n_an_%m.nc'),0],
        ['WOA','p_an',os.path.join(inps,'woa','p_an_%m.nc'),0],
        ['WOA','i_an',os.path.join(inps,'woa','i_an_%m.nc'),0],
        ]
    cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag,append=True)

if run_neural:
    import neural_network_train as nnt
    nnt.daily_neural_driver(os.path.join(model_save_loc2,'inputs','glodap.csv'),fco2_sst = 'G2talk', prov = 'TA_prov_smoothed',var = ['G2temperature','G2salinity','p_an','i_an'],sep=',',
       learning_rate = 0.001,model_save_loc = model_save_loc2 +'/',unc =[0.35,0.2,0.05,1],bath = 'GEBCO_elevation',bath_cutoff = None,sea_ice = None,
       mapping_var=['CCI_SST_analysed_sst','CCI_SSS_sss_CMEMS_so','WOA_p_an','WOA_i_an'],mapping_prov = 'TA_prov_smoothed',mapping_file = data_file,ktoc='CCI_SST_analysed_sst',c=[600,2700],
       name = 'ta',longname='Total Alkalinity',unit = 'umol/kg',plot_parameter='Total Alkalinity',plot_unit='$\mu$mol kg$^{-1}$', tot_lut_val = 6000,lat_v = 'G2latitude',lon_v='G2longitude',
       year_v = 'G2year',mon_v ='G2month',day_v = 'G2day')
    nnt.plot_residuals(model_save_loc2,'G2latitude','G2longitude','G2talk','fco2',log = log,lag=lag,bin=True,geopan=geopan)
    nnt.plot_residuals(model_save_loc2,'G2latitude','G2longitude','G2talk','fco2',zoom_lon=[-15,15],zoom_lat=[40,65],plot_file='NorthWestEuro_residuals.png',log = log,lag=lag,bin=True,geopan=geopan)

if run_pyco2:
    import construct_input_netcdf as cinp
    cinp.copy_netcdf_vars(model_save_loc2+'/output.nc',['ta','ta_net_unc','ta_tot_unc','ta_val_unc','ta_para_unc'],model_save_loc+'/output.nc')
    import pyCO2sys_func as co2sys
    co2sys.run_pyCO2sys(model_save_loc+'/output.nc',data_file,phosphate_var = 'WOA_p_an',phosphate_var_unc=0.3,phosphate_unc_perc=True,silicate_var='WOA_i_an',silicate_var_unc=0.3,silicate_unc_perc=True,sst_var = 'CCI_SST_analysed_sst',sss_var='CCI_SSS_sss_CMEMS_so')
    co2sys.plot_pyCO2sys_out(os.path.join(model_save_loc,'output.nc'),model_save_loc,geopan=geopan)
    du.insitu_grid(glodap_file.split('.')[0],log,lag,start_yr,end_yr,data_file,year_col = 'G2year',month_col='G2month',lat_col = 'G2latitude',lon_col='G2longitude',chla_col = 'G2tco2',unit = 'umolkg-1',out_var_name = 'insitu_DIC',out_var_long_name='insitu Dissolved Inorganic Carbon')
    co2sys.plot_carbonate_validation(model_save_loc,data_file,'insitu_DIC',os.path.join(model_save_loc,'output.nc'),'dic',geopan=geopan)
    du.insitu_grid(glodap_file.split('.')[0],log,lag,start_yr,end_yr,data_file,year_col = 'G2year',month_col='G2month',lat_col = 'G2latitude',lon_col='G2longitude',chla_col = 'G2phtsinsitutp',unit = 'total scale',out_var_name = 'insitu_pH',out_var_long_name='insitu pH')
    co2sys.plot_carbonate_validation(model_save_loc,data_file,'insitu_pH',os.path.join(model_save_loc,'output.nc'),'pH',lims = [7.8,8.4],var_name = 'pH', unit='Total Scale',vma = [-0.1,0.1],geopan=geopan)
    du.output_binned_insitu(data_file,'insitu_DIC',model_save_loc+'/output.nc','dic',os.path.join(model_save_loc,'validation','dic_independent_validation.csv'),'DIC','umol kg-1')
    du.output_binned_insitu(data_file,'insitu_pH',model_save_loc+'/output.nc','pH',os.path.join(model_save_loc,'validation','pH_independent_validation.csv'),'pH','total_scale')

if add_land:
    area = np.transpose(du.area_grid(log,lag,1)* 1e6)
    du.netcdf_append_basic(os.path.join(model_save_loc,'output.nc'),area,'area',units='m^2',lon_dim='longitude',lat_dim='latitude')
    ocean = du.load_netcdf_var(os.path.join(inps,'bath.nc'),'ocean_proportion')
    du.netcdf_append_basic(os.path.join(model_save_loc,'output.nc'),ocean,'ocean_proportion',units='',lon_dim='longitude',lat_dim='latitude')
