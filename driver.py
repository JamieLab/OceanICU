#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""
#This is needed or the code crashes with the reanalysis step...
if __name__ == '__main__':
    import os
    import Data_Loading.data_utils as du
    import sys
    sys.path.append(os.path.join(os.getcwd(),'Data_Loading'))
    create_inp =False
    run_neural =False
    run_flux = False
    plot_final = True
    coare = False

    fluxengine_config = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/OceanICU/fluxengine_config/fluxengine_config_night.conf'

    model_save_loc = 'D:/ESA_CONTRACT/NN/testing'
    inps = os.path.join(model_save_loc,'inputs')
    data_file = os.path.join(inps,'neural_network_input.nc')
    start_yr = 1985
    end_yr = 2022
    log,lag = du.reg_grid(lat=0.1,lon=0.1,latm=[-58,-30],lonm=[-72,-48])

    if create_inp:
        from neural_network_train import make_save_tree
        make_save_tree(model_save_loc)
        cur = os.getcwd()
        os.chdir('Data_Loading')

        from Data_Loading.cmems_glorysv12_download import cmems_average

        #cmems_average('D:/Data/CMEMS/SSS/MONTHLY',os.path.join(inps,'SSS_Test'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='so')
        # cmems_average('D:/Data/CMEMS/MLD/MONTHLY',os.path.join(inps,'MLD'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='mlotst',log_av = True)
        # #
        # from Data_Loading.cci_sstv2 import cci_sst_spatial_average
        # cci_sst_spatial_average(data='D:/Data/SST-CCI/monthly',out_loc=os.path.join(inps,'SST'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag)
        # #
        # from Data_Loading.interpolate_noaa_ersl import interpolate_noaa
        # interpolate_noaa('D:/Data/NOAA_ERSL/2023_download.txt',grid_lon = log,grid_lat = lag,out_dir = os.path.join(inps,'xco2atm'),start_yr=start_yr,end_yr = end_yr)
        #
        # from Data_Loading.ERA5_data_download import era5_average
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msl'),log=log,lag=lag,var='msl',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'blh'),log=log,lag=lag,var='blh',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'d2m'),log=log,lag=lag,var='d2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'t2m'),log=log,lag=lag,var='t2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msdwlwrf'),log=log,lag=lag,var='msdwlwrf',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msdwswrf'),log=log,lag=lag,var='msdwswrf',start_yr = start_yr,end_yr =end_yr)
        #
        # import Data_Loading.gebco_resample as ge
        # ge.gebco_resample('D:/Data/Bathymetry/GEBCO_2023.nc',log,lag,save_loc = os.path.join(inps,'bath.nc'))
        #
        # import Data_Loading.ccmp_average as cc
        # cc.ccmp_average('D:/Data/CCMP/v3.0/monthly',outloc=os.path.join(inps,'ccmp'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag)
        import run_reanalysis as rean
        # socat_file = 'D:/Data/_DataSets/SOCAT/v2023/SOCATv2023_reanalysed/SOCATv2023with_header_ESACCI.tsv'
        # rean.regrid_fco2_data(socat_file,latg=lag,long=log,save_loc=inps,grid=False)
        # import Data_Loading.cci_sstv2 as cci_sst
        # cci_sst.cci_socat_append(os.path.join(inps,'socat','socat.tsv'))
        # import Data_Loading.interpolate_noaa_ersl as noaa
        # noaa.append_noaa(os.path.join(inps,'socat','socat.tsv'),'D:/Data/NOAA_ERSL/2023_download.txt')
        #
        import Data_Loading.cmems_glorysv12_download as cmems
        # cmems.cmems_socat_append(os.path.join(inps,'socat','socat.tsv'),data_loc = 'D:/Data/CMEMS/SSS/DAILY',variable='so')
        cmems.cmems_socat_append(os.path.join(inps,'socat','socat.tsv'),data_loc = 'D:/Data/CMEMS/MLD/DAILY',variable='mlotst',log=True)

        import construct_input_netcdf as cinp
        #Vars should have each entry as [Extra_Name, netcdf_variable_name,data_location,produce_anomaly]
        vars = [['CCI_SST','analysed_sst',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        ,['CCI_SST','sea_ice_fraction',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        ,['NOAA_ERSL','xCO2',os.path.join(inps,'xco2atm','%Y','%Y_%m*.nc'),1]
        ,['ERA5','blh',os.path.join(inps,'blh','%Y','%Y_%m*.nc'),0]
        ,['ERA5','d2m',os.path.join(inps,'d2m','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msdwlwrf',os.path.join(inps,'msdwlwrf','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msdwswrf',os.path.join(inps,'msdwswrf','%Y','%Y_%m*.nc'),0]
        ,['ERA5','msl',os.path.join(inps,'msl','%Y','%Y_%m*.nc'),0]
        ,['ERA5','t2m',os.path.join(inps,'t2m','%Y','%Y_%m*.nc'),0]
        ,['CMEMS','so',os.path.join(inps,'SSS','%Y','%Y_%m*.nc'),1]
        ,['CMEMS','mlotst',os.path.join(inps,'MLD','%Y','%Y_%m*.nc'),1]
        ,['CCMP','w',os.path.join(inps,'ccmp','%Y','%Y_%m*.nc'),0]
        ]
        #cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag)
        # rean.reanalyse(name='CCI-SST',out_dir=os.path.join(inps,'socat'),outfile = data_file,start_yr = start_yr,end_yr = end_yr,prefix = 'GL_from_%Y_to_%Y_%m.nc',socat_files = [socat_file],flip = False)
        import convex as cox
        sh_file = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_Covex_Seascape/Carbon Budget/data/LMEs/LMEs66.shp'
        LME,ind = cox.load_LME_file(sh_file)
        x,y,tit = cox.load_LMEs(LME,ind,14)
        #cinp.province_shape(data_file,'si_prov',log,lag,start_yr,end_yr,x,y)
        #cox.socat_append_prov(os.path.join(inps,'socat','socat.tsv'),x,y,1)
        #rean.correct_fco2_daily(os.path.join(inps,'socat','socat.tsv'),'fCO2_reanalysed [uatm]','T_reynolds [C]','cci_sst [C]')

    if run_neural:
        import neural_network_train as nnt
        nnt.daily_socat_neural_driver(os.path.join(inps,'socat','socat.tsv'),fco2_sst = 'cci_sst [C]_fco2', prov = 'prov',var = ['cci_sst [C]','noaa_atm [ppm]','cmems_so','cmems_mlotst'],
           model_save_loc = model_save_loc +'/',unc =[0.4,1,0.1,0.05],bath = 'GEBCO_elevation',bath_cutoff = None,fco2_cutoff_low = None,fco2_cutoff_high = None,sea_ice = None,
           mapping_var=['CCI_SST_analysed_sst','NOAA_ERSL_xCO2','CMEMS_so','CMEMS_mlotst'],mapping_prov = 'si_prov',mapping_file = data_file,ktoc='CCI_SST_analysed_sst')
        #nnt.plot_total_validation_unc(fco2_sst = 'CCI_SST',model_save_loc = model_save_loc,ice = 'CCI_SST_sea_ice_fraction')
    if run_flux:
        import fluxengine_driver as fl
        print('Running flux calculations....')
        #fl.GCB_remove_prov17(model_save_loc)
        # fl.fluxengine_netcdf_create(model_save_loc,input_file = data_file,tsub='CCI_SST_analysed_sst',ws = 'CCMP_w',seaice = 'CCI_SST_sea_ice_fraction',
        #      sal='CMEMS_so',msl = 'ERA5_msl',xCO2 = 'NOAA_ERSL_xCO2',start_yr=start_yr,end_yr=end_yr, coare_out = os.path.join(inps,'coare'), tair = 'ERA5_t2m', dewair = 'ERA5_d2m',
        #      rs = 'ERA5_msdwswrf', rl = 'ERA5_msdwlwrf', zi = 'ERA5_blh',coolskin = 'COARE3.5')
        # fl.fluxengine_run(model_save_loc,fluxengine_config,start_yr,end_yr)
        # fl.flux_uncertainty_calc(model_save_loc,start_yr = start_yr,end_yr=end_yr)
        fl.calc_annual_flux(model_save_loc,lon=log,lat=lag)
        # fl.plot_example_flux(model_save_loc)
        #import custom_flux_av.ofluxghg_flux_budgets as bud
        #bud.run_flux_budgets(indir = os.path.join(model_save_loc,'flux'),outroot=model_save_loc+'/')

    if plot_final:
        #import fluxengine_driver as fl
        #fl.plot_annual_flux('temporal_flux.png',['D:/ESA_CONTRACT/NN/xCO2_SST','D:/ESA_CONTRACT/NN/xCO2_SST_SSS','D:/ESA_CONTRACT/NN/xCO2_SST_SSS_MLD','D:/ESA_CONTRACT/NN/GCB_2023_Watson'],['xCO2_atm+CCI_SST','xCO2_atm+CCI_SST+CMEMS_SSS','xCO2_atm+CCI_SST+CMEMS_SSS+CMEMS_MLD','GCB_Watson_2023'])
        # fl.plot_annual_flux('GCB_submissions.png',['D:/ESA_CONTRACT/NN/GCB_2023_Watson','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens5','D:/ESA_CONTRACT/NN/GCB_2023_CCISSTv3','D:/ESA_CONTRACT/NN/GCB_2023_Prelim'],
        #     ['GCB_Watson_2023_CCI_SSTv2','GCB_Watson_2023_OISST','GCB_Watson_2023_CCI_SSTv3','GCB_2023_Prelim (SOCATv2023)'])
        #fl.plot_annual_flux('temporal_flux.png',['D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens3','D:/ESA_CONTRACT/NN/NEW_TESTING'],['Watson et al. GCB submission','Ford nerual network version'])
        import animated_plotting as ap
        ap.animated_output(model_save_loc,start_yr=start_yr)
