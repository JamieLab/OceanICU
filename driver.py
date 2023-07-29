#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""
#This is needed or the code crashes with the reanalysis step...
if __name__ == '__main__':
    import os

    create_inp =False
    run_neural = True
    run_flux = False
    plot_final = False
    coare = False
    # Vars should have each entry as [Extra_Name, netcdf_variable_name,data_location]
    # vars = [['CCI_OC','chlor_a','D:/Data/OC-CCI/monthly/chlor_a/1DEG']]
    vars = [['CCI_SST','analysed_sst','D:/Data/SST-CCI/MONTHLY_1DEG',1]
        ,['CCI_SSTv3','analysed_sst','D:/Data/SST-CCI/v3/MONTHLY_1_DEG',1]
        ,['NOAA_ERSL','xCO2','D:/Data/NOAA_ERSL/DATA/MONTHLY',1]
        ,['CCI_OC','chlor_a','D:/Data/OC-CCI/monthly/chlor_a/1DEG',0]
        ,['CCI_SST','sea_ice_fraction','D:/Data/SST-CCI/MONTHLY_1DEG',0]
        ,['CCI_SSTv3','sea_ice_fraction','D:/Data/SST-CCI/v3/MONTHLY_1_DEG',0]
        ,['OI_SST','sst','D:/Data/OISSTv2_1/monthly',1]
        ,['GEBCO','elevation','D:/Data/Bathymetry/LOWER_RES',0]
        ,['PROV','longhurst','D:/Data/Longhurst',0]
        ,['ERA5','si10','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['ERA5','msl','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['CMEMS','so','D:/Data/CMEMS/SSS/MONTHLY/1DEG',1]
        ,['CMEMS','mlotst','D:/Data/CMEMS/MLD/MONTHLY/1DEG',1]
        ,['ERA5','blh','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['ERA5','d2m','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['ERA5','t2m','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['ERA5','msdwlwrf','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['ERA5','msdwswrf','D:/Data/ERA5/MONTHLY/1DEG',0]
        ,['Takahashi','taka','D:/Data/Takahashi_Clim/monthly',0]
        ,['BICEP','pp','D:/Data/BICEP/marine_primary_production/v4.2/monthly/1DEG',0]
        ,['BICEP','POC','D:/Data/BICEP/particulate_organic_carbon/v5.0/monthly/GEO/1DEG',0]
        ,['BICEP','EP_Dunne','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',0]
        ,['BICEP','EP_Li','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',0]
        ,['BICEP','EP_Henson','D:/Data/BICEP/oceanic_export_production/v1.0/monthly/1DEG',0]
        ,['Watson','biome','C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/GCB_Submission_Watsonetal/GCB_Dan_Ford/output/networks/monthly',0]
        ,['CCMP','w','D:/Data/CCMP/v3.0/monthly/1DEG',0]]

    reanal = [['CCI_SST','D:/Data/SST-CCI/MONTHLY_1DEG', '_ESA_CCI_MONTHLY_SST_1_DEG.nc','D:/ESA_CONTRACT/reanalysis/SST_CCI_S2023'],
        ['OI_SST','D:/Data/OISSTv2_1/monthly', '_OISSTv2.nc','D:/ESA_CONTRACT/reanalysis/OI_SST_S2023'],
        ['OI_SST_Watson','', '','D:/ESA_CONTRACT/reanalysis/Watson_OISST_v2022'],
        ['CCI_SSTv3','D:/Data/SST-CCI/v3/MONTHLY_1_DEG', '_ESA_CCI_MONTHLY_SST_1_DEG.nc','D:/ESA_CONTRACT/reanalysis/SST_CCIv3'],]
    socat = ['D:/Data/_DataSets/SOCAT/v2023','SOCATv2023.tsv']
    socatflagE = ['D:/Data/_DataSets/SOCAT/v2023','SOCATv2023_FlagE.tsv']

    data_file = 'D:/ESA_CONTRACT/neural_network_input.nc'
    fluxengine_config = 'fluxengine_config/fluxengine_config_night.conf'
    coare_out = 'D:/ESA_CONTRACT/coare'
    start_yr = 1985
    end_yr = 2022
    if create_inp:
        import construct_input_netcdf as cinp
        import run_reanalysis as rean
        cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr)
        #CCI SST v2.1
        rean.reanalyse(socat_dir=socat[0],socat_files=[socat[1]],sst_dir=reanal[0][1],sst_tail=reanal[0][2],out_dir = reanal[0][3],
            force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[0][0],var='analysed_sst')
        # rean.reanalyse(socat_dir=socatflagE[0],socat_files=[socatflagE[1]],sst_dir=reanal[0][1],sst_tail=reanal[0][2],out_dir = reanal[0][3]+'_FlagE',
        #     force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[0][0]+'_FlagE',var='analysed_sst')
        #OISST v2.1
        # rean.reanalyse(socat_dir=socat[0],socat_files=[socat[1]],sst_dir=reanal[1][1],sst_tail=reanal[1][2],out_dir = reanal[1][3],
        #     force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[1][0],var='sst')
        # rean.reanalyse(socat_dir=socatflagE[0],socat_files=[socatflagE[1]],sst_dir=reanal[1][1],sst_tail=reanal[1][2],out_dir = reanal[1][3]+'_FlagE',
        #     force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[1][0]+'_FlagE',var='sst')

        # rean.reanalyse(socat_dir=socat[0],socat_files=[socat[1]],sst_dir=reanal[2][1],sst_tail=reanal[2][2],out_dir = reanal[2][3],
        #     force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[2][0],var='analysed_sst')
        #
        # rean.reanalyse(socat_dir=socat[0],socat_files=[socat[1]],sst_dir=reanal[3][1],sst_tail=reanal[3][2],out_dir = reanal[3][3],
        #     force_reanalyse=False,start_yr = start_yr,end_yr = end_yr,outfile=data_file,name=reanal[3][0],var='analysed_sst')

    model_save_loc = 'D:/ESA_CONTRACT/NN/NEW_TESTING'
    if run_neural:
        import neural_network_train as nnt
        # nnt.driver(data_file,fco2_sst = 'CCI_SST', prov = 'Watson_biome',var = ['CCI_SST_analysed_sst','NOAA_ERSL_xCO2','CMEMS_so','CMEMS_mlotst','CCI_SST_analysed_sst_anom','NOAA_ERSL_xCO2_anom','CMEMS_so_anom','CMEMS_mlotst_anom'],
        #   model_save_loc = model_save_loc +'/',unc =[0.4,1,0.1,0.05,0.4,1,0.1,0.05],bath = 'GEBCO_elevation',bath_cutoff = None,fco2_cutoff_low = 50,fco2_cutoff_high = 750,sea_ice = 'CCI_SST_sea_ice_fraction')
        nnt.plot_total_validation_unc(input_file = data_file,fco2_sst = 'CCI_SST',model_save_loc = model_save_loc,ice = 'CCI_SST_sea_ice_fraction')
    if run_flux:
        import fluxengine_driver as fl
        print('Running flux calculations....')
        fl.GCB_remove_prov17(model_save_loc)
        fl.fluxengine_netcdf_create(model_save_loc,input_file = data_file,tsub='CCI_SST_analysed_sst',ws = 'CCMP_w',seaice = 'CCI_SST_sea_ice_fraction',
            sal='CMEMS_so',msl = 'ERA5_msl',xCO2 = 'NOAA_ERSL_xCO2',start_yr=start_yr,end_yr=end_yr, coare_out = coare_out, tair = 'ERA5_t2m', dewair = 'ERA5_d2m',
            rs = 'ERA5_msdwswrf', rl = 'ERA5_msdwlwrf', zi = 'ERA5_blh',coolskin = 'COARE3.5')
        fl.fluxengine_run(model_save_loc,fluxengine_config,start_yr,end_yr)
        import custom_flux_av.ofluxghg_flux_budgets as bud
        bud.run_flux_budgets(indir = os.path.join(model_save_loc,'flux'),outroot=model_save_loc+'/')

    if plot_final:
        import fluxengine_driver as fl
        #fl.plot_annual_flux('temporal_flux.png',['D:/ESA_CONTRACT/NN/xCO2_SST','D:/ESA_CONTRACT/NN/xCO2_SST_SSS','D:/ESA_CONTRACT/NN/xCO2_SST_SSS_MLD','D:/ESA_CONTRACT/NN/GCB_2023_Watson'],['xCO2_atm+CCI_SST','xCO2_atm+CCI_SST+CMEMS_SSS','xCO2_atm+CCI_SST+CMEMS_SSS+CMEMS_MLD','GCB_Watson_2023'])
        # fl.plot_annual_flux('GCB_submissions.png',['D:/ESA_CONTRACT/NN/GCB_2023_Watson','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens5','D:/ESA_CONTRACT/NN/GCB_2023_CCISSTv3','D:/ESA_CONTRACT/NN/GCB_2023_Prelim'],
        #     ['GCB_Watson_2023_CCI_SSTv2','GCB_Watson_2023_OISST','GCB_Watson_2023_CCI_SSTv3','GCB_2023_Prelim (SOCATv2023)'])
        fl.plot_annual_flux('temporal_flux.png',['D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens3','D:/ESA_CONTRACT/NN/NEW_TESTING'],['Watson et al. GCB submission','Ford nerual network version'])
