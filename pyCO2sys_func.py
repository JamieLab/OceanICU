#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 04/2024
Adding functionality to calculate the whole carbonate system from fCO2(sw) and Total Alkalinity with propagated uncertainties
"""
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import Data_Loading.data_utils as du
import PyCO2SYS as pyco2
import construct_input_netcdf as cinp

def run_pyCO2sys(data_file,aux_file,fco2_var='fco2',ta_var = 'ta',sst_var = 'CCI_SST_analysed_sst',sst_kelvin = True,sss_var = 'CMEMS_so',sst_var_unc = 'CCI_SST_analysed_sst_uncertainty',
    sss_var_unc = 0.1,phosphate_var=False,phosphate_var_unc=False,phosphate_unc_perc=False,silicate_var=False,silicate_var_unc=False,
    silicate_unc_perc=False):
    """
    data_file: The file containing the fCO2(sw) and Total Alkalinity
    aux_file: The file containing the SST and SSS data
    fco2_var: variable name in data_file for fCO2(sw)
    ta_var: Variable name in data_file for TA
    sst_var: Variable name in aux_file for SST
    sst_kelvin: Is the sst_var is in kelvin (If True, then we convert to degC)
    sss_var: Variable name in aux_file for SSS
    """
    var_out = [['dic','dic','umol/kg','Dissolved Inorganic Carbon in seawater'],
        ['u_dic','dic_tot_unc','umol/kg','Dissolved Inorganic Carbon in seawater total uncertainty'],
        ['u_dic__par2','dic_ta_unc','umol/kg','Dissolved Inorganic Carbon in seawater alkalinity uncertainty'],
        ['u_dic__par1','dic_fco2_unc','umol/kg','Dissolved Inorganic Carbon in seawater fCO2(sw) uncertainty'],
        ['u_dic__temperature','dic_sst_unc','umol/kg','Dissolved Inorganic Carbon in seawater SST uncertainty'],
        ['u_dic__salinity','dic_sss_unc','umol/kg','Dissolved Inorganic Carbon in seawater SSS uncertainty'],
        ['u_dic__total_phosphate','dic_phos_unc','umol/kg','Dissolved Inorganic Carbon in seawater phosphate uncertainty'],
        ['u_dic__total_silicate','dic_sili_unc','umol/kg','Dissolved Inorganic Carbon in seawater silicate uncertainty'],
        ['pH','pH','-log([H+])','pH on total scale'],
        ['u_pH','pH_tot_unc','-log([H+])','pH on total scale total uncertainty'],
        ['u_pH__par2','pH_ta_unc','-log([H+])','pH on total scale alkalinity uncertainty'],
        ['u_pH__par1','pH_fco2_unc','-log([H+])','pH on total scale fCO2(sw) uncertainty'],
        ['u_pH__temperature','pH_sst_unc','-log([H+])','pH on total scale SST uncertainty'],
        ['u_pH__salinity','pH_sss_unc','-log([H+])','pH on total scale SSS uncertainty'],
        ['u_pH__total_phosphate','pH_phos_unc','-log([H+])','pH on total scale phosphate uncertainty'],
        ['u_pH__total_silicate','pH_sili_unc','-log([H+])','pH on total scale silicate uncertainty']
        ]
    # var_out_list = ['dic',
    #     'u_dic',
    #     'u_dic__par2',
    #     'u_dic__par1',
    #     'u_dic__temperature',
    #     'u_dic__salinity',
    #     'pH',
    #     'u_pH',
    #     'u_pH__par2',
    #     'u_pH__par1',
    #     'u_pH__temperature',
    #     'u_pH__salinity']
    # var_renamed_list = ['dic',
    #     'dic_tot_unc',
    #     'dic_ta_unc',
    #     'dic_fco2_unc',
    #     'dic_sst_unc',
    #     'dic_sss_unc',
    #     'pH',
    #     'pH_tot_unc',
    #     'pH_ta_unc',
    #     'pH_fco2_unc',
    #     'pH_sst_unc',
    #     'pH_sss_unc']
    print('Running pyCO2sys...')
    print('Loading fCO2(sw) and TA data...')
    c = Dataset(data_file,'r')
    fco2 = np.array(c[fco2_var])
    ta = np.array(c[ta_var])
    fco2_unc = np.array(c[fco2_var+'_tot_unc'])
    ta_unc = np.array(c[ta_var+'_tot_unc'])
    c.close()
    print('Loading SST and SSS data...')
    c = Dataset(aux_file,'r')
    sst = np.array(c[sst_var])
    if sst_kelvin:
        sst = sst - 273.15
    sss = np.array(c[sss_var])
    if sst_var_unc:
        if isinstance(sst_var_unc,str):
            sst_unc = np.array(c[sst_var_unc])
        elif isinstance(sst_var_unc,float):
            sst_unc = np.ones((sst.shape))*sst_var_unc
    else:
        sst_unc = np.zeros((sst.shape))

    if sss_var_unc:
        if isinstance(sss_var_unc,str):
            sss_unc = np.array(c[sss_var_unc])
        elif isinstance(sss_var_unc,float):
            sss_unc = np.ones((sss.shape))*sss_var_unc
    else:
        sss_unc = np.zeros((sss.shape))
    """
    Phosphate variables (if not avaiable everything set to 0; 0 is default for pyCO2sys)
    """
    if phosphate_var:
        phosphate = np.array(c[phosphate_var])
        if isinstance(phosphate_var_unc,str):
            phosphate_unc = np.array(c[phosphate_var_unc])
        elif isinstance(phosphate_var_unc,float):
            if phosphate_unc_perc:
                phosphate_unc = phosphate*phosphate_var_unc
            else:
                phosphate_unc = np.ones((phosphate.shape))*phosphate_var_unc
        else:
            phosphate_unc = np.zeros((sss.shape))
    else:
        phosphate = np.zeros((sss.shape))
        phosphate_unc = np.zeros((sss.shape))

    """
    Silicate variables (if not avaiable everything set to 0; 0 is default for pyCO2sys)
    """
    if silicate_var:
        silicate = np.array(c[silicate_var])
        if isinstance(silicate_var_unc,str):
            silicate_unc = np.array(c[silicate_var_unc])
        elif isinstance(silicate_var_unc,float):
            if silicate_unc_perc:
                silicate_unc = silicate*silicate_var_unc
            else:
                silicate_unc = np.ones((silicate.shape))*silicate_var_unc
        else:
            silicate_unc = np.zeros((sss.shape))
    else:
        silicate = np.zeros((sss.shape))
        silicate_unc = np.zeros((sss.shape))

    c.close()
    direct = {}
    units = {}
    longname = {}
    for a in var_out:
        direct[a[1]] = np.zeros((sss.shape)); direct[a[1]][:] = np.nan
        units[a[1]] = a[2]
        longname[a[1]] = a[3]
    print('Running CO2sys...')
    for i in range(sst.shape[2]):
        print('Timestep: '+str(i))

        py_out = pyco2.sys(par1 = fco2[:,:,i],
            par2 = ta[:,:,i],
            par1_type = 5,
            par2_type = 1,
            temperature=sst[:,:,i],
            salinity = sss[:,:,i],
            total_phosphate=phosphate[:,:,i],
            total_silicate=silicate[:,:,i],
            uncertainty_into = ['dic','pH'],
            uncertainty_from = {'par2':ta_unc[:,:,i], 'par1':fco2_unc[:,:,i], 'temperature': sst_unc[:,:,i],'salinity': sss_unc[:,:,i],'total_phosphate': phosphate_unc[:,:,i],
                'total_silicate':silicate_unc[:,:,i]})
        for a in var_out:
            direct[a[1]][:,:,i] = py_out[a[0]]


    cinp.append_netcdf(data_file,direct,1,1,1,units=units,longname=longname)

def plot_pyCO2sys_out(data_file,model_save_loc):
    import geopandas as gpd
    from matplotlib.gridspec import GridSpec
    import cmocean
    import matplotlib.transforms
    font = {'weight' : 'normal',
            'size'   :26}
    matplotlib.rc('font', **font)
    worldmap = gpd.read_file(gpd.datasets.get_path("ne_50m_land"))
    fig = plt.figure(figsize=(40,35))
    row = 4; col=2;
    gs = GridSpec(row,col, figure=fig, wspace=0.10,hspace=0.15,bottom=0.025,top=0.975,left=0.05,right=0.975)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]).ravel()
    print(axs.shape)
    for i in range(len(axs)):
        worldmap.plot(color="lightgrey", ax=axs[i])
    c = Dataset(data_file,'r')
    lon = np.array(c['longitude'])
    lat = np.array(c['latitude'])

    fco2 = np.nanmean(c['fco2'],axis=2)
    a = axs[0].pcolor(lon,lat,np.transpose(fco2),vmin=280,vmax =500,cmap=cmocean.cm.balance)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ ($\mu$atm)')

    fco2 = np.nanmean(c['fco2_tot_unc'],axis=2)
    a = axs[1].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =50,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ total uncertainty ($\mu$atm)')

    fco2 = np.nanmean(c['ta'],axis=2)
    a = axs[2].pcolor(lon,lat,np.transpose(fco2),vmin=2000,vmax =2600,cmap=cmocean.cm.haline)
    cbar = fig.colorbar(a); cbar.set_label('TA ($\mu$mol kg$^{-1}$)')

    fco2 = np.nanmean(c['ta_tot_unc'],axis=2)
    a = axs[3].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =50,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('TA total uncertainty ($\mu$mol kg$^{-1}$)')

    fco2 = np.nanmean(c['dic'],axis=2)
    a = axs[4].pcolor(lon,lat,np.transpose(fco2),vmin=1900,vmax =2400,cmap=cmocean.cm.deep)
    cbar = fig.colorbar(a); cbar.set_label('DIC ($\mu$mol kg$^{-1}$)')

    fco2 = np.nanmean(c['dic_tot_unc'],axis=2)
    a = axs[5].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =50,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('DIC total uncertainty ($\mu$mol kg$^{-1}$)')

    fco2 = np.nanmean(c['pH'],axis=2)
    a = axs[6].pcolor(lon,lat,np.transpose(fco2),vmin=7.9,vmax =8.3,cmap=cmocean.cm.matter)
    cbar = fig.colorbar(a); cbar.set_label('pH (total scale)')

    fco2 = np.nanmean(c['pH_tot_unc'],axis=2)
    a = axs[7].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =0.1,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('pH total uncertainty (total scale)')

    c.close()
    fig.savefig(os.path.join(model_save_loc,'plots','carbonate_system.png'),dpi=300)

def calc_nstar(file,delimiter,nitrate_var,phosphate_var,gridded=False,prefix=''):
    """
    This function
    """
    import pandas as pd
    import construct_input_netcdf as cinp
    def nstar(nitrate, phosphate):
        ns = (nitrate - 16*phosphate + 2.9)*0.87
        return ns

    if gridded:
        print('Gridded')
        c = Dataset(file,'r')
        nitrate = np.array(c[nitrate_var])
        phosphate = np.array(c[phosphate_var])
        c.close()
        ns = nstar(nitrate,phosphate)
        direct = {}
        direct['n_star'] = ns
        cinp.append_netcdf(file,direct,1,1,1)
    else:
        print('Text File based calcualtion')
        data = pd.read_table(file,sep=delimiter)
        nit = data[nitrate_var]; nit[nit<0] = np.nan
        phos = data[phosphate_var]; phos[phos<0] = np.nan
        data[prefix+'n*_Calculated'] = nstar(nit,phos)
        data.to_csv(file,sep=delimiter,index=False)

def load_DIVA_glodap(file,delimiter,out_file,log,lag,var_name='ta'):
    """
    This functions loads a DIVA interpolation of total alkalinity data that represents a quasi-annual grid - This
    DIVA interpolation must be manually done in Ocean Data Viewer, with the input glodap file for the nerual network.
    I.e we use all the surface data to produce the annual grid.
    After loading that file, we interpolate the values onto the grid resolution we are using. As this is only use to inform
    the SOM provinces. This then outputs into a netcdf file that can be ingested within the construct_input_netcdf framework.
    #
    file: txt file output from Ocean Data Viewer with the DIVA interpolated values.
    delimiter: is the delimiter of file (should be \t but depends on how it is manually exported)
    out_file: is the netcdf file for the interpolated data (on the latitude/longitude grid for the nerual network) to be output to.
    log: longitude of the neural network grid
    lag: latitude of the neural network grid
    var_name: variable name for the output data to be put under in the netcdf (default is 'ta')
    """
    from scipy.interpolate import griddata
    from Data_Loading.data_utils import netcdf_create_basic
    lag_o,log_o = np.meshgrid(lag,log)
    print(log_o.shape)
    import pandas as pd
    data = pd.read_table(file,sep=delimiter)
    print(data)
    f = data[data['Latitude']==-90.0]
    g = data[data['Longitude']==-180.0]

    var = np.transpose(np.array(data['Estimated G2talk @ G2talk=first']).reshape((len(g),len(f))))
    var[var==-1.000000e+10] = np.nan
    print(np.nanmin(var))
    print(var.shape)
    lat,lon = np.meshgrid(g['Latitude'],f['Longitude'])
    print(lon.shape)
    out = griddata((lon.ravel(),lat.ravel()), var.ravel(), (log_o.ravel(), lag_o.ravel()), method='linear').reshape(log_o.shape)
    netcdf_create_basic(out_file,out,var_name,lag,log)
