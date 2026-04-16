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
from matplotlib.gridspec import GridSpec
import matplotlib.transforms
import Data_Loading.data_utils as du
import PyCO2SYS as pyco2
import construct_input_netcdf as cinp
import geopandas as gpd
import cmocean
import weight_stats as ws
import pandas as pd

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
    var_out = [['dic','dic','umol/kg','Dissolved Inorganic Carbon in seawater',''],
        ['u_dic','dic_tot_unc','umol/kg','Dissolved Inorganic Carbon in seawater total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__par2','dic_ta_unc','umol/kg','Dissolved Inorganic Carbon in seawater alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__par1','dic_fco2_unc','umol/kg','Dissolved Inorganic Carbon in seawater fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__temperature','dic_sst_unc','umol/kg','Dissolved Inorganic Carbon in seawater SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__salinity','dic_sss_unc','umol/kg','Dissolved Inorganic Carbon in seawater SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__total_phosphate','dic_phos_unc','umol/kg','Dissolved Inorganic Carbon in seawater phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_dic__total_silicate','dic_sili_unc','umol/kg','Dissolved Inorganic Carbon in seawater silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['pH','pH','-log([H+] + [HSO4-])','pH on total scale',''],
        ['u_pH','pH_tot_unc','-log([H+] + [HSO4-])','pH on total scale total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__par2','pH_ta_unc','-log([H+] + [HSO4-])','pH on total scale alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__par1','pH_fco2_unc','-log([H+] + [HSO4-])','pH on total scale fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__temperature','pH_sst_unc','-log([H+] + [HSO4-])','pH on total scale SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__salinity','pH_sss_unc','-log([H+] + [HSO4-])','pH on total scale SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__total_phosphate','pH_phos_unc','-log([H+] + [HSO4-])','pH on total scale phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__total_silicate','pH_sili_unc','-log([H+] + [HSO4-])','pH on total scale silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['saturation_aragonite','saturation_aragonite','unitless','Saturation state of Aragonite',''],
        ['u_saturation_aragonite','saturation_aragonite_tot_unc','unitless','Saturation state of Aragonite total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__par2','saturation_aragonite_ta_unc','unitless','Saturation state of Aragonite alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__par1','saturation_aragonite_fco2_unc','unitless','Saturation state of Aragonite fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__temperature','saturation_aragonite_sst_unc','unitless','Saturation state of Aragonite SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__salinity','saturation_aragonite_sss_unc','unitless','Saturation state of Aragonite SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__total_phosphate','saturation_aragonite_phos_unc','unitless','Saturation state of Aragonite phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__total_silicate','saturation_aragonite_sili_unc','unitless','Saturation state of Aragonite silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['pH_free','pH_free','-log([H+])','pH on free scale',''],
        ['u_pH_free','pH_free_tot_unc','-log([H+])','pH on free scale total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__par2','pH_free_ta_unc','-log([H+])','pH on free scale alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__par1','pH_free_fco2_unc','-log([H+])','pH on free scale fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__temperature','pH_free_sst_unc','-log([H+])','pH on free scale SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__salinity','pH_free_sss_unc','-log([H+])','pH on free scale SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__total_phosphate','pH_free_phos_unc','-log([H+])','pH on free scale phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH_free__total_silicate','pH_free_sili_unc','-log([H+])','pH on free scale silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
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
    comment = {}
    for a in var_out:
        direct[a[1]] = np.zeros((sss.shape)); direct[a[1]][:] = np.nan
        units[a[1]] = a[2]
        longname[a[1]] = a[3]
        comment[a[1]] = a[4]
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
            uncertainty_into = ['dic','pH','saturation_aragonite','pH_free'],
            uncertainty_from = {'par2':ta_unc[:,:,i], 'par1':fco2_unc[:,:,i], 'temperature': sst_unc[:,:,i],'salinity': sss_unc[:,:,i],'total_phosphate': phosphate_unc[:,:,i],
                'total_silicate':silicate_unc[:,:,i]})
        #print(py_out)
        for a in var_out:
            direct[a[1]][:,:,i] = py_out[a[0]]


    cinp.append_netcdf(data_file,direct,1,1,1,units=units,longname=longname,comment=comment)

def plot_pyCO2sys_out(data_file,model_save_loc,geopan = True):


    font = {'weight' : 'normal',
            'size'   :30}
    matplotlib.rc('font', **font)
    label_size = 38
    if geopan:
        worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig = plt.figure(figsize=(40,35))
    row = 4; col=2;
    gs = GridSpec(row,col, figure=fig, wspace=0.10,hspace=0.15,bottom=0.025,top=0.975,left=0.05,right=0.975)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]).ravel()
    print(axs.shape)
    if geopan:
        for i in range(len(axs)):
            worldmap.plot(color="lightgrey", ax=axs[i],zorder=2)
    c = Dataset(data_file,'r')
    lon = np.array(c['longitude'])
    lat = np.array(c['latitude'])

    fco2 = np.nanmean(c['fco2'],axis=2)
    a = axs[0].pcolor(lon,lat,np.transpose(fco2),vmin=280,vmax =500,cmap=cmocean.cm.balance)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ ($\mu$atm)',fontsize = label_size)

    fco2 = np.nanmean(c['fco2_tot_unc'],axis=2)
    a = axs[1].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =80,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ total\nuncertainty ($\mu$atm)',fontsize = label_size)

    fco2 = np.nanmean(c['ta'],axis=2)
    a = axs[2].pcolor(lon,lat,np.transpose(fco2),vmin=2000,vmax =2600,cmap=cmocean.cm.haline)
    cbar = fig.colorbar(a); cbar.set_label('TA ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['ta_tot_unc'],axis=2)
    a = axs[3].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =100,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('TA total\nuncertainty ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['dic'],axis=2)
    a = axs[4].pcolor(lon,lat,np.transpose(fco2),vmin=1900,vmax =2400,cmap=cmocean.cm.deep)
    cbar = fig.colorbar(a); cbar.set_label('DIC ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['dic_tot_unc'],axis=2)
    a = axs[5].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =100,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('DIC total\nuncertainty ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['pH'],axis=2)
    a = axs[6].pcolor(lon,lat,np.transpose(fco2),vmin=7.9,vmax =8.3,cmap=cmocean.cm.matter)
    cbar = fig.colorbar(a); cbar.set_label('pH (total scale)',fontsize = label_size)

    fco2 = np.nanmean(c['pH_tot_unc'],axis=2)
    a = axs[7].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =0.2,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('pH total\nuncertainty (total scale)',fontsize = label_size)

    c.close()
    fig.savefig(os.path.join(model_save_loc,'plots','carbonate_system.png'),dpi=150)
    plt.close(fig)

def plot_pyCO2sys_out_flux(data_file,model_save_loc,geopan = True):


    font = {'weight' : 'normal',
            'size'   :30}
    matplotlib.rc('font', **font)
    label_size = 38
    if geopan:
        worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig = plt.figure(figsize=(40,42))
    row = 5; col=2;
    gs = GridSpec(row,col, figure=fig, wspace=0.10,hspace=0.15,bottom=0.025,top=0.975,left=0.05,right=0.975)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]).ravel()
    print(axs.shape)
    if geopan:
        for i in range(len(axs)):
            worldmap.plot(color="lightgrey", ax=axs[i],zorder=2)
    c = Dataset(data_file,'r')
    lon = np.array(c['longitude'])
    lat = np.array(c['latitude'])

    fco2 = np.nanmean(c['fco2'],axis=2)
    a = axs[0].pcolor(lon,lat,np.transpose(fco2),vmin=280,vmax =500,cmap=cmocean.cm.balance)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ ($\mu$atm)',fontsize = label_size)

    fco2 = np.nanmean(c['fco2_tot_unc'],axis=2)
    a = axs[1].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =80,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('fCO$_{2}$ $_{(sw)}$ total\nuncertainty ($\mu$atm)',fontsize = label_size)

    fco2 = np.nanmean(c['ta'],axis=2)
    a = axs[2].pcolor(lon,lat,np.transpose(fco2),vmin=2000,vmax =2600,cmap=cmocean.cm.haline)
    cbar = fig.colorbar(a); cbar.set_label('TA ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['ta_tot_unc'],axis=2)
    a = axs[3].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =100,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('TA total\nuncertainty ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['dic'],axis=2)
    a = axs[4].pcolor(lon,lat,np.transpose(fco2),vmin=1900,vmax =2400,cmap=cmocean.cm.deep)
    cbar = fig.colorbar(a); cbar.set_label('DIC ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['dic_tot_unc'],axis=2)
    a = axs[5].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =100,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('DIC total\nuncertainty ($\mu$mol kg$^{-1}$)',fontsize = label_size)

    fco2 = np.nanmean(c['pH'],axis=2)
    a = axs[6].pcolor(lon,lat,np.transpose(fco2),vmin=7.9,vmax =8.3,cmap=cmocean.cm.matter)
    cbar = fig.colorbar(a); cbar.set_label('pH (total scale)',fontsize = label_size)

    fco2 = np.nanmean(c['pH_tot_unc'],axis=2)
    a = axs[7].pcolor(lon,lat,np.transpose(fco2),vmin=0,vmax =0.2,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('pH total\nuncertainty (total scale)',fontsize = label_size)

    fco2 = np.nanmean(c['flux'],axis=2)
    a = axs[8].pcolor(lon,lat,np.transpose(fco2),vmin=-0.2,vmax =0.2,cmap=cmocean.cm.balance)
    cbar = fig.colorbar(a); cbar.set_label('Air-sea CO$_2$ flux\n(g C m$^{-2}$ d$^{-1}$)',fontsize = label_size)

    fco22 = np.nanmean(c['flux_unc'] * np.abs(c['flux']),axis=2)
    a = axs[9].pcolor(lon,lat,np.transpose(fco22),vmin=0,vmax =0.2,cmap=cmocean.cm.thermal)
    cbar = fig.colorbar(a); cbar.set_label('Air-sea CO$_2$ flux\n uncertainty (g C m$^{-2}$ d$^{-1}$)',fontsize = label_size)
    c.close()
    fig.savefig(os.path.join(model_save_loc,'plots','carbonate_system+flux.png'),dpi=150)
    plt.close(fig)

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

def load_DIVA_glodap(file,delimiter,out_file,log,lag,var_name='ta',inp_name = 'Estimated G2talk @ G2talk=first'):
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

    var = np.transpose(np.array(data[inp_name]).reshape((len(g),len(f))))
    var[var==-1.000000e+10] = np.nan
    print(np.nanmin(var))
    print(var.shape)
    lat,lon = np.meshgrid(g['Latitude'],f['Longitude'])
    print(lon.shape)
    out = griddata((lon.ravel(),lat.ravel()), var.ravel(), (log_o.ravel(), lag_o.ravel()), method='linear').reshape(log_o.shape)
    netcdf_create_basic(out_file,out,var_name,lag,log)

def plot_carbonate_validation(model_save_loc,insitu_file,insitu_var,nn_file,nn_var,lims = [1200,2400],var_name = 'DIC',unit='umol kg$^{-1}$',vma = [-40,40],geopan=True):
    from neural_network_train import unweight
    font = {'weight' : 'normal',
            'size'   :14}
    matplotlib.rc('font', **font)
    c = Dataset(insitu_file,'r')
    insitu = np.array(c[insitu_var])
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    c.close()

    c = Dataset(nn_file,'r')
    nn = np.array(c[nn_var])
    c.close()

    fig = plt.figure(figsize=(21,7))
    gs = GridSpec(1,3, figure=fig, wspace=0.25,hspace=0.25,bottom=0.1,top=0.95,left=0.05,right=0.95)
    ax = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:])

    ax.scatter(insitu,nn,s=2)
    unweight(insitu.ravel(),nn.ravel(),ax,np.array(lims),unit = unit,plot=True,loc = [0.52,0.26])
    stats_temp = ws.unweighted_stats(insitu.ravel(),nn.ravel(),'b')
    lims = np.array(lims)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.fill_between(lims,lims-stats_temp['rmsd'],lims+stats_temp['rmsd'],color='k',alpha=0.6,linestyle='')
    ax.fill_between(lims,lims-(stats_temp['rmsd']*2),lims+(stats_temp['rmsd']*2),color='k',alpha=0.4,linestyle='')
    ax.set_ylabel('UExP-FNN-U ' + var_name +' (' + unit +')')
    ax.set_xlabel('in situ ' + var_name +' (' + unit +')')
    ax.plot(lims,lims,'k-')
    if geopan:
        worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        worldmap.plot(color="lightgrey", ax=ax2)
    dif = np.nanmean(insitu - nn,axis=2)

    a = ax2.pcolor(lon,lat,np.transpose(dif),cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 20, which='both', N=None),vmin = vma[0],vmax=vma[1])
    cbar = fig.colorbar(a); cbar.set_label(var_name + ' Bias (' + unit+')\n (in situ - nerual network)')
    fig.savefig(os.path.join(model_save_loc,'plots',var_name+'_validation_bias.png'),dpi=300)
    plt.close(fig)


def montecarlo_mean_testing(model_save_loc,start_yr = 1985,end_yr = 2022,decor=[2000,200],flux_var = '',flux_variable='flux',
    inp_file=False,single_output=False,ens=100,bath_cutoff=False,output_file = 'annual_flux.csv',mask_file=False,mask_var = '',):
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

    # We have our default location for the output file (i.e where all the output data from the Nerual network and flux uncertainty computation per pixel)
    # but we can specify here if this file is different
    if inp_file:
        c = Dataset(inp_file,'r')
    else:
        c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
    c_flux = np.array(c.variables[flux_variable])
    print(c_flux.shape)
    c_flux_unc = np.array(c.variables[flux_var])

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
    c.close()
    if bath_cutoff:
        c=Dataset(bath_file,'r')
        elev=  np.squeeze(np.array(c.variables[bath_var]))
        c.close()
    print(land.shape)

    if mask_file:
        c = Dataset(mask_file,'r')
        mask = np.array(c.variables[mask_var])
        time_mask = np.array(c.variables['time'])
        c.close()
        f = np.where(time[0] == time_mask)[0]
        f2 = np.where(time[-1] == time_mask)[0]
        print(f)
        print(f2)
        mask = mask[:,:,int(f):int(f2+1)]
    # Here we cycle through the time dimension (third dimension) and set the areas where the water is deeper than the bathymetry cutoff to nan
    # then save back to the flux array.
    # Only applied if bath_cutoff is specified
    if bath_cutoff:
        for i in range(c_flux.shape[2]):
            flu = c_flux[:,:,i]  ; flu[elev<=bath_cutoff] = np.nan; c_flux[:,:,i] = flu
    if mask_file:
        flu = c_flux; flu[mask!=1.0] = np.nan; c_flux = flu
    #fco2_tot_unc =  fco2_tot_unc[:,:,:,np.newaxis]
    # Get a list of the years we are computing the fluxes for
    a = list(range(start_yr,end_yr+1))

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
            decor_loaded = np.loadtxt(decor,delimiter=',') # And then if a actual path is specified the first call will fail and then load it from the absolute path supplied#

        #Extracts the decorrelation lenght between the start and end years (so allows a decorrelation file with a longer time frame to be used)
        f = np.where(decor_loaded[:,0] == start_yr)[0]
        g = np.where(decor_loaded[:,0] == end_yr)[0]
        print(f)
        print(g)
        decors[:,0] = decor_loaded[f[0]:g[0]+1,1] # Median decorrelation length loaded

        decors[:,1] = decor_loaded[f[0]:g[0]+1,2] # IQR is left as is - we want 2 sigma equivalent, so I'd divide by 2 to get the IQR as a +-, then times by 2 to get 2 sigma equivalent.

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


        # Were dealing with a precomputed flux uncertainty, so we multiple by the perturbation vlaue and add to the flux
        e_flux = c_flux + (unc*c_flux_unc)
        temp_area = np.repeat((land*area)[:, :, np.newaxis], 12, axis=2)
        t = 0
        for i in range(0,c_flux.shape[2],12):# Cycle through the flux array, in annual increments and sum the flux thats in Pg C mon-1 into a Pg C yr-1
            flu = e_flux[:,:,i:i+12] #Extract the years values
            # print(flu)
            f = np.where(np.isnan(flu) == 0)
            # print(f)
            print(np.average(flu[f],weights=temp_area[f]))
            c_flu = c_flux[:,:,i:i+12] # Extract the unperturbed values - I think this only applies when we are doing a single output... so we have the actual calulated flux and the uncertainty
            out[t,j] = np.average(flu[f],weights=temp_area[f])
            out2[t] = np.average(c_flu[f],weights=temp_area[f])

            t = t+1# Keep a count so we can put the annual values in the right output row

    data = pd.DataFrame(a,columns=['Year']) # So we save the year
    st = np.std(out,axis=1) # Calcualte the standard devaiiton of the ensembles
    data['std'] = st# Save these into the table
    data[flux_variable] = out2 # This is our calcualted ocena carbon sink with no perturbations (so flux as is calculated)
    data.to_csv(os.path.join(model_save_loc,output_file),index=False,na_rep='nan') # save the data back on to the file.
