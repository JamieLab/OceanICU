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
        ['pH','pH','-log([H+])','pH on total scale',''],
        ['u_pH','pH_tot_unc','-log([H+])','pH on total scale total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__par2','pH_ta_unc','-log([H+])','pH on total scale alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__par1','pH_fco2_unc','-log([H+])','pH on total scale fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__temperature','pH_sst_unc','-log([H+])','pH on total scale SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__salinity','pH_sss_unc','-log([H+])','pH on total scale SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__total_phosphate','pH_phos_unc','-log([H+])','pH on total scale phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_pH__total_silicate','pH_sili_unc','-log([H+])','pH on total scale silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['saturation_aragonite','saturation_aragonite','unitless','Saturation state of Aragonite',''],
        ['u_saturation_aragonite','saturation_aragonite_tot_unc','unitless','Saturation state of Aragonite total uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__par2','saturation_aragonite_ta_unc','unitless','Saturation state of Aragonite alkalinity uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__par1','saturation_aragonite_fco2_unc','unitless','Saturation state of Aragonite fCO2(sw) uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__temperature','saturation_aragonite_sst_unc','unitless','Saturation state of Aragonite SST uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__salinity','saturation_aragonite_sss_unc','unitless','Saturation state of Aragonite SSS uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__total_phosphate','saturation_aragonite_phos_unc','unitless','Saturation state of Aragonite phosphate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
        ['u_saturation_aragonite__total_silicate','saturation_aragonite_sili_unc','unitless','Saturation state of Aragonite silicate uncertainty','Uncertainties considered 95% confidence (2 sigma)'],
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
            uncertainty_into = ['dic','pH','saturation_aragonite'],
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
            worldmap.plot(color="lightgrey", ax=axs[i])
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
    fig.savefig(os.path.join(model_save_loc,'plots','carbonate_system.png'),dpi=300)
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
