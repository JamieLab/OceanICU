#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script takes SOCAT tsv files for both the main and FlagE datasets and produces a reanlysed fCO2
dataset within the neural network input netcdf. Currently hardcoded to 1deg outputs (working to modify...)
"""

import custom_reanalysis.reanalyse_socat_v2 as rean
import shutil
import os
import datetime
from netCDF4 import Dataset
import Data_Loading.data_utils as du
import numpy as np

def reanalyse(socat_dir=None,socat_files=None,sst_dir=None,sst_tail=None,out_dir=None,force_reanalyse=False,
    start_yr = 1990,end_yr = 2020,name = '',outfile = None, var = None):
    """
    Function to run the fluxengine reanalysis code in custom_reanalysis. DJF to put a pull request into FluxEngine with
    the updates made within the custom code.
    """
    #if __name__ == '__main__':
    if not os.path.exists(out_dir):
        print('No reanalysis done previously - running reanalysis...')
        rean.RunReanalyseSocat(socatdir=socat_dir, socatfiles=socat_files, sstdir=sst_dir, ssttail=sst_tail, startyr=1985,
                          endyr=2023,socatversion=2020,regions=["GL"],customsst=True,
                          customsstvar=var,customsstlat='latitude',customsstlon='longitude'
                          ,output=out_dir)
    elif force_reanalyse:
        print('Forcing Reanalysis....')
        shutil.rmtree(out_dir)
        rean.RunReanalyseSocat(socatdir=socat_dir, socatfiles=socat_files, sstdir=sst_dir, ssttail=sst_tail, startyr=1985,
                          endyr=2023,socatversion=2020,regions=["GL"],customsst=True,
                          customsstvar=var,customsstlat='latitude',customsstlon='longitude'
                          ,output=out_dir)
    fco2,fco2_std = retrieve_fco2(out_dir,start_yr = start_yr,end_yr=end_yr)
    append_to_file(outfile,fco2,fco2_std,name,socat_files[0])

def load_prereanalysed(input_file,output_file,start_yr=1990, end_yr = 2020,name=''):
    """
    Function loads the reanlysed fCO2 data produced by JamieLab, instead of running the reanalysis process.
    """
    c = Dataset(input_file)
    ti = np.array(c.variables['tmnth'])
    time = []
    for t in ti:
        time.append((datetime.datetime(1970,1,1) + datetime.timedelta(days=int(t))).year)
    time = np.array(time)
    #print(time)
    fco2 = np.array(c.variables['fco2_reanalysed_ave_weighted'])
    fco2_std = np.array(c.variables['fco2_reanalysed_std_weighted'])
    c.close()
    f = np.where((time <= end_yr) & (time >= start_yr))
    #print(fco2.shape)
    fco2 = np.transpose(fco2[f[0],:,:],(2,1,0))
    fco2_std = np.transpose(fco2_std[f[0],:,:],(2,1,0))

    fco2[fco2<0] = np.nan
    fco2_std[fco2_std<0] = np.nan
    #print(fco2.shape)
    append_to_file(output_file,fco2,fco2_std,name,input_file)

def append_to_file(output_file,fco2,fco2_std,name,socat_files):
    """
    Function to append the reanalysed fCO2 to the neural network input file
    """
    c = Dataset(output_file,'a',format='NETCDF4_CLASSIC')
    var_o = c.createVariable(name+'_reanalysed_fCO2_sw','f4',('longitude','latitude','time'))
    var_o[:] = fco2
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files

    var_o = c.createVariable(name+'_reanalysed_fCO2_sw_std','f4',('longitude','latitude','time'))
    var_o[:] = fco2_std
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) std using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files
    c.close()

def retrieve_fco2(rean_dir,start_yr=1990,end_yr=2020):
    """
    Function to iteratively load the fCO2sw from the reanalysis folder (i.e one netcdf per month_year combo)
    """
    print('Retrieving fCO2 from reanlysis')
    months = (end_yr-start_yr+1)*12
    yr = start_yr
    mon = 1
    t = 0
    inits = 0
    while yr <= end_yr:
        da = datetime.datetime(yr,mon,1)
        loc = os.path.join(rean_dir,'reanalysed_global',da.strftime('%m'),da.strftime('%Y%m01-OCF-CO2-GLO-1M-100-SOCAT-CONV.nc'))
        print(loc)
        if du.checkfileexist(loc):
            #print('True')
            c = Dataset(loc,'r')
            fco2_temp = np.squeeze(c.variables['weighted_fCO2_Tym'])
            if fco2_temp.shape[0] < fco2_temp.shape[1]:
                fco2_temp = np.fliplr(np.transpose(fco2_temp))

            fco2_std_temp = np.squeeze(c.variables['stdev_fCO2_Tym'])
            if fco2_std_temp.shape[0] < fco2_std_temp.shape[1]:
                fco2_std_temp = np.fliplr(np.transpose(fco2_std_temp))
            c.close()
            if inits == 0:
                fco2 = np.empty((fco2_temp.shape[0],fco2_temp.shape[1],months))
                fco2[:] = np.nan
                fco2_std = np.copy(fco2)
                inits = 1
            fco2[:,:,t] = fco2_temp
            fco2_std[:,:,t] = fco2_std_temp
        t += 1
        mon += 1
        if mon == 13:
            yr += 1
            mon=1
    fco2[fco2<0] = np.nan
    fco2_std[fco2_std<0] = np.nan
    return fco2,fco2_std
