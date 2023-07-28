#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 04/2023

"""

import getpass
import motuclient
import datetime
import data_utils as du
import os
from netCDF4 import Dataset
import numpy as np
class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None

def motu_option_parser(script_template, usr, pwd, output_filename,output_directory,date_min,date_max):
    dictionary = dict(
        [e.strip().partition(" ")[::2] for e in script_template.split('--')])
    dictionary['variable'] = [value for (var, value) in [e.strip().partition(" ")[::2] for e in script_template.split('--')] if var == 'variable']  # pylint: disable=line-too-long
    for k, v in list(dictionary.items()):
        if v == '<OUTPUT_FILENAME>':
            dictionary[k] = output_filename
        if v == '<USERNAME>':
            dictionary[k] = usr
        if v == '<PASSWORD>':
            dictionary[k] = pwd
        if v == '<OUTPUT_DIRECTORY>':
            dictionary[k] = output_directory
        if v == '<DATEMIN>':
            dictionary[k] = date_min
        if v == '<DATEMAX>':
            dictionary[k] = date_max
        if k in ['longitude-min', 'longitude-max', 'latitude-min',
                 'latitude-max', 'depth-min', 'depth-max']:
            dictionary[k] = float(v)
        # if k in ['date-min', 'date-max']:
        #     dictionary[k] = v[1:-1]
        dictionary[k.replace('-','_')] = dictionary.pop(k)
    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    return dictionary

def cmems_sss_load(loc,start_yr = 1993,end_yr = 2020):
    """
    Reanalysis Dataset DOI: https://doi.org/10.48670/moi-00021
    Renalaysis/Forecast Dataset DOI: https://doi.org/10.48670/moi-00016
    """
    # Year reanalysis ends - check https://doi.org/10.48670/moi-00021 for year
    # After this year we use the forecast dataset from: https://doi.org/10.48670/moi-00016
    transition_yr = 2020
    script_template_fore = 'python -m motuclient \
    --motu https://my.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_MULTIYEAR_PHY_001_030-TDS \
    --product-id cmems_mod_glo_phy_my_0.083_P1M-m \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --depth-min 0.49402499198913574 --depth-max 0.49402499198913574 \
    --variable so \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'

    script_template_nrt = 'python -m motuclient \
    --motu https://nrt.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS \
    --product-id cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --depth-min 0.49402499198913574 --depth-max 0.49402499198913574 \
    --variable so \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'

    """
    Edit information below!
    """
    USERNAME = 'dford1'
    PASSWORD = 'Jokers!99.88\\'
    OUTPUT_DIRECTORY = loc

    yr = start_yr
    mon = 1
    while yr <= end_yr:
        date_min = datetime.datetime(yr,mon,1,0,0,0);
        date_max = datetime.datetime(yr,mon,27,23,59,59); date_max = date_max.strftime('%Y-%m-%d %H:%M:%S')
        OUTPUT_TEMP = os.path.join(OUTPUT_DIRECTORY,str(yr))
        du.makefolder(OUTPUT_TEMP)
        OUTPUT_FILENAME = date_min.strftime('%Y_%m_CMEMS_GLORYSV12_SSS.nc')
        date_min = date_min.strftime('%Y-%m-%d %H:%M:%S')
        print(OUTPUT_FILENAME)
        if not du.checkfileexist(os.path.join(OUTPUT_TEMP,OUTPUT_FILENAME)):
            if yr > transition_yr:
                script_template = script_template_nrt
            else:
                script_template = script_template_fore
            data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME,OUTPUT_TEMP,date_min,date_max)
            motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_automated))
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def cmems_average_sss(loc,outloc,start_yr=1990,end_yr=2023,res=1):
    du.makefolder(outloc)
    log,lag = du.reg_grid(lon=res,lat=res)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CMEMS_GLORYSV12_SSS.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CMEMS_GLORYSV12_SSS_'+str(res)+'_deg.nc')
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables['so'][:])))

            va_da[va_da < 0.0] = np.nan
            va_da[va_da > 60.0] = np.nan
            #print(va_da)
            c.close()
            #lon,va_da=du.grid_switch(lon,va_da)

            if t == 0:
                lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                t = 1
            va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            du.netcdf_create_basic(outfile,va_da_out,'so',lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def cmems_mld_load(loc,start_yr = 1993,end_yr = 2020):
    """
    Reanalysis Dataset DOI: https://doi.org/10.48670/moi-00021
    Renalaysis/Forecast Dataset DOI: https://doi.org/10.48670/moi-00016
    """
    # Year reanalysis ends - check https://doi.org/10.48670/moi-00021 for year
    # After this year we use the forecast dataset from: https://doi.org/10.48670/moi-00016
    transition_yr = 2020
    script_template_fore = 'python -m motuclient \
    --motu https://my.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_MULTIYEAR_PHY_001_030-TDS \
    --product-id cmems_mod_glo_phy_my_0.083_P1M-m \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --depth-min 0.49402499198913574 --depth-max 0.49402499198913574 \
    --variable mlotst \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'

    script_template_nrt = 'python -m motuclient \
    --motu https://nrt.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS \
    --product-id cmems_mod_glo_phy_anfc_0.083deg_P1M-m \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --variable mlotst \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'

    """
    Edit information below!
    """
    USERNAME = 'dford1'
    PASSWORD = 'Jokers!99.88\\'
    OUTPUT_DIRECTORY = loc

    yr = start_yr
    mon = 1
    while yr <= end_yr:
        date_min = datetime.datetime(yr,mon,1,0,0,0);
        date_max = datetime.datetime(yr,mon,27,23,59,59); date_max = date_max.strftime('%Y-%m-%d %H:%M:%S')
        OUTPUT_TEMP = os.path.join(OUTPUT_DIRECTORY,str(yr))
        du.makefolder(OUTPUT_TEMP)
        OUTPUT_FILENAME = date_min.strftime('%Y_%m_CMEMS_GLORYSV12_MLD.nc')
        date_min = date_min.strftime('%Y-%m-%d %H:%M:%S')
        print(OUTPUT_FILENAME)
        if not du.checkfileexist(os.path.join(OUTPUT_TEMP,OUTPUT_FILENAME)):
            if yr > transition_yr:
                script_template = script_template_nrt
            else:
                script_template = script_template_fore
            data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME,OUTPUT_TEMP,date_min,date_max)
            motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_automated))
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def cmems_average_mld(loc,outloc,start_yr=1990,end_yr=2023,res=1):
    du.makefolder(outloc)
    log,lag = du.reg_grid(lon=res,lat=res)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CMEMS_GLORYSV12_MLD.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_CMEMS_GLORYSV12_MLD_'+str(res)+'_deg.nc')
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables['mlotst'][:])))
            va_da = np.log10(va_da)
            # va_da[va_da < 0.0] = np.nan
            # va_da[va_da > 60.0] = np.nan
            #print(va_da)
            c.close()
            #lon,va_da=du.grid_switch(lon,va_da)

            if t == 0:
                lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                t = 1
            va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            du.netcdf_create_basic(outfile,va_da_out,'mlotst',lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
