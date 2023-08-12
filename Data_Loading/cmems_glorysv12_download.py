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

def motu_option_parser(script_template, usr, pwd, output_filename,output_directory,date_min,date_max,vari,product):
    dictionary = dict(
        [e.strip().partition(" ")[::2] for e in script_template.split('--')])
    #dictionary['variable'] = [value for (var, value) in [e.strip().partition(" ")[::2] for e in script_template.split('--')] if var == 'variable']  # pylint: disable=line-too-long
    dictionary['variable'] = [vari]
    #print(list(dictionary.items()))
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
        if v == '<PRODUCTID>':
            dictionary[k] = product
        if k in ['longitude-min', 'longitude-max', 'latitude-min',
                 'latitude-max', 'depth-min', 'depth-max']:
            dictionary[k] = float(v)
        # if k in ['date-min', 'date-max']:
        #     dictionary[k] = v[1:-1]
        dictionary[k.replace('-','_')] = dictionary.pop(k)
    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    return dictionary

def script_fore():
    script_template_fore = 'python -m motuclient \
    --motu https://my.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_MULTIYEAR_PHY_001_030-TDS \
    --product-id <PRODUCTID> \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --depth-min 0.49402499198913574 --depth-max 0.49402499198913574 \
    --variable <VAR> \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'
    product = 'cmems_mod_glo_phy_my_0.083_P1M-m'
    return script_template_fore, product

def script_aft(variable):
    script_template_nrt = 'python -m motuclient \
    --motu https://nrt.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS \
    --product-id <PRODUCTID> \
    --longitude-min -180 --longitude-max 180 \
    --latitude-min -90 --latitude-max 90 \
    --date-min <DATEMIN> --date-max <DATEMAX> \
    --depth-min 0.49402499198913574 --depth-max 0.49402499198913574 \
    --variable <VAR> \
    --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> \
    --user <USERNAME> --pwd <PASSWORD>'
    if variable == 'thetao':
        product = 'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m'
    elif variable == 'so':
        product = 'cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m'
    elif variable == 'uo':
        product = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m'
    elif variable == 'vo':
        product = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m'
    else:
        product = 'cmems_mod_glo_phy_anfc_0.083deg_P1M-m'
    return script_template_nrt,product

def load_glorysv12_monthly(loc,start_yr = 1993,end_yr = 2020,variable=None):
    """
    Reanalysis Dataset DOI: https://doi.org/10.48670/moi-00021
    Renalaysis/Forecast Dataset DOI: https://doi.org/10.48670/moi-00016
    """
    USERNAME = 'dford1'
    PASSWORD = 'Jokers!99.88\\'
    OUTPUT_DIRECTORY = loc
    # Year reanalysis ends - check https://doi.org/10.48670/moi-00021 for year
    # After this year we use the forecast dataset from: https://doi.org/10.48670/moi-00016
    transition_yr = 2020

    yr = start_yr
    mon = 1
    while yr <= end_yr:
        date_min = datetime.datetime(yr,mon,1,0,0,0);
        date_max = datetime.datetime(yr,mon,27,23,59,59); date_max = date_max.strftime('%Y-%m-%d %H:%M:%S')
        OUTPUT_TEMP = os.path.join(OUTPUT_DIRECTORY,str(yr))
        du.makefolder(OUTPUT_TEMP)
        OUTPUT_FILENAME = date_min.strftime(f'%Y_%m_CMEMS_GLORYSV12_{variable}.nc')
        date_min = date_min.strftime('%Y-%m-%d %H:%M:%S')
        print(OUTPUT_FILENAME)
        if not du.checkfileexist(os.path.join(OUTPUT_TEMP,OUTPUT_FILENAME)):
            if yr > transition_yr:
                script_template,product = script_aft(variable)
            else:
                script_template,product = script_fore()
            data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME,OUTPUT_TEMP,date_min,date_max,variable,product)
            #print(data_request_options_dict_automated)
            motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_automated))
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def cmems_average(loc,outloc,start_yr=1990,end_yr=2023,log=[],lag=[],variable='',log_av=False):
    du.makefolder(outloc)
    res = np.abs(log[0] - log[1])
    #log,lag = du.reg_grid(lon=res,lat=res)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),str(yr)+'_'+du.numstr(mon)+f'_CMEMS_GLORYSV12_{variable}.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+f'_CMEMS_GLORYSV12_{variable}_'+str(res)+'_deg.nc')
        print(file)
        print(outfile)
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables[variable][:])))
            va_da[va_da<-100] = np.nan
            if log_av:
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
            du.netcdf_create_basic(outfile,va_da_out,variable,lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
