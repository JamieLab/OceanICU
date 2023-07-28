#!/usr/bin/env python
"""
"""
import datetime
import os
import data_utils as du
from netCDF4 import Dataset
import numpy as np
import urllib.request
import requests

def cci_sst_v3_trailblaze(loc,start_yr=1990,end_yr=2023):
    du.makefolder(loc)
    htt = 'https://gws-access.jasmin.ac.uk/public/esacci-sst/CDR3.0_release/Analysis/L4/v3.0.1/'
    d = datetime.datetime(start_yr,1,1)
    # t = 1
    while d.year < end_yr:
        if d.day == 1:
            du.makefolder(os.path.join(loc,str(d.year)))
            du.makefolder(os.path.join(loc,str(d.year),du.numstr(d.month)))

        file = os.path.join(loc,str(d.year),du.numstr(d.month),d.strftime('%Y%m%d120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc'))
        htt_file = htt+'/'+d.strftime('%Y/%m/%d')+'/'+d.strftime('%Y%m%d120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc')
        print(file)
        #print(htt_file)
        if not du.checkfileexist(file):
            print('Downloading... ' + file)
            urllib.request.urlretrieve(htt_file,file)
        #open(file).write(requests.get(htt_file))
        d = d + datetime.timedelta(days=1)
        # t = t+1
        # if t == 30:
        #     break
