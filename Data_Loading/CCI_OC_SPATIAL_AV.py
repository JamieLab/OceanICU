#!/usr/bin/env python3
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import data_utils as du

def oc_cci_average(loc,start_yr = 1993,end_yr = 2020,log='',lag=''):
    if start_yr <= 1997:
        start_ye = 1997
        st_mon = 9 # Needs to be manually modified - OC-CCI starts in 09 / 1997
    else:
        st_mon = 1
    res = np.abs(log[0] - log[1])
    du.makefolder(os.path.join(out_folder,str(st_ye)))
    ye = start_yr
    mon = st_mon
    t = 0
    while ye <= end_yr:
        du.makefolder(os.path.join(out_folder,str(ye)))

        file = os.path.join(data,str(ye),'ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-'+str(ye)+numstr(mon)+'-fv6.0.nc')
        file_o = os.path.join(out_folder,str(ye),'ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_'+str(ye)+numstr(mon)+f'-fv6.0_{res}_deg.nc')
        print(file)
        if t == 0:
            [lon,lat] = du.load_grid(file)
            [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
            #print(lo_grid)
            #print(la_grid)
            t = 1
        if du.checkfileexist(file_o) == 0:
            c = Dataset(file,'r')
            chl = np.transpose(np.array(c.variables['chlor_a'][0,:,:]))
            chl[chl>2000] = np.nan
            # Averaging done in log10 space due to log normal distribution of chl-a values.
            chl = np.log10(chl)
            chl_o = du.grid_average(chl,lo_grid,la_grid)
            # Convert back to normal unit space.
            chl_o = 10**chl_o
            du.netcdf_create_basic(file_o,chl_o,'chlor_a',lag,log)

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1
