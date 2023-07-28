#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script to split the takahashi fCO2 climatology into its month values for input into
the OceanICU create_input_netcdf.py structure.
"""



import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import data_utils as du

loc = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/GCB_Submission_Watsonetal/GCB_Dan_Ford/output/networks'
in_file = 'biomes_26-May-2023_1523.nc'
out_loc = loc + '/monthly'
start_yr = 1985
du.makefolder(out_loc)

c = Dataset(os.path.join(loc,in_file))
prov = np.array(c.variables['prov'])
print(prov.shape)
c.close()

lon,lat = du.reg_grid()
i = 0
mon = 1
yr = start_yr
while i < prov.shape[2]:
    outfile = str(yr) + du.numstr(mon) + '_watson_som.nc'
    du.netcdf_create_basic(os.path.join(out_loc,outfile),prov[:,:,i],'biome',lat,lon)
    mon = mon+1
    i = i +1
    if mon == 13:
        yr = yr+1
        mon=1
