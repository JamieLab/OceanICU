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
import scipy.io

file = 'D:/Data/Takahashi_Clim/taka_for_nn_2_1985_2019.mat'
output = 'D:/Data/Takahashi_Clim/monthly'
mat = scipy.io.loadmat(file)
lat = np.array(mat['lat'])
lon = np.array(mat['lon'])
taka = np.array(mat['taka_for_nn_2'])
print(taka.shape)
taka = taka.transpose((2,1,0))
taka = taka[:,:,0:12]
print(taka.shape)
for i in range(12):
    du.netcdf_create_basic(os.path.join(output,'takahashi_'+du.numstr(i+1)+'_.nc'),taka[:,:,i],'taka',lat,lon)
