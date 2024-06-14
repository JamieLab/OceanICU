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

def split_landschutzer_mat(file='D:/Data/Takahashi_Clim/taka_for_nn_2_1985_2019.mat',output='D:/Data/Takahashi_Clim/monthly'):
    """
    This function takes the Takahashi climatology extended to the coast line provided by Peter Landschutzer, and splits into
    monthly netcdf files for input into the construct_input_netcdf input file construction.
    file = the Mat file provided by Peter Landschutzer
    output = the location to save the output netcdf files
    """
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

def extend_fay_et_al(file,output,latg,long,bath_file):
    """
    This function takes the updated Takahashi climaotlogy described in Fay et al. (2024; https://doi.org/10.5194/essd-16-2123-2024) and outputs
    it in the form needed for the neural network framework. We extend the climatology to the coast and to the Arctic and smooth using a 3x3 filter to remove
    the hard boundaries.
    file = the netcdf file from Fay et al. with the takahashi climatology
    output = the output location for the processed climatology
    latg = the latitude grid the output needs to be on
    long = the longitude grid the output needs to be on
    bath_file = the bathmetry/ocean proportion file generated from GEBCO - this is so the output can be masked to just the ocean considered.
    """
    from scipy.ndimage import generic_filter
    import scipy.interpolate as interp
    du.makefolder(output)

    #Loading the takahashi climatology - this comes as dfCO2 so we need to convert back to fCO2sw with the fCO2atm
    c = Dataset(file,'r')
    lat = np.array(c['lat'])
    lon = np.array(c['lon'])
    lont = np.copy(lon) #Temporary lon array for grid switching
    dfco2 = np.array(c['dfco2'])
    atm = np.array(c['atmfco2'])
    c.close()
    # dfCO2 seems to be on a different grid (0-360E instead of -180 to 180) so we switch thje grid
    for i in range(12):
        lont,dfco2[:,:,i] = du.grid_switch(lont,dfco2[:,:,i])
    lon = lon-180
    # Calcualte the fCO2sw
    fco2sw = atm + dfco2

    # As the data are produced on a 4x5 deg grid that is then interpolated to 1 degree, we need to extend the
    # data into the coast and the Arctic underice region. Here we fill data that is a nan with a 15x15 degree nanmean.
    # We only subsitute this data in when the orginial data is a nan.
    for i in range(12):
        print(i)
        fco2sw_t = generic_filter(fco2sw[:,:,i],np.nanmean,[15,15])
        fco2sw_o = fco2sw[:,:,i]
        fco2sw_o[np.isnan(fco2sw_o) == 1] = fco2sw_t[np.isnan(fco2sw_o) == 1]
        #fco2sw_o = generic_filter(fco2sw_o,np.nanmean,[5,5])
        fco2sw[:,:,i] = fco2sw_o

    # We then fill any remaining gaps with nearest neighbour interpolation. So we take the extended fCO2sw data,
    # remove the points that are nan from the interpolation and then do the interpolation to produce a complete
    # grid (that also covers the land...).
    lon_m,lat_m = np.meshgrid(lon,lat)
    for i in range(12):
        print(i)
        points = np.stack([lon_m.ravel(),lat_m.ravel()],-1)
        #print(points.shape)
        fco2 = fco2sw[:,:,i].ravel()
        points2 = points[np.isnan(fco2)==0,:]
        fco2 = fco2[np.isnan(fco2)==0]
        interped = interp.griddata(points2,fco2,points,method='nearest')
        interped = interped.reshape((len(lon),len(lat)))
        fco2sw[:,:,i] = interped

    # fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
    # ax1.pcolor(lon,lat,np.transpose(dfco2[:,:,0]))
    # ax2.pcolor(lon,lat,np.transpose(atm[:,:,0]))
    # abar = ax3.pcolor(lon,lat,np.transpose(fco2sw[:,:,0]))
    # plt.colorbar(abar)
    # plt.show()

    # We then use the complete grid and linearly interpolate to the require grid output.
    # We then do a 7x7 filter to smooth out the 4x5 deg boxes on the final output (that may affect the Self Organising Map
    # introducing hard boundaries...)
    # DJF note: Need to check this works for 0.25 deg as we may get missing values on the edges of the grid...
    long_m,latg_m = np.meshgrid(long,latg)
    points = np.stack([lon_m.ravel(),lat_m.ravel()],-1)
    fco2_out = np.zeros((len(long),len(latg),12)); fco2_out[:] = np.nan
    for i in range(12):
        print(i)
        points_i = np.stack([long_m.ravel(),latg_m.ravel()],-1)

        interped = interp.griddata(points,fco2sw[:,:,i].ravel(),points_i,method='linear')
        interped = interped.reshape((len(lon),len(lat)))
        interped =generic_filter(interped,np.nanmean,[7,7])
        fco2_out[:,:,i] = interped

    #Finally we mask the array to only areas with "ocean" as determined from the ocean proportion array in the GEBCO bathymetry file.
    # This file needs to be generated before running this script...
    c = Dataset(bath_file,'r')
    ocean = np.array(c['ocean_proportion'])
    c.close()
    for i in range(12):
        print(i)
        fco2_t = fco2_out[:,:,i]
        fco2_t[ocean == 0] = np.nan
        fco2_out[:,:,i] = fco2_t

    #Finally save the outputs into individual monthly arrays for input into the construct_input_netcdf functionality.
    for i in range(12):
        du.netcdf_create_basic(os.path.join(output,'Fayetal_Takahashi_' + du.numstr(i+1) + '.nc'),fco2_out[:,:,i],'taka',latg,long,units='uatm')
