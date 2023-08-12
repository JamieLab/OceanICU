#!/usr/bin/env python
import os
import glob
from netCDF4 import Dataset
import numpy as np

# Function to produce month numbers as 01, instead of 1 (for example) - mainly for file names
def numstr(num):
    if num < 10:
        return '0'+str(num)
    else:
        return str(num)

# Function to load the lat and lon grid of CCI-SST files, so we can put this in each files
# and so we can get the size to construct the array
def lat_lon(file):
    c = Dataset(file,'r')
    lat = np.array(c.variables['lat'][:])
    lon = np.array(c.variables['lon'][:])
    c.close()
    return lon,lat

# Function to load the analysed sst from the CCI-SST file and remove the time dimension from
# the varaible - transpose also to get into lon - lat as opposed to lat - lon.
def load_sst(file):
    c = Dataset(file,'r')
    sst = np.array(c.variables['analysed_sst'][:])
    sst = np.transpose(sst[0,:,:])
    ice = np.array(c.variables['sea_ice_fraction'][:])
    ice = np.transpose(ice[0,:,:])
    ice[ice < -0] = np.nan
    c.close()
    return sst,ice

# Function to get the mean monthly sst (averaging along the time dimension; thrid axis)
def mean_sst(sst):
    sst_o = np.nanmean(sst,axis=2)
    return sst_o

def netcdf_create(file,sst,ice,lat,lon):
    copts={"zlib":True,"complevel":5} # Compression variables to save space :-)
    outp = Dataset(file,'w',format='NETCDF4_CLASSIC')
    outp.createDimension('lon',lon.shape[0])
    outp.createDimension('lat',lat.shape[0])
    sst_o = outp.createVariable('analysed_sst','i2',('lon','lat'),fill_value = -27315,**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = sst
    sst_o.standard_name = 'sea_water_temperature'
    sst_o.long_name = 'Monthly mean analaysed sea surface temperature'
    sst_o.units = 'Kelvin'
    sst_o.scale_factor = 0.01 # Scale factor and offset from ESA files, and defined below in function convert_sst_to_sc
    sst_o.add_offset =273.15

    sst_o = outp.createVariable('sea_ice_fraction','f4',('lon','lat'),**copts)
    sst_o[:] = ice
    sst_o.standard_name = 'sea_ice_fraction'
    sst_o.long_name = 'Monthly mean sea ice fraction'
    sst_o.units = '%'

    lat_o = outp.createVariable('latitude','f4',('lat'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = outp.createVariable('longitude','f4',('lon'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    outp.close()

# Function to convert the kelvin sst in float to kelvin sst in short, with a
# scale factor and offset constient with the ESA files
def convert_sst_to_sc(sst):
    offset = 273.15
    scale_factor = 0.01
    sst = np.short((sst - offset)/scale_factor)
    return sst

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

# Input directory for ESA data
