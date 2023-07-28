#!/usr/bin/env python3
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np

def load_grid(file):
    c = Dataset(file,'r')
    lat = np.array(c.variables['lat'][:])
    lon = np.array(c.variables['lon'][:])
    c.close()
    return lon,lat

def load_sst(file):
    c = Dataset(file,'r')
    sst = np.transpose(np.array(c.variables['elevation'][:])).astype('f')

    sst[sst==9.96921E36] = np.nan
    print(sst)
    #sst = np.transpose(sst[0,:,:])
    c.close()
    return sst

def numstr(num):
    if num < 10:
        return '0'+str(num)
    else:
        return str(num)

def determine_grid_average(hlon,hlat,llon,llat):
    res = abs(llon[0] - llon[1])/2
    print(res)
    lo_grid = []
    la_grid = []
    for i in range(len(llon)):
        print(i/len(llon))
        #print(np.where(np.logical_and(hlon < llon[i]+res,hlon >= llon[i]-res)))
        lo_grid.append(np.where(np.logical_and(hlon < llon[i]+res,hlon >= llon[i]-res)))
    for i in range(len(llat)):
        print(i/len(llat))
        la_grid.append(np.where(np.logical_and(hlat < llat[i]+res,hlat >= llat[i]-res)))
        # grid[i,j] =

    return lo_grid,la_grid

def grid_average(var,lo_grid,la_grid):
    print(var.shape)
    print(len(lo_grid))
    print(len(la_grid))
    var_o = np.empty((len(lo_grid),len(la_grid)))
    var_o[:] = np.nan
    print(var_o.shape)
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            var_o[i,j] = np.nanmean(var[lo_grid[i],la_grid[j]])
    return var_o

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def netcdf_create(file,sst,lat,lon):
    copts={"zlib":True,"complevel":5} # Compression variables to save space :-)
    outp = Dataset(file,'w',format='NETCDF4_CLASSIC')
    outp.code_used = 'GEBCO_BATHYMETRY_SPATIAL_AV.py'
    outp.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outp.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    outp.createDimension('lon',lon.shape[0])
    outp.createDimension('lat',lat.shape[0])
    sst_o = outp.createVariable('elevation','f4',('lon','lat'),**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = sst
    # sst_o.standard_name = 'sea_water_temperature'
    # sst_o.long_name = 'Monthly mean analaysed sea surface temperature'
    # sst_o.units = 'Kelvin'
    # sst_o.scale_factor = 0.01 # Scale factor and offset from ESA files, and defined below in function convert_sst_to_sc
    # sst_o.add_offset =273.15

    lat_o = outp.createVariable('latitude','f4',('lat'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = outp.createVariable('longitude','f4',('lon'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    outp.close()


def checkfileexist(file):
    #print(file)
    g = glob.glob(file)
    #print(g)
    if not g:
        return 0
    else:
        return 1

def reg_grid(lat=1,lon=1):
    lat_g = np.arange(-90+(lat/2),90-(lat/2)+lat,lat)
    lon_g = np.arange(-180.0+(lon/2),180-(lon/2)+lon,lon)
    return lon_g,lat_g

in_folder = 'D:/Data/Bathymetry'
out_folder = 'D:/Data/Bathymetry/LOWER_RES'
res = 1
makefolder(out_folder)

# file_9km = 'E:/Data/PAR/MERGED/1997/199709_MERGED_NASA_PAR_9km.nc'
# [lon_9,lat_9] = load_grid(file_9km)
[lon_9,lat_9] = reg_grid(lat=res,lon=res)




file = os.path.join(in_folder,'GEBCO_2022_sub_ice_topo.nc')
file_o = os.path.join(out_folder,'GEBCO_2022_sub_ice_topo_'+str(res)+'_deg.nc')
print(file)

[lon,lat] = load_grid(file)
[lo_grid,la_grid] = determine_grid_average(lon,lat,lon_9,lat_9)
#print(lo_grid)
#print(la_grid)

if checkfileexist(file_o) == 0:
    sst = load_sst(file)
    sst_o = grid_average(sst,lo_grid,la_grid)
    #sst_o = convert_sst_to_sc(sst_o)
    netcdf_create(file_o,sst_o,lat_9,lon_9)
