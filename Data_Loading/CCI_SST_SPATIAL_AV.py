#!/usr/bin/env python3
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import data_utils as du

def load_grid(file):
    c = Dataset(file,'r')
    lat = np.array(c.variables['latitude'][:])
    lon = np.array(c.variables['longitude'][:])
    c.close()
    return lon,lat

def load_sst(file):
    c = Dataset(file,'r')
    sst = np.array(c.variables['analysed_sst'][:])
    sst[sst==-27315] = np.nan
    ice = np.array(c.variables['sea_ice_fraction'][:])
    #sst = np.transpose(sst[0,:,:])
    c.close()
    return sst,ice

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
    var_o = np.empty((len(lo_grid),len(la_grid)))
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            var_o[i,j] = np.nanmean(var[lo_grid[i],la_grid[j]])
    return var_o

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def netcdf_create(file,sst,ice,lat,lon):
    copts={"zlib":True,"complevel":5} # Compression variables to save space :-)
    outp = Dataset(file,'w',format='NETCDF4_CLASSIC')
    outp.code_used = 'cci_sst_spatial_average.py'
    outp.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outp.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    outp.createDimension('lon',lon.shape[0])
    outp.createDimension('lat',lat.shape[0])
    sst_o = outp.createVariable('analysed_sst','i2',('lon','lat'),fill_value = -32767,**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = sst
    sst_o.standard_name = 'sea_water_temperature'
    sst_o.long_name = 'Monthly mean analaysed sea surface temperature'
    sst_o.units = 'Kelvin'
    sst_o.scale_factor = 0.01 # Scale factor and offset from ESA files, and defined below in function convert_sst_to_sc
    sst_o.add_offset =273.15

    sst_o = outp.createVariable('sea_ice_fraction','f4',('lon','lat'),**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = ice
    sst_o.standard_name = 'sea_ice fraction'
    sst_o.long_name = 'Monthly mean analaysed sea ice fraction'
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

def netcdf_create2(file,sst,ice,lat,lon):
    copts={"zlib":True,"complevel":5} # Compression variables to save space :-)
    outp = Dataset(file,'w',format='NETCDF4_CLASSIC')
    outp.code_used = 'cci_sst_spatial_average.py'
    outp.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outp.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    outp.createDimension('lon',lon.shape[0])
    outp.createDimension('lat',lat.shape[0])
    sst_o = outp.createVariable('sst','i2',('lat','lon'),fill_value = -32767,**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = np.transpose(sst)
    sst_o.standard_name = 'sea_water_temperature'
    sst_o.long_name = 'Monthly mean analaysed sea surface temperature'
    sst_o.units = 'Kelvin'
    sst_o.scale_factor = 0.01 # Scale factor and offset from ESA files, and defined below in function convert_sst_to_sc
    sst_o.add_offset =273.15

    sst_o = outp.createVariable('sea_ice_fraction','f4',('lat','lon'),**copts) #Fill Value is set as this (can't make it the same as ESA for some reason?)
    sst_o[:] = np.transpose(ice)
    sst_o.standard_name = 'sea_ice fraction'
    sst_o.long_name = 'Monthly mean analaysed sea ice fraction'
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

def convert_sst_to_sc(sst):
    offset = 273.15
    scale_factor = 0.01
    a = np.isnan(sst)
    sst = np.short((sst - offset)/scale_factor)
    sst[a==1] = -32767
    return sst




data = 'D:/Data/SST-CCI/monthly'
out_folder = 'D:/Data/SST-CCI/MONTHLY_1DEG_RE'
makefolder(out_folder)

# file_9km = 'E:/Data/PAR/MERGED/1997/199709_MERGED_NASA_PAR_9km.nc'
# [lon_9,lat_9] = load_grid(file_9km)
[lon_9,lat_9] = du.reg_grid(lat=1,lon=1)


st_ye = 1981
st_mon = 9

en_ye = 2022
makefolder(os.path.join(out_folder,str(st_ye)))
ye = st_ye
mon = st_mon
t = 0
while ye <= en_ye:
    if mon == 1:
        makefolder(os.path.join(out_folder,str(ye)))

    file = os.path.join(data,str(ye),'ESA_CCI_MONTHLY_SST_'+str(ye)+numstr(mon)+'.nc')
    file_o = os.path.join(out_folder,str(ye),str(ye)+numstr(mon)+'_ESA_CCI_MONTHLY_SST_1_deg.nc')
    print(file)
    if t == 0:
        [lon,lat] = load_grid(file)
        [lo_grid,la_grid] = determine_grid_average(lon,lat,lon_9,lat_9)
        #print(lo_grid)
        #print(la_grid)
        t = 1
    if du.checkfileexist(file_o) == 0:
        sst,ice = load_sst(file)
        #print(sst.shape)
        sst_o = grid_average(sst,lo_grid,la_grid)
        ice_o = grid_average(ice,lo_grid,la_grid)
        sst_o = convert_sst_to_sc(sst_o)
        netcdf_create2(file_o,sst_o,ice_o,lat_9,lon_9)

    mon = mon+1
    if mon == 13:
        ye = ye+1
        mon = 1
