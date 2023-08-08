#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
import data_utils as du

def gebco_resample(file,log,lag):
    res = np.abs(log[0] - log[1])
    file_p = file.split('.')
    file_o = file_p[0] + f'_{res}_deg.' + file_p[1]

    print(file_o)

    [lon,lat] = du.load_grid(file,latv = 'lat',lonv='lon')
    [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)

    if du.checkfileexist(file_o) == 0:
        c = Dataset(file,'r')
        elev = np.transpose(np.array(c.variables['elevation'][:])).astype('f')
        c.close()
        elev_o = du.grid_average(elev,lo_grid,la_grid)
        #sst_o = convert_sst_to_sc(sst_o)
        du.netcdf_create_basic(file_o,elev_o,'elevation',lag,log)
        land = land_proportion_calc(elev,lo_grid,la_grid)
        du.netcdf_append_basic(file_o,land,'ocean_proportion')

def land_proportion_calc(var,lo_grid,la_grid):
    out = np.empty((len(lo_grid),len(la_grid)))
    out[:] = np.nan
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            if (la_grid[j][0].size == 0) or (lo_grid[i][0].size == 0):
                out[i,j] = np.nan
            else:
                v = var[lo_grid[i],la_grid[j]]
                #print(v)
                p = np.where(v < 0)
                #print(len(p[0]))
                #print(v.size)
                out[i,j] = len(p[0]) / v.size
    return out
