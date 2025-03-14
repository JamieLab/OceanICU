#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
import data_utils as du

def gebco_resample(file,log,lag,save_loc = False,save_loc_fluxengine=False):
    res = np.round(np.abs(log[0]-log[1]),2)
    if not save_loc:
        file_p = file.split('.')
        file_o = file_p[0] + f'_{res}_deg.' + file_p[1]
    else:
        file_o = save_loc

    print(file_o)

    [lon,lat] = du.load_grid(file,latv = 'lat',lonv='lon')
    [lo_grid,la_grid] = du.determine_grid_average(lon,lat,log,lag)
    area = np.transpose(du.area_grid(lat = lag,lon = log,res=res) * 1e6)

    if du.checkfileexist(file_o) == 0:
        c = Dataset(file,'r')
        elev = np.transpose(np.array(c.variables['elevation'][:])).astype('f')
        c.close()
        elev_o = du.grid_average(elev,lo_grid,la_grid)
        #sst_o = convert_sst_to_sc(sst_o)
        du.netcdf_create_basic(file_o,elev_o,'elevation',lag,log)
        land = land_proportion_calc(elev,lo_grid,la_grid)

        du.netcdf_append_basic(file_o,land,'ocean_proportion')
        land = np.abs(land-1)
        du.netcdf_append_basic(file_o,area,'area')
        du.netcdf_append_basic(file_o,land,'land_proportion')
        land = np.flipud(np.transpose(land))
        c = Dataset(file_o,'a')
        c['elevation'].units = 'm'
        c['area'].units = 'm^2'
        c.close()
        if save_loc_fluxengine:
            outp = Dataset(save_loc_fluxengine,'w',format='NETCDF4_CLASSIC')
            outp.createDimension('lon',lag.shape[0])
            outp.createDimension('lat',log.shape[0])
            outp.createDimension('time',1)
            sst_o = outp.createVariable('land_proportion','f4',('time','lon','lat'),zlib=True)
            sst_o[:] = land[np.newaxis,:,:]
            outp.close()
        #du.netcdf_create_basic(save_loc_fluxengine,land,'land_proportion',log,lag)

def land_proportion_calc(var,lo_grid,la_grid):
    out = np.empty((len(lo_grid),len(la_grid)))
    out[:] = np.nan
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            if (la_grid[j][0].size == 0) or (lo_grid[i][0].size == 0):
                out[i,j] = np.nan
            else:
                temp_lo,temp_la = np.meshgrid(lo_grid[i],la_grid[j])
                temp_lo = temp_lo.ravel(); temp_la = temp_la.ravel()
                v = var[temp_lo,temp_la]
                #print(v)
                p = np.where(v < 0)
                #print(len(p[0]))
                #print(v.size)
                out[i,j] = len(p[0]) / v.size
    return out
