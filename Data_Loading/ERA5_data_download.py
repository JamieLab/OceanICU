#!/usr/bin/env python

import datetime
import os
import data_utils as du
from netCDF4 import Dataset
import numpy as np

def download_era5(loc,start_yr=1990,end_yr=2023):
    import cdsapi
    #loc = "D:/Data/ERA5/MONTHLY/DATA"
    end_year = end_yr
    ye = start_yr
    mon = 1

    while ye < end_year:
        d = datetime.datetime(ye,mon,1)
        mo = d.strftime("%m")
        p = os.path.join(loc,str(ye))
        du.makefolder(p)

        if not du.checkfileexist(p+'/'+d.strftime("%Y_%m")+'*.nc'):
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': [
                        '10m_wind_speed', '2m_dewpoint_temperature', '2m_temperature',
                        'boundary_layer_height', 'mean_sea_level_pressure', 'mean_surface_downward_long_wave_radiation_flux',
                        'mean_surface_downward_short_wave_radiation_flux',
                    ],
                    'year': str(ye),
                    'month': mo,
                    'time': '00:00',
                    'format': 'netcdf',
                },
                os.path.join(p,str(ye)+'_'+mo+'_ERA5.nc'))

        mon = mon+1
        if mon == 13:
            ye = ye+1
            mon = 1

def era5_average(loc,outloc,start_yr=1990,end_yr=2023,log=[],lag=[],var=None,orgi_res = 0.25):
    du.makefolder(outloc)
    res = np.round(np.abs(log[0]-log[1]),2)

    yr = start_yr
    mon = 1
    t = 0
    while yr < end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),str(yr)+'_'+du.numstr(mon)+'_ERA5.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+f'_ERA5_{var}_'+str(res)+'_deg.nc')
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            if t == 0:
                lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.transpose(np.squeeze(np.array(c.variables[var][:])))
            c.close()
            lon,va_da=du.grid_switch(lon,va_da)
            
            if res > orgi_res:
                # If we are averaging to a 1 deg grid for example then we use the grid averaging.
                # However if the new grid is higher resolution then we need to spatially interpolate.
                if t == 0:
                    lo_grid,la_grid = du.determine_grid_average(lon,lat,log,lag)
                    t = 1
                va_da_out = du.grid_average(va_da,lo_grid,la_grid)
            else:
                va_da_out = du.grid_interp(lon,lat,va_da,log,lag)
            du.netcdf_create_basic(outfile,va_da_out,var,lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
