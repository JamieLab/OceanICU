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

def era5_daily(loc,start_yr,end_yr):
    import cdsapi
    import calendar

    d = datetime.datetime(start_yr,1,1)

    while d.year <= end_yr:
        print(d.year)
        print(d.month)
        p = os.path.join(loc,str(d.year))
        du.makefolder(p)
        p = os.path.join(loc,str(d.year),d.strftime('%m'))
        du.makefolder(p)
        day = []
        for i in range(1,calendar.monthrange(d.year,d.month)[1]+1):
            day.append(du.numstr(i))
        if not du.checkfileexist(p+'/'+d.strftime("%Y_%m")+'*.nc'):


            client = cdsapi.Client()
            dataset = "reanalysis-era5-single-levels"
            request = {
                'product_type': ['reanalysis'],
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': [d.strftime('%Y')],
                'month': [d.strftime('%m')],
                'day': day,
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'data_format': 'netcdf',
            }
            target = os.path.join(p,d.strftime("%Y_%m")+'_hourly_ERA5.nc')
            client.retrieve(dataset, request, target)
        d = d + datetime.timedelta(days=int(day[-1]))

def era5_wind_time_average(loc,outloc,start_yr,end_yr):
    du.makefolder(outloc)
    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),du.numstr(mon),str(yr)+'_'+du.numstr(mon)+'_hourly_ERA5.nc')
        print(file)
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+'_ERA5.nc')
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            lon,lat = du.load_grid(file)
            c = Dataset(file,'r')
            u = np.array(c['u10'])
            v = np.array(c['v10'])
            c.close()
            ws = np.sqrt(u**2 + v**2)
            print(ws.shape)
            ws2 = ws**2
            ws = np.transpose(np.mean(ws,axis=0))
            ws2 = np.transpose(np.mean(ws2,axis=0))
            print(ws.shape)
            lon,ws = du.grid_switch(lon,ws)
            l,ws2 = du.grid_switch(np.array(180),ws2)
            du.netcdf_create_basic(outfile,ws,'ws',lat,lon)
            du.netcdf_append_basic(outfile,ws2,'ws2')
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1

def era5_average(loc,outloc,start_yr=1990,end_yr=2023,log=[],lag=[],var=None,orgi_res = 0.25):
    du.makefolder(outloc)
    res = np.round(np.abs(log[0]-log[1]),2)

    yr = start_yr
    mon = 1
    t = 0
    while yr <= end_yr:
        if mon == 1:
            du.makefolder(os.path.join(outloc,str(yr)))
        file = os.path.join(loc,str(yr),str(yr)+'_'+du.numstr(mon)+'_ERA5.nc')
        outfile = os.path.join(outloc,str(yr),str(yr)+'_'+du.numstr(mon)+f'_ERA5_{var}_'+str(res)+'_deg.nc')
        if du.checkfileexist(file) and not du.checkfileexist(outfile):
            #if t == 0:
            lon,lat = du.load_grid(file)

            c = Dataset(file,'r')
            va_da = np.squeeze(np.array(c.variables[var][:]))
            if va_da.shape[0] < va_da.shape[1]:
                va_da = np.transpose(va_da)
            c.close()
            if lon[0] >= 0:
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
                t = 1
            #print(va_da_out)
            du.netcdf_create_basic(outfile,va_da_out,var,lag,log)
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
