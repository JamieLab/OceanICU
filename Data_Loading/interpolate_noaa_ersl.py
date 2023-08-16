#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script takes a NOAA_ERSL surface file, and interpolates the values to a complete equal grid.
Final output is netCDF files for each month and year in the dataset. Default is a monthly
1 deg grid, but can handle higher spatial resolution inteprolations

To do:
Add ability to modify temporal resolution
"""
from netCDF4 import Dataset
import numpy as np
import numpy.matlib
import datetime
import time
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import data_utils as du
import os

def interpolate_noaa(file,lat=1,lon=1,grid_lat=[],grid_lon=[],grid_time=[],out_dir='D:/Data/NOAA_ERSL/DATA/MONTHLY/',end_yr = [],start_yr=[]):
    du.makefolder(out_dir)
    print('Generating sine latitudes for NOAA/ERSL...')
    noaa_grid = sinelat()
    print('Loading NOAA_file: ' +file)
    data = load_noaa_file(file)
    noaa_time = np.ascontiguousarray(data[:,0])
    data = data[:,1:-1:2]

    # plt.pcolor(noaa_grid,noaa_time,data)
    # plt.show()
    print('Setting up interpolation variable...')
    interp = RegularGridInterpolator((noaa_time,noaa_grid),data)
    if (grid_lon == []) & (grid_lat == []):
        print('Generating regular lat/lon grid at '+str(lat)+' deg...')
        grid_lon,grid_lat = reg_grid(lat=lat,lon=lon)

    if (grid_time == []):
        print('Generating regular monthly time grid....')
        date_ti,grid_time = generate_monthly_time(int(start_yr),int(np.floor(noaa_time[-1])))
        print(grid_time)

    grid_lat_vec = grid_lat
    print('Interpolating values...')
    grid_time,grid_lat = np.meshgrid(grid_time,grid_lat)
    interp_vals = interp((grid_time,grid_lat))
    #print(interp_vals.shape)
    for i in range(0,interp_vals.shape[1]):
        du.makefolder(os.path.join(out_dir,date_ti[i].strftime('%Y')))
        out = np.matlib.repmat(interp_vals[:,i],len(grid_lon),1)
        #print(out.shape)
        file_o = os.path.join(out_dir,date_ti[i].strftime('%Y'),date_ti[i].strftime('%Y_%m_NOAA_ERSL_xCO2.nc'))
        if du.checkfileexist(file_o) == 0:
            du.netcdf_create_basic(file_o,out,'xCO2',grid_lat_vec,grid_lon)
    if (end_yr != []):
        print('Extending data beyond NOAA ERSL end year! End year = ' + date_ti[-1].strftime('%Y'))
        yr = int(date_ti[-1].strftime('%Y')) + 1
        mon = 1
        while yr <= end_yr:
            temp_date = datetime.datetime(yr-1,mon,1)
            c = Dataset(os.path.join(out_dir,temp_date.strftime('%Y'), temp_date.strftime('%Y_%m_NOAA_ERSL_xCO2.nc')))
            xdata = np.array(c.variables['xCO2']) + 2.4 # Growth rate of atmospheric CO2 for recent years from https://gml.noaa.gov/ccgg/trends/gr.html
            c.close()
            temp_date = datetime.datetime(yr,mon,1)
            du.makefolder(os.path.join(out_dir,temp_date.strftime('%Y')))
            file_o = os.path.join(out_dir,temp_date.strftime('%Y'),temp_date.strftime('%Y_%m_NOAA_ERSL_xCO2.nc'))
            du.netcdf_create_basic(file_o,xdata,'xCO2',grid_lat_vec,grid_lon)
            mon = mon+1
            if mon == 13:
                yr = yr+1
                mon=1


def load_noaa_file(file):
    data = np.loadtxt(file,skiprows=0)
    return data

def generate_monthly_time(start,end):
    # Generates the monthly grid for the intepolation. We assume that the middle of the month (i.e 15th) is
    # representative of the full month
    dateti = []
    datedec = []
    yr = start
    mon = 1
    while yr < end:
        da = datetime.datetime(year=yr,month=mon,day=15)
        dateti.append(da)
        datedec.append(toYearFraction(da))
        mon = mon + 1
        if mon == 13:
            yr = yr+1
            mon = 1
    return np.array(dateti),np.array(datedec)

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)
    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def sinelat():
    # Converts the sin latitude values that NOAA ERSL is defined for into actual latitudes (i.e deg)
    a = np.arange(-1.0,1.00,0.05) # Weird bug in np.range that means final value doesnt = 1 but is ~= 1.
    a = np.append(a,1.0) # So I append 1 as the last value.
    return np.rad2deg(np.arcsin(a))
