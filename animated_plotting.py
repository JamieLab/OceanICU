#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 08/2023
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter,FFMpegWriter
import os
import cmocean
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import geopandas as gpd
import matplotlib as mpl
import Data_Loading.data_utils as du
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\df391\\OneDrive - University of Exeter\\Python\ffmpeg\\bin\\ffmpeg.exe'

def animated_output(model_save_loc,start_yr,gcb_output = 'D:\\ESA_CONTRACT\\NN\\GCB_2023_Prelim_Ens3\\output.nc'):
    worldmap = gpd.read_file(gpd.datasets.get_path("ne_50m_land"))
    def animate(i):
        fig.clear()
        gs = GridSpec(1,2, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.1,right=0.98)
        ax = fig.add_subplot(gs[0,1])
        worldmap.plot(color="lightgrey", ax=ax)
        vmin = np.nanmean(atm[:,:,i])-80
        vmax = np.nanmean(atm[:,:,i])+80
        c = ax.pcolor(lon,lat,np.transpose(fco2[:,:,i]),cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
        c2 = fig.colorbar(c,ax=ax)
        c2.set_label('fCO$_{2 (sw)}$ ($\mu$atm)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


        yr = 1985
        m = 1+i
        g = np.floor(m/12)
        yr = yr+g
        mon = int(np.round(((m/12) - g)*12,0)+1)
        ax.set_title(f'Year: {int(yr)} - Month: {mon}')
        ax = fig.add_subplot(gs[0,0])
        worldmap.plot(color="lightgrey", ax=ax)
        vmin = np.nanmean(atm[:,:,i])-80
        vmax = np.nanmean(atm[:,:,i])+80
        c = ax.pcolor(lon_gcb,lat_gcb,np.transpose(fco2_gcb[:,:,i]),cmap=cmocean.cm.balance,vmin=vmin,vmax=vmax)
        c2 = fig.colorbar(c,ax=ax)
        c2.set_label('fCO$_{2 (sw)}$ ($\mu$atm)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Year: {int(yr)} - Month: {mon}')

        #plt.show()
        #return c,c2

    file = os.path.join(model_save_loc,'output.nc')

    c = Dataset(file,'r')
    fco2 = np.array(c.variables['fco2'])
    lat = np.array(c.variables['latitude'])
    lon = np.array(c.variables['longitude'])
    c.close()
    c = Dataset(gcb_output,'r')
    fco2_gcb = np.array(c.variables['fco2'])
    lon_gcb,lat_gcb = du.reg_grid()
    c.close()

    file = os.path.join(model_save_loc,'input_values.nc')
    c = Dataset(file,'r')
    atm = np.array(c.variables['NOAA_ERSL_xCO2'])
    c.close()

    yr = start_yr
    mon = 1
    i = 0

    pad = 3
    ylim = [np.min(lat)-pad,np.max(lat)+pad]
    xlim = [np.min(lon)-pad,np.max(lon)+pad]
    fig = plt.figure(figsize=(15,7))
    #animate(0)
    ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=True,frames=fco2.shape[2])
    #ani.save(os.path.join(model_save_loc,'plots','animated.gif'), dpi=300, writer=PillowWriter(fps=10))
    ani.save(os.path.join(model_save_loc,'plots','animated.mp4'), dpi=300, writer=FFMpegWriter(fps=6))
