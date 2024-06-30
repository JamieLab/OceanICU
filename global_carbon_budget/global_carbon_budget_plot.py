#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 06/2023

Script to plot flux output for GCB submission

"""
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
from matplotlib.gridspec import GridSpec
import matplotlib.transforms

font = {'weight' : 'normal',
        'size'   : 19}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

model_save_loc = 'D:/OceanCarbon4Climate/NN/GCB2024_full_version'
gcb_file = model_save_loc+'/GCB_output.nc'

c = Dataset(gcb_file,'r')
time = (np.array(c['time'])/12) + 1985
fl = np.array(c['fgco2_reg'])
c.close()

fig = plt.figure(figsize=(15,15))
gs = GridSpec(2,2, figure=fig, wspace=0.15,hspace=0.15,bottom=0.1,top=0.95,left=0.1,right=0.98)
ax =[fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
tit = ['Global','North of 30N','30N to 30S','South of 30S']
for i in range(4):
    ax[i].plot(time,fl[i,:])
    if i == 0:
        ax[i].set_ylabel('Air-sea CO2 flux (Pg C yr-1; +ve into the ocean)')
    if i == 1:
        ax[i].set_ylabel('Air-sea CO2 flux (Pg C yr-1; +ve into the ocean)')
        ax[i].set_xlabel('Time (year)')
    if i == 3:
        ax[i].set_xlabel('Time (year)')
    ax[i].set_title(tit[i])

fig.savefig(os.path.join(model_save_loc,'plots/gcb_submission.png'),dpi=300)
