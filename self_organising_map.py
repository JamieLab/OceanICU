#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2024
Functions to implement a Self Organising Map (as in Landschutzer et al. 2013, 2014,2016), in a Python/TensorFlow implementation
"""
import datetime
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import Data_Loading.data_utils as du
import tf_som
from sklearn.preprocessing import StandardScaler
import construct_input_netcdf as cinp

# This is needed due to duplicate librarys... I spent hours trying to fix the duplication,
# but couldn't. I haven't seen an adverse effects of this yet...
# Believe it's to do with the tf_som package AND matplotlib using same librarys but
# different versions.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def som_feed_forward(model_save_loc,data_file,inp_vars,ref_year = 1970,o_var = 'prov',box=5,m=4, plot=True):
    """
    This function takes variables out of the data file generated with
    construct_input_netcdf.py, and uses them to train and implement a SOM approach. Here
    the approach is run on a climatology based approach.
    """
    #Loading basic time, lat and lon data
    c = Dataset(data_file,'r')
    time = np.array(c['time'])
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])

    # Then we load the variables for the SOM into a dictionary
    vars = {}
    for v in inp_vars:
        vars[v] = np.array(c[v])
    c.close()

    # As the implementation uses a climatology, we calculate the month that each timestep represents...
    # So an array of 1-12's
    time_month = []
    for i in range(len(time)):
        time_month.append((datetime.datetime(ref_year,1,15) + datetime.timedelta(days= int(time[i]))).month)
    time_month = np.array(time_month)
    #print(time_month)

    #Now we generate the climatologies for each of the variables provided and save within
    # a 4D numpy array (lon,lat,time,variable)
    clims = np.zeros((len(lon),len(lat),12,len(inp_vars)))
    clims[:] = np.nan
    # Cycle through the dictionary, generate climatology and then save in numpy array
    for i in range(len(inp_vars)):
        clims[:,:,:,i] = cinp.construct_climatology(vars[inp_vars[i]],time_month)
    #print(clims.shape)

    #Now checking where we have all variables (i.e for each point in the monthly climatology
    #checking the values are all real and not NAN)
    na = np.sum(np.isnan(clims),axis=3)
    f = np.where(na == 0)
    # We extract the first variable and ravel (convert to 1D array), to see the length
    a = clims[f[0],f[1],f[2],0].ravel()
    # Now build an array with columns for each variable
    som_inp = np.zeros((len(a),len(inp_vars)))
    # And fill with each variable
    for i in range(len(inp_vars)):
        som_inp[:,i] = clims[f[0],f[1],f[2],i].ravel()

    # Now we need to transform these variables so they have a mean of ~0,
    # so we use StandardScaler like in the Neural network training
    StdScl = StandardScaler()
    StdScl.fit(som_inp)
    som_inp = StdScl.transform(som_inp)

    #Now we setup the SOM training, where m is the number nodes. So a m=4 would be
    #a 16 node SOM (we only run with consitent rows and columns)
    # We run this for 100 epochs
    som = tf_som.SOM(m, m, len(inp_vars), dtype = np.float32, learning_rate = 0.3, sigma = 1, epochs = 100)
    som.train(som_inp)
    # Now we extract the winners (or the SOM node that corresponds to each varibale combination)
    win = som.winners(som_inp)
    # And then save back onto the mapping grid (so go from 1D back to 3D, with everything in the right place)
    map = np.zeros((na.shape)); map[:] = np.nan
    map[f[0],f[1],f[2]] = win
    if plot:
        # Plotting to check it looks reasonable
        fig,ax = plt.subplots(3,4)
        ax = ax.ravel()
        for i in range(0,12):
            ax[i].pcolor(lon,lat,np.transpose(map[:,:,i]))

        plt.show()

    # Now the orginial output from the SOM can have single pixel provinces, or very
    # small provinces embedded in a large province.
    # So we use a mode filter to smooth the province and remove the majority of the small
    # few pixel provinces, which is run seperately for each month in the climatology
    map_filtered = np.zeros((map.shape)); map_filtered[:] = np.nan
    for i in range(map.shape[2]):
        print(i)
        map_filtered[:,:,i] = mode_smooth(map[:,:,i],box=box)
    # THis filtering (or how it's implemented) extends the provinces far into the land,
    # so we whereever the orginial SOM output was NAN we set this to NAN.
    # To maintain consistency
    map_filtered[np.isnan(map) == 1] = np.nan

    # Now we calculate how many years the orginial data covered so we can replicate
    # the SOM output to the number of years present.
    s = vars[inp_vars[0]].shape[2]/12
    map = np.tile(map,(1,1,int(s)))
    map_filtered = np.tile(map_filtered,(1,1,int(s)))
    print(vars[inp_vars[0]].shape[2])
    print(map.shape)
    # Now setting up the dictionary to save the data back into the data file, where
    # the input data came from
    direct = {}
    direct[o_var] = map
    direct[o_var+'_smoothed'] = map_filtered
    cinp.append_netcdf(data_file,direct,lon,lat,int(s))

def mode_smooth(data,box = 3):
    """
    Function to implement a mode filter that can be applied to 2 dimensional arrays
    data = the 2D data array
    box = the dimensions of the array the mode filter should be applied (so 3 is a 3x3 pixel box)

    box should be an odd number (I haven't tried an even number but theoertically it shouldn't work
    or will produce wierd values... even number at own risk)
    """
    from scipy.ndimage import generic_filter
    data_filt = generic_filter(data,mode,[box,box])

    return data_filt

def mode(data):
    """
    Implementing a mode function that removes NAN values before the mode is calculated
    i.e so like a nanmode function.
    data = is a 1D array of the data the mode should be calculated on
    """
    from scipy import stats
    #print(data)
    f = np.where(np.isnan(data) == 0)
    #print(f)
    out = stats.mode(data[f],axis=None)
    #print(out)
    return out[0]
