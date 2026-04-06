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
import os
from pickle import dump, load

# This is needed due to duplicate librarys... I spent hours trying to fix the duplication,
# but couldn't. I haven't seen an adverse effects of this yet...
# Believe it's to do with the tf_som package AND matplotlib using same librarys but
# different versions.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def som_feed_forward(model_save_loc,data_file,inp_vars,ref_year = 1970,o_var = 'prov',box=[7,5,3],m=4, plot=False,normalise = False):
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
    if normalise:
        StdScl = StandardScaler()
        StdScl.fit(som_inp)
        som_inp = StdScl.transform(som_inp)
        dump(som_inp,open(os.path.join(model_save_loc,'scalars','SOM_scalar.pkl'),'wb'))

    #Now we setup the SOM training, where m is the number nodes. So a m=4 would be
    #a 16 node SOM (we only run with consitent rows and columns)
    # We run this for 100 epochs
    som = tf_som.SOM(m, m, len(inp_vars), dtype = np.float32, learning_rate = 0.3, sigma = 1, epochs = 200)
    som.train(som_inp)
    dump(som,open(os.path.join(model_save_loc,'networks','SOM_network.pkl'),'wb'))
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
    if isinstance(box, list):
        map_filtered = np.copy(map)
        for j in box:
            print('Filtering by ' + str(j))
            for i in range(map.shape[2]):
                print(i)
                map_filtered[:,:,i] = mode_smooth(map_filtered[:,:,i],box=j)
    else:
        for i in range(map.shape[2]):
            print(i)
            map_filtered[:,:,i] = mode_smooth(map[:,:,i],box=j)
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

    c = Dataset(data_file,'a')
    c.variables[o_var].input_variables = str(inp_vars)
    c.variables[o_var].normalise = str(normalise)
    c.variables[o_var+'_smoothed'].input_variables = str(inp_vars)
    c.variables[o_var+'_smoothed'].normalise = str(normalise)
    c.variables[o_var+'_smoothed'].smoothing_boxes = str(box)
    c.close()

def som_feed_forward_probability(model_save_loc,data_file,inp_vars,ref_year = 1970,o_var = 'prov',box=[7,5,3],m=4, plot=False,normalise = False,unc = False,ens=20,data_file_out=''):
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
    if normalise:
        StdScl = StandardScaler()
        StdScl.fit(som_inp)
        som_inp_raw = np.copy(som_inp)
        som_inp = StdScl.transform(som_inp)
        dump(StdScl,open(os.path.join(model_save_loc,'scalars','SOM_scalar.pkl'),'wb'))

    #Now we setup the SOM training, where m is the number nodes. So a m=4 would be
    #a 16 node SOM (we only run with consitent rows and columns)
    # We run this for 100 epochs
    som = tf_som.SOM(m, m, len(inp_vars), dtype = np.float32, learning_rate = 0.3, sigma = 1, epochs = 200)
    som.train(som_inp)
    dump(som,open(os.path.join(model_save_loc,'networks','SOM_network.pkl'),'wb'))
    # som = load(open(os.path.join(model_save_loc,'networks','SOM_network.pkl'),'rb'))
    map = np.zeros((na.shape[0],na.shape[1],na.shape[2],m*m));
    for i in range(ens):
        print('Ensemble ' + str(i))
        if normalise:
            temp_som_inp = np.copy(som_inp_raw)
        else:
            temp_som_inp = np.copy(som_inp)
        n = np.random.normal(0,0.5,len(temp_som_inp[:,0]))
        for j in range(len(inp_vars)):
            temp_som_inp[:,j] = temp_som_inp[:,j] + (n * unc[j])
        if normalise:
            temp_som_inp = StdScl.transform(temp_som_inp)
        win = som.winners(temp_som_inp)
        temp_som_map = np.zeros((na.shape)); temp_som_map[:] = np.nan;
        temp_som_map[f[0],f[1],f[2]] = win

        for t in range(0,m*m):
            g = np.where(temp_som_map == t)
            map[g[0],g[1],g[2],t] = map[g[0],g[1],g[2],t]+1
    map_main = np.zeros((na.shape[0],na.shape[1],na.shape[2])); map_main[:] = np.nan
    win = som.winners(som_inp)
    map_main[f[0],f[1],f[2]] = win

    c=Dataset(data_file_out,'w')
    dims = list(c.dimensions)
    print(dims)
    c.createDimension('latitude',lat.shape[0])
    c.createDimension('longitude',lon.shape[0])
    lat_o = c.createVariable('latitude','f4',('latitude'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = c.createVariable('longitude','f4',('longitude'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    c.createDimension('province',m*m)
    c.createDimension('clim_time',12)
    dims = list(c.dimensions)
    print(dims)
    var_o = c.createVariable(o_var+'_ensemble','f4',('longitude','latitude','clim_time','province'))
    var_o[:] = map/ens
    var_o = c.createVariable(o_var+'prov','f4',('longitude','latitude','clim_time'))
    var_o[:] = map_main

    # Now the orginial output from the SOM can have single pixel provinces, or very
    # small provinces embedded in a large province.
    # So we use a mode filter to smooth the province and remove the majority of the small
    # few pixel provinces, which is run seperately for each month in the climatology
    map_filtered = np.zeros((map_main.shape)); map_filtered[:] = np.nan
    if isinstance(box, list):
        map_filtered = np.copy(map_main)
        for j in box:
            print('Filtering by ' + str(j))
            for i in range(map.shape[2]):
                print(i)
                map_filtered[:,:,i] = mode_smooth(map_filtered[:,:,i],box=j)
    else:
        for i in range(map.shape[2]):
            print(i)
            map_filtered[:,:,i] = mode_smooth(map_main[:,:,i],box=j)
    # THis filtering (or how it's implemented) extends the provinces far into the land,
    # so we whereever the orginial SOM output was NAN we set this to NAN.
    # To maintain consistency
    map_filtered[np.isnan(map_main) == 1] = np.nan
    var_o = c.createVariable(o_var+'_smoothed','f4',('longitude','latitude','clim_time'))
    var_o[:] = map_filtered
    #
    # # Now we calculate how many years the orginial data covered so we can replicate
    # # the SOM output to the number of years present.
    # s = vars[inp_vars[0]].shape[2]/12
    # map = np.tile(map,(1,1,int(s)))
    # map_filtered = np.tile(map_filtered,(1,1,int(s)))
    # print(vars[inp_vars[0]].shape[2])
    # print(map.shape)
    # # Now setting up the dictionary to save the data back into the data file, where
    # # the input data came from
    # direct = {}
    # direct[o_var] = map
    # direct[o_var+'_smoothed'] = map_filtered
    # cinp.append_netcdf(data_file,direct,lon,lat,int(s))
    c.close()
    # c = Dataset(data_file,'a')
    # c.variables[o_var].input_variables = str(inp_vars)
    # c.variables[o_var].normalise = str(normalise)
    # c.variables[o_var+'_smoothed'].input_variables = str(inp_vars)
    # c.variables[o_var+'_smoothed'].normalise = str(normalise)
    # c.variables[o_var+'_smoothed'].smoothing_boxes = str(box)
    # c.close()

def som_probability_append_longhurst_prov(model_save_loc,data_file,longhurstfile,long_prov,prov_val,prov_var,prob_var,m=2):
    c = Dataset(os.path.join(data_file),'a')
    prov = np.array(c.variables[prov_var][:])

    d = Dataset(longhurstfile,'r')
    long = np.array(d.variables['longhurst'])
    d.close()

    d = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
    bath = np.array(d.variables['ocean_proportion'][:])
    d.close()
    f_bath = np.where(bath == 0)
    # long[f_bath[0],f_bath[1]] = np.nan
    for i in range(len(long_prov)):
        f = np.where(long == long_prov[i])
        print(f[0].shape)
        print(np.nanmax(prov))
        prov[f[0],f[1],:] = prov_val

    prov[f_bath[0],f_bath[1],:] = np.nan
    c.variables[prov_var][:] = prov

    # Now to deal with the probability side...
    prob_variable = np.array(c.variables[prob_var][:])
    prob_variable = np.concatenate((prob_variable,np.zeros((prob_variable.shape[0],prob_variable.shape[1],prob_variable.shape[2],1))),axis=3)
    print(prob_variable.shape)
    for i in range(len(long_prov)):
        f = np.where(long == long_prov[i])
        prob_variable[f[0],f[1],:,-1] = 1
        for j in range(prob_variable.shape[2]):
            prob_variable[:,:,j,-1] = spill2(prob_variable[:,:,j,-1],m=m)
            prob_variable[:,:,j,-1] = prob_variable[:,:,j,-1]/np.max(prob_variable[:,:,j,-1])
            for t in range(prob_variable.shape[3]-1):
                prob_variable[:,:,j,t] = prob_variable[:,:,j,t] * (1-prob_variable[:,:,j,-1])
    f = np.where(bath == 0)
    prob_variable[f[0],f[1],:,:] = np.nan


    dims = list(c.dimensions)
    print(dims)
    cou = 0
    for j in dims:
        if 'province' in j:
            print(str(j) + ' Yes')
            cou=cou+1
    c.createDimension('province_'+str(cou),prob_variable.shape[3])
    print(list(c.variables.keys()))
    if 'prov_ensemble_'+str(cou) not in list(c.variables.keys()):
        var_o = c.createVariable('prov_ensemble_'+str(cou),'f4',('longitude','latitude','clim_time','province_'+str(cou)))

    c.variables['prov_ensemble_'+str(cou)][:] = prob_variable
    c.close()

def som_probability_manual_prov(model_save_loc,data_file,lat_g,lon_g,prov_var,prob_var,fill=np.nan):
    c = Dataset(data_file,'a')
    prov = np.array(c.variables[prov_var][:])
    lat = np.array(c.variables['latitude'][:])
    lon = np.array(c.variables['longitude'][:])

    f = np.where((lat>lat_g[0]) & (lat <lat_g[1]) )[0]
    g = np.where((lon>lon_g[0]) & (lon <lon_g[1]) )[0]
    [g,f] = np.meshgrid(g,f)
    for i in range(prov.shape[2]):
        prov[g,f,i] = fill
    c.variables[prov_var][:] = prov
    prob = np.array(c.variables[prob_var])
    prob[g,f,:,:] = fill
    c.variables[prob_var][:] = prob
    c.close()

def merge_provinces(som_file,data_file,prov_var,prob_var,out_prov=False,out_prob=False,cutoff=0.1,province_dim_name='province'):
    if not out_prov:
        out_prov=prov_var
    if not out_prob:
        out_prob=prob_var
    c = Dataset(som_file,'r')
    prov = np.array(c[prov_var])
    prob = np.array(c[prob_var])
    c.close()
    prob[prob<cutoff] = 0.0
    su = np.sum(prob,axis=3)

    for i in range(prob.shape[3]):
        prob[:,:,:,i] = prob[:,:,:,i] * (1/su)

    c = Dataset(data_file,'a')
    time = len(np.array(c['time']))
    s = time/12
    prov = np.tile(prov,(1,1,int(s)))
    prob = np.tile(prob,(1,1,int(s),1))
    if province_dim_name not in list(c.dimensions):
        c.createDimension(province_dim_name,prob.shape[-1])
    if out_prov not in list(c.variables.keys()):
        var_o = c.createVariable(out_prov,'f4',('longitude','latitude','time'))
    c.variables[out_prov][:] = prov

    if out_prob not in list(c.variables.keys()):
        var_o = c.createVariable(out_prob,'f4',('longitude','latitude','time',province_dim_name))
    c.variables[out_prob][:] = prob

    c.close()


def spill2(arr, nval=0, m=2):
    import scipy.ndimage
    """
    Function to blur the edge of the province in the probability masking...
    """
    sigma = [m, m]
    y = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='nearest')

    return y

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
    out = stats.mode(data[f].ravel(),keepdims=False)
    #print(out)
    return out[0]
