#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script provides function for the training of neural networks to estimate
fCO2(sw,subskin) for different province areas driven from driver.py.
This script loads data from an input netcdf produced by construct_input_netcdf.py, formats the
inputs ready for use in the neural network training and trains a neural network. The networks are
saved in an output folder with the scaling elements and an uncertainty lookup
table to allow for complete fCO2(sw,subskin) fields to be produced by a subsequent script.

These functions are driven from driver.py.

Credits to Josh Blannin who provided example code for TensorFlow implementation of the neural network.
"""
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pickle import dump, load
import weight_stats as ws
import os
import tensorflow as tf
import glob
import datetime
import weight_stats as ws
import matplotlib.transforms
font = {'weight' : 'normal',
        'size'   : 19}
matplotlib.rc('font', **font)

tf.autograph.set_verbosity(0)

def driver(data_file,fco2_sst = None, prov = None,var = [],unc = None, model_save_loc = None,
    bath = None, bath_cutoff = None, fco2_cutoff_low = None, fco2_cutoff_high = None,sea_ice=None):
    """
    """
    print('Creating output directory tree...')
    make_save_tree(model_save_loc)
    print('Neural_Network_Start...')
    vars = [fco2_sst+'_reanalysed_fCO2_sw',fco2_sst+'_reanalysed_fCO2_sw_std']
    vars.append(prov)
    if not bath_cutoff == None:
        vars.append(bath)
    if not sea_ice == None:
        vars.append(sea_ice)

    for v in var:
        vars.append(v)
    tabl,output_size,lon,lat = load_data(data_file,vars)
    if not sea_ice == None:
        print(f'Setting where sea ice is greater than 95% to nan...')
        tabl[tabl[sea_ice] > 0.95] = np.nan

    mapping_data = tabl
    # Where the std_dev is 0 but an fCO2 value exists indicates a single cruise. So we input a stddev of 0 for
    # these values, so then we can combine the measurement unc (assumed to be ~5uatm from Bakker et al. 2016) later.
    tabl[vars[1]][((np.isnan(tabl[vars[0]]) == 0) & (np.isnan(tabl[vars[1]]) == 1))] = 0
    # Only keep rows where we have all the data, i.e no gaps
    tabl = tabl[(np.isnan(tabl) == 0).all(axis=1)]
    #print(tabl)
    #Combine the stdev with an assumed measurement unc of 5 uatm (Bakker et al. 2016).
    tabl[vars[1]] = np.sqrt(tabl[vars[1]]**2 + 5**2)

    if not fco2_cutoff_low == None:
        print(f'Trimming fCO2(sw) less than {fco2_cutoff_low} uatm...')
        print(tabl.shape)
        tabl = tabl[(tabl[vars[0]] > fco2_cutoff_low)]
        print(tabl.shape)

    if not fco2_cutoff_high == None:
        print(f'Trimming fCO2(sw) greater than {fco2_cutoff_high} uatm...')
        print(tabl.shape)
        tabl = tabl[(tabl[vars[0]] < fco2_cutoff_high)]
        print(tabl.shape)

    #Bathymetry cutoff - Here we remove data shallower then the bathymetry cutoff (i.e coastal data)
    if not bath_cutoff == None:
        print(f'Trimming data shallower than {bath_cutoff} m...')
        print(tabl.shape)
        tabl = tabl[(tabl[bath] < -bath_cutoff)]
        print(tabl.shape)

    # For sanity and to check that we have an arbitary amount of data (at least ~1000 observations) in each
    # province we plan to train on. If this needs adjusting, go back to your province map building
    # and modify the map there.
    for v in np.unique(tabl[vars[2]]):
        print(str(v) + ' : '+str(np.argwhere(np.array(tabl[vars[2]]) == v).shape))

    run_neural_network(tabl,fco2 = vars[0], prov = prov, var = var, model_save_loc = model_save_loc,unc = unc)

    # flag_e_validation(data_file,fco2_sst = fco2_sst, prov = prov, var = var, model_save_loc = model_save_loc,bath = bath,bath_cutoff=bath_cutoff,fco2_cutoff_low=fco2_cutoff_low,
    #     fco2_cutoff_high = fco2_cutoff_high,unc = unc)
    mapped,mapped_net_unc,mapped_para_unc = neural_network_map(mapping_data,var=var,model_save_loc=model_save_loc,prov = prov,output_size=output_size,unc = unc)
    # mapped_net_unc[mapped<50] = np.nan
    # mapped_para_unc[mapped<50] = np.nan
    # mapped[mapped<50] = np.nan
    save_mapped_fco2(mapped,mapped_net_unc,mapped_para_unc,data_shape = output_size, model_save_loc = model_save_loc, lon = lon,lat = lat)
    plot_total_validation_unc(input_file = data_file,fco2_sst = fco2_sst,model_save_loc = model_save_loc,ice = sea_ice)
    add_validation_unc(model_save_loc)
    add_total_unc(model_save_loc)
    plot_mapped(model_save_loc)


# def flag_e_validation(data_file=None,fco2_sst = None,prov = None,var = None,model_save_loc = None,bath_cutoff = None, bath = None, fco2_cutoff_low = None, fco2_cutoff_high = None,unc=None):
#     """
#     """
#     print('Running SOCAT Flag E validation...')
#     vars = [fco2_sst+'_FlagE_reanalysed_fCO2_sw',fco2_sst+'_FlagE_reanalysed_fCO2_sw_std']
#     vars.append(prov)
#     vars.append(bath)
#     for v in var:
#         vars.append(v)
#     tabl,output_size,lon,lat = load_data(data_file,vars)
#     # Where the std_dev is 0 but an fCO2 value exists indicates a single cruise. So we input a stddev of 0 for
#     # these values, so then we can combine the measurement unc (assumed to be ~5uatm from Bakker et al. 2016) later.
#     tabl[vars[1]][((np.isnan(tabl[vars[0]]) == 0) & (np.isnan(tabl[vars[1]]) == 1))] = 0
#     # Only keep rows where we have all the data, i.e no gaps
#     tabl = tabl[(np.isnan(tabl) == 0).all(axis=1)]
#     #print(tabl)
#     #Combine the stdev with an assumed measurement unc of 5 uatm (Bakker et al. 2016).
#     tabl[vars[1]] = np.sqrt(tabl[vars[1]]**2 + 5**2)
#     # For sanity and to check that we have an arbitary amount of data (at least ~1000 observations) in each
#     # province we plan to train on. If this needs adjusting, go back to your province map building
#     # and modify the map there.
#     if not bath_cutoff == None:
#         print(f'Trimming data shallower than {bath_cutoff} m...')
#         print(tabl.shape)
#         tabl = tabl[(tabl[bath] < -bath_cutoff)]
#         print(tabl.shape)
#
#     if not fco2_cutoff_low == None:
#         print(f'Trimming fCO2(sw) less than {fco2_cutoff_low} uatm...')
#         print(tabl.shape)
#         tabl = tabl[(tabl[vars[0]] > fco2_cutoff_low)]
#         print(tabl.shape)
#
#     if not fco2_cutoff_high == None:
#         print(f'Trimming fCO2(sw) greater than {fco2_cutoff_high} uatm...')
#         print(tabl.shape)
#         tabl = tabl[(tabl[vars[0]] < fco2_cutoff_high)]
#         print(tabl.shape)
#
#     for v in np.unique(tabl[vars[2]]):
#         print(str(v) + ' : '+str(np.argwhere(np.array(tabl[vars[2]]) == v).shape))
#
#     X = np.array(tabl[var])
#     #print(X.shape)
#     y = np.array(tabl[vars[0:2]])
#     prov = np.array(tabl[prov])
#     for v in np.unique(prov):
#         print(f'Running province number {v}...')
#         mask = np.argwhere(prov == v)
#         model = tf.keras.models.load_model(os.path.join(model_save_loc,'networks',f'prov_{v}_model'))
#         scalar = load(open(os.path.join(model_save_loc,'scalars',f'prov_{v}_scalar.pkl'),'rb'))
#         if unc:
#             lut = load(open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'rb'))
#             lut = lut[0]
#         X_test=np.squeeze(X[mask,:])
#         print(X_test.shape)
#         if len(X_test.shape) > 1:
#             X_test = scalar.transform(X_test)
#             y_test = np.squeeze(y[mask,:])
#             y_test_preds = model.predict(X_test)
#
#             if unc:
#                 out = lut_retr(lut,X_test); y_test_preds = np.stack((np.squeeze(y_test_preds),out),axis=1)
#             # print(y_test)
#             # print(y_test_preds)
#             dump([y_test,y_test_preds], open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation_Eflag.pkl'), 'wb'))

def load_data(load_loc,vars):
    """
    """
    print('Loading data...')
    c = Dataset(load_loc,'r')
    lon = np.array(c.variables['longitude'][:])
    lat = np.array(c.variables['latitude'][:])
    t = 0
    for v in vars:
        data = np.squeeze(np.array(c.variables[v][:]))
        if t == 0:
            d = np.append(data.shape,len(vars))
            output = np.empty((d))
            output[:] = np.nan
        output[:,:,:,t] = data
        t += 1
    for i in range(0,len(vars)):
        if i == 0:
            tabl = pd.DataFrame()
        temp = np.squeeze(np.reshape(output[:,:,:,i],(-1,1)))
        #print(temp.shape)
        tabl[vars[i]] = temp
    c.close()
    #Returns the loaded data, the size of the data in longitude,latitude,time, the longitude array
    #and the latitude array
    return tabl,d[0:3],lon,lat

def make_save_tree(model_save_loc):
    """
    Function to create the output save tree if the folders don't exist
    """
    if not os.path.isdir(model_save_loc):
        os.mkdir(model_save_loc)
    networks = os.path.join(model_save_loc,'networks')
    if not os.path.isdir(networks):
        os.mkdir(networks)
    scalars = os.path.join(model_save_loc,'scalars')
    if not os.path.isdir(scalars):
        os.mkdir(scalars)
    validation = os.path.join(model_save_loc,'validation')
    if not os.path.isdir(validation):
        os.mkdir(validation)
    plots = os.path.join(model_save_loc,'plots')
    if not os.path.isdir(plots):
        os.mkdir(plots)
    unc = os.path.join(model_save_loc,'unc_lut')
    if not os.path.isdir(unc):
        os.mkdir(unc)

def run_neural_network(data,fco2 = None,prov = None,var=None,model_save_loc=None,plot=False, unc = None, ens = 10):
    """
    Function to run the nerual network training, and saving the best performing model. This function
    produces the model, scaler, uncertainty look up table and validation results for use in producing
    global maps of fCO2 (sw,subskin), with per pixel uncertainties.
    """
    """
    Neural Network intilisation settings
    """
    # Set early stopping condition: 1e-4
    #es = tf.keras.callbacks.EarlyStopping(monitor='val_rmse', mode='min', patience=10, min_delta=1e-4)
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10, min_delta=5)
    # Set metric to measure: RMSE
    #rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse', dtype=None)
    mse = tf.keras.losses.MeanSquaredError()
    #rmse = tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)
    # Set optimizer: ADAM
    # initial_learning_rate = 0.01
    # final_learning_rate = 0.000001
    # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/(40000/50))
    # steps_per_epoch = int(40000/50)
    #
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #             initial_learning_rate=initial_learning_rate,
    #             decay_steps=steps_per_epoch,
    #             decay_rate=learning_rate_decay_factor,
    #             staircase=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    #opt = tf.keras.optimizers.Adam()
    #Set the node increments (set initially as 2**i (i.e base 2 values)) - changed 07/2023 to mutliples of 5.
    node_increments = [i*10 for i in range(6,31,3)]
    """
    End of intilisation definitions
    """

    print('Running Neural Network training....')
    # Extract the input datasets from the data table
    X = np.array(data[var])
    print(X.shape)
    # Extract the fCO2(sw,subskin) data from the table, along with standard deviation.
    y = np.transpose(np.array([data[fco2],data[fco2+'_std']]))
    print(y.shape)
    # Extract the province number dataset.
    prov =data[prov]
    for v in np.unique(prov):
        print(f'Running province number {v}...')
        mask = np.argwhere(np.array(prov) == v)
        #print(len(mask))
        # Here we split  20% of the data to act as an independent dataset (i.e not included in the nerual network training at all)
        # This leaves 80% to be split into two groups; the training and validation sets at ~70% and ~20% of the total dataset respectively
        inds = mask
        X_train, X_val, X_test, y_train, y_val, y_test,ind_train,ind_val,ind_test = train_val_test_split(X[mask,:], y[mask,:],inds, indep = 0.05, train = 0.2)

        # The standard scaler normalises each input parameter by its mean and std dev
        StdScl = StandardScaler()
        StdScl.fit(X_train)
        # This needs to be saved so it can be loaded later to normalise future inputs to the neural network
        dump(StdScl, open(os.path.join(model_save_loc,'scalars',f'prov_{v}_scalar.pkl'), 'wb'))
        #Transform the input datasets for the 3 training sets
        X_train = StdScl.transform(X_train); X_val = StdScl.transform(X_val); X_test = StdScl.transform(X_test);
        print('Mask length: ' + str(len(mask)))
        print('Training length: '+ str(X_train.shape))
        print('Validation length: '+ str(X_val.shape))
        print('Test length: ' + str(X_test.shape))

        """
        Modified the training step 07/2023 - instead of running a single neural network, a ensemble is run and the mean of this ensemble taken.
        For uncertainites we are able to provide a neural network unc (noise in the nerual networks) and the propagated input parameter unc.
        First we run a training step to check the number of nodes to use. Then we run the ensemble, where each ensemble is saved- and run with
        the same training/validation/test data.
        """
        min_rmse = 10000 # Set really large so the first is always better.
        min_n = None # This gets set to the node number that has the lowest RMSD
        for n in node_increments:
            print(n)
            # Here we setup the Tensor Flow Model (Thanks to Josh Blannin for example code)
            model = tf.keras.models.Sequential(name='PROV_'+str(v)) # Setup the model
            model.add(tf.keras.layers.Dense(n, input_dim=len(var), activation='sigmoid')) # Add Hidden layer
            model.add(tf.keras.layers.Dense(1, activation='linear')) # Add Output layer
            model.compile(optimizer=opt, loss='mse') # Set the loss function and metric to determine the training results
            # Here we train the neural network.
            history = model.fit(X_train, y_train[:,0], epochs=200, validation_data=[X_val,y_val[:,0]], verbose=0, callbacks=[es],shuffle=False, batch_size=int(len(X_train)/50))
            # We then use the model to predict fCO2(sw,subskin) from the input training datasets
            #y_test_preds = model.predict(X_test); y_val_preds = model.predict(X_val); y_train_preds = model.predict(X_train);

            # Calculate the independent test dataset RMSD
            rms_error = history.history['loss'][-1]
            print(f'prov = {v}; n = {n}; loss = {rms_error}') # Print the province number, node increment and RMSD.

            # If the model "substantially" improves the fit then it becomes the best model (.4 is an arbitary number that can be changed)
            # The added performance requires for a small gain in this case is not beneficial. If too many nodes are added the model may become overtrained.
            # and won't be generalised.
            if (rms_error - min_rmse < -20):
                print('New Best Model!')
                min_rmse = rms_error
                min_n = n
                # If it's a better model based on the independent test dataset then the model is save...
                #model.save(os.path.join(model_save_loc,'networks',f'prov_{v}_model'))
                # And then the training datasets are saved, for validation plots.
                #dump([y_train,y_train_preds,y_val,y_val_preds,y_test,y_test_preds], open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'), 'wb'))
        # min_n = 100
        print(f'Minimum n found: {min_n} - Running neural network ensemble for prov {v}')
        #y_test_preds = np.zeros((X_test.shape[0],ens)); y_test_preds[:] = np.nan;
        #y_val_preds = np.zeros((X_val.shape[0],ens)); y_val_preds[:] = np.nan;
        #y_train_preds = np.zeros((X_train.shape[0],ens)); y_train_preds[:] = np.nan;

        for i in range(ens):
            X_train, X_val, X_test, y_train, y_val, y_test,ind_train,ind_val,ind_test = train_val_test_split(X[mask,:], y[mask,:],inds, indep = 0.05, train = 0.2)
            X_train = StdScl.transform(X_train); X_val = StdScl.transform(X_val); X_test = StdScl.transform(X_test);

            print(f'Running ensemble {i} for prov {v}')
            model = tf.keras.models.Sequential(name='PROV_'+str(v)) # Setup the model
            model.add(tf.keras.layers.Dense(min_n, input_dim=len(var), activation='sigmoid')) # Add Hidden layer
            model.add(tf.keras.layers.Dense(1, activation='linear')) # Add Output layer
            model.compile(optimizer=opt, loss='mse') # Set the loss function and metric to determine the training results
            # Here we train the neural network.
            history = model.fit(X_train, y_train[:,0], epochs=200, validation_data=[X_val,y_val[:,0]], verbose=0, callbacks=[es],shuffle=False, batch_size=int(len(X_train)/50))
            loss = history.history['loss'][-1]
            print(f'Ensemble Loss = { loss } ensemble {i}')
            # Compute the predictions for the validation, and save within an array so we have all the ensembles together
            #y_test_preds[:,i] = np.squeeze(model.predict(X_test)); y_val_preds[:,i] = np.squeeze(model.predict(X_val)); y_train_preds[:,i] = np.squeeze(model.predict(X_train));
            model.save(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens{i}'))
            #print(y_test_preds)
        #y_test_preds = np.nanmean(y_test_preds,axis=1); y_val_preds = np.nanmean(y_val_preds,axis=1); y_train_preds = np.nanmean(y_train_preds,axis=1);
        #print(y_test_preds)
        dump([ind_train,ind_val,ind_test], open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'), 'wb'))

        # Once the best model is selected, we reproduce the matchup dataset with added uncertainity information
        # if uncertainties are provided.
        if unc != None:
            # Here we provide unc information on the predicted fCO2(sw,subskin) values by propagating the
            # uncertainites in the input parameters. If no uncertainity values are provided, the in situ
            # std_dev + measurement uncertainity is provided.
            # Here to be computationally efficient a look up table for the uncertainties are calculated,
            # and linear interpolation used to fill in the look up table.

            # Load the best performing model for the
            model = tf.keras.models.load_model(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens0'))
            # Calculate the min and maximum of the input dataset values so the range of data can be determined
            minmax = np.squeeze(np.stack((np.amin(X[mask,:],axis=0),np.amax(X[mask,:],axis=0))))
            # Generate the look up table for uncertainty
            lut = unc_lut_generate(minmax,model,StdScl,unc)
            # and save the look up table.
            dump([lut],open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'wb'))
            #print(X_train)
            # Here we produce the uncertainty informaiton for the predicated fCO2 (sw,subskin) values for the
            # test dataset.
            # out = lut_retr(lut,X_train); y_train_preds = np.stack((np.squeeze(y_train_preds),out),axis=1)
            # out = lut_retr(lut,X_val); y_val_preds = np.stack((np.squeeze(y_val_preds),out),axis=1)
            # out = lut_retr(lut,X_test); y_test_preds = np.stack((np.squeeze(y_test_preds),out),axis=1)
            # # And we resave the training datasets for the validation plotting.
            # dump([y_train,y_train_preds,y_val,y_val_preds,y_test,y_test_preds], open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'), 'wb'))

def unc_lut_generate(minmax,model,scalar,unc,res = False):
    """
    """
    print('Calculating uncertainty look up table...')
    if not res:
        res = int(np.ceil(20 / minmax.shape[1]))
    print(f'Resolution of LUT = {res}...')
    rang = (minmax[1,:] - minmax[0,:])
    me = np.mean(minmax,axis=0)
    m = [np.linspace(i,j,res) for i,j in zip(me-rang,me+rang)]
    grids = np.meshgrid(*m)
    sh = grids[0].shape
    for i in range(len(grids)):
        grids[i] = np.reshape(grids[i],(-1,1))
    grid_o = np.transpose(np.squeeze(np.stack(grids)))
    unc_o = uncertainty_montecarlo(model,scalar,grid_o,unc)
    unc_o = np.reshape(unc_o,(sh))

    m2 = np.transpose(np.squeeze(np.stack(m)))
    m2 = scalar.transform(m2)
    m = []
    for i in range(m2.shape[1]):
        m.append(m2[:,i])
    outp = [m,unc_o]
    return outp

def lut_retr(lut,values):
    """

    """
    from scipy.interpolate import interpn
    vals = interpn(lut[0], lut[1], values,bounds_error=False,fill_value=np.amax(lut[1]))
    return vals

def uncertainty_montecarlo(model,scalar,X_vals,unc,repeat = 100):
    """
    """
    print('Running MonteCarlo...')
    print(X_vals.shape)
    #X_vals = scalar.inverse_transform(X_vals)
    outstd = np.empty((X_vals.shape[0]))
    outstd[:] = np.nan
    for i in range(X_vals.shape[0]):
        print(f'{i} in {X_vals.shape[0]}')
        mod_inp = np.empty((repeat,X_vals.shape[1]))
        mod_inp[:] = np.nan
        for j in range(X_vals.shape[1]):
            mod_inp[:,j] = np.random.normal(X_vals[i,j],unc[j],repeat)
        #print(mod_inp)
        mod_inp = scalar.transform(mod_inp)
        mod_out = model.predict(mod_inp)
        #print(mod_out)
        outstd[i] = 2*np.std(mod_out)
    return outstd

def train_val_test_split(X,y,ind, indep = 0.2, train = 0.4):
    """
    """
    X_train, X_test, y_train, y_test,ind_train,ind_test = train_test_split(X, y,ind, test_size=indep, random_state=2)
    X_train, X_val, y_train, y_val,ind_train,ind_val = train_test_split(X_train, y_train, ind_train, test_size=train) # 0.25 x 0.8 = 0.2
    return np.squeeze(X_train), np.squeeze(X_val), np.squeeze(X_test), np.squeeze(y_train), np.squeeze(y_val), np.squeeze(y_test), np.squeeze(ind_train), np.squeeze(ind_val),np.squeeze(ind_test)

def plot_total_validation(model_save_loc,gcb = False):
    """
    """
    print('Plotting validation....')
    files = glob.glob(os.path.join(model_save_loc,'validation','*_validation.pkl'))
    #files = files[0:-1]
    #print(files)
    y_test_total = np.array(np.nan)
    y_test_preds_total = np.array(np.nan)
    fig = plt.figure()
    c = np.array([0,800])
    t = 0
    for v in files:
        print(v)
        y_train,y_train_preds,y_val,y_val_preds,y_test,y_test_preds = load(open(v,'rb'))
        if t == 0:
            y_test_all = np.copy(y_test[:,0])
            y_test_preds_all = np.copy(y_test_preds)
            t = 1
            if gcb:
                y_test_all = np.concatenate((y_test_all,y_train[:,0],y_val[:,0]))
                y_test_preds_all =  np.concatenate((y_test_preds_all,y_train_preds,y_val_preds))
        else:
            if gcb:
                y_test_all = np.concatenate((y_test_all,y_test[:,0],y_train[:,0],y_val[:,0]))
                y_test_preds_all =  np.concatenate((y_test_preds_all,y_test_preds,y_train_preds,y_val_preds))
            else:
                y_test_all = np.concatenate((y_test_all,y_test[:,0]))
                y_test_preds_all = np.concatenate((y_test_preds_all,y_test_preds))

        # print(y_test.shape)
        # print(y_test_preds.shape)
        if gcb:
            plt.scatter(np.concatenate((y_test[:,0],y_val[:,0],y_train[:,0])),np.concatenate((y_test_preds,y_val_preds,y_train_preds)),s=4)
        else:
            plt.scatter(y_test[:,0],y_test_preds,s=4)
    print(y_test_all.shape)
    print(y_test_preds_all.shape)
    y_test_preds_all = np.squeeze(y_test_preds_all)
    plt.xlim(c); plt.ylim(c); plt.plot(c,c,'k-');
    stats_un = ws.unweighted_stats(y_test_all,y_test_preds_all,'val')
    h2 = plt.plot(c,c*stats_un['slope']+stats_un['intercept'],'k-.',zorder=5, label = 'Unweighted')
    plt.xlabel('in situ fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
    plt.ylabel('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
    s = model_save_loc.split('/')
    if gcb:
        plt.title(s[-1] + ' - GCB Evaluation = True')
    else:
        plt.title(s[-1] + ' - GCB Evaluation = False')
    ax = plt.gca()
    rmsd = np.round(stats_un['rmsd'],2)
    bias = np.round(stats_un['rel_bias'],2)
    sl = np.round(stats_un['slope'],2)
    ip = np.round(stats_un['intercept'],2)
    n = stats_un['n']
    ax.text(0.70,0.3,f'Unweighted Stats\nRMSD = {rmsd} $\mu$atm\nBias = {bias} $\mu$atm\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    if gcb:
        fig.savefig(os.path.join(model_save_loc,'plots','total_validation_gcb.png'),format='png',dpi=300)
    else:
        fig.savefig(os.path.join(model_save_loc,'plots','total_validation.png'),format='png',dpi=300)

def unweight(x,y,ax,c):
    stats_un = ws.unweighted_stats(x,y,'val')
    h2 = ax.plot(c,c*stats_un['slope']+stats_un['intercept'],'k-.',zorder=5, label = 'Unweighted')
    rmsd = '%.2f' %np.round(stats_un['rmsd'],2)
    bias = '%.2f' %np.round(stats_un['rel_bias'],2)
    sl = '%.2f' %np.round(stats_un['slope'],2)
    ip = '%.2f' %np.round(stats_un['intercept'],2)
    n = stats_un['n']
    ax.text(0.55,0.3,f'Unweighted Stats\nRMSD = {rmsd} $\mu$atm\nBias = {bias} $\mu$atm\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    return h2

def weighted(x,y,weights,ax,c):
    #weights = 1/(np.sqrt(y_test_all[:,1]**2 + y_test_preds_all[:,1]**2))
    stats = ws.weighted_stats(x,y,weights,'val')
    h1 = ax.plot(c,c*stats['slope']+stats['intercept'],'k--',zorder=5, label = 'Weighted')
    rmsd = '%.2f' % np.round(stats['rmsd'],2)
    bias = '%.2f' %np.round(stats['rel_bias'],2)
    sl = '%.2f' %np.round(stats['slope'],2)
    ip = '%.2f' %np.round(stats['intercept'],2)
    n = stats['n']
    ax.text(0.02,0.95,f'Weighted Stats\nRMSD = {rmsd} $\mu$atm\nBias = {bias} $\mu$atm\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    return h1

def plot_total_validation_unc(input_file=False,fco2_sst=False,model_save_loc=False, save_file=False,fco2_cutoff_low = 50,fco2_cutoff_high = 750,ice = None,per_prov=True):
    """
    """
    if not save_file:
        save_file = os.path.join(model_save_loc,'plots','total_validation.png')
        if per_prov:
            save_file_p = os.path.join(model_save_loc,'plots','per_prov_validation.png')
    print('Plotting validation....')

    c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
    fco2 = c.variables['fco2'][:]
    fco2_unc = np.sqrt(c.variables['fco2_net_unc'][:]**2 + c.variables['fco2_para_unc'][:]**2)
    fco2 = np.reshape(fco2,(-1,1))
    fco2_unc = np.reshape(fco2_unc,(-1,1))
    c.close()

    c = Dataset(input_file,'r')
    soc = np.array(c.variables[fco2_sst+'_reanalysed_fCO2_sw'][:])
    soc_s = np.array(c.variables[fco2_sst+'_reanalysed_fCO2_sw_std'][:])
    soc_s[np.isnan(soc_s)==True] = 0
    soc_s = np.sqrt(soc_s**2 + 5**2)
    if ice:
        ic = c.variables[ice][:]
        soc[ic > 0.95] = np.nan
    soc = np.reshape(soc,(-1,1))
    soc_s = np.reshape(soc_s,(-1,1))
    c.close()
    soc[soc < fco2_cutoff_low] = np.nan
    soc[soc > fco2_cutoff_high] = np.nan

    c = Dataset(os.path.join(model_save_loc,'networks','biomes.nc'))
    prov = c.variables['prov'][:]
    prov = np.reshape(prov,(-1,1))
    c.close()

    f = np.where((np.isnan(soc) == False) & (np.isnan(prov) == False) & (np.isnan(fco2) == False))
    fco2 = fco2[f]
    soc = soc[f]
    prov=prov[f]
    soc_s = soc_s[f]
    fco2_unc = fco2_unc[f]


    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(2,2, figure=fig, wspace=0.2,hspace=0.2,bottom=0.1,top=0.95,left=0.1,right=0.98)
    c = np.array([0,800])
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    t = 0
    for v in np.unique(prov)[:]:
        ind_train,ind_val,ind_test = load(open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'),'rb'))
        print(f'Train = {len(ind_train)} - Val = {len(ind_val)} - Test = {len(ind_test)}')
        ind_trval = np.concatenate([ind_train, ind_val])
        if t == 0:
            tot_s = np.stack((soc[ind_trval],soc_s[ind_trval]),axis=1)
            tot_n = np.stack((fco2[ind_trval],fco2_unc[ind_trval]),axis=1)
            t = 1
        else:
            tot_s = np.concatenate((tot_s,np.stack((soc[ind_trval],soc_s[ind_trval]),axis=1)))
            tot_n = np.concatenate((tot_n,np.stack((fco2[ind_trval],fco2_unc[ind_trval]),axis=1)))
    ax[0].scatter(tot_s[:,0],tot_n[:,0],s=2)
    h2 = unweight(tot_s[:,0],tot_n[:,0],ax[0],c)
    h1 = weighted(tot_s[:,0],tot_n[:,0],1/np.sqrt(tot_s[:,1]**2 + tot_n[:,1]**2),ax[0],c)
    t=0
    if per_prov:
        col = 6
        row = int(np.ceil(len(np.unique(prov))/col))
        fig2 = plt.figure(figsize=(col*8,row*8))
        gs = GridSpec(row,col, figure=fig2, wspace=0.25,hspace=0.25,bottom=0.05,top=0.98,left=0.05,right=0.98)
        axs = [[fig2.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]
        flatList = [element for innerList in axs for element in innerList]
        axs = flatList
        tp = 0
    for v in np.unique(prov)[:]:
        ind_train,ind_val,ind_test = load(open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'),'rb'))
        w = ws.weighted_stats(soc[ind_test],fco2[ind_test],1/np.sqrt(soc_s[ind_test]**2 + fco2_unc[ind_test]**2),'b')
        if per_prov:
            axs[tp].scatter(soc[ind_test],fco2[ind_test])
            unweight(soc[ind_test],fco2[ind_test],axs[tp],c)
            weighted(soc[ind_test],fco2[ind_test],1/np.sqrt(soc_s[ind_test]**2 + fco2_unc[ind_test]**2),axs[tp],c)
            axs[tp].set_title(f'Province {v}')
            axs[tp].set_xlim(c); axs[tp].set_ylim(c); axs[tp].plot(c,c,'k-');
            axs[tp].set_xlabel('in situ fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
            axs[tp].set_ylabel('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
            tp = tp+1
        if t == 0:
            tot_ts = np.stack((soc[ind_test],soc_s[ind_test]),axis=1)
            tot_tn = np.stack((fco2[ind_test],fco2_unc[ind_test]),axis=1)
            rmsd = np.array([v,w['rmsd']])
            #print(rmsd)
            t = 1
        else:
            tot_ts = np.concatenate((tot_ts,np.stack((soc[ind_test],soc_s[ind_test]),axis=1)))
            tot_tn = np.concatenate((tot_tn,np.stack((fco2[ind_test],fco2_unc[ind_test]),axis=1)))
            rmsd = np.vstack((rmsd,np.array([v,w['rmsd']])))
            #print(rmsd)
    print(rmsd)
    np.savetxt(os.path.join(model_save_loc,'validation','independent_test_rmsd.csv'), rmsd, delimiter=",")
    ax[1].scatter(tot_ts[:,0],tot_tn[:,0],s=2)
    h2 = unweight(tot_ts[:,0],tot_tn[:,0],ax[1],c)
    h1 = weighted(tot_ts[:,0],tot_tn[:,0],1/np.sqrt(tot_ts[:,1]**2 + tot_tn[:,1]**2),ax[1],c)

    ts = np.concatenate((tot_ts,tot_s))
    tn = np.concatenate((tot_tn,tot_n))
    ax[2].scatter(ts[:,0],tn[:,0],s=2)
    h2 = unweight(ts[:,0],tn[:,0],ax[2],c)
    h1 = weighted(ts[:,0],tn[:,0],1/np.sqrt(ts[:,1]**2 + tn[:,1]**2),ax[2],c)

    ax[3].scatter(ts[:,0],tn[:,0],s=2,c='r',zorder = 3)
    ax[3].errorbar(ts[:,0],tn[:,0],xerr=ts[:,1],yerr=tn[:,1],linestyle='none')

    for i in [0,1,2,3]:
        ax[i].set_xlim(c); ax[i].set_ylim(c); ax[i].plot(c,c,'k-');
        ax[i].set_xlabel('in situ fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
        ax[i].set_ylabel('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
    ax[0].set_title('Training and Validation')
    ax[1].set_title('Independent Test')
    ax[2].set_title('Train, Val and Test')
    ax[3].set_title('Train, Val and Test with errorbars')

    fig.savefig(save_file,format='png',dpi=300)
    if per_prov:
        fig2.savefig(save_file_p,format='png',dpi=300)

def add_validation_unc(model_save_loc):
    c = Dataset(os.path.join(model_save_loc,'networks','biomes.nc'))
    prov = c.variables['prov'][:]
    s = prov.shape
    prov = np.reshape(prov,(-1,1))
    c.close()
    val = np.loadtxt(os.path.join(model_save_loc,'validation','independent_test_rmsd.csv'),delimiter=',')

    val_unc = np.zeros((prov.shape))
    val_unc[:] = np.nan
    for v in range(0,val.shape[0]):
        val_unc[prov == v+1] = val[v,1]
    val_unc = np.reshape(val_unc,s)

    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    var_o = c.createVariable('fco2_val_unc','f4',('longitude','latitude','time'))
    var_o[:] = val_unc
    c.close()

def add_total_unc(model_save_loc):
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    ne = c.variables['fco2_net_unc'][:]
    pa = c.variables['fco2_para_unc'][:]
    va = c.variables['fco2_val_unc'][:]

    tot = np.sqrt(ne**2 + pa**2 + va**2)
    var_o = c.createVariable('fco2_tot_unc','f4',('longitude','latitude','time'))
    var_o[:] = tot
    c.close()

def plot_mapped(model_save_loc):
    c = Dataset(model_save_loc+'/output.nc')
    lat = np.squeeze(c.variables['latitude'][:])
    lon = np.squeeze(c.variables['longitude'][:])
    fco2 = c.variables['fco2'][:]
    fco2_net_unc = c.variables['fco2_net_unc'][:]
    fco2_para_unc = c.variables['fco2_para_unc'][:]
    fco2_val_unc = c.variables['fco2_val_unc'][:]
    fco2_tot_unc = c.variables['fco2_tot_unc'][:]
    c.close()
    print(lat.shape)
    print(lon.shape)
    print(fco2.shape)
    fco2_plot = np.transpose(np.nanmean(fco2[:,:,-12:-1],axis=2))
    fco2_u_plot = np.transpose(np.nanmean(fco2_net_unc[:,:,-12:-1],axis=2))
    fco2_up_plot = np.transpose(np.nanmean(fco2_para_unc[:,:,-12:-1],axis=2))
    fco2_val_plot = np.transpose(np.nanmean(fco2_val_unc[:,:,-12:-1],axis=2))
    fco2_tot_plot = np.transpose(np.nanmean(fco2_tot_unc[:,:,-12:-1],axis=2))
    fig = plt.figure(figsize=(21,21))
    gs = GridSpec(3,2, figure=fig, wspace=0.1,hspace=0.10,bottom=0.05,top=0.95,left=0.05,right=1.00)
    ax1 = fig.add_subplot(gs[0,0]);
    pc = ax1.pcolor(lon,lat,fco2_plot)
    cbar = plt.colorbar(pc,ax=ax1)
    cbar.set_label('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)');
    pc.set_clim([300,500])
    s = model_save_loc.split('/')
    ax1.set_title(s[-1])

    ax2 = fig.add_subplot(gs[0,1]);
    pc = ax2.pcolor(lon,lat,fco2_u_plot)
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ network uncertainty ($\mu$atm)');
    pc.set_clim([0,50])

    ax3 = fig.add_subplot(gs[1,0]);
    pc = ax3.pcolor(lon,lat,fco2_up_plot)
    cbar = plt.colorbar(pc,ax=ax3)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ input parameter uncertainty ($\mu$atm)');
    pc.set_clim([0,50])

    ax4 = fig.add_subplot(gs[1,1]);
    pc = ax4.pcolor(lon,lat,fco2_val_plot)
    cbar = plt.colorbar(pc,ax=ax4)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ validation uncertainty ($\mu$atm)');
    pc.set_clim([0,50])

    ax5 = fig.add_subplot(gs[2,0]);
    pc = ax5.pcolor(lon,lat,fco2_tot_plot)
    cbar = plt.colorbar(pc,ax=ax5)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ total uncertainty ($\mu$atm)');
    pc.set_clim([0,50])

    s = model_save_loc.split('/')
    ax1.set_title(s[-1])

    fig.savefig(os.path.join(model_save_loc,'plots','mapped_example.png'),format='png',dpi=300)

def neural_network_map(mapping_data,var=None,model_save_loc=None,prov = None,output_size=None,unc = None,ens=10):
    """
    """
    inp = np.array(mapping_data[var])
    prov = np.array(mapping_data[prov])
    print(inp.shape)
    out = np.empty((inp.shape[0]))
    out[:] = np.nan
    unc_net = np.copy(out)
    unc_para = np.copy(out)
    for v in np.unique(prov[~np.isnan(prov)]):
        scalar = load(open(os.path.join(model_save_loc,'scalars',f'prov_{v}_scalar.pkl'),'rb'))
        f = np.squeeze(np.argwhere(prov == v))
        print(f'Number of samples for {v}: {len(f)}')
        mod_inp = scalar.transform(inp[f,:])
        #mod_inp = inp[f,:]
        out_t = np.zeros((len(f),ens))
        for i in range(ens):
            mod = tf.keras.models.load_model(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens{i}'),compile=False)
            out_t[:,i] = np.squeeze(mod.predict(mod_inp))
        out[f] = np.nanmean(out_t,axis=1)
        unc_net[f] = np.nanstd(out_t,axis=1)*2
        if unc:
            lut = load(open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'rb'))
            lut = lut[0]
            unc_para[f] = np.squeeze(lut_retr(lut,mod_inp))
    out = np.reshape(out,(output_size))
    unc_net = np.reshape(unc_net,(output_size))
    unc_para = np.reshape(unc_para,(output_size))
    return out,unc_net,unc_para

def save_mapped_fco2(data,net_unc,para_unc,data_shape = None, model_save_loc = None, lon = None,lat = None):
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'w',format='NETCDF4_CLASSIC')
    c.code_used = 'neural_network_train.py'
    c.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    c.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'

    c.createDimension('longitude',data_shape[0])
    c.createDimension('latitude',data_shape[1])
    c.createDimension('time',data_shape[2])
    var_o = c.createVariable('fco2','f4',('longitude','latitude','time'))
    var_o[:] = data
    data_smooth = spatial_smooth(data)
    var_o = c.createVariable('fco2_smoothed','f4',('longitude','latitude','time'))
    var_o[:] = data_smooth
    var_o = c.createVariable('fco2_net_unc','f4',('longitude','latitude','time'))
    var_o[:] = net_unc
    var_o = c.createVariable('fco2_para_unc','f4',('longitude','latitude','time'))
    var_o[:] = para_unc
    lat_o = c.createVariable('latitude','f4',('latitude'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = c.createVariable('longitude','f4',('longitude'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    c.close()

def spatial_smooth(data):
    import scipy.signal as sg
    kernel_shape = (3, 3)
    kernel = np.full(kernel_shape, 1/np.prod(kernel_shape))
    data_smooth = np.empty((data.shape))
    data_smooth[:] = np.nan
    for i in range(data.shape[2]):
        data_smooth[:,:,i] = sg.convolve2d(data[:,:,i], kernel, mode='same',boundary = 'wrap')
    data_smooth[np.isnan(data_smooth) == 1] = data[np.isnan(data_smooth) == 1]
    return data_smooth
