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
import Data_Loading.data_utils as du
font = {'weight' : 'normal',
        'size'   : 19}
matplotlib.rc('font', **font)

tf.autograph.set_verbosity(0)

def driver(data_file,fco2_sst = None, prov = None,var = [],unc = None, model_save_loc = None,
    bath = None, bath_cutoff = None, fco2_cutoff_low = None, fco2_cutoff_high = None,sea_ice=None,tot_lut_val=6000,activ = 'sigmoid',socat_sst=False,ens=10,name='fco2'):
    """
    This is the driver function to load the data from the input file, trim the data extremes(?) and then call
    the neural network package.
    """
    print('Creating output directory tree...')
    make_save_tree(model_save_loc)
    print('Neural_Network_Start...')

    vars = [fco2_sst+'_reanalysed_fCO2_sw',fco2_sst+'_reanalysed_fCO2_sw_std', fco2_sst+'_reanalysed_count_obs']
    if socat_sst:
        vars.append(fco2_sst+'_reanalysed_sst')
    vars.append(prov)
    if not bath_cutoff == None:
        vars.append(bath)
    if not sea_ice == None:
        vars.append(sea_ice)

    for v in var:
        vars.append(v)
    tabl,output_size,lon,lat,time = load_data(data_file,vars,model_save_loc,outp=True)
    # if not sea_ice == None:
    #     print(f'Setting where sea ice is greater than 95% to nan...')
    #     tabl[tabl[sea_ice] > 0.95] = np.nan

    mapping_data = tabl
    print(mapping_data)
    # Where the std_dev is 0 but an fCO2 value exists indicates a single cruise. So we input a stddev of 0 for
    # these values, so then we can combine the measurement unc (assumed to be ~5uatm from Bakker et al. 2016) later.
    tabl[vars[1]][((np.isnan(tabl[vars[0]]) == 0) & (np.isnan(tabl[vars[1]]) == 1))] = 0
    tabl[vars[2]][tabl[vars[2]]<0] = 0
    # Only keep rows where we have all the data, i.e no gaps
    tabl = tabl[(np.isnan(tabl) == 0).all(axis=1)]
    #print(tabl)
    #Combine the stdev with an assumed measurement unc of 5 uatm (Bakker et al. 2016).
    tabl[vars[1]] = np.sqrt((tabl[vars[1]]/np.sqrt(tabl[vars[2]]))**2 + 5**2)

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
    for v in np.unique(tabl[prov]):
        print(str(v) + ' : '+str(np.argwhere(np.array(tabl[prov]) == v).shape))

    if socat_sst:
        net_train_var = [fco2_sst+'_reanalysed_sst']
        for v in var[1:]:
            net_train_var.append(v)
    else:
        net_train_var = var
    print(net_train_var)
    run_neural_network(tabl,fco2 = vars[0], prov = prov, var = net_train_var, model_save_loc = model_save_loc,unc = unc,tot_lut_val = tot_lut_val,activ = activ,ens=ens)

    # Next function runs the neural network ensemble to produce complete maps of fCO2(sw), alongside the network (standard dev of neural net ensembles) and parameter uncertainties
    # (propagated input parameter uncertainties)
    mapped,mapped_net_unc,mapped_para_unc = neural_network_map(mapping_data,var=var,model_save_loc=model_save_loc,prov = prov,output_size=output_size,unc = unc,ens=ens)
    # Then we save the fCO2 data
    save_mapped_fco2(mapped,mapped_net_unc,mapped_para_unc,data_shape = output_size, model_save_loc = model_save_loc, lon = lon,lat = lat,time = time)
    # Once saved the validation can be extracted, and used to determine the validation uncertainty for each province,
    # which can then be mapped. This function produces validation statistics for the training/validation, test and all data together
    # using both weighted and unweighted statistics.
    plot_total_validation_unc(fco2_sst = fco2_sst,model_save_loc = model_save_loc,ice = sea_ice,prov = prov)
    add_validation_unc(model_save_loc,data_file=data_file,prov=prov)
    # This then produces the total uncertainty (combine parameter, network and validation uncertainties in quadrature)
    add_total_unc(model_save_loc)
    # Plot the mean of the last year of the timeseries for a sanity check.
    plot_mapped(model_save_loc)

    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    c.variables[name].predictor_parameters = str(var)
    c.variables[name].ensemble_size = str(ens)
    c.variables[name].province_variable = str(prov)
    c.variables[name+'_para_unc'].uncertainty_vals_lut = str(unc)
    c.variables[name+'_para_unc'].uncertainty_vals_lut_parameters = str(var)
    c.variables[name+'_para_unc'].lut_table_max_size = str(tot_lut_val)
    c.close()

def daily_neural_driver(data_file,fco2_sst = None, prov = None,var = [],mapping_var=[],mapping_prov = [],unc = None, model_save_loc = None,
    bath = None, bath_cutoff = None, cutoff_low = None, cutoff_high = None,sea_ice=None,tot_lut_val=6000,mapping_file=[],ktoc = None,epochs=200,node_in = range(6,31,3),
    sep = '\t',name='fco2',longname='Fugacity of CO2 in seawater',unit = 'uatm',c = [0,1500],learning_rate=0.01,plot_unit = '$\mu$atm',plot_parameter = 'fCO$_{2 (sw)}$',
    lat_v = '',lon_v='',year_v='',mon_v='',day_v='',ens=10):
    vars = [fco2_sst,fco2_sst+'_std',lat_v,lon_v,year_v,mon_v,day_v]
    vars.append(prov)
    if not bath_cutoff == None:
        vars.append(bath)
    if not sea_ice == None:
        vars.append(sea_ice)

    for v in var:
        vars.append(v)
    data = pd.read_table(data_file,sep=sep)
    data[fco2_sst+'_std'] = np.zeros((len(data)))
    data = data[vars]
    if not cutoff_low == None:
        print(f'Trimming fCO2(sw) less than {cutoff_low} uatm...')
        print(data.shape)
        data = data[(data[vars[0]] > cutoff_low)]
        print(data.shape)

    if not cutoff_high == None:
        print(f'Trimming fCO2(sw) greater than {cutoff_high} uatm...')
        print(data.shape)
        data = data[(data[vars[0]] < cutoff_high)]
        print(data.shape)
    data = data[(np.isnan(data) == 0).all(axis=1)]

    for v in np.unique(data[prov]):
        print(str(v) + ' : '+str(np.argwhere(np.array(data[prov]) == v).shape))

    run_neural_network(data,fco2 = vars[0], prov = prov, var = var, model_save_loc = model_save_loc,unc = unc,tot_lut_val = tot_lut_val,epochs=epochs,node_in = node_in,learning_rate = learning_rate,ens=ens)
    plot_total_validation_unc(fco2_sst = fco2_sst,model_save_loc = model_save_loc,ice = sea_ice,prov = prov,daily=True,var=var,fco2_cutoff_low = cutoff_low,fco2_cutoff_high=cutoff_high,c_plot=np.array(c),unit = plot_unit,parameter = plot_parameter,
        year_col = year_v,month_col=mon_v,day_col=day_v, lat_col = lat_v,lon_col=lon_v)
    print(mapping_var)
    map_vars = mapping_var.copy()
    map_vars.append(mapping_prov)
    tabl,output_size,lon,lat,time = load_data(mapping_file,map_vars,model_save_loc,outp=False)
    if ktoc:
        tabl[ktoc] = tabl[ktoc] - 273.15
    print(tabl)
    print(mapping_var)
    mapped,mapped_net_unc,mapped_para_unc = neural_network_map(tabl,var=mapping_var,model_save_loc=model_save_loc,prov = mapping_prov,output_size=output_size,unc = unc)

    save_mapped_fco2(mapped,mapped_net_unc,mapped_para_unc,data_shape = output_size, model_save_loc = model_save_loc, lon = lon,lat = lat,name = name,longname=longname,unit = unit)
    add_validation_unc(model_save_loc,mapping_file,prov,name = name,longname=longname,unit = unit)
    add_total_unc(model_save_loc,name = name,longname=longname,unit = unit)

    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    c.variables[name].predictor_parameters = str(var)
    c.variables[name].ensemble_size = str(ens)
    c.variables[name].province_variable = str(prov)
    c.variables[name+'_para_unc'].uncertainty_vals_lut = str(unc)
    c.variables[name+'_para_unc'].uncertainty_vals_lut_parameters = str(var)
    c.variables[name+'_para_unc'].lut_table_max_size = str(tot_lut_val)
    c.close()

"""
Flag E valdiation needs updating to the new construct. Treat this as a independent test dataset (29/07/2023).
Need to work out why reanalysis script currently doesn't work for SOCAT2023...
"""
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

def load_data(load_loc,vars,output_loc=[],outp=True):
    """
    Function to load the data variables into a pandas dataframe for use in the neural
    network training.
    """
    from construct_input_netcdf import save_netcdf
    print('Loading data...')
    c = Dataset(load_loc,'r')
    lon = np.array(c.variables['longitude'][:])
    lat = np.array(c.variables['latitude'][:])
    time = list(c.variables['time'][:])
    t = 0
    # This loop cycles through the netcdf variables and loads the ones defined in vars
    ins_save = {}
    for v in vars:
        data = np.squeeze(np.array(c.variables[v][:]))
        if t == 0:
            # Create the output array with the correct size (allows the inputs to be any shape size etc.)
            d = np.append(data.shape,len(vars))
            output = np.empty((d))
            output[:] = np.nan
        output[:,:,:,t] = data
        ins_save[v] = data
        t += 1
    if outp:
        save_netcdf(os.path.join(output_loc,'inputs','neural_network_train_input_values.nc'),ins_save,lon,lat,output.shape[2],time_track=time)
    # Reshape the 3d numpy arrays into a single colum and put into a dataframe.
    for i in range(0,len(vars)):
        if i == 0:
            tabl = pd.DataFrame()
        temp = np.squeeze(np.reshape(output[:,:,:,i],(-1,1)))
        #print(temp.shape)
        tabl[vars[i]] = temp
    c.close()
    #Returns the loaded data, the size of the data in longitude,latitude,time, the longitude array
    #and the latitude array
    return tabl,d[0:3],lon,lat,time

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
    inputs = os.path.join(model_save_loc,'inputs')
    if not os.path.isdir(inputs):
        os.mkdir(inputs)
    decor = os.path.join(model_save_loc,'decorrelation')
    if not os.path.isdir(decor):
        os.mkdir(decor)

def run_neural_network(data,fco2 = None,prov = None,var=None,model_save_loc=None,plot=False, unc = None, ens = 10,tot_lut_val=6000,epochs=200,node_in = range(6,31,3),activ = 'sigmoid',
    learning_rate =0.01):
    """
    Function to run the nerual network training, and saving the best performing model. This function
    produces the model, scaler, uncertainty look up table and validation results for use in producing
    global maps of fCO2 (sw,subskin), with per pixel uncertainties.
    """
    """
    Neural Network intilisation settings
    """
    data.to_csv(os.path.join(model_save_loc,'training.tsv'),sep='\t')
    # Set early stopping condition: 1e-4
    #es = tf.keras.callbacks.EarlyStopping(monitor='val_rmse', mode='min', patience=10, min_delta=1e-4)
    # Here we set a early stop on the loss function (i.e the MSE) so that if it doesn't change by a certain
    # amount the nerual network training is stopped. For the loss function in this application it ranges between
    # 400 - 1700 depending on the province.
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10, min_delta=5)

    #rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse', dtype=None)
    # Mean square error used instead of RMSD - found the neural network output wasn't producing "good" results when
    # using RMSD.
    mse = tf.keras.losses.MeanSquaredError()
    #rmse = tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)

    #Optimizer was set to Adam in Josh Blannin's code but the SGD optimizer on testing produced better overall results.
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    #opt = tf.keras.optimizers.Adam()
    #Set the node increments (set initially as 2**i (i.e base 2 values)) - changed 07/2023 to mutliples of 5. and then to multiples of 10 (26/07/2023)
    # Found that using low numbers of neurons were hurting the neural network performance so upped the numbers.
    # A pretraining step is used so the number of neurons actually used is kept to a minimum.
    node_increments = [i*10 for i in node_in] # This ranges neurons from 60 to 300 in 30 increments
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
    # For each province the neural network training procedure is roughly
    # 1. Pre training step to determine number of neruons.
    # 2. Train 10 enembles of the neural network with that number of neurons.
    # 3. For ensemble 0, train the input parameter uncertainty look up table.
    for v in np.unique(prov):
        print(f'Running province number {v}...')
        mask = np.argwhere(np.array(prov) == v)
        #print(len(mask))
        # Here we split  5% of the data to act as an independent dataset (i.e not included in the nerual network training at all)
        # This leaves 95% to be split into two groups; the training and validation sets at ~80% and ~20% of the train/val dataset respectively
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
        min_n = None # This gets set to the node number that has the lowest loss function

        """
        1. Start neruon pretraining step...
        """
        for n in node_increments:
            print(n)
            # Here we setup the Tensor Flow Model (Thanks to Josh Blannin for example code)
            model = tf.keras.models.Sequential(name='PROV_'+str(v)) # Setup the model
            model.add(tf.keras.layers.Dense(n, input_dim=len(var),activation=activ)) # Add Hidden layer - changed from relu to sigmoid as seems to produce better results.
            #model.add(tf.keras.layers.Dense(n, activation='sigmoid')) # Add Hidden layer - changed from relu to sigmoid as seems to produce better results.
            model.add(tf.keras.layers.Dense(1, activation='linear')) # Add Output layer
            model.compile(optimizer=opt, loss='mse') # Set the loss function and metric to determine the training results
            # Here we train the neural network.
            history = model.fit(X_train, y_train[:,0], epochs=epochs, validation_data=[X_val,y_val[:,0]], verbose=0, callbacks=[es],shuffle=False, batch_size=int(len(X_train)/50))

            # Calculate the independent test dataset loss function
            rms_error = history.history['loss'][-1]
            print(f'prov = {v}; n = {n}; loss = {rms_error}') # Print the province number, node increment and RMSD.

            # If the model "substantially" improves the fit then it becomes the best model
            # The added performance requires for a small gain in this case is not beneficial. If too many nodes are added the model may become overtrained.
            # and won't be generalised.
            # In this case we are looking at the loss function, so a arbitary improvement of 20 is selected (i.e the new model must improve by 20 to be used as the new best model)
            if (rms_error - min_rmse < -20):
                print('New Best Model!')
                min_rmse = rms_error
                min_n = n

        # min_n = 100
        print(f'Minimum n found: {min_n} - Running neural network ensemble for prov {v}')
        """
        2. Start nerual network ensemble training.
        """
        for i in range(ens):
            loss_check=0
            loss_count = 0
            while loss_check == 0:
                # Extract the same independent test data for all nerual networks, but trainign and validation split varies with each nerual network.
                # Trainign and validation are therefore quasi seperate... and should be deemed as 1 dataset when validating.
                X_train, X_val, X_test, y_train, y_val, y_test,ind_train,ind_val,ind_test = train_val_test_split(X[mask,:], y[mask,:],inds, indep = 0.05, train = 0.2)
                # Apply the standard scalar...
                X_train = StdScl.transform(X_train); X_val = StdScl.transform(X_val); X_test = StdScl.transform(X_test);

                print(f'Running ensemble {i} for prov {v}')
                """
                Neural network setup and training.
                """

                model = tf.keras.models.Sequential(name='PROV_'+str(v)) # Setup the model
                model.add(tf.keras.layers.Dense(min_n, input_dim=len(var), activation=activ)) # Add Hidden layer
                #model.add(tf.keras.layers.Dense(n, activation='sigmoid'))
                model.add(tf.keras.layers.Dense(1, activation='linear')) # Add Output layer
                model.compile(optimizer=opt, loss='mse') # Set the loss function and metric to determine the training results
                # Here we train the neural network.
                history = model.fit(X_train, y_train[:,0], epochs=epochs, validation_data=[X_val,y_val[:,0]], verbose=0, callbacks=[es],shuffle=False, batch_size=int(len(X_train)/50))
                loss = history.history['loss'][-1]
                print(f'Ensemble Loss = { loss } ensemble {i}')
                # Save each of the ensemble outputs
                model.save(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens{i}'))
                if (np.abs(min_rmse - loss)/min_rmse <0.4) or (loss < min_rmse):
                    loss_check=1
                    print(f'Loss similar - ensemble member {i} accepted...')
                else:
                    print(f'Loss not similar - restarting ensemble {i}')
                    loss_count = loss_count+1
                if loss_count == 5:
                    print(f'Loss appears to not be achievable... Doubling the achieveable loss...')
                    min_rmse = (min_rmse + loss)/2
                    loss_count = 0
        # Here we dump the indexes of each of the trainign datasets. These are reuse for the validation step and producing the validation unc...
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
            lut = unc_lut_generate(minmax,model,StdScl,unc,tot_lut_val=tot_lut_val)
            # and save the look up table.
            dump([lut],open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'wb'))

def unc_lut_generate(minmax,model,scalar,unc,res = False,tot_lut_val=6000):
    """
    This function generates a lookup table to determine the input parameter uncertainity.
    We generate a regular linear grid between the min and max values of all inputs with a set number of steps
    so that the LUT table is manageable to produce.
    """
    print('Calculating uncertainty look up table...')
    if not res:
        # Here we calculate the number of steps in the LUT, so that the full lookup table is ~6000 values.
        # So the more input parameters the less resolved the LUT is.
        step = 0
        vals = 0
        while vals < tot_lut_val:
            step = step+1
            vals = step**len(unc)
        res = step-1

    print(f'Resolution of LUT = {res}... total values = {res**len(unc)}')
    rang = (((minmax[1,:] - minmax[0,:])*1.5)+0.01)/2 # Calculate the range of each value, and divide by two
    me = np.mean(minmax,axis=0) # Find the middle value of the the range
    m = [np.linspace(i,j,res) for i,j in zip(me-rang,me+rang)] # For each value we produce a linearly space grid between the min and max values.
    grids = np.meshgrid(*m) # Then mesh these together into one big grid.
    sh = grids[0].shape # Save the shape of the meshgird for later
    #Reshape the grids so they are 1 column.
    for i in range(len(grids)):
        grids[i] = np.reshape(grids[i],(-1,1))
    # Stack these individual grids together
    grid_o = np.transpose(np.squeeze(np.stack(grids)))
    # Then run the uncertainity determination
    unc_o = uncertainty_montecarlo(model,scalar,grid_o,unc)
    # Reshape the grid back to the orginial shape
    unc_o = np.reshape(unc_o,(sh))
    # Stack the linear spaced input arrays together
    m2 = np.transpose(np.squeeze(np.stack(m)))
    # Transform the linear spaced inputs into scalar space.
    m2 = scalar.transform(m2)
    m = []
    for i in range(m2.shape[1]):
        m.append(m2[:,i])
    # Combine the linear spaced grid in scalar space with the uncertainty information.
    outp = [m,unc_o]
    return outp

def lut_retr(lut,values):
    """
    Function to retrieve values from the uncertainty lookup table produced by unc_lut_generate
    """
    from scipy.interpolate import interpn
    vals = interpn(lut[0], lut[1], values,bounds_error=False,fill_value=np.amax(lut[1]))
    return vals

def uncertainty_montecarlo(model,scalar,X_vals,unc,repeat = 100):
    """
    Function to run a montecarlo uncertainty approach to workout the sensitivity of the nerual
    network to the input parameters. For each combination of values, a 100 ensemble of the mean perturbed
    randomly by the uncertainity is produced, and the resulting fCO2 distribution is used to calculate the
    2 standard deviation which indicates the uncertainty.
    """
    print('Running MonteCarlo...')
    print(X_vals.shape)
    #X_vals = scalar.inverse_transform(X_vals)
    outstd = np.empty((X_vals.shape[0]))
    outstd[:] = np.nan
    # For each combination of values we run a monte carlo approac
    for i in range(X_vals.shape[0]):
        print(f'{i} in {X_vals.shape[0]}')
        #produce an array for the montecarlo values (i.e 100 ensembles of mean + random component)
        mod_inp = np.empty((repeat,X_vals.shape[1]))
        mod_inp[:] = np.nan
        # Add the random value for each of the inputs perturbed by its uncertainty
        for j in range(X_vals.shape[1]):
            mod_inp[:,j] = np.random.normal(X_vals[i,j],unc[j],repeat)
        # Transform these usign the scalar to put them into the neural network input space
        mod_inp = scalar.transform(mod_inp)
        # Predict the output with neural network
        mod_out = model.predict(mod_inp)
        #print(mod_out)
        # Out value is the 1 standard deviaiton of this as we assume the input uncertainty values are the 95% confidence.
        outstd[i] = np.std(mod_out)
    return outstd

def train_val_test_split(X,y,ind, indep = 0.2, train = 0.4):
    """
    Function to split the input arrays into a training, validation and test dataset
    The test dataset is set to be the same every time, but the train/val split can change.
    """
    X_train, X_test, y_train, y_test,ind_train,ind_test = train_test_split(X, y,ind, test_size=indep, random_state=2) # Random state makes the test dataset the same everytime.
    X_train, X_val, y_train, y_val,ind_train,ind_val = train_test_split(X_train, y_train, ind_train, test_size=train) # 0.25 x 0.8 = 0.2
    return np.squeeze(X_train), np.squeeze(X_val), np.squeeze(X_test), np.squeeze(y_train), np.squeeze(y_val), np.squeeze(y_test), np.squeeze(ind_train), np.squeeze(ind_val),np.squeeze(ind_test)

"""
This function is not needed - unweighted statistics are produced with the weighted ones.
The neural network approach here is all about defined uncertainties, so this is obsolete.
"""
# def plot_total_validation(model_save_loc,gcb = False):
#     """
#     Function to produce a validation statistic without determine if the data was training/validation/test. I.e just combine all the data.
#     This fucntion can be called seperate on other datasets to produce a GCB like validation.
#     Need to rewrite!!! - 29/07/2023 DJF
#     """
#     print('Plotting validation....')
#     files = glob.glob(os.path.join(model_save_loc,'validation','*_validation.pkl'))
#     #files = files[0:-1]
#     #print(files)
#     y_test_total = np.array(np.nan)
#     y_test_preds_total = np.array(np.nan)
#     fig = plt.figure()
#     c = np.array([0,800])
#     t = 0
#     for v in files:
#         print(v)
#         y_train,y_train_preds,y_val,y_val_preds,y_test,y_test_preds = load(open(v,'rb'))
#         if t == 0:
#             y_test_all = np.copy(y_test[:,0])
#             y_test_preds_all = np.copy(y_test_preds)
#             t = 1
#             if gcb:
#                 y_test_all = np.concatenate((y_test_all,y_train[:,0],y_val[:,0]))
#                 y_test_preds_all =  np.concatenate((y_test_preds_all,y_train_preds,y_val_preds))
#         else:
#             if gcb:
#                 y_test_all = np.concatenate((y_test_all,y_test[:,0],y_train[:,0],y_val[:,0]))
#                 y_test_preds_all =  np.concatenate((y_test_preds_all,y_test_preds,y_train_preds,y_val_preds))
#             else:
#                 y_test_all = np.concatenate((y_test_all,y_test[:,0]))
#                 y_test_preds_all = np.concatenate((y_test_preds_all,y_test_preds))
#
#         # print(y_test.shape)
#         # print(y_test_preds.shape)
#         if gcb:
#             plt.scatter(np.concatenate((y_test[:,0],y_val[:,0],y_train[:,0])),np.concatenate((y_test_preds,y_val_preds,y_train_preds)),s=4)
#         else:
#             plt.scatter(y_test[:,0],y_test_preds,s=4)
#     print(y_test_all.shape)
#     print(y_test_preds_all.shape)
#     y_test_preds_all = np.squeeze(y_test_preds_all)
#     plt.xlim(c); plt.ylim(c); plt.plot(c,c,'k-');
#     stats_un = ws.unweighted_stats(y_test_all,y_test_preds_all,'val')
#     h2 = plt.plot(c,c*stats_un['slope']+stats_un['intercept'],'k-.',zorder=5, label = 'Unweighted')
#     plt.xlabel('in situ fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
#     plt.ylabel('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)')
#     s = model_save_loc.split('/')
#     if gcb:
#         plt.title(s[-1] + ' - GCB Evaluation = True')
#     else:
#         plt.title(s[-1] + ' - GCB Evaluation = False')
#     ax = plt.gca()
#     rmsd = np.round(stats_un['rmsd'],2)
#     bias = np.round(stats_un['rel_bias'],2)
#     sl = np.round(stats_un['slope'],2)
#     ip = np.round(stats_un['intercept'],2)
#     n = stats_un['n']
#     ax.text(0.70,0.3,f'Unweighted Stats\nRMSD = {rmsd} $\mu$atm\nBias = {bias} $\mu$atm\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
#     if gcb:
#         fig.savefig(os.path.join(model_save_loc,'plots','total_validation_gcb.png'),format='png',dpi=300)
#     else:
#         fig.savefig(os.path.join(model_save_loc,'plots','total_validation.png'),format='png',dpi=300)

def unweight(x,y,ax,c,unit = '$\mu$atm',plot=False,loc = [0.52,0.35]):
    """
    Function to calculate the unweighted statistics and add them to a scatter plot (well any plot really, in the bottom right corner)
    """
    stats_un = ws.unweighted_stats(x,y,'val')
    if plot:
        h2 = ax.plot(c,c*stats_un['slope']+stats_un['intercept'],'k--',zorder=5, label = 'Unweighted')
    rmsd = '%.2f' %np.round(stats_un['rmsd'],2)
    bias = '%.2f' %np.round(stats_un['rel_bias'],2)
    sl = '%.2f' %np.round(stats_un['slope'],2)
    ip = '%.2f' %np.round(stats_un['intercept'],2)
    n = stats_un['n']
    ax.text(loc[0],loc[1],f'Unweighted Stats\nRMSD = {rmsd} {unit}\nBias = {bias} {unit}\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    #return h2

def weighted(x,y,weights,ax,c,unit = '$\mu$atm'):
    """
    Function to calculate the weighted statistics and add them to a scatter plot (well any plot really, in the top left corner)
    """
    #weights = 1/(np.sqrt(y_test_all[:,1]**2 + y_test_preds_all[:,1]**2))
    stats = ws.weighted_stats(x,y,weights,'val')
    h1 = ax.plot(c,c*stats['slope']+stats['intercept'],'k--',zorder=5, label = 'Weighted')
    rmsd = '%.2f' % np.round(stats['rmsd'],2)
    bias = '%.2f' %np.round(stats['rel_bias'],2)
    sl = '%.2f' %np.round(stats['slope'],2)
    ip = '%.2f' %np.round(stats['intercept'],2)
    n = stats['n']
    ax.text(0.02,0.95,f'Weighted Stats\nRMSD = {rmsd} {unit}\nBias = {bias} {unit}\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    return h1

def neural_val_run(data,model_save_loc,var,provs,ens=10,unc=True,name='fco2'):
    inp = np.array(data[var])
    #print(inp.shape)
    prov = np.array(data[provs])

    out = np.empty((inp.shape[0]))
    out[:] = np.nan
    unc_net = np.copy(out)
    unc_para = np.copy(out)
    for v in np.unique(prov[~np.isnan(prov)]):
        scalar = load(open(os.path.join(model_save_loc,'scalars',f'prov_{v}_scalar.pkl'),'rb')) # Load the scalar
        f = np.squeeze(np.where(prov == v)) # Find the data within the province
        #print(f)
        print(f'Number of samples for {v}: {len(f)}')
        mod_inp = scalar.transform(inp[f,:]) # Transform the data using the scalar
        #mod_inp = inp[f,:]
        out_t = np.zeros((len(f),ens))
        #print(out_t.shape)
        # For each ensemble run the neural network and get the output
        for i in range(ens):
            #print(mod_inp.shape)
            mod = tf.keras.models.load_model(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens{i}'),compile=False)
            out_t[:,i] = np.squeeze(mod.predict(mod_inp))
        out[f] = np.nanmean(out_t,axis=1) # The output fCO2 is the mean of the ensembles
        unc_net[f] = np.nanstd(out_t,axis=1)*2 # The output fCO2 network uncertainity
        if unc:
            #Load the parameter uncertainty look up table and apply.
            lut = load(open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'rb'))
            lut = lut[0]
            unc_para[f] = np.squeeze(lut_retr(lut,mod_inp))
    data[name] = out
    data[name+'_net_unc'] = unc_net
    data[name+'_para_unc'] = unc_para
    data.to_csv(os.path.join(model_save_loc,'training_addedneural.tsv'),sep='\t',index=False)
    return data


def plot_total_validation_unc(fco2_sst=False,model_save_loc=False, save_file=False,fco2_cutoff_low = 50,fco2_cutoff_high = 750,ice = None,per_prov=True,prov = None,daily = False,var = [],
    c_plot = np.array([0,800]),name='fco2',unit = '$\mu$atm',parameter = 'fCO$_{2 (sw)}$',year_col='',month_col='',day_col='',lat_col='',lon_col=''):
    """
    Function to produce validation statistics with respect to the train/validation/test datasets. This step is extremely sensitive to the indexes used, see note below as to
    a change needed in the code to stop issues.
    DJF - need to save all the input parameters used in the neural network within the neural network folder so that issue of mismatched inputs in the construct_input_netcdf netcdf file,
    don't propagate through to the validaiton step.
    """
    header = 'Year, Month, Day, Latitude (deg N), Longitude (deg E), in situ '+name+' ('+unit+'), in situ '+name+' standard_deviation ('+unit+'), UExP-FNN-U '+name+' ('+unit+'), UExP-FNN-U network+parameter uncertainty ('+unit+'), UExP-FNN-U region (unitless)'
    if not save_file:
        save_file = os.path.join(model_save_loc,'plots','total_validation.png')
        if per_prov:
            save_file_p = os.path.join(model_save_loc,'plots','per_prov_validation.png')
    if daily:
        # if du.checkfileexist(os.path.join(model_save_loc,'training_addedneural.tsv')):
        #     data = pd.read_table(os.path.join(model_save_loc,'training_addedneural.tsv'),sep='\t')
        # else:
        data = pd.read_table(os.path.join(model_save_loc,'training.tsv'),sep='\t')
        print(data)
        data = neural_val_run(data,model_save_loc,var=var,provs=prov,name=name)
        prov = np.array(data[prov])
        fco2 = np.array(data[name])
        fco2_unc = np.array(np.sqrt(data[name+'_net_unc']**2 + data[name+'_para_unc']**2))
        soc = np.array(data[fco2_sst])
        soc_s = np.array(data[fco2_sst+'_std'])
        year = np.array(data[year_col])
        month = np.array(data[month_col])
        day = np.array(data[day_col])
        longitude = np.array(data[lon_col])
        latitude = np.array(data[lat_col])

    else:
        input_file = os.path.join(model_save_loc,'inputs','neural_network_train_input_values.nc')
        # Here we create the save file names if they arent defined.

        print('Plotting validation....')

        # Here we load the output arrays from the neural network
        c = Dataset(os.path.join(model_save_loc,'output.nc'),'r')
        latitude = c.variables['latitude'][:]
        longitude = c.variables['longitude'][:]
        latitude,longitude = np.meshgrid(latitude,longitude)
        time = c.variables['time'][:]
        time_units = c.variables['time'].units
        latitude = np.repeat(latitude[:, :, np.newaxis], time.shape[0], axis=2); longitude = np.repeat(longitude[:, :, np.newaxis], time.shape[0], axis=2);
        print(latitude.shape)
        time2 = np.zeros((latitude.shape))
        for i in range(len(time)):
            time2[:,:,i] = time[i]

        fco2 = c.variables['fco2'][:]
        fco2_unc = np.sqrt(c.variables['fco2_net_unc'][:]**2 + c.variables['fco2_para_unc'][:]**2) # Combining the network unc, and input parameter unc so we have a single uncertainty values
        # Reshape these into a single column array.
        fco2 = np.reshape(fco2,(-1,1))
        fco2_unc = np.reshape(fco2_unc,(-1,1))
        c.close()

        # Now we load the SOCAT data, and the datasets needed to trim the data in the same way as was done in the run_neural_network() function. This step must be identical
        # or the output will be wrong, because the input index wont correspond correctly...
        c = Dataset(input_file,'r')
        soc = np.array(c.variables[fco2_sst+'_reanalysed_fCO2_sw'][:])
        soc_s = np.array(c.variables[fco2_sst+'_reanalysed_fCO2_sw_std'][:])
        soc_s[np.isnan(soc_s)==True] = 0 # If the std is np.nan but there is an fCO2 value we set std to 0
        soc_s = np.sqrt(soc_s**2 + 5**2) # Combine in quadrature the std and a measurment unc of 5 (Bakker et al. 2016)
        # Remove the data where ice is greater than 0.95.
        # if ice:
        #     ic = c.variables[ice][:]
        #     soc[ic > 0.95] = np.nan
        soc = np.reshape(soc,(-1,1))
        soc_s = np.reshape(soc_s,(-1,1))
        c.close()
        # Remove the high and low socat values
        soc[soc < fco2_cutoff_low] = np.nan
        soc[soc > fco2_cutoff_high] = np.nan

        # Load the provinces so we can do the per province evaluation for the validation uncertainty.
        c = Dataset(input_file,'r')
        prov = c.variables[prov][:]
        prov = np.reshape(prov,(-1,1))
        c.close()
        latitude = np.reshape(latitude,(-1,1)); longitude = np.reshape(longitude,(-1,1)); time2 = np.reshape(time2,(-1,1))
        # Find where we have values for the SOCAT data, neural network data, and the province.
        f = np.where((np.isnan(soc) == False) & (np.isnan(prov) == False) & (np.isnan(fco2) == False))
        fco2 = fco2[f]; soc = soc[f]; prov=prov[f]; soc_s = soc_s[f]; fco2_unc = fco2_unc[f]; latitude = latitude[f]; longitude = longitude[f]; time2 = time2[f] # Trim the arrays.
        year = np.zeros((time2.shape));month = np.zeros((time2.shape));day = np.zeros((time2.shape))

        for i in range(len(time2)):
            date = datetime.datetime.strptime(time_units.split(' ')[-1],'%Y-%m-%d') + datetime.timedelta(days = int(time2[i]))
            year[i] = date.year
            month[i] = date.month
            day[i] = date.day

        #Convert the time to year, month, day

    """
    Plotting time!
    """
    # Setting up the figure for the training/validaiton, test and total validation plots
    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.25,bottom=0.1,top=0.95,left=0.1,right=0.98)
    #c = np.array([0,800])
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]

    # Training/validaiton plotting
    t = 0 # Counter so we can properly concatenate the arrays from each province (i.e we load each provinces validation index then we can append the values to the array correctly)
    for v in np.unique(prov)[:]:
        ind_train,ind_val,ind_test = load(open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'),'rb')) # Load the validation data
        print(f'Train = {len(ind_train)} - Val = {len(ind_val)} - Test = {len(ind_test)}') # Sanity check print...
        ind_trval = np.concatenate([ind_train, ind_val]) # As training and validation are quasi independent we merge the two datasets.
        if t == 0: # Start the combination array
            tot_s = np.stack((soc[ind_trval],soc_s[ind_trval]),axis=1)
            tot_n = np.stack((fco2[ind_trval],fco2_unc[ind_trval]),axis=1)
            ind_trval_all = ind_trval
            t = 1
        else: # Append the values to the array produced when t = 0
            tot_s = np.concatenate((tot_s,np.stack((soc[ind_trval],soc_s[ind_trval]),axis=1)))
            tot_n = np.concatenate((tot_n,np.stack((fco2[ind_trval],fco2_unc[ind_trval]),axis=1)))
            ind_trval_all = np.concatenate((ind_trval_all,ind_trval))
    ax[0].scatter(tot_s[:,0],tot_n[:,0],s=2) # Scatter the data onto the plot
    # Produce the scatter validation text and put onto the plot (both weighted and unweighted versions)
    h2 = unweight(tot_s[:,0],tot_n[:,0],ax[0],c_plot,unit=unit)
    h1 = weighted(tot_s[:,0],tot_n[:,0],1/np.sqrt(tot_s[:,1]**2 + tot_n[:,1]**2),ax[0],c_plot,unit=unit)
    stats_temp = ws.weighted_stats(tot_s[:,0],tot_n[:,0],1/np.sqrt(tot_s[:,1]**2 + tot_n[:,1]**2),'b')
    ax[0].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6)
    ax[0].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4)

    ind_trval_all_out = np.transpose(np.stack((year[ind_trval_all],month[ind_trval_all],day[ind_trval_all],latitude[ind_trval_all],longitude[ind_trval_all],soc[ind_trval_all],soc_s[ind_trval_all],fco2[ind_trval_all],fco2_unc[ind_trval_all],prov[ind_trval_all])))
    np.savetxt(os.path.join(model_save_loc,'validation','train_val_data.csv'), ind_trval_all_out,header = header,delimiter = ',')
    # Start independent test plotting
    # Here we have an additional step to do per province test dataset plots so we have the validation uncertainty for the mapping.
    t=0 # Counter so we can properly concatenate the arrays from each province (i.e we load each provinces validation index then we can append the values to the array correctly)
    if per_prov:
        col = 6 # 6 columns on the figure
        row = int(np.ceil(len(np.unique(prov))/col)) # Calculate the rows based on number of provinces
        font = {'weight' : 'normal',
                'size'   : 25}
        matplotlib.rc('font', **font)
        #Setting up the figure for the per province validation
        fig2 = plt.figure(figsize=(col*9,row*9))
        gs = GridSpec(row,col, figure=fig2, wspace=0.25,hspace=0.25,bottom=0.05,top=0.98,left=0.05,right=0.98)
        axs = [[fig2.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]
        flatList = [element for innerList in axs for element in innerList]
        axs = flatList
        tp = 0
    for v in np.unique(prov)[:]:
        ind_train,ind_val,ind_test = load(open(os.path.join(model_save_loc,'validation',f'prov_{v}_validation.pkl'),'rb'))
        w = ws.weighted_stats(soc[ind_test],fco2[ind_test],1/np.sqrt(soc_s[ind_test]**2 + fco2_unc[ind_test]**2),'b') # Weighted stats so we can extract the RMSD for each province and save it.

        if per_prov: # Plot the scatter plot for each province with the in plot statistics

            axs[tp].scatter(soc[ind_test],fco2[ind_test])
            unweight(soc[ind_test],fco2[ind_test],axs[tp],c_plot,unit=unit)
            weighted(soc[ind_test],fco2[ind_test],1/np.sqrt(soc_s[ind_test]**2 + fco2_unc[ind_test]**2),axs[tp],c_plot,unit=unit)
            axs[tp].set_title(f'Province {v}')
            axs[tp].set_xlim(c_plot); axs[tp].set_ylim(c_plot); axs[tp].plot(c_plot,c_plot,'k-');
            stats_temp = ws.weighted_stats(soc[ind_test],fco2[ind_test],1/np.sqrt(soc_s[ind_test]**2 + fco2_unc[ind_test]**2),'b')
            axs[tp].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6,zorder=-1)
            axs[tp].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4,zorder=-2)
            axs[tp].set_xlabel('in situ '+parameter+' ('+ unit+')')
            axs[tp].set_ylabel('Neural Network '+parameter+' ('+unit+')')
            tp = tp+1
        if t == 0: # Start the combination array
            tot_ts = np.stack((soc[ind_test],soc_s[ind_test]),axis=1)
            tot_tn = np.stack((fco2[ind_test],fco2_unc[ind_test]),axis=1)
            ind_test_all = ind_test
            rmsd = np.array([v,w['rmsd']]) # Attach RMSD to province number
            #print(rmsd)
            t = 1
        else: # Append the values to the array produced when t = 0
            tot_ts = np.concatenate((tot_ts,np.stack((soc[ind_test],soc_s[ind_test]),axis=1)))
            tot_tn = np.concatenate((tot_tn,np.stack((fco2[ind_test],fco2_unc[ind_test]),axis=1)))
            ind_test_all = np.concatenate((ind_test_all,ind_test))
            rmsd = np.vstack((rmsd,np.array([v,w['rmsd']]))) # Append further rmsd values with its province number
            #print(rmsd)
    ind_test_all_out = np.transpose(np.stack((year[ind_test_all],month[ind_test_all],day[ind_test_all],latitude[ind_test_all],longitude[ind_test_all],soc[ind_test_all],soc_s[ind_test_all],fco2[ind_test_all],fco2_unc[ind_test_all],prov[ind_test_all])))
    np.savetxt(os.path.join(model_save_loc,'validation','independent_test_data.csv'), ind_test_all_out,header = header,delimiter = ',')

    font = {'weight' : 'normal',
            'size'   : 19}
    matplotlib.rc('font', **font)
    print(rmsd)
    np.savetxt(os.path.join(model_save_loc,'validation','independent_test_rmsd.csv'), rmsd, delimiter=",") # Save this rmsd data to a csv file to load later...
    # Produce the scatter plot for the test dataset
    ax[1].scatter(tot_ts[:,0],tot_tn[:,0],s=2)
    h2 = unweight(tot_ts[:,0],tot_tn[:,0],ax[1],c_plot,unit=unit)
    h1 = weighted(tot_ts[:,0],tot_tn[:,0],1/np.sqrt(tot_ts[:,1]**2 + tot_tn[:,1]**2),ax[1],c_plot,unit=unit)
    stats_temp = ws.weighted_stats(tot_ts[:,0],tot_tn[:,0],1/np.sqrt(tot_ts[:,1]**2 + tot_tn[:,1]**2),'b')
    ax[1].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6)
    ax[1].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4)
    # Merged all the data together and produce a final validation plot
    ts = np.concatenate((tot_ts,tot_s))
    tn = np.concatenate((tot_tn,tot_n))
    ax[2].scatter(ts[:,0],tn[:,0],s=2)
    h2 = unweight(ts[:,0],tn[:,0],ax[2],c_plot,unit=unit)
    h1 = weighted(ts[:,0],tn[:,0],1/np.sqrt(ts[:,1]**2 + tn[:,1]**2),ax[2],c_plot,unit=unit)
    stats_temp = ws.weighted_stats(ts[:,0],tn[:,0],1/np.sqrt(ts[:,1]**2 + tn[:,1]**2),'b')
    ax[2].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6)
    ax[2].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4)

    # Produce a scatter plot with the errorbars as well...
    ax[3].scatter(ts[:,0],tn[:,0],s=2,c='r',zorder = 3)
    ax[3].errorbar(ts[:,0],tn[:,0],xerr=ts[:,1],yerr=tn[:,1],linestyle='none')
    ax[3].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6,zorder=5)
    ax[3].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4,zorder=4)

    # Plot tidying up and adding axis labels and plot titles
    let = ['a','b','c','d','e','f','g','h']
    for i in [0,1,2,3]:
        ax[i].set_xlim(c_plot); ax[i].set_ylim(c_plot); ax[i].plot(c_plot,c_plot,'k-');
        ax[i].set_xlabel('in situ '+parameter+' ('+ unit+')')
        ax[i].set_ylabel('Neural Network '+parameter+' ('+unit+')')
        ax[i].text(0.03,1.07,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 26)
    ax[0].set_title('Training and Validation')
    ax[1].set_title('Independent Test')
    ax[2].set_title('All datasets')
    ax[3].set_title('All datasets with errorbars')

    # Save the figures.
    fig.savefig(save_file,format='png',dpi=300)
    plt.close(fig)
    if per_prov:
        fig2.savefig(save_file_p,format='png',dpi=300)
        plt.close(fig2)

def add_validation_unc(model_save_loc,data_file,prov,name='fco2',longname='Fugacity of CO2 in seawater',unit = 'uatm',file = 'independent_test_rmsd.csv'):
    """
    Function to take the independent test rmsd values produced in plot_total_validation_unc and produce a array within the output netcdf
    with the validation uncertainity for each province mapped
    """
    # Load the province data
    c = Dataset(data_file)
    prov = c.variables[prov][:]
    s = prov.shape
    prov = np.reshape(prov,(-1,1))
    c.close()
    # Load the test validation RMSD.
    val = np.loadtxt(os.path.join(model_save_loc,'validation',file),delimiter=',')

    val_unc = np.zeros((prov.shape))
    val_unc[:] = np.nan
    if len(val.shape)>1:# So if we have a single province this should be 1, otherwise it should be 2 (i.e 2 dimensions)
        for v in range(0,val.shape[0]):# Iterate through the provinces and add the validation to the province areas
            print(v)
            val_unc[prov == val[v,0]] = val[v,1]*2 # For each province we put the validaiton RMSD as the value...
    else: #Single province must be handled differently
        val_unc[prov == val[0]] = val[1]*2 # val[0] is the province number, and val[1] is the RMSD
    val_unc = np.reshape(val_unc,s)

    # Need to add a check if the fCO2_val_unc variable has been created already - if it has we just overwrite with the new data...
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    keys = c.variables.keys()
    if name+'_val_unc' in keys:
        c.variables[name+'_val_unc'][:] = val_unc
        c.variables[name+'_val_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    else:

        var_o = c.createVariable(name+'_val_unc','f4',('longitude','latitude','time'))
        var_o[:] = val_unc
    c.variables[name+'_val_unc'].long_name = longname+' evaluation uncertainty'
    c.variables[name+'_val_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables[name+'_val_unc'].units = unit
    c.variables[name+'_val_unc'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    c.close()

def add_total_unc(model_save_loc,name='fco2',longname='Fugacity of CO2 in seawater',unit = 'uatm'):
    """
    Function to produce the total fCO2 uncertainity by combining the uncertainity components in quadrature.
    """
    # Load the uncertainity components
    c = Dataset(os.path.join(model_save_loc,'output.nc'),'a')
    keys = c.variables.keys()
    ne = c.variables[name+'_net_unc'][:]
    pa = c.variables[name+'_para_unc'][:]
    va = c.variables[name+'_val_unc'][:]

    tot = np.sqrt(ne**2 + pa**2 + va**2) # Combine these in quadrature
    # Need to add a check if the fCO2_tot_unc variable has been created already - if it has we just overwrite with the new data...
    if name+'_tot_unc' in keys:
        c.variables[name+'_tot_unc'][:] = tot
        c.variables[name+'_tot_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    else:
        var_o = c.createVariable(name+'_tot_unc','f4',('longitude','latitude','time'))
        var_o[:] = tot
    c.variables[name+'_tot_unc'].long_name = longname+' total uncertainty'
    c.variables[name+'_tot_unc'].date_generated = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c.variables[name+'_tot_unc'].comment = 'Combination of'+name+'_val_unc, '+name+'_net_unc and '+name+'_para_unc in quadrature'
    c.variables[name+'_tot_unc'].uncertainties = 'Uncertainties considered 95% confidence (2 sigma)'
    c.variables[name+'_tot_unc'].units = unit
    c.close()

def plot_mapped(model_save_loc,dat=300):
    """
    Function to plot a mapped example of the fCO2 and the uncertainty components for the final year of the array.
    More a sanity check that things have worked correctly.
    """
    #dat =300
    #Load all the variables needed for plotting
    c = Dataset(model_save_loc+'/output.nc')
    lat = np.squeeze(c.variables['latitude'][:])
    lon = np.squeeze(c.variables['longitude'][:])
    fco2 = c.variables['fco2'][:]
    fco2_net_unc = c.variables['fco2_net_unc'][:]
    fco2_para_unc = c.variables['fco2_para_unc'][:]
    fco2_val_unc = c.variables['fco2_val_unc'][:]
    fco2_tot_unc = c.variables['fco2_tot_unc'][:]
    c.close()

    #Take the mean of the final year for each variable
    fco2_plot = np.transpose(fco2[:,:,dat])
    fco2_u_plot = np.transpose(fco2_net_unc[:,:,dat])
    fco2_up_plot = np.transpose(fco2_para_unc[:,:,dat])
    fco2_val_plot = np.transpose(fco2_val_unc[:,:,dat])
    fco2_tot_plot = np.transpose(fco2_tot_unc[:,:,dat])

    # And plot these... (Maybe a more line efficient way to do this...)
    fig = plt.figure(figsize=(28,28))
    gs = GridSpec(3,2, figure=fig, wspace=0.18,hspace=0.18,bottom=0.05,top=0.95,left=0.05,right=1.00)
    ax1 = fig.add_subplot(gs[0,0]);
    pc = ax1.pcolor(lon,lat,fco2_plot)
    cbar = plt.colorbar(pc,ax=ax1)
    cbar.set_label('Predicted fCO$_{2 (sw,subskin)}$ ($\mu$atm)');
    pc.set_clim([300,500])
    ax1.set_facecolor('gray')
    s = model_save_loc.split('/')
    ax1.set_title(s[-1])

    ax2 = fig.add_subplot(gs[0,1]);
    pc = ax2.pcolor(lon,lat,fco2_u_plot)
    cbar = plt.colorbar(pc,ax=ax2)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ network uncertainty ($\mu$atm)');
    pc.set_clim([0,50])
    ax2.set_facecolor('gray')

    ax3 = fig.add_subplot(gs[1,0]);
    pc = ax3.pcolor(lon,lat,fco2_up_plot)
    cbar = plt.colorbar(pc,ax=ax3)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ input parameter uncertainty ($\mu$atm)');
    pc.set_clim([0,50])
    ax3.set_facecolor('gray')

    ax4 = fig.add_subplot(gs[1,1]);
    pc = ax4.pcolor(lon,lat,fco2_val_plot)
    cbar = plt.colorbar(pc,ax=ax4)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ evaluation uncertainty ($\mu$atm)');
    pc.set_clim([0,50])
    ax4.set_facecolor('gray')

    ax5 = fig.add_subplot(gs[2,0]);
    pc = ax5.pcolor(lon,lat,fco2_tot_plot)
    cbar = plt.colorbar(pc,ax=ax5)
    cbar.set_label('fCO$_{2 (sw,subskin)}$ total uncertainty ($\mu$atm)');
    pc.set_clim([0,50])
    ax5.set_facecolor('gray')

    # put the model folder name as the title, so we know what this was generated for.
    s = model_save_loc.split('/')
    ax1.set_title(s[-1])
    # Save the figure.
    fig.savefig(os.path.join(model_save_loc,'plots','mapped_example.png'),format='png',dpi=300)
    plt.close(fig)

def neural_network_map(mapping_data,var=None,model_save_loc=None,prov = None,output_size=None,unc = None,ens=10):
    """
    Function to apply the trained neural network to the full data to produce global maps of fCO2 sw with the network unc and input parameter unc.
    """
    print(var)
    inp = np.array(mapping_data[var])
    prov = np.array(mapping_data[prov])
    print(inp.shape)
    # Produce the output arrays
    out = np.empty((inp.shape[0]))
    out[:] = np.nan
    unc_net = np.copy(out)
    unc_para = np.copy(out)
    # For each province we run the neural network
    for v in np.unique(prov[~np.isnan(prov)]):
        scalar = load(open(os.path.join(model_save_loc,'scalars',f'prov_{v}_scalar.pkl'),'rb')) # Load the scalar
        f = np.squeeze(np.argwhere(prov == v)) # Find the data within the province
        print(f'Number of samples for {v}: {len(f)}')
        mod_inp = scalar.transform(inp[f,:]) # Transform the data using the scalar
        #mod_inp = inp[f,:]
        out_t = np.zeros((len(f),ens))
        # For each ensemble run the neural network and get the output
        for i in range(ens):
            mod = tf.keras.models.load_model(os.path.join(model_save_loc,'networks',f'prov_{v}_model_ens{i}'),compile=False)
            out_t[:,i] = np.squeeze(mod.predict(mod_inp))
        out[f] = np.nanmean(out_t,axis=1) # The output fCO2 is the mean of the ensembles
        unc_net[f] = np.nanstd(out_t,axis=1)*2 # The output fCO2 network uncertainity
        if unc:
            #Load the parameter uncertainty look up table and apply.
            lut = load(open(os.path.join(model_save_loc,'unc_lut',f'prov_{v}_lut.pkl'),'rb'))
            lut = lut[0]
            unc_para[f] = np.squeeze(lut_retr(lut,mod_inp))
    # Reshape the outputs to the correct lon,lat,time dimensions.
    out = np.reshape(out,(output_size))
    unc_net = np.reshape(unc_net,(output_size))
    unc_para = np.reshape(unc_para,(output_size))
    return out,unc_net,unc_para

def save_mapped_fco2(data,net_unc,para_unc,data_shape = None, model_save_loc = None, lon = None,lat = None,time = [],name = 'fco2',longname='Fugacity of CO2 in seawater',unit = 'uatm'):
    """
    Function to save the mapped fco2 data produced by nerual_network_map.
    """
    from construct_input_netcdf import save_netcdf
    direct = {}; direct[name] = data; direct[name+'_net_unc'] = net_unc; direct[name+'_para_unc'] = para_unc
    units = {}; units[name] = unit; units[name+'_net_unc'] = unit; units[name+'_para_unc'] = unit
    comment = {}; comment[name] = ''; comment[name+'_net_unc'] = 'Uncertainties considered 95% confidence (2 sigma)'; comment[name+'_para_unc'] = 'Uncertainties considered 95% confidence (2 sigma)'
    long_name={};
    long_name[name] = longname
    long_name[name+'_net_unc'] = longname+' network uncertainty'
    long_name[name+'_para_unc'] = longname+' parameter uncertainty'

    save_netcdf(os.path.join(model_save_loc,'output.nc'),direct,lon,lat,data.shape[2],time_track=time,units = units,long_name=long_name,comment = comment)

def plot_residuals(model_save_loc,latv,lonv,var,out_var,zoom_lon = False,zoom_lat = False,plot_file = 'mapped_residuals.png',bin = False,log = False,lag = False,geopan = True):
    import geopandas as gpd
    import cmocean
    data = pd.read_table(os.path.join(model_save_loc,'training_addedneural.tsv'),sep='\t')
    if geopan:
        worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    cmap = cmocean.cm.balance
    cmap = cmocean.tools.crop_by_percent(cmap, 20, which='both', N=None)
    if bin:
        fig = plt.figure(figsize=(15,15))
        gs = GridSpec(2,1, figure=fig, wspace=0.18,hspace=0.18,bottom=0.05,top=0.95,left=0.05,right=0.95)
    else:
        fig = plt.figure(figsize=(15,7))
        gs = GridSpec(1,1, figure=fig, wspace=0.18,hspace=0.18,bottom=0.05,top=0.95,left=0.05,right=0.95)
    ax = fig.add_subplot(gs[0,0])
    if geopan:
        worldmap.plot(color="lightgrey", ax=ax)
    a = plt.scatter(data[lonv],data[latv],c = data[var] - data[out_var],vmin = -30, vmax=30,cmap = cmap)
    cbar = fig.colorbar(a); cbar.set_label(var + ' - ' + out_var)
    if zoom_lat:
        ax.set_xlim(zoom_lon)
        ax.set_ylim(zoom_lat)
    if bin:
        ax = fig.add_subplot(gs[1,0])
        if geopan:
            worldmap.plot(color="lightgrey", ax=ax)
        lat = np.array(data[latv])
        lon = np.array(data[lonv])
        res = np.abs(log[0]-log[1])/2
        bi = np.zeros((len(log),len(lag)))
        for i in range(len(log)):
            for j in range(len(lag)):
                f = np.where((lon > log[i]-res) & (lon < log[i]+res) & (lat> lag[j]-res) & (lat<lag[j]+res))[0]
                #print(f)
                bi[i,j] = np.nanmean(data[var][f] - data[out_var][f])

        a = ax.pcolor(log,lag,np.transpose(bi),vmin=-30,vmax=30,cmap = cmap)
        cbar = fig.colorbar(a); cbar.set_label(var + ' - ' + out_var)
        if zoom_lat:
            ax.set_xlim(zoom_lon)
            ax.set_ylim(zoom_lat)

    fig.savefig(os.path.join(model_save_loc,'plots',plot_file),dpi=300)
    plt.close(fig)

def OC4C_calc_independent_test_rmsd(model_save_loc,input_file,sst_name,fco2_file,province_file,prov_var,output_file = 'OC4C_independent_test.csv',name='fco2',unit = '$\mu$atm',parameter = 'fCO$_{2 (sw)}$',
    c_plot = np.array([0,800])):
    c = Dataset(province_file,'r')
    prov = np.array(c.variables[prov_var])
    c.close()

    c = Dataset(input_file,'r')
    fco2 = np.array(c.variables[sst_name+'_reanalysed_fCO2_sw_indpendent'])
    fco2_std = np.array(c.variables[sst_name+'_reanalysed_fCO2_sw_std_indpendent'])
    fco2_count = np.array(c.variables[sst_name+'_reanalysed_count_obs_indpendent'])
    c.close()
    fco2_std[(np.isnan(fco2)==0) & (np.isnan(fco2_std)==1)] = 0
    fco2_std = np.sqrt((fco2_std/np.sqrt(fco2_count))**2 + 5**2)

    c = Dataset(fco2_file,'r')
    fco2_nn = np.array(c.variables['fco2'])
    c.close()
    print(prov.shape)
    if len(prov.shape) == 2:
        prov2 = np.repeat(prov[:, :, np.newaxis], fco2.shape[2], axis=2)
    else:
        prov2 = prov
    print(prov2.shape)
    uniq = np.unique(prov2).tolist()
    print(uniq[-1])
    uniq.remove(uniq[-1])

    col = 6 # 6 columns on the figure
    row = int(np.ceil(len(uniq)/col)) # Calculate the rows based on number of provinces
    font = {'weight' : 'normal',
            'size'   : 25}
    matplotlib.rc('font', **font)
    #Setting up the figure for the per province validation
    fig2 = plt.figure(figsize=(col*9,row*9))
    gs = GridSpec(row,col, figure=fig2, wspace=0.25,hspace=0.25,bottom=0.05,top=0.98,left=0.05,right=0.98)
    axs = [[fig2.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]
    flatList = [element for innerList in axs for element in innerList]
    axs = flatList
    tp = 0
    for i in uniq:
        f = np.where((prov2 == i) & (np.isnan(fco2) == 0))
        axs[tp].scatter(fco2[f],fco2_nn[f])
        w = ws.weighted_stats(fco2[f],fco2_nn[f],1/np.sqrt(fco2_std[f]**2),'b') # Weighted stats so we can extract the RMSD for each province and save it.
        unweight(fco2[f],fco2_nn[f],axs[tp],c_plot,unit=unit)
        weighted(fco2[f],fco2_nn[f],1/np.sqrt(fco2_std[f]**2),axs[tp],c_plot,unit=unit)
        axs[tp].set_title(f'Province {i}')
        axs[tp].set_xlim(c_plot); axs[tp].set_ylim(c_plot); axs[tp].plot(c_plot,c_plot,'k-');

        stats_temp = ws.weighted_stats(fco2[f],fco2_nn[f],1/np.sqrt(fco2_std[f]**2),'b')
        axs[tp].fill_between(c_plot,c_plot-stats_temp['rmsd'],c_plot+stats_temp['rmsd'],color='k',alpha=0.6,zorder=-1)
        axs[tp].fill_between(c_plot,c_plot-(stats_temp['rmsd']*2),c_plot+(stats_temp['rmsd']*2),color='k',alpha=0.4,zorder=-2)
        axs[tp].set_xlabel('in situ '+parameter+' ('+ unit+')')
        axs[tp].set_ylabel('Neural Network '+parameter+' ('+unit+')')
        if tp == 0:
            rmsd = np.array([i,w['rmsd']])
        else:
            rmsd = np.vstack((rmsd,np.array([i,w['rmsd']]))) # Append further rmsd values with its province number
        tp = tp+1
    fig2.savefig(os.path.join(model_save_loc,'plots','per_prov_validation.png'),format='png',dpi=300)
    plt.close(fig2)
    np.savetxt(os.path.join(model_save_loc,'validation',output_file), rmsd, delimiter=",")
