#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023
Script takes SOCAT tsv files for both the main and FlagE datasets and produces a reanlysed fCO2
dataset within the neural network input netcdf. Currently hardcoded to 1deg outputs (working to modify...)
"""

import custom_reanalysis.reanalyse_socat_v2 as rean
import shutil
import os
import datetime
from netCDF4 import Dataset
import Data_Loading.data_utils as du
import numpy as np
import pandas as pd
import multiprocessing

statvariables=['fCO2_Tym','pCO2_Tym','fCO2_SST','pCO2_SST','Tcl_C','SST_C']


def reanalyse(socat_dir=None,socat_files=None,sst_dir=None,sst_tail=None,out_dir=None,force_reanalyse=False,
    start_yr = 1990,end_yr = 2020,name = '',outfile = None, var = None,prefix = '%Y%m01-OCF-CO2-GLO-1M-100-SOCAT-CONV.nc',flip=True,
    kelvin = False):
    """
    Function to run the fluxengine reanalysis code in custom_reanalysis. DJF to put a pull request into FluxEngine with
    the updates made within the custom code.
    """
    #if __name__ == '__main__':
    if not os.path.exists(out_dir):
        print('No reanalysis done previously - running reanalysis...')
        rean.RunReanalyseSocat(socatdir=socat_dir, socatfiles=socat_files, sstdir=sst_dir, ssttail=sst_tail, startyr=1985,
                          endyr=2023,socatversion=2020,regions=["GL"],customsst=True,
                          customsstvar=var,customsstlat='latitude',customsstlon='longitude'
                          ,output=out_dir)
    elif force_reanalyse:
        print('Forcing Reanalysis....')
        shutil.rmtree(out_dir)
        rean.RunReanalyseSocat(socatdir=socat_dir, socatfiles=socat_files, sstdir=sst_dir, ssttail=sst_tail, startyr=1985,
                          endyr=2023,socatversion=2020,regions=["GL"],customsst=True,
                          customsstvar=var,customsstlat='latitude',customsstlon='longitude'
                          ,output=out_dir)
    fco2,fco2_std,fco2_sst = retrieve_fco2(out_dir,start_yr = start_yr,end_yr=end_yr,prefix=prefix)
    if kelvin:
        fco2_sst = fco2_sst+273.15
    append_to_file(outfile,fco2,fco2_std,fco2_sst,name,socat_files[0])

def load_prereanalysed(input_file,output_file,start_yr=1990, end_yr = 2020,name='',kelvin=True,socat_notreanalysed=False):
    """
    Function loads the reanlysed fCO2 data produced by JamieLab, instead of running the reanalysis process.

    input_file = the location of the prereanalysed NetCDF file
    output_file = the NetCDF file the data should be appended to
    start_yr = the start year for data extraction
    end_yr = the end year for data extraction (inclusive)
    name = the name of the SST dataset used for reanalysis
    kelvin = toggle to output the SST data as Kelvin or degress C
    """
    c = Dataset(input_file)
    ti = np.array(c.variables['tmnth'])
    time = []
    #Construct time array of years, to allow slicing of the data
    for t in ti:
        time.append((datetime.datetime(1970,1,1) + datetime.timedelta(days=int(t))).year)
    time = np.array(time)
    #print(time)
    if socat_notreanalysed:
        #Extracting the non-reanalysed SOCAT data...
        fco2 = np.array(c.variables['fco2_ave_weighted'])
        fco2_std = np.array(c.variables['fco2_std_weighted'])
        fco2_nobs = np.array(c.variables['count_nobs_reanalysed'])
        sst = np.array(c.variables['sst_ave_weighted'])
    else:
        #Extracting the reanalysed fCO2 data
        fco2 = np.array(c.variables['fco2_reanalysed_ave_weighted'])
        fco2_std = np.array(c.variables['fco2_reanalysed_std_weighted'])
        fco2_nobs = np.array(c.variables['count_nobs_reanalysed'])
        sst = np.array(c.variables['sst_reynolds'])
    #Toggle to allow the function to output the SST paired to the observations as degC or K
    if kelvin:
        #Kelvin output (as prereanalysed files are in degC)
        sst =  sst +273.15 #sst comes in in degC prefer in K

    c.close()
    # Find where the data within the reanalysed file match the specified time period
    f = np.where((time <= end_yr) & (time >= start_yr))
    #print(fco2.shape)
    # Trim the arrays to the time (first dimension is time), then transpose to
    # (lon,lat,time) dimensions
    fco2 = np.transpose(fco2[f[0],:,:],(2,1,0))
    fco2_std = np.transpose(fco2_std[f[0],:,:],(2,1,0))
    fco2_nobs = np.transpose(fco2_nobs[f[0],:,:],(2,1,0))
    sst = np.transpose(sst[f[0],:,:],(2,1,0))

    if time[-1] != end_yr:
        #If the end_year of the SOCAT data is not the end year we want to extrapolate to then pad the end,
        #so the data can be included in the netcdf. For example this is used by the UoE to prepare input
        #datasets for the GCB release, and allows running of an addiitonal year in preparation
        deltatime = (end_yr - time[-1])*12
        add_field = np.zeros((fco2.shape[0],fco2.shape[1],deltatime)); add_field[:] = np.nan

        #Add the additional padding fields on the time axis :-)
        fco2 = np.concatenate((fco2,add_field),axis=2)
        fco2_std = np.concatenate((fco2_std,add_field),axis=2)
        fco2_nobs = np.concatenate((fco2_nobs,add_field),axis=2)
        sst = np.concatenate((sst,add_field),axis=2)

    # Remove data that is either fill or isn't representative (i.e fCO2sw of 0 isn't really possible)
    # in seawater.
    # Need better solution to this...
    fco2[fco2<0] = np.nan
    fco2_std[fco2_std<0] = np.nan
    if kelvin:
        sst[sst<0] = np.nan
    else:
        sst[sst<-2.5] = np.nan
    #print(fco2.shape)
    # Append to data to the output file.
    append_to_file(output_file,fco2,fco2_std,sst,fco2_nobs,name,input_file,not_reanalysed=socat_notreanalysed)

def append_to_file(output_file,fco2,fco2_std,sst,obs,name,socat_files,not_reanalysed = False):
    """
    Function to append the reanalysed fCO2 to the neural network input file

    output_file = the file to append the fCO2 data to
    fco2 = 3D array of fCO2 data
    fco2_std = 3D array of fCO2 standard deviation data
    sst = 3D array of SST data coincident to the fCO2
    name = name of the SST data used within the reanalysis
    socat_files = string of the reanalysed SOCAT file location
    """
    c = Dataset(output_file,'a',format='NETCDF4_CLASSIC')
    var_o = c.createVariable(name+'_reanalysed_fCO2_sw','f4',('longitude','latitude','time'))
    var_o[:] = fco2
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files
    if not_reanalysed:
        var_o.not_reanalysed = 'These data are not reanalysed and are the orginial SOCAT data. NetCDF variable name is just for compatiability with the nerual network setup'

    var_o = c.createVariable(name+'_reanalysed_fCO2_sw_std','f4',('longitude','latitude','time'))
    var_o[:] = fco2_std
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) std using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files
    if not_reanalysed:
        var_o.not_reanalysed = 'These data are not reanalysed and are the orginial SOCAT data. NetCDF variable name is just for compatiability with the nerual network setup'
    var_o = c.createVariable(name+'_reanalysed_sst','f4',('longitude','latitude','time'))
    var_o[:] = sst
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) std using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files
    if not_reanalysed:
        var_o.not_reanalysed = 'These data are not reanalysed and are the orginial SOCAT data. NetCDF variable name is just for compatiability with the nerual network setup'
        var_o.comment = 'SST data that is coincident to the SOCAT fCO2(sw)'
    else:
        var_o.comment = 'SST data that is coincident to the reanalysed fCO2(sw)'

    var_o = c.createVariable(name+'_reanalysed_count_obs','f4',('longitude','latitude','time'))
    var_o[:] = obs
    # var_o.standard_name = 'Reanalysed fCO2(sw, subskin) std using ' + name + ' datset'
    var_o.sst = name
    #var_o.units = 'uatm'
    var_o.created_from = socat_files
    c.close()

def model_fco2_append(input_file,output_file,start_yr=1990, end_yr = 2020,name='',ref_yr = 1970):
    """
    Function to append GCB model fCO2 into the neural network framework input file
    """

    c = Dataset(input_file,'r')
    data = np.array(c['sfco2'])
    c.close()
    #Setup a time array so we know the year that the data is from.
    time = []
    yr = ref_yr
    mon = 1
    for ti in range(data.shape[0]):
        time.append(yr)
        mon = mon+1
        if mon == 13:
            mon = 1
            yr=yr+1
    time = np.array(time)
    print(time)

    f = np.where((time <= end_yr) & (time >= start_yr))[0]
    data = data[f,:,:]
    data = du.lon_switch(data)
    data = np.transpose(data,[2,1,0])
    fco2_std = np.zeros((data.shape))
    data[data>10000] = np.nan

    c = Dataset(output_file,'a')
    var_o = c.createVariable('model_reanalysed_fCO2_sw','f4',('longitude','latitude','time'))
    var_o[:] = data
    var_o.model_data = 'Data from ' + input_file
    var_o.model_subsampled_data = 'These data are model subsampled data - NetCDF variable name is just for compatiability with the nerual network setup'
    var_o = c.createVariable('model_reanalysed_fCO2_sw_std','f4',('longitude','latitude','time'))
    var_o[:] = fco2_std
    var_o.model_data = 'Data from ' + input_file
    var_o.model_subsampled_data = 'These data are model subsampled data - NetCDF variable name is just for compatiability with the nerual network setup'
    var_o.padded_zeros = 'This data array is just padded with zeros...'
    c.close()


def retrieve_fco2(rean_dir,start_yr=1990,end_yr=2020,prefix = '%Y%m01-OCF-CO2-GLO-1M-100-SOCAT-CONV.nc',flip=False):
    """
    Function to iteratively load the fCO2sw from the reanalysis folder (i.e one netcdf per month_year combo)

    rean_dir = Top level location of the reanalysis data (i.e the folder that contains 'reanalysed_global' folder)
    start_yr = the start year for data extraction
    end_yr = the end year for data extraction (inclusive)
    prefix = the file name of the reanalysed files (where %Y becomes the 4 digit year and %m becomes two digit month)
    flip = Flip data on the latitude axis - I don't think this needs to ever be set to true now...
    """
    print('Retrieving fCO2 from reanlysis')
    # Calculate how many months of data should be avaiable based on start and end years
    months = (end_yr-start_yr+1)*12
    # Setup the while loop, to cycle through all of the files in the time period
    yr = start_yr
    mon = 1
    t = 0 # Counting variable
    inits = 0 # A toggle for the first time we are able to load data
    while yr <= end_yr:
        da = datetime.datetime(yr,mon,1) # Datetime to make getting file names easier
        loc = os.path.join(rean_dir,'reanalysed_global',da.strftime('%m'),da.strftime(prefix))
        print(loc)
        if du.checkfileexist(loc): # Checking the file exists
            #print('True')
            # Loading the data
            c = Dataset(loc,'r')
            fco2_temp = np.squeeze(c.variables['unweighted_fCO2_Tym']) #FluxEngine files have time dimension of 1 so we squeeze that out
            fco2_temp = np.transpose(fco2_temp) # Tranpose from lat/lon to lon/lat
            if flip:
                fco2_temp = np.fliplr(fco2_temp)

            fco2_std_temp = np.squeeze(c.variables['unweighted_std_fCO2_Tym'])
            fco2_std_temp = np.transpose(fco2_std_temp)
            if flip:
                fco2_std_temp = np.fliplr(fco2_std_temp)

            fco2_sst_temp = np.squeeze(c.variables['unweighted_Tcl_C'])
            fco2_sst_temp = np.transpose(fco2_sst_temp)
            if flip:
                fco2_sst_temp = np.fliplr(fco2_sst_temp)
            c.close()
            if inits == 0: # On first avaiable file we create the output arrays
                fco2 = np.empty((fco2_temp.shape[0],fco2_temp.shape[1],months))
                fco2[:] = np.nan
                fco2_std = np.copy(fco2)
                fco2_sst = np.copy(fco2)
                inits = 1 # Set toggle to 1 so we don't create recreating the output arrays
            fco2[:,:,t] = fco2_temp # Putting the data into the array
            fco2_std[:,:,t] = fco2_std_temp
            fco2_sst[:,:,t] = fco2_sst_temp
        t += 1 # Add to counting
        mon += 1 # Add another month
        if mon == 13: # If month gets to 13 we need to increment the year and set month to 1
            yr += 1
            mon=1
    # Remove data that is either fill or isn't representative (i.e fCO2sw of 0 isn't really possible)
    # in seawater.
    # Need better solution to this...
    fco2[fco2<0] = np.nan
    fco2_std[fco2_std<0] = np.nan
    fco2_sst[fco2_sst<-5] = np.nan
    return fco2,fco2_std,fco2_sst

def read_socat(file,pad=7302):
    """
    Function to read the SOCAT tsv file

    file = is the file to open (should be a .tsv file)
    pad = is the number of rows at the top of file to miss (7302 seems to be the number for the reanalysed files)
    """
    data=pd.read_csv(file,sep='\t',skiprows=pad)#,low_memory=False)#,usecols=list(range(3,36)))
    lon = data['longitude [dec.deg.E]']
    lon[lon>180] = lon[lon>180] - 360 # Prefer longitude in -180 to 180 then 0-360
    data['longitude [dec.deg.E]'] = lon # Save back
    return data

def find_socat(data,lat,lon):
    """
    Function to find all the SOCAT data in a latitude/longitude region

    data = is the SOCAT data as a Pandas table, read with 'read_socat' function
    lat = latitude grid
    lon = longitude grid
    """
    res_lat = np.abs(lat[0]-lat[1])/2 # Find the resoltion of the latitude grid
    mlat = [np.min(lat)-res_lat,np.max(lat)+res_lat] # Find the max and min of the latitude, and add resolution back so we have the whole box at grid ends
    print(mlat)
    #Same as above for the Longitude
    res_lon = np.abs(lon[0]-lon[1])/2
    mlon = [np.min(lon)-res_lon,np.max(lon)+res_lon]
    print(mlon)
    soclat = data['latitude [dec.deg.N]']
    #print(soclat)
    soclon = data['longitude [dec.deg.E]']
    # Find where the data falls within these bounds...
    d = data[(soclat > mlat[0]) & (soclat < mlat[1]) & (soclon > mlon[0]) & (soclon < mlon[1])]
    return d

def regrid_fco2_data(file,latg,long,start_yr=1990,end_yr=2022,save_loc = [],grid=True,fco2var = 'fCO2_reanalysed [uatm]',pco2var = 'pCO2_reanalysed [uatm]',sstvar = 'T_reynolds [C]',
    save_file ='socat.tsv',save_fold = False,pad = 7302):
    """
    Function to regrid the SOCAT tsv into a gridded file consistent to that produced by SOCAT and JamieLab, but with additional extras
    to allow daily data, different spatial resolutions or other data to also be appended.

    file = reanalysed SOCAT.tsv file data_location
    latg = regularly gridded latitude array (1D)
    long = regularly gridded longitude array (1D)
    start_yr = the start year for data extraction
    end_yr = the end year for data extraction (inclusive)
    save_loc = the save folder location the data should be saved (this is top level, as an additional folder of 'socat' is created one level down)
    grid = Whether to actually grid the data, or just produce a SOCAT.tsv file with data within the latg - long bounds
    fco2var = the column name in the SOCAT tsv file for the fCO2sw data
    pco2var = the column name in the SOCAT tsv file for the pCO2sw data
    sstvar = the column name in the SOCAT tsv file for the SST data paired to the f/pCO2sw
    save_file = the name of the .tsv save file
    save_fold = the direct name of the folder the data should be saved in (allowing the function to be used outside the OceanICU framework)
    pad = the number of rows at the top of file to miss (7302 seems to be the number for the reanalysed files)
    """
    import custom_reanalysis.combine_nc_files as combine_nc_files
    import glob
    if not save_fold:
        save_fold = os.path.join(save_loc,'socat')
    else:
        save_fold = save_loc
    du.makefolder(save_fold)
    if du.checkfileexist(os.path.join(save_fold,save_file)):
        data = pd.read_table(os.path.join(save_fold,save_file),sep='\t')
    else:
        data = read_socat(file,pad = pad)
        data = find_socat(data,latg,long)
        data.to_csv(os.path.join(save_fold,save_file),sep='\t')
    #print(data)
    if grid:
        print(data.columns)
        result=np.recarray((np.array(data['yr']).size,),dtype=[('yr',np.int32),
                                             ('mon',np.int32),
                                             ('day', np.int32),
                                             ('hh', np.int32),
                                             ('mm', np.int32),
                                             ('ss', np.int32),
                                             ('lat',np.float64),
                                             ('lon',np.float64),
                                             ('SST_C',np.float64),
                                             ('Tcl_C',np.float64),
                                             ('fCO2_SST',np.float64),
                                             ('fCO2_Tym',np.float64),
                                             ('pCO2_SST',np.float64),
                                             ('pCO2_Tym',np.float64),
                                             ('expocode',np.dtype('O'))]);
        result['yr']=data['yr'];
        result['mon']=data['mon'];
        result['day']=data['day'];
        result['hh']=data['hh'];
        result['mm']=data['mm'];
        result['ss']=data['ss'];
        result['lat']=data['latitude [dec.deg.N]']
        result['lon']=data['longitude [dec.deg.E]']
        result['SST_C']=data['SST [deg.C]']
        result['Tcl_C']=data[sstvar]
        result['fCO2_SST']=data['fCO2rec [uatm]']
        result['fCO2_Tym']=data[fco2var]
        result['pCO2_SST']=data['pCO2_SST [uatm]']
        result['pCO2_Tym']=data[pco2var]
        result['expocode'] = data['Expocode']

        for yrs in set(result['yr']):
            year_data = result[np.where(result['yr'] == yrs)]
            for mon in set(year_data['mon']):
                print(f'Year: {yrs} - Month: {mon}')
                month_data = year_data[np.where(year_data['mon'] == mon)]
                data_loc = os.path.join(save_fold,'reanalysed_global',"%02d"%mon)

                data_loc_per = os.path.join(data_loc,'per_cruise')
                du.makefolder(data_loc)
                du.makefolder(data_loc_per)
                outputfile='GL_from_%s_to_%s_%02d.nc'%(yrs,yrs,mon)

                final_output_path = os.path.join(data_loc,outputfile)
                for expo in set(month_data['expocode']):
                    print(expo)
                    output_cruise_file=os.path.join(data_loc_per,outputfile.replace('.nc','-%s.nc'%expo))
                    expo_indices=np.where(month_data['expocode']==expo)
                    expo_data=month_data[expo_indices]
                    variabledictionary=CreateBinnedData(expo_data,latg,long)
                    #print(variabledictionary)
                    half_days_in_month=15.5
                    datadate=datetime.datetime(yrs,mon,int(half_days_in_month),0,0,0)
                    myprocess=multiprocessing.Process(target=WriteOutToNCAsGrid,args=(variabledictionary,output_cruise_file,None,long,latg,datadate))
                    myprocess.start()
                    myprocess.join()
                common_prefix=outputfile.replace('.nc','')
                cruise_files=glob.glob("%s/%s*"%(data_loc_per,common_prefix))
                if len(cruise_files) !=0:
                   print()
                   print("Combining cruises from region, year and month: ",cruise_files)
                   print()
                   combine_nc_files.FromFilelist(filelist=cruise_files,output=final_output_path,
                                                 weighting="cruise-weighted",outputtime=datadate)

                   #get the binned data for the whole month to add to the nc file as other variables
                   allnewvars=CreateBinnedData(month_data,latg,long)
                   newvars={v : allnewvars[v] for v in statvariables+['stds']}
                   combine_nc_files.AddNewVariables(filename=final_output_path,newvars=newvars)

def geo_idx(dd, dd_array):
    """
     - dd - the decimal degree (latitude or longitude)
     - dd_array - the list of decimal degrees to search.
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.

     From: https://stackoverflow.com/questions/43777853/
   """
    geo_idx = (np.abs(dd_array - dd)).argmin()
    return geo_idx

def CreateBinnedData(month_data,latg,long):
    """
    Updated FluxEngine reanalysis function to allow the creation of varying reanalysis files (both the orginial but also with different SST data i.e daily)

    month_data = Pandas table of all the variable used to create the gridded data (produced within 'regrid_fco2_data' function)
    latg = regularly gridded latitude array (1D)
    long = regularly gridded longitude array (1D)
    """
    import custom_reanalysis.netcdf_helper as netcdf_helper
    #import pandas as pd;
    #allData = pd.DataFrame(month_data);

    #grid information
    nlon = len(long) # number of longitude pixels
    lon0 = long[0] # start longitude
    lon1 = long[-1] # end longitude
    nlat = len(latg) # number of latitude pixels
    lat0 = latg[0] # start latitude
    lat1 = latg[-1] # end latitude
    dlon = (lon1 - lon0) / nlon
    print(dlon)
    dlat = (lat1 - lat0) / nlat
    print(dlat)
    #Set up arrays with default values
    dTs = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    fCO2_Tyms = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    fCO2_SSTs = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    dFs = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    pCO2_Tyms = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    pCO2_SSTs = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    dPs = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    Tcl_C = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    SST_C = np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
    ndata = np.zeros((nlat,nlon))# keep track of multiple entries

    maximums={}
    minimums={}
    stds={}
    #Other data we want to keep track of
    for var in statvariables:
        #print(var)
        maximums[var]=np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
        minimums[var]=np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE
        stds[var]=np.zeros((nlat,nlon))+netcdf_helper.MISSINGDATAVALUE

    #Calculate differences of data for this month
    #Difference in temperature
    dT=month_data['Tcl_C'] - month_data['SST_C']
    #Difference in fugacity
    dF=month_data['fCO2_Tym'] - month_data['fCO2_SST']
    #Difference in partial pressure
    dP=month_data['pCO2_Tym'] - month_data['pCO2_SST']

    #Have changed method here as with previous one if lat data was an integer
    #then it fell into the wrong grid cell.
    #i.e. grid cells were 0<x<=1, 1<x<=2, ...
    #and we wanted 0<=x<1, 1<=x<2, ...
    #so round down to integer first, then subtract 1 [for lats only]
    ilons = []
    ilats = []
    for i in range(0,len(month_data['lon'])):
        ilons.append(geo_idx(month_data['lon'][i],long))
        ilats.append(geo_idx(month_data['lat'][i],latg))
    ilons = np.array(ilons); ilats = np.array(ilats);
    #indicies where the ilons,ilats are within the grid bounds
    w=np.where((ilons >= 0)&(ilons < nlon)&(ilats >= 0)&(ilats < nlat))
    w=w[0]
    if w.size==0: return
    #update ilons,ilats to ones which fall in grid (all of them?)
    if len(w)>1:
      ilons, ilats = ilons[w], ilats[w]
    else:
      #edge case where w only has 1 element - in which case ilons/ilats are scalars not arrays
      #to fix it we convert them to a list of 1 element
      ilons=[ilons[0]]
      ilats=[ilats[0]]
    #update these data points to 0 (from -999) for all output arrays
    dTs[ilats,ilons]=0
    fCO2_Tyms[ilats, ilons] = 0.
    fCO2_SSTs[ilats,ilons]=0
    pCO2_Tyms[ilats, ilons] = 0.
    pCO2_SSTs[ilats,ilons]=0
    Tcl_C[ilats,ilons]=0
    SST_C[ilats,ilons]=0

    for var in statvariables:
        maximums[var][ilats,ilons]=0
        minimums[var][ilats,ilons]=0
        stds[var][ilats,ilons]=0

    #get a list of all the lat,lon pairs we have
    indices=list(set(zip(ilats,ilons)))
    for index in indices:
      #for each index in this list find all the points that fall into that grid cell index
      points=np.where(((ilats==index[0])&(ilons==index[1])))
      #Now we can bin these data into the grid cell
      fCO2_Tyms[index] += np.mean(month_data['fCO2_Tym'][points])
      pCO2_Tyms[index] += np.mean(month_data['pCO2_Tym'][points])
      fCO2_SSTs[index] += np.mean(month_data['fCO2_SST'][points])
      pCO2_SSTs[index] += np.mean(month_data['pCO2_SST'][points])
      Tcl_C[index] += np.mean(month_data['Tcl_C'][points])
      SST_C[index] += np.mean(month_data['SST_C'][points])
      dTs[index] += np.mean(dT[points])
      ndata[index] += points[0].size

      dFs[index]=fCO2_Tyms[index] - fCO2_SSTs[index]
      dPs[index]=pCO2_Tyms[index] - pCO2_SSTs[index]
      for var in statvariables:
         maximums[var][index]=np.nanmax(month_data[var][points])
         minimums[var][index]=np.nanmin(month_data[var][points])
         #we use ddof=1 to get the unbiased estimator (i.e. divide by N-1 in stdev formula)
         #unless only 1 element then set std = NAN (so that std are consistent)
         if len(points[0])>1:
            stds[var][index]=np.std(month_data[var][points],ddof=1)
         else:
            stds[var][index]=np.nan

    vardict={}
    vardict['fCO2_Tym']=fCO2_Tyms
    vardict['pCO2_Tym']=pCO2_Tyms
    vardict['fCO2_SST']=fCO2_SSTs
    vardict['pCO2_SST']=pCO2_SSTs
    vardict['Tcl_C'] = Tcl_C
    vardict['SST_C'] = SST_C
    vardict['dT']=dTs
    vardict['dF']=dFs
    vardict['dP']=dPs
    vardict['ndata']=ndata
    vardict['maximums']=maximums
    vardict['minimums']=minimums
    vardict['stds']=stds

    return vardict

def WriteOutToNCAsGrid(vardict,outputfile,extrapolatetoyear,long,latg,outputtime=1e9):
    """
    Updated FluxEngine function to write out the data created by 'CreateBinnedData' into a consistent netCDF file format

    vardict = the dictionary output by 'CreateBinnedData' with all the required data
    outputfile = outputfile name
    extrapolatetoyear = year to extrapolate the data to (I think this is a year number but I don't use it...)
    latg = regularly gridded latitude array (1D)
    long = regularly gridded longitude array (1D)
    outputtime = a time number given to 'standard_setup_SOCAT' function
    """
    import custom_reanalysis.netcdf_helper as netcdf_helper
    #Write out the data into a netCDF file
    print("Writing to: %s"%outputfile)
    #Test directory exists
    if not os.path.exists(os.path.dirname(outputfile)):
        raise Exception("Directory to write file to does not exist: %s"%(os.path.dirname(outputfile)))


    #update ndata in order to put as a variable in output file
    vardict['ndata']=np.where(vardict['ndata']==0,netcdf_helper.MISSINGDATAVALUE,vardict['ndata'])

    with Dataset(outputfile, 'w', format = 'NETCDF4') as ncfile:
        #Add the standard dims and variables
        netcdf_helper.standard_setup_SOCAT(ncfile,timedata=outputtime,londata=long,
                                          latdata=latg)
        #set output names to allow extrapolation to any year or no extrapolation
        #so that the variable names make sense in the netcdf file
        if extrapolatetoyear is None:
         varext=""
         nameext=""
        else:
         varext="_%d"%extrapolatetoyear
         nameext=" extrapolated to %d"%extrapolatetoyear

        #Add the newly calculated fugacity, partial pressures and differences
        fCO2_SST_data = ncfile.createVariable('fCO2_SST','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        fCO2_SST_data[:] = vardict['fCO2_SST']
        fCO2_SST_data.units = 'uatm'
        fCO2_SST_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        fCO2_SST_data.valid_min = 0.
        fCO2_SST_data.valid_max = 1e6
        fCO2_SST_data.scale_factor = 1.
        fCO2_SST_data.add_offset = 0.
        fCO2_SST_data.standard_name = "fCO2_SST"
        fCO2_SST_data.long_name = "CO2 fugacity using SOCAT methodology"

        SST_data = ncfile.createVariable('SST_C','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        SST_data[:] = vardict['SST_C']
        SST_data.units = 'degC'
        SST_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        SST_data.valid_min = -5
        SST_data.valid_max = 1e6
        SST_data.scale_factor = 1.
        SST_data.add_offset = 0.
        SST_data.standard_name = "SST_C"
        SST_data.long_name = "SST using SOCAT methodology"

        Tym_data = ncfile.createVariable('Tcl_C','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        Tym_data[:] = vardict['Tcl_C']
        Tym_data.units = 'degC'
        Tym_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        Tym_data.valid_min = -5
        Tym_data.valid_max = 1e6
        Tym_data.scale_factor = 1.
        Tym_data.add_offset = 0.
        Tym_data.standard_name = "Tcl_C"
        Tym_data.long_name = "SST using OC-FLUX methodology"

        fCO2_Tym_data = ncfile.createVariable('fCO2_Tym'+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        fCO2_Tym_data[:] = vardict['fCO2_Tym']
        fCO2_Tym_data.units = 'uatm'
        fCO2_Tym_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        fCO2_Tym_data.valid_min = 0.
        fCO2_Tym_data.valid_max = 1e6
        fCO2_Tym_data.scale_factor = 1.
        fCO2_Tym_data.add_offset = 0.
        fCO2_Tym_data.standard_name = "fCO2_Tym"+varext
        fCO2_Tym_data.long_name = "CO2 fugacity using OC-FLUX methodology"+nameext

        pCO2_SST_data = ncfile.createVariable('pCO2_SST','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        pCO2_SST_data[:] = vardict['pCO2_SST']
        pCO2_SST_data.units = 'uatm'
        pCO2_SST_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        pCO2_SST_data.valid_min = 0.
        pCO2_SST_data.valid_max = 1e6
        pCO2_SST_data.scale_factor = 1.
        pCO2_SST_data.add_offset = 0.
        pCO2_SST_data.standard_name = "pCO2_SST"
        pCO2_SST_data.long_name = "CO2 partial pressure using SOCAT methodology"



        pCO2_Tym_data = ncfile.createVariable('pCO2_Tym'+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        pCO2_Tym_data[:] = vardict['pCO2_Tym']
        pCO2_Tym_data.units = 'uatm'
        pCO2_Tym_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        pCO2_Tym_data.valid_min = 0.
        pCO2_Tym_data.valid_max = 1e6
        pCO2_Tym_data.scale_factor = 1.
        pCO2_Tym_data.add_offset = 0.
        pCO2_Tym_data.standard_name = "pCO2_Tym"+varext
        pCO2_Tym_data.long_name = "CO2 partial pressure using OC-FLUX methodology"+nameext

        dT_data = ncfile.createVariable('dT','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        dT_data[:] = vardict['dT']
        dT_data.units = 'Degree C'
        dT_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        dT_data.valid_min = -999.
        dT_data.valid_max = 999.
        dT_data.scale_factor = 1.
        dT_data.add_offset = 0.
        dT_data.standard_name = "dT"
        dT_data.long_name = "difference Tym - SST"

        dfCO2_data = ncfile.createVariable('dfCO2','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        dfCO2_data[:] = vardict['dF']
        dfCO2_data.units = 'uatm'
        dfCO2_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        dfCO2_data.valid_min = -999.
        dfCO2_data.valid_max = 999.
        dfCO2_data.scale_factor = 1.
        dfCO2_data.add_offset = 0.
        dfCO2_data.standard_name = "dfCO2"
        dfCO2_data.long_name = "difference fCO2,Tym - fCO2,SST"

        dpCO2_data = ncfile.createVariable('dpCO2','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        dpCO2_data[:] = vardict['dP']
        dpCO2_data.units = 'uatm'
        dpCO2_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        dpCO2_data.valid_min = -999.
        dpCO2_data.valid_max = 999.
        dpCO2_data.scale_factor = 1.
        dpCO2_data.add_offset = 0.
        dpCO2_data.standard_name = "dpCO2"
        dpCO2_data.long_name = "difference pCO2,Tym - pCO2,SST"

        N_data = ncfile.createVariable('count_nobs','f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
        N_data[:] = vardict['ndata']
        N_data.units = 'count'
        N_data.missing_value = netcdf_helper.MISSINGDATAVALUE
        N_data.valid_min = 0.
        N_data.valid_max = 10000000.
        N_data.scale_factor = 1.
        N_data.add_offset = 0.
        N_data.standard_name = "count_nobs"
        N_data.long_name = "Number of observations mean-averaged in cell"

        for var in statvariables:
            if var in ['fCO2_Tym','pCO2_Tym']:
                minf_data = ncfile.createVariable('min_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                minf_data[:] = vardict['minimums'][var]
                minf_data.units = 'uatm'
                minf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                minf_data.valid_min = 0.
                minf_data.valid_max = 1000000.
                minf_data.scale_factor = 1.
                minf_data.add_offset = 0.
                minf_data.standard_name = "min_"+var+varext
                minf_data.long_name = "Minimum "+var+varext+" occupying binned cell"

                maxf_data = ncfile.createVariable('max_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                maxf_data[:] = vardict['maximums'][var]
                maxf_data.units = 'uatm'
                maxf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                maxf_data.valid_min = 0.
                maxf_data.valid_max = 1000000.
                maxf_data.scale_factor = 1.
                maxf_data.add_offset = 0.
                maxf_data.standard_name = "max_"+var+varext
                maxf_data.long_name = "Maximum "+var+varext+" occupying binned cell"

                stdf_data = ncfile.createVariable('std_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                stdf_data[:] = vardict['stds'][var]
                stdf_data.units = 'uatm'
                stdf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                stdf_data.valid_min = 0.
                stdf_data.valid_max = 1000000.
                stdf_data.scale_factor = 1.
                stdf_data.add_offset = 0.
                stdf_data.standard_name = "stdev_"+var+varext
                stdf_data.long_name = "Standard deviation of "+var+varext+" occupying binned cell"

        for var in statvariables:
            if var in ['Tcl_C','SST_C']:
                minf_data = ncfile.createVariable('min_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                minf_data[:] = vardict['minimums'][var]
                minf_data.units = 'degC'
                minf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                minf_data.valid_min = -5.
                minf_data.valid_max = 1000000.
                minf_data.scale_factor = 1.
                minf_data.add_offset = 0.
                minf_data.standard_name = "min_"+var+varext
                minf_data.long_name = "Minimum "+var+varext+" occupying binned cell"

                maxf_data = ncfile.createVariable('max_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                maxf_data[:] = vardict['maximums'][var]
                maxf_data.units = 'degC'
                maxf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                maxf_data.valid_min = -5
                maxf_data.valid_max = 1000000.
                maxf_data.scale_factor = 1.
                maxf_data.add_offset = 0.
                maxf_data.standard_name = "max_"+var+varext
                maxf_data.long_name = "Maximum "+var+varext+" occupying binned cell"

                stdf_data = ncfile.createVariable('std_'+var+varext,'f4',('time','latitude','longitude'),fill_value=netcdf_helper.MISSINGDATAVALUE,zlib=True)
                stdf_data[:] = vardict['stds'][var]
                stdf_data.units = 'degC'
                stdf_data.missing_value = netcdf_helper.MISSINGDATAVALUE
                stdf_data.valid_min = 0
                stdf_data.valid_max = 1000000.
                stdf_data.scale_factor = 1.
                stdf_data.add_offset = 0.
                stdf_data.standard_name = "stdev_"+var+varext
                stdf_data.long_name = "Standard deviation of "+var+varext+" occupying binned cell"

def correct_fco2_daily(socat_file,month_fco2,month_sst,daily_sst,co2 = '_fco2'):
    """
    Function to correct the monthly reanalysed fCO2sw data (at a monthly SST) to a respective daily SST data

    socat_file = the SOCAT tsv file with the reanalysed data and additional daily SST column (this should be a saved tsv file that's already been loaded previously by 'read_socat' function)
    month_fco2 = the monthly f/pCO2sw data column name
    month_sst = the monthly SST data column name, paired to the monthly fCO2_sw data
    daily_sst = the daily SST data column name
    co2 = string defining whether the month_fCO2 column is fCO2sw or pCO2sw
    """
    data = pd.read_table(socat_file,sep='\t')
    daily_fco2 = daily_sst + co2 + ' [uatm]'
    # Conversion value from Wanninkhof et al. (2022)
    data[daily_fco2] = data[month_fco2] * np.exp(0.0413 * (data[daily_sst] - data[month_sst]))
    data.to_csv(socat_file,sep='\t',index=False)
