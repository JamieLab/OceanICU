import os
import sys
from netCDF4 import Dataset
import numpy as np
import datetime

sys.path.append(os.path.join('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU','Data_Loading'))
sys.path.append(os.path.join('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU'))

import fluxengine_driver as fl
import data_utils as du

model_save_loc = 'F:/OceanCarbon4Climate/GCB2025'
flux_loc = os.path.join(model_save_loc,'flux')
# flux_loc = os.path.join(model_save_loc,'flux_wanninkhof_era5')
file_loc = os.path.join(model_save_loc,'output.nc')
start_yr = 2010
end_yr = 2020

ref_time = datetime.datetime(1970,1,15)
flux_dict = {
    'atmfco2': 'OAPC1',
    'wind_speed': 'WS1_mean',
    'wind_speed_second': 'windu10_moment2',
    'pressure': 'air_pressure',
    'sst_skin': 'ST1_Kelvin_mean',
    'sst_subskin': 'FT1_Kelvin_mean',
    'salinity_subskin': 'OKS1',
    'salinity_skin': 'salinity_skin',
    'v_gas':'V_gas'
}
variable_dict = [
    'fco2',
    'fco2_tot_unc',
    'flux',
    'flux_unc_fco2sw',
    'flux_unc_k',
    'flux_unc_ph2o',
    'flux_unc_ph2o_fixed',
    'flux_unc_schmidt',
    'flux_unc_schmidt_fixed',
    'flux_unc_solskin_unc',
    'flux_unc_solskin_unc_fixed',
    'flux_unc_solsubskin_unc',
    'flux_unc_solsubskin_unc_fixed',
    'flux_unc_wind',
    'flux_unc_xco2atm',
]
c = Dataset(file_loc,'r')
lon = np.array(c['longitude'])
lat = np.array(c['latitude'])
time = np.array(c['time'])
time2=[]
for i in range(len(time)):
    time2.append( [(datetime.timedelta(days=int(time[i])) + ref_time).year,(datetime.timedelta(days=int(time[i])) + ref_time).month])
time2 = np.array(time2)

f = np.where((time2[:,0] >=start_yr) & (time2[:,0] <= end_yr))[0]
time2 = time2[f]
direct = {}
direct_vars = {}
for i in variable_dict:
    direct[i] = np.array(c[i][:,:,f])
    direct[i][direct[i] > 10000] = np.nan


direct['atmfco2'] = np.transpose(fl.load_flux_var(flux_loc,'OAPC1',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['wind_speed'] = np.transpose(fl.load_flux_var(flux_loc,'WS1_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['wind_speed_second'] = np.transpose(fl.load_flux_var(flux_loc,'windu10_moment2',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['pressure'] = np.transpose(fl.load_flux_var(flux_loc,'air_pressure',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['sst_skin'] = np.transpose(fl.load_flux_var(flux_loc,'ST1_Kelvin_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['sst_subskin'] = np.transpose(fl.load_flux_var(flux_loc,'FT1_Kelvin_mean',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['salinity_subskin'] = np.transpose(fl.load_flux_var(flux_loc,'OKS1',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['salinity_skin'] = np.transpose(fl.load_flux_var(flux_loc,'salinity_skin',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])
direct['v_gas'] = np.transpose(fl.load_flux_var(flux_loc,'V_gas',start_yr,end_yr,lon.shape[0],lat.shape[0],time.shape[0]),[1,0,2])

direct_out = {}

for i in list(direct.keys()):
    direct_out[i] = np.zeros((len(lon),len(lat),12))
direct_out['flux_unc'] = np.zeros((len(lon),len(lat),12))
vars = list(direct_out.keys())

for i in range(1,13):
    f = np.where(time2[:,1] == i)[0]
    print(f)
    for j in vars:
        if j == 'fco2_tot_unc':
            direct_out[j][:,:,i-1] = np.sqrt(np.nansum(direct[j][:,:,f]**2,axis=2)) / len(f)
            direct_out[j][:,:,i-1][direct_out[j][:,:,i-1] <0.001] = np.nan
        elif j == 'flux_unc':
            v = ['flux_unc_k','flux_unc_ph2o_fixed','flux_unc_schmidt_fixed','flux_unc_solskin_unc_fixed','flux_unc_solsubskin_unc_fixed']
            unc ={}
            for k in v:
                unc[k] = np.nanmean(direct[k][:,:,f] * np.abs(direct['flux'][:,:,f]),axis=2)
                direct_out[k][:,:,i-1] = unc[k]

            v = ['flux_unc_fco2sw','flux_unc_ph2o','flux_unc_schmidt','flux_unc_solskin_unc','flux_unc_solsubskin_unc','flux_unc_wind','flux_unc_xco2atm']

            for k in v:
                unc[k] = np.sqrt(np.nansum((direct[k][:,:,f]*np.abs(direct['flux'][:,:,f]))**2,axis=2))/len(f)
                direct_out[k][:,:,i-1] = unc[k]
            t = list(unc.keys())
            out = np.zeros((len(lon),len(lat),len(t)))
            for tp in range(len(t)):
                out[:,:,tp] = unc[t[tp]]
            direct_out[j][:,:,i-1] = np.sqrt(np.nansum(out**2,axis=2))
            #direct_out[j][:,:,i-1][direct_out[j][:,:,i-1] <0.001] = np.nan
        else:
            if 'flux_unc' in j:
                print('Skipping flux uncertainties...')
            else:
                direct_out[j][:,:,i-1] = np.nanmean(direct[j][:,:,f],axis=2)

for v in vars:
    if 'flux_unc' in v:
        direct_out[v][np.isnan(direct_out['fco2']) == 1] = np.nan

# outs = Dataset('GCB-2025_dataprod_UExP-FNN-U_1980-2024_climatologyperiod_'+str(start_yr)+'-'+str(end_yr)+'.nc','w')
outs = Dataset('Fordetal_UExP-FNN-U_surface-carbonate-system_v2025-1_climatologyperiod_'+str(start_yr)+'-'+str(end_yr)+'.nc','w')

outs.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
outs.created_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
outs.created_from = 'Data created from ' + model_save_loc
outs.climatology_file_generated = 'True'
outs.climatology = 'Climatology generated between ' + str(start_yr) + ' and ' + str(end_yr)
outs.createDimension('lon',lon.shape[0])
outs.createDimension('lat',lat.shape[0])
outs.createDimension('month',12)
var = outs.createVariable('lat','f4',('lat'))
var[:] = lat
var.Long_name = 'Latitude'
var.units = 'Degrees North'

var = outs.createVariable('lon','f4',('lon'))
var[:] = lon #Convert form -180 to 180, 0 to 360
var.Long_name = 'Longitude'
var.units = 'Degrees East'

for i in vars:
    var_o = outs.createVariable(i,'f4',('lon','lat','month'),fill_value=np.nan)
    var_o[:] = direct_out[i]
    try:
        var_o.setncatts(c[i].__dict__)
        print('Added attributes for '+ i)
    except:
        print('No attributes for ' + i)
    var_o.climatology = 'Climatology generated between ' + str(start_yr) + ' and ' + str(end_yr)

d=Dataset(os.path.join(flux_loc,str(start_yr),'01','OceanFluxGHG-month01-jan-'+str(start_yr)+'-v0.nc'),'r')
for i in vars:
    print(i)
    if i in list(flux_dict.keys()):
        if flux_dict[i] in list(d.variables.keys()):
            a = d[flux_dict[i]].__dict__
            a.pop('_FillValue')
            a.pop('missing_value')
            a.pop('scale_factor')
            a.pop('add_offset')
            outs.variables[i].setncatts(a)
d.close()

v = list(outs.variables.keys())
for i in v:
    if 'flux_unc' in i:
        outs[i].units = 'g C m-2 d-1'
        outs[i].comment = ''

outs['atmfco2'].Long_name = 'Atmospheric fCO2'
outs['atmfco2'].units = 'uatm'
outs['atmfco2'].description = 'Atmospheric fCO2 evaluated at the skin temperature'

d = Dataset(os.path.join(model_save_loc,'inputs','bath.nc'),'r')
mask = np.array(d['ocean_proportion'][:])
d.close()

var_o = outs.createVariable('area','f4',('lon','lat'),fill_value=np.nan)
var_o[:] =np.transpose(du.area_grid(lat=lat,lon=lon,res=1) * 1e6) * mask
var_o.long_name = 'Total surface area of each grid cell'
var_o.units = 'm^2'
var_o.description = 'Calculated assuming the Earth is a oblate sphere with major and minor radius of 6378.137 km and 6356.7523 km respectively and a ocean proportion mask'

c.close()
outs.close()
