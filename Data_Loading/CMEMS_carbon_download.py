import copernicusmarine
import datetime
import os
import data_utils as du

start_yr = 1985
end_yr = 2024
version = "202603"
output_loc = 'D:/Data/CMEMS-CARBON-'+version
output_file_struct = 'CMEMS-CARBON-'+version+'_%Y_%M.nc'

yr = start_yr
mon=1

while yr<=end_yr:
    d = datetime.datetime(yr,mon,1)
    if not du.checkfileexist(os.path.join(output_loc,output_file_struct.replace('%Y',str(d.year)).replace('%M',du.numstr(d.month)))):

        copernicusmarine.subset(
          dataset_id="cmems_obs-mob_glo_bgc-car_my_irr-i",
          dataset_version=version,
          variables=["fgco2", "fgco2_uncertainty", "omega_ar", "omega_ar_uncertainty", "omega_ca", "omega_ca_uncertainty", "ph", "ph_uncertainty", "spco2", "spco2_uncertainty", "talk", "talk_uncertainty", "tco2", "tco2_uncertainty"],
          minimum_longitude=-179.875,
          maximum_longitude=179.875,
          minimum_latitude=-89.875,
          maximum_latitude=89.875,
          start_datetime=d.strftime('%Y-%m-%dT%H:%M:%S'),
          end_datetime=d.strftime('%Y-%m-%dT%H:%M:%S'),
          coordinates_selection_method="strict-inside",
          netcdf_compression_level=1,
          disable_progress_bar=True,
          output_filename = output_file_struct.replace('%Y',str(d.year)).replace('%M',du.numstr(d.month)),
          output_directory = output_loc,
        )
    else:
        print('File Exists!')
    mon=mon+1
    if mon==13:
        yr=yr+1
        mon=1
