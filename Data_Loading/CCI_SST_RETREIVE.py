#!/usr/bin/env python
import cdsapi
import zipfile
import datetime
import os
import glob
import time

def numstr(num):
    if num < 10:
        return '0'+str(num)
    else:
        return str(num)

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def checkfileexist(file):
    #print(file)
    g = glob.glob(file)
    #print(g)
    if not g:
        return 0
    else:
        return 1


loc = "D:/Data/SST-CCI/"
end_year = 2023
d = datetime.datetime(1981,9,1)

while d.year < end_year:
    ye = d.strftime("%Y")
    mon = d.strftime("%m")
    day = d.strftime("%d")
    if d.day == 1:
        p = loc+str(d.year)
        makefolder(p)
        p = p+'/'+mon
        makefolder(p)

    if checkfileexist(p+'/'+d.strftime("%Y%m%d")+'*.nc') == 0:
        cou = 0
        while True:
            try:
                time.sleep(5)
                if cou == 3:
                    break
                c = cdsapi.Client()
                c.retrieve(
                'satellite-sea-surface-temperature',
                {
                'version': '2_1',
                'variable': 'all',
                'format': 'zip',
                'processinglevel': 'level_4',
                'sensor_on_satellite': 'combined_product',
                'year': ye,
                'month': mon,
                'day': day,
                },
                loc+'download.zip')
            except:
                cou = cou+1
            break
        if cou == 3:
            raise RetryError

        with zipfile.ZipFile(loc+'download.zip', 'r') as zip_ref:
            zip_ref.extractall(p)
    d = d+datetime.timedelta(days=1)
