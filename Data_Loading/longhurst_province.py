#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""

from netCDF4 import Dataset
import numpy as np
import shapefile as shp
import data_utils as du
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

def longhurst_map(file,time_s,time_e,lon=1,lat=1):
    data = shp.Reader(file)
    lon,lat = du.reg_grid(lon,lat)
    longhurst_produce(data,lon,lat)
    # print(data)
    # print(data.shape(0).points)
    # print(data.records())#.shape.__geo_interface__['coordinates'])

def longhurst_produce(shape,lon,lat):
    le = len(shape.shapes())
    long,latg = np.meshgrid(lon,lat)
    print(lon.shape)
    print(lat.shape)

    points = np.concatenate((np.reshape(long,[-1,1]),np.reshape(latg,[-1,1])),axis=1)
    map = np.empty((points.shape[0],1))
    map[:] = np.nan
    print(points.shape)
    for i in range(0,le):
        print(shape.records()[i])
        #path = mpltPath.Path(shape.shape(i).points)
        #print(path)
        #print(shape.shape(i).points[0].shape)
        inside = du.inpoly2(points,np.array(shape.shape(i).points))
        print(inside)
        map[inside[0] == True] = i

    plt.figure()
    plt.pcolor(long,latg,np.reshape(map,[180,360]))
    plt.show()

# def plot_test(file):
#     sf = shp.Reader(file)
#     print(sf.shape(1).__geo_interface__)
    # plt.figure()
    # for shape in sf.shapeRecords():
    #     x = [i[0] for i in shape.shape.points[:]]
    #     y = [i[1] for i in shape.shape.points[:]]
    #     plt.plot(x,y)
    # plt.show()


longhurst_map('D:/Data/Longhurst/Longhurst_world_v4_2010.shp',1990,2020)
#plot_test('D:/Data/Longhurst/Longhurst_world_v4_2010.shp')
