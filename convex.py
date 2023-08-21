#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 07/2023

This script adds functionality required by the Convex Seascape Survey.

"""
import numpy as np;
import shapefile as shp
from shapely.ops import cascaded_union
from shapely.geometry import Polygon
from Data_Loading.data_utils import inpoly2

def merge_LMEs(shape,ind,vals):
    l = []
    for i in vals:
        f = np.squeeze(np.where((ind == i)))
        lm = cut_LMEs(np.array(shape.shape(f).points))
        #g = np.squeeze(np.where( (lm[0,0] == lm[:,0]) & (lm[0,1] == lm[:,1] )))
        l.append(Polygon(zip(lm[:,0],lm[:,1])))
    bound = cascaded_union(l)
    x,y = bound.exterior.xy
    return x,y

def cut_LMEs(points):
    g = np.squeeze(np.where( (points[0,0] == points[:,0]) & (points[0,1] == points[:,1] )))
    points = points[0:g[1]+1,:]
    return points

def load_LMEs(shape,ind,vals):
    f = np.squeeze(np.where((ind == vals)))
    lm = cut_LMEs(np.array(shape.shape(f).points))
    return lm[:,0],lm[:,1],shape.record(f)[2]

def load_LME_file(sh_file):
    LME = shp.Reader(sh_file)
    ind = []
    for i in range(0,len(LME.shapes())):
        print(LME.record(i))
        ind.append(int(LME.record(i)[1]))
    ind = np.array(ind)
    return LME,ind
