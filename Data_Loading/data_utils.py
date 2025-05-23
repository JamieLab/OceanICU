#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""
import glob
import datetime
import os
from netCDF4 import Dataset
import numpy as np
import scipy.interpolate as interp
from pathlib import Path
from dateutil.relativedelta import relativedelta

def load_netcdf_var(file, variable):
    """
    Function to load a specified netcdf variable from a file. [data<-10000] used to remove
    fill values (need to think of a better way...)
    """
    c = Dataset(file,'r')
    data = np.array(c.variables[variable][:])
    data[data<-10000] = np.nan
    c.close()
    #print(data)
    return data

def oneDegreeArea(latDegrees,res):
    re, rp = 6378.137, 6356.7523
    dtor = np.radians(1.0)
    # area of a 1x1 degree box at a given latitude in radians
    latRadians = latDegrees * dtor
    cosLat, sinLat = np.cos(latRadians), np.sin(latRadians)
    rc, rs = re * cosLat, rp * sinLat
    r2c, r2s = re * re * cosLat, rp * rp * sinLat
    earth_radius = np.sqrt((r2c * r2c + r2s * r2s) / (rc * rc + rs * rs))
    erd = earth_radius * dtor
    return erd * erd * cosLat / (1/res)**2

def reg_grid(lat=1,lon=1,latm = [-90,90],lonm = [-180,180],pad = False):
    """
    Definition of a regular grid of resolution in degrees. Able to specify the resolution
    for potential future higher spatial resoltions. Defaults to 1 deg, global grid.
    latm and lonm allows for regional grids to be constructed, i.e regular grid defined
    between latm, lonm
    """
    lat_g = np.arange(latm[0]+(lat/2),latm[1]-(lat/2)+lat,lat)
    lon_g = np.arange(lonm[0]+(lon/2),lonm[1]-(lon/2)+lon,lon)

    if pad:
        lat_g = lat_g - (lat/2); lat_g = np.append(lat_g,latm[1])
        lon_g = lon_g - (lon/2)

    return lon_g,lat_g

def area_grid(lon,lat,res):
    print(res)
    #lon,lat = reg_grid(lon=lon,lat=lat)
    area = np.zeros((len(lat),1))
    for i in range(len(lat)):
        area[i] = oneDegreeArea(lat[i],res)
    area = np.tile(area,(1,len(lon)))
    return area

def determine_grid_average(hlon,hlat,llon,llat):
    res = abs(llon[0] - llon[1])/2
    print(res)
    lo_grid = []
    la_grid = []
    for i in range(len(llon)):
        print(i/len(llon))
        #print(np.where(np.logical_and(hlon < llon[i]+res,hlon >= llon[i]-res)))
        lo_grid.append(np.where(np.logical_and(hlon < llon[i]+res,hlon >= llon[i]-res)))
    for i in range(len(llat)):
        print(i/len(llat))
        la_grid.append(np.where(np.logical_and(hlat < llat[i]+res,hlat >= llat[i]-res)))
        # grid[i,j] =
    return lo_grid,la_grid

def determine_grid_average_nonreg(hlon,hlat,llon,llat):
    res = abs(llon[0] - llon[1])/2
    print(res)
    llon,llat = np.meshgrid(llon,llat)
    llon = llon.ravel()
    llat = llat.ravel()
    out = []
    for i in range(len(llon)):
        out.append(np.where((hlon < llon[i]+res) & (hlon >= llon[i]-res) & (hlat < llat[i]+res) & (hlat >= llat[i]-res))[0])
    #print(out)
    return out

def grid_average_nonreg(var,in_grid):
    var_o = np.empty((len(in_grid)))
    var_o[:] = np.nan
    for i in range(len(in_grid)):
        #print(in_grid[i])
        if in_grid[i].size > 0:
            var_o[i] = np.nanmean(var[in_grid[i]])
    return var_o

def grid_average(var,lo_grid,la_grid,lon=[],lat=[],area_wei = False,land_mask=False,gebco_file=False,gebco_out=False,app=''):
    """

    """
    if area_wei:
        res = np.abs(lon[0] - lon[1])
        area = area_grid(lon,lat,res)
        area = np.transpose(area)

    if land_mask:
        import gebco_resample as geb

        file_t = Path(gebco_file).stem
        geb.gebco_resample(gebco_file,lon,lat,save_loc = os.path.join(gebco_out,file_t+'_'+str(res)+app+'.nc'))
        c=Dataset(os.path.join(gebco_out,file_t+'_'+str(res)+app+'.nc'),'r')
        ocean_proportion = np.array(c['ocean_proportion'])
        c.close()
        print(area.shape)
        area = area * ocean_proportion

    print(var.shape)
    print(len(lo_grid))
    print(len(la_grid))
    var_o = np.empty((len(lo_grid),len(la_grid)))
    var_o[:] = np.nan
    print(var_o.shape)
    for i in range(len(lo_grid)):
        print(i/len(lo_grid))
        for j in range(len(la_grid)):
            # print(lo_grid[i][0].size)
            # print(la_grid[j])
            # print(la_grid[j][0].size)
            # print(var[lo_grid[i],la_grid[j]])
            # print(np.nanmean(var[lo_grid[i],la_grid[j]]))
            if (la_grid[j][0].size == 0) or (lo_grid[i][0].size == 0):
                var_o[i,j] = np.nan
            else:
                # print(lo_grid[i][0])
                # print(la_grid[j][0])
                temp_lo,temp_la = np.meshgrid(lo_grid[i],la_grid[j])
                temp_lo = temp_lo.ravel(); temp_la = temp_la.ravel()
                # print(temp_lo)
                # print(temp_la)
                # print(var[temp_lo,temp_la])
                var_t = var[temp_lo,temp_la]
                if area_wei:
                    area_t = area[temp_lo,temp_la]
                    a = np.where(np.isnan(var_t) ==0)
                    # print(a)
                    if len(a[0]) == 0:
                        var_o[i,j] = np.nan
                    else:
                        if np.sum(area_t[a]) == 0:
                            var_o[i,j] = np.nan
                        else:
                            var_o[i,j] = np.average(var_t[a],weights = area_t[a])
                else:
                    var_o[i,j] = np.nanmean(var[temp_lo,temp_la])
    return var_o

def grid_switch(lon,var):
    """
    Function to switch a global grid from 0-360E to -180 to 180E. This requires the
    array to be manually modified.
    """
    var_sh = int(var.shape[0]/2)
    print(var_sh)
    var_temp = np.empty((var.shape))
    var_temp[0:var_sh,:] = var[var_sh:,:]
    var_temp[var_sh:,:] = var[0:var_sh,:]
    lon_temp = np.empty((var_sh*2))
    lon = lon-180
    # lon_temp[var_sh:] = lon[0:var_sh]
    # lon_temp[0:var_sh] = lon[var_sh:]
    return lon,var_temp

def insitu_grid(file,lon,lat,start_yr,end_yr,out_file,format='.csv',sep=',',
    year_col = '# Year',month_col = 'Month',day_col = 'Day',lat_col = 'Latitude (deg N)',lon_col = 'Longitude (deg W)',chla_col = 'chlorphyll-a (mgm-3)'
        ,skiprows=0,dateti=False,datecol='',unit = '',out_var_name='',out_var_long_name = '',min_val = -9999):
    """
    """
    import pandas as pd
    from construct_input_netcdf import append_netcdf
    res_h = np.abs(lon[0]-lon[1])/2
    t_len = (end_yr-start_yr+1)*12

    # data = np.genfromtxt('csv/'+file+'.csv', delimiter=',')
    data = pd.read_table(file+format,sep = sep,skiprows=skiprows)
    if dateti:
        time = data[datecol]
        temp = np.zeros((np.array(time).size,3)); temp[:] = np.nan
        for i in range(time.size):
            tt = datetime.datetime.strptime(time[i],dateformat)
            temp[i,0] = tt.year
            temp[i,1] = tt.month
            temp[i,2] = tt.day
        data[year_col] = temp[:,0]
        data[month_col] = temp[:,1]
        data[day_col] = temp[:,2]
    data = np.transpose(np.vstack((np.array(data[year_col]),np.array(data[month_col]),np.array(data[lat_col]),np.array(data[lon_col]),np.array(data[chla_col]))))
    data[:,-1][data[:,-1] <= min_val] = np.nan
    chl = np.zeros((len(lon),len(lat),t_len))
    chl[:] = np.nan
    print(chl.shape)
    print(data)
    print(data.shape)

    yr = start_yr
    mon=1
    i=0
    time = []
    while yr<=end_yr:
        print(str(yr) + ' - ' + str(mon))
        f = np.where((data[:,0] == yr) & (data[:,1] == mon))[0]
        if f.size!=0:
            print('Not Zeros')
            for j in range(0,len(lat)):
                for k in range(0,len(lon)):
                    g = np.where((data[f,2] > lat[j]-res_h) & (data[f,2] < lat[j]+res_h) & (data[f,3] > lon[k]-res_h) & (data[f,3] < lon[k]+res_h))
                    chl[k,j,i] = np.nanmean(data[f[g],4])

        time.append(datetime.datetime(yr,mon,15))
        i = i+1
        mon = mon+1
        if mon == 13:
            yr = yr+1
            mon=1
    direct ={}
    direct[out_var_name] = chl
    longname = {}
    longname[out_var_name] = out_var_long_name
    units = {}
    units[out_var_name] = unit
    append_netcdf(out_file,direct,lon,lat,0,units = units,longname=longname)
    c = Dataset(out_file,'a')
    c[out_var_name].file_input = file
    c.close()

def output_binned_insitu(data_file,in_var,nn_file,nn_var,output_file,name,unit):
    header = 'Year, Month, Day, Latitude (deg N), Longitude (deg E), in situ '+name+' ('+unit+'), in situ '+name+' standard_deviation ('+unit+'), UExP-FNN-U '+name+' ('+unit+'), UExP-FNN-U total uncertainty ('+unit+'), UExP-FNN-U region (unitless)'

    c = Dataset(data_file,'r')
    time = c.variables['time'][:]
    time_unit = c.variables['time'].units
    lat = c.variables['latitude'][:]
    lon = c.variables['longitude'][:]
    lat,lon = np.meshgrid(lat,lon)
    lat = np.repeat(lat[:, :, np.newaxis], time.shape[0], axis=2); lon = np.repeat(lon[:, :, np.newaxis], time.shape[0], axis=2);
    ins_data = c.variables[in_var][:]
    ins_unc = np.zeros((ins_data.shape))
    c.close()
    c = Dataset(nn_file,'r')
    nn_data = c.variables[nn_var][:]
    nn_unc = c.variables[nn_var+'_tot_unc'][:]
    c.close()

    time2 = np.zeros((lat.shape))
    for i in range(len(time)):
        time2[:,:,i] = time[i]

    time2 = np.reshape(time2,(-1,1)); lat = np.reshape(lat,(-1,1)); lon = np.reshape(lon,(-1,1)); ins_data = np.reshape(ins_data,(-1,1))
    ins_unc = np.reshape(ins_unc,(-1,1)); nn_data = np.reshape(nn_data,(-1,1)); nn_unc = np.reshape(nn_unc,(-1,1))

    f = np.where((np.isnan(ins_data) == False) & (np.isnan(nn_data) == False))
    time2 = time2[f]; lat = lat[f]; lon = lon[f]; ins_data = ins_data[f]; ins_unc = ins_unc[f]; nn_data=  nn_data[f]; nn_unc = nn_unc[f]

    year = np.zeros((time2.shape));month = np.zeros((time2.shape));day = np.zeros((time2.shape)); prov = np.zeros((time2.shape))
    for i in range(len(time2)):
        date = datetime.datetime.strptime(time_unit.split(' ')[-1],'%Y-%m-%d') + datetime.timedelta(days = int(time2[i]))
        year[i] = date.year
        month[i] = date.month
        day[i] = date.day

    out = np.transpose(np.stack((year,month,day,lat,lon,ins_data,ins_unc,nn_data,nn_unc,prov)))
    np.savetxt(output_file, out,header = header,delimiter = ',')

def load_grid(file,latv = 'latitude',lonv = 'longitude'):
    c = Dataset(file,'r')
    lat = np.array(c.variables[latv][:])
    lon = np.array(c.variables[lonv][:])
    c.close()
    return lon,lat

def checkfileexist(file):
    #print(file)
    g = glob.glob(file)
    #print(g)
    if not g:
        return False
    else:
        return True

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def numstr(num):
    if num < 10:
        return '0'+str(num)
    else:
        return str(num)

def numstr3(num):
    if num < 10:
        return '00'+str(num)
    if num < 100:
        return '0'+str(num)
    else:
        return str(num)

def time_con_str(date,units):
    uni = datetime.datetime.strptime(units.split(' ')[-1],'%Y-%m-%d')
    time2 = []
    for i in range(len(date)):
        time2.append([(uni + relativedelta(days=date[i])).year,(uni + relativedelta(days=date[i])).month])
    time2=np.array(time2)
    return time2

def netcdf_create_basic(file,var,var_name,lat,lon,flip=False,units=''):
    #copts={"zlib":True,"complevel":5} # Compression variables to save space :-)
    outp = Dataset(file,'w',format='NETCDF4_CLASSIC')
    outp.date_created = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outp.code_by = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    outp.code_location = 'https://github.com/JamieLab/OceanICU'
    outp.createDimension('lon',lon.shape[0])
    outp.createDimension('lat',lat.shape[0])
    if flip:
        sst_o = outp.createVariable(var_name,'f4',('lat','lon'),zlib=True)
        sst_o[:] = np.transpose(var)
    else:
        sst_o = outp.createVariable(var_name,'f4',('lon','lat'),zlib=True)
        sst_o[:] = var
    sst_o.units = units
    sst_o.time_generated = datetime.datetime.now().strftime(('%d/%m/%Y'))

    lat_o = outp.createVariable('latitude','f4',('lat'))
    lat_o[:] = lat
    lat_o.units = 'Degrees'
    lat_o.standard_name = 'Latitude'
    lon_o = outp.createVariable('longitude','f4',('lon'))
    lon_o.units = 'Degrees'
    lon_o.standard_name = 'Longitude'
    lon_o[:] = lon
    outp.close()

def netcdf_append_basic(file,var,var_name,flip=False,units=''):
    outp = Dataset(file,'a',format='NETCDF4_CLASSIC')
    if flip:
        if var_name in outp.variables.keys():
            sst_o = outp[var_name]
            sst_o[:] = np.tranpose(var)
        else:
            sst_o = outp.createVariable(var_name,'f4',('lat','lon'),zlib=True)
            sst_o[:] = np.transpose(var)
    else:
        if var_name in outp.variables.keys():
            sst_o = outp[var_name]
            sst_o[:] = var
        else:
            sst_o = outp.createVariable(var_name,'f4',('lon','lat'),zlib=True)
            sst_o[:] = var
    sst_o.units = units
    sst_o.time_generated = datetime.datetime.now().strftime(('%d/%m/%Y'))
    outp.close()

def lon_switch(var):
    temp = np.zeros((var.shape))
    temp[:,:,0:180] = var[:,:,180:]
    temp[:,:,180:] = var[:,:,0:180]
    return temp

def lon_switch_2d(var):
    temp = np.zeros((var.shape))
    temp[:,0:180] = var[:,180:]
    temp[:,180:] = var[:,0:180]
    return temp

def grid_interp(o_lon,o_lat,o_data,n_lon,n_lat,plot=False):

    o_data = o_data.T
    o_lon,o_lat = np.meshgrid(o_lon,o_lat)

    n_lon,n_lat = np.meshgrid(n_lon,n_lat)

    s = n_lon.shape
    points = np.stack([o_lon.ravel(),o_lat.ravel()],-1)
    out = interp.griddata(points,o_data.ravel(),(n_lon.ravel(),n_lat.ravel()))
    out = np.transpose(out.reshape(s))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolor(o_lon); plt.colorbar()
        plt.figure()
        plt.pcolor(o_lat); plt.colorbar()
        plt.figure()
        plt.pcolor(n_lon)
        plt.figure()
        plt.pcolor(n_lat); plt.colorbar()
        plt.figure()
        plt.pcolor(o_data);plt.colorbar()
        plt.figure()
        plt.pcolor(n_lon,n_lat,out.T)
        plt.show()
    return out

def point_interp(o_lon,o_lat,o_data,n_lon,n_lat,plot=False):

    o_lon,o_lat = np.meshgrid(o_lon,o_lat)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolor(o_lon,o_lat,o_data)
        plt.scatter(n_lon,n_lat)
        plt.show()
    points = np.stack([o_lon.ravel(),o_lat.ravel()],-1)
    out = interp.griddata(points,o_data.ravel(),(n_lon,n_lat))
    return out


def inpoly2(vert, node, edge=None, ftol=5.0e-14):
    """
    INPOLY2: compute "points-in-polygon" queries.
    STAT = INPOLY2(VERT, NODE, EDGE) returns the "inside/ou-
    tside" status for a set of vertices VERT and a polygon
    NODE, EDGE embedded in a two-dimensional plane. General
    non-convex and multiply-connected polygonal regions can
    be handled. VERT is an N-by-2 array of XY coordinates to
    be tested. STAT is an associated N-by-1 boolean array,
    with STAT[II] = TRUE if VERT[II, :] is an inside point.
    The polygonal region is defined as a piecewise-straight-
    line-graph, where NODE is an M-by-2 array of polygon ve-
    rtices and EDGE is a P-by-2 array of edge indexing. Each
    row in EDGE represents an edge of the polygon, such that
    NODE[EDGE[KK, 0], :] and NODE[EDGE[KK, 2], :] are the
    coordinates of the endpoints of the KK-TH edge. If the
    argument EDGE is omitted it assumed that the vertices in
    NODE are connected in ascending order.
    STAT, BNDS = INPOLY2(..., FTOL) also returns an N-by-1
    boolean array BNDS, with BNDS[II] = TRUE if VERT[II, :]
    lies "on" a boundary segment, where FTOL is a floating-
    point tolerance for boundary comparisons. By default,
    FTOL ~ EPS ^ 0.85.
    --------------------------------------------------------
    This algorithm is based on a "crossing-number" test,
    counting the number of times a line extending from each
    point past the right-most end of the polygon intersects
    with the polygonal boundary. Points with odd counts are
    "inside". A simple implementation requires that each
    edge intersection be checked for each point, leading to
    O(N*M) complexity...
    This implementation seeks to improve these bounds:
  * Sorting the query points by y-value and determining can-
    didate edge intersection sets via binary-search. Given a
    configuration with N test points, M edges and an average
    point-edge "overlap" of H, the overall complexity scales
    like O(M*H + M*LOG(N) + N*LOG(N)), where O(N*LOG(N))
    operations are required for sorting, O(M*LOG(N)) operat-
    ions required for the set of binary-searches, and O(M*H)
    operations required for the intersection tests, where H
    is typically small on average, such that H << N.
  * Carefully checking points against the bounding-box asso-
    ciated with each polygon edge. This minimises the number
    of calls to the (relatively) expensive edge intersection
    test.
    Updated: 19 Dec, 2020
    Authors: Darren Engwirda, Keith Roberts
    """

    vert = np.asarray(vert, dtype=np.float64)
    node = np.asarray(node, dtype=np.float64)

    STAT = np.full(
        vert.shape[0], False, dtype=np.bool_)
    BNDS = np.full(
        vert.shape[0], False, dtype=np.bool_)

    if node.size == 0: return STAT, BNDS

    if edge is None:
#----------------------------------- set edges if not passed
        indx = np.arange(0, node.shape[0] - 1)

        edge = np.zeros((
            node.shape[0], +2), dtype=np.int32)

        edge[:-1, 0] = indx + 0
        edge[:-1, 1] = indx + 1
        edge[ -1, 0] = node.shape[0] - 1

    else:
        edge = np.asarray(edge, dtype=np.int32)

#----------------------------------- prune points using bbox
    xdel = np.nanmax(node[:, 0]) - np.nanmin(node[:, 0])
    ydel = np.nanmax(node[:, 1]) - np.nanmin(node[:, 1])

    lbar = (xdel + ydel) / 2.0

    veps = (lbar * ftol)

    mask = np.logical_and.reduce((
        vert[:, 0] >= np.nanmin(node[:, 0]) - veps,
        vert[:, 1] >= np.nanmin(node[:, 1]) - veps,
        vert[:, 0] <= np.nanmax(node[:, 0]) + veps,
        vert[:, 1] <= np.nanmax(node[:, 1]) + veps)
    )

    vert = vert[mask]

    if vert.size == 0: return STAT, BNDS

#------------------ flip to ensure y-axis is the `long` axis
    xdel = np.amax(vert[:, 0]) - np.amin(vert[:, 0])
    ydel = np.amax(vert[:, 1]) - np.amin(vert[:, 1])

    lbar = (xdel + ydel) / 2.0

    if (xdel > ydel):
        vert = vert[:, (1, 0)]
        node = node[:, (1, 0)]

#----------------------------------- sort points via y-value
    swap = node[edge[:, 1], 1] < node[edge[:, 0], 1]
    temp = edge[swap]
    edge[swap, :] = temp[:, (1, 0)]

#----------------------------------- call crossing-no kernel
    stat, bnds = \
        _inpoly(vert, node, edge, ftol, lbar)

#----------------------------------- unpack array reindexing
    STAT[mask] = stat
    BNDS[mask] = bnds

    return STAT, BNDS


def _inpoly(vert, node, edge, ftol, lbar):
    """
    _INPOLY: the local pycode version of the crossing-number
    test. Loop over edges; do a binary-search for the first
    vertex that intersects with the edge y-range; crossing-
    number comparisons; break when the local y-range is met.
    """

    feps = ftol * (lbar ** +1)
    veps = ftol * (lbar ** +1)

    stat = np.full(
        vert.shape[0], False, dtype=np.bool_)
    bnds = np.full(
        vert.shape[0], False, dtype=np.bool_)

#----------------------------------- compute y-range overlap
    ivec = np.argsort(vert[:, 1], kind="quicksort")

    XONE = node[edge[:, 0], 0]
    XTWO = node[edge[:, 1], 0]
    YONE = node[edge[:, 0], 1]
    YTWO = node[edge[:, 1], 1]

    XMIN = np.minimum(XONE, XTWO)
    XMAX = np.maximum(XONE, XTWO)

    XMIN = XMIN - veps
    XMAX = XMAX + veps
    YMIN = YONE - veps
    YMAX = YTWO + veps

    YDEL = YTWO - YONE
    XDEL = XTWO - XONE

    EDEL = np.abs(XDEL) + YDEL

    ione = np.searchsorted(
        vert[:, 1], YMIN,  "left", sorter=ivec)
    itwo = np.searchsorted(
        vert[:, 1], YMAX, "right", sorter=ivec)

#----------------------------------- loop over polygon edges
    for epos in range(edge.shape[0]):

        xone = XONE[epos]; xtwo = XTWO[epos]
        yone = YONE[epos]; ytwo = YTWO[epos]

        xmin = XMIN[epos]; xmax = XMAX[epos]

        edel = EDEL[epos]

        xdel = XDEL[epos]; ydel = YDEL[epos]

    #------------------------------- calc. edge-intersection
        for jpos in range(ione[epos], itwo[epos]):

            jvrt = ivec[jpos]

            if bnds[jvrt]: continue

            xpos = vert[jvrt, 0]
            ypos = vert[jvrt, 1]

            if xpos >= xmin:
                if xpos <= xmax:
                #------------------- compute crossing number
                    mul1 = ydel * (xpos - xone)
                    mul2 = xdel * (ypos - yone)

                    if feps * edel >= abs(mul2 - mul1):
                #------------------- BNDS -- approx. on edge
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == yone) and (xpos == xone):
                #------------------- BNDS -- match about ONE
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == ytwo) and (xpos == xtwo):
                #------------------- BNDS -- match about TWO
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (mul1 <= mul2) and (ypos >= yone) \
                            and (ypos < ytwo):
                #------------------- advance crossing number
                        stat[jvrt] = not stat[jvrt]

            elif (ypos >= yone) and (ypos < ytwo):
            #----------------------- advance crossing number
                stat[jvrt] = not stat[jvrt]

    return stat, bnds
