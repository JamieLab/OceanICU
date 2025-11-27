 #!/usr/bin/env python3
# Python implmentation of the weighted statistics presented in Ford et al. (2021; https://doi.org/10.1016/j.rse.2021.112435).
# The functions can also be called in a unweighted approach, where weights are all set to 1 (equal importance).
# Created by Daniel Ford (d.ford@exeter.ac.uk)
# -----------------------------
# Version 1.0 - 20/09/2022
# - Inital porting to Python
# ---------------------------
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import bootstrap

def round_to(x,sig=2):
    return round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)

def weighted_stats(x,y,weights,meth,n_resamples=200,confidence_level=0.95,bootstraping=False):
    f = np.squeeze(np.argwhere( (np.isnan(x) == 0) & (np.isnan(y) == 0) &  (np.isnan(weights) == 0)))
    if f.size >= 1:
        x = x[f]
        y = y[f]
        weights = weights[f]
        weights = weights/np.sum(weights)
        bi = rel_bias(x,y,weights)

        bi_m = med_rel_bias(x,y,weights)
        abi = abs_bias(x,y,weights)
        rms = rmsd(x,y,weights)

        ap = apd(x,y,weights)
        rp = rpd(x,y,weights)
        cor = pearson_corr(x,y,weights)
        ma = mad(x,y,weights)
        if f.size > 1:
            slope,intercept = regress(x,y,weights)
            if bootstraping:
                regress_err = bootstrap((x,y,weights),regress,confidence_level =confidence_level,paired=True,n_resamples=n_resamples)
            else:
                regress_err = np.nan
        else:
            slope = np.nan
            intercept = np.nan

        if bootstraping:
            bi_err = bootstrap((x,y,weights),rel_bias,confidence_level =confidence_level,paired=True,n_resamples=n_resamples)
            rms_err = bootstrap((x,y,weights),rmsd,confidence_level =confidence_level,paired=True,n_resamples=n_resamples)
            struct = {
                'meth':meth,
                'rel_bias':bi,
                'rel_bias_err':bi_err.standard_error*2,
                'med_rel_bias':bi_m,
                'abs_bias':abi,
                'rmsd':rms,
                'rmsd_err':rms_err.standard_error*2,
                'slope':slope,
                'slope_err':regress_err.standard_error[0]*2,
                'intercept':intercept,
                'intercept_err':regress_err.standard_error[1]*2,
                'apd':ap,
                'rpd':rp,
                'mad':ma,
                'n':f.size,
                'pearson_corr':cor
                }
        else:
            struct = {
                'meth':meth,
                'rel_bias':bi,
                'rel_bias_err':np.nan,
                'med_rel_bias':bi_m,
                'abs_bias':abi,
                'rmsd':rms,
                'rmsd_err':np.nan,
                'slope':slope,
                'slope_err':np.nan,
                'intercept':intercept,
                'intercept_err':np.nan,
                'apd':ap,
                'rpd':rp,
                'mad':ma,
                'n':f.size,
                'pearson_corr':cor
                }
    else:
        struct = {
        'meth':meth,
            'rel_bias':np.nan,
            'rel_bias_err':np.nan,
            'med_rel_bias':np.nan,
            'abs_bias':np.nan,
            'rmsd':np.nan,
            'rmsd_err':np.nan,
            'slope':np.nan,
            'slope_err':np.nan,
            'intercept':np.nan,
            'intercept_err':np.nan,
            'apd':np.nan,
            'rpd':np.nan,
            'mad':np.nan,
            'n':f.size,
            'pearson_corr':np.nan
            }
    return struct

def unweighted_stats(x,y,meth):
    f = np.squeeze(np.argwhere( (np.isnan(x) == 0) & (np.isnan(y) == 0) ))
    #print(f.size)
    if f.size >= 1:
        #print(f)
        x = x[f]
        y = y[f]
        weights = np.ones((f.size))
        if f.size == 1:
            weights =weights[0]
        #print(x)
        #print(weights)
        su = np.sum(weights)
        weights = weights/su
        #print(weights)
        bi = rel_bias(x,y,weights)
        bi_m = med_rel_bias(x,y,weights)
        abi = abs_bias(x,y,weights)
        rms = rmsd(x,y,weights)
        ap = apd(x,y,weights)
        rp = rpd(x,y,weights)
        cor = pearson_corr(x,y,weights)
        ma = mad(x,y,weights)
        if f.size > 1:
            #weights = weights*su
            #print(weights)
            slope,intercept = regress(x,y,weights)
        else:
            slope = np.nan
            intercept = np.nan

        struct = {'meth':meth,'rel_bias':bi,'med_rel_bias':bi_m,'abs_bias':abi,'rmsd':rms,'slope':slope,'intercept':intercept,'apd':ap,'rpd':rp,'n':f.size,'pearson_corr':cor,'mad':ma}
    else:
        struct = {'meth':meth,'rel_bias':np.nan,'med_rel_bias':np.nan,'abs_bias':np.nan,'rmsd':np.nan,'slope':np.nan,'intercept':np.nan,'apd':np.nan,'rpd':np.nan,'n':f.size,'pearson_corr':np.nan,'mad':np.nan}
    return struct

def rel_bias(x,y,weights):
    bi = np.average((x)-(y), weights=weights)
    return bi

def med_rel_bias(x,y,weights):
    bi = np.median((x)-(y))
    return bi

def abs_bias(x,y,weights):
    bi = np.average(np.abs(x)-np.abs(y), weights=weights)
    return bi

def rmsd(x,y,weights):
    rms = np.sqrt(np.average((x-y)**2, weights=weights))
    return rms

def apd(x,y,weights):
    ap =  np.average( np.abs(x-y) / (np.abs(np.abs(x)+np.abs(y))/2), weights=weights) *100
    return ap

def rpd(x,y,weights):
    ap =  np.average( (x-y) / (np.abs(np.abs(x)+np.abs(y))/2), weights=weights ) *100
    return ap

def mad(x,y,weights):
    ma = np.average(np.abs(y - x),weights=weights)
    return ma

def regress(x,y,weights):
    co = np.cov((np.stack((x,y))),aweights=weights,ddof=1)
    #print(co)
    x_s = DescrStatsW(x, weights=weights)
    y_s = DescrStatsW(y, weights=weights)
    #print('Var X = '+str(x_s.var))
    #print('Var Y = '+str(y_s.var))
    #print('Cov'+str(co[0,1]))

    lam = 0.5*(x_s.var + y_s.var + np.sqrt( ( x_s.var + y_s.var)**2 - 4 * (x_s.var * y_s.var - co[0,1]**2)))
    slope = co[0,1] / (lam - y_s.var)
    intercept = y_s.mean - (slope*x_s.mean)

    return slope,intercept

def pearson_corr(x,y,weights):
    c = DescrStatsW(np.transpose(np.stack((x,y))),weights=np.transpose((weights)))
    #print(c.corrcoef[0,1])
    return c.corrcoef[0,1]
