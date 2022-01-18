#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:20:17 2021

@author: chosila
"""

import numpy as np
import scipy
import scipy.stats as sci
import time
from numba import jit

#@jit 
def calc_pull(D_raw, R_list_raw, tol, optAB):
    intD = D_raw.sum()  ## Total number of data entries

    nRef = len(R_list_raw)  ## Number of reference histograms
    intR_list = np.array([x.sum() for x in R_list_raw])  ## Total number of entries per ref hist
    intR_total = intR_list.sum()         ## Total number of entries summed across all ref hists

    scaleTol = np.power(1 + np.power(R_list_raw * tol**2, 2), -0.5)  ## 2D array of scaling factors, nRef x nBins
    ## Multiply along only the first axis: see https://stackoverflow.com/questions/49435353/numpy-multiply-arbitrary-shape-array-along-first-axis
    intR_list_tol = (scaleTol.T * intR_list).T 

    R_list_raw_tol = R_list_raw * scaleTol  ## 2D array of scaled number of entries per ref hist bin



    ## get the TF array for selecting which one to do betabinom on 
    ## lowstatcond should have all positions where any ref or d_raw positions have stats lower than 100
    low_stat_cut = 20
    lowStatCond = np.zeros_like(D_raw, dtype=bool)
    for i in R_list_raw: 
        lowStatCond = lowStatCond | (i < low_stat_cut)
    lowStatCond = lowStatCond | (D_raw < low_stat_cut)
    #lowStatCond = np.ones_like(D_raw, dtype=bool)

    ## Initialize arrays probHi and probLo with same dimensions as D_raw, values all 0

    probHi = np.zeros(np.count_nonzero(lowStatCond))
    probLo = np.zeros(np.count_nonzero(lowStatCond))

    for iRef in range(nRef):
        if optAB == 'A':  ## For approach 'A', all reference histograms receive equal weight
            wgt_AB = 1.0 / nRef
        if optAB == 'B':  ## For approach 'B', histograms weighted by number of entries        
            wgt_AB = 1.0 * intR_list[iRef] / intR_total
        
        ## create array where only the ones that meet conditions are filled. rest of array zero. this doens't work with optAB==B
        draw = D_raw[lowStatCond]#np.where(lowStatCond, D_raw, 0)
        draw1 = draw -1 #np.where(lowStatCond, D_raw, 0) - np.where(lowStatCond, np.ones_like(D_raw), 0)
        rlistrawtol1 = R_list_raw_tol[iRef][lowStatCond] + 1 #np.where(lowStatCond, R_list_raw_tol[iRef], 0 ) + np.where(lowStatCond, np.ones_like(D_raw), 0)
        intRtol = intR_list_tol[iRef][lowStatCond]#np.where(lowStatCond, intR_list_tol[iRef], 0)
    
        probHi += wgt_AB * np.nan_to_num(sci.betabinom.sf(draw1, intD, rlistrawtol1, intRtol - rlistrawtol1))
        probLo += wgt_AB * np.nan_to_num(sci.betabinom.cdf(draw,  intD, rlistrawtol1, intRtol - rlistrawtol1))

        #probHi += wgt_AB * sci.betabinom.sf(D_raw-1, intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)
        #probLo += wgt_AB * sci.betabinom.cdf(D_raw,  intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)


    ## For each bin, take the lower probability (i.e. the one that is < 50%)
    prob = probLo
    condition = probHi < probLo
    prob[condition] = probHi[condition]

    prob *= 2  ## Multiply by 2 for 2-sided probability  
    prob = np.clip(prob, a_max=1, a_min=None)

    pull_lowStat = calcPull(prob)


    ## calculate the rest of them using normal squareroot method 
    ## first normalize all of theme to refavg, then do the sqrt
    intRavg = R_list_raw.mean(axis=0).sum() 
    D_norm = D_raw*intRavg/D_raw.sum()
    pull_highStat = []
    for i in R_list_raw:
        R_norm = i*intRavg/i.sum()
        pull_highStat.append(sqrtpull(D_norm, R_norm))

    print(f'{pull_highStat=}')
    pull_highStat = np.array(pull_highStat).mean(axis=0)
    
    pull = pull_highStat
    pull[lowStatCond] = pull_lowStat#[lowStatCond]    

    ## give direction to pull. needed to normalize the data to avg ref
    pull[R_list_raw.mean() < D_norm]*=-1

    print(f'{pull=}')
    print('-----------------------------------')
    return pull


def calcPull(prob):
    prob = np.clip(prob, a_min = np.power(0.1,16), a_max=None)## recalc prob so that it's not too small
    pull = np.sqrt(scipy.stats.chi2.ppf(1-prob,1)) ## convert to pull value
        
    return pull

def sqrtpull(data, ref):
    dataErrs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * data)) / 2 - 1 - data))
    refErrs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * ref)) / 2 - 1 - ref))
    
    ## if data[i] > ref[i], use dataErr[0][i], refErr[1][i] and vice versa
    pull = (data - ref) / ((dataErrs[0]**2 + refErrs[1]**2)**0.5)
    cond = data < ref
    pull[cond] = (data[cond] - ref[cond]) / ((dataErrs[1][cond]**2 + refErrs[0][cond]**2)**0.5)


    #print(f'{dataErrs[1]=}')
    #print(f'{refErrs[0]=}')
    #print(f'{data=}')
    #print(f'{ref=}')

    return pull
