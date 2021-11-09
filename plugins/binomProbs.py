#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:20:17 2021

@author: chosila
"""

import numpy as np
import scipy
import scipy.stats as sci

def calc_pull_A(D_raw, R_list_raw, tol):
    intD = D_raw.sum()  ## Total number of data entries
    D_norm = D_raw / intD  ## Data entries per bin, normalized to 1.0

    nRef = len(R_list_raw)  ## Number of reference histograms
    intR_list = np.array([x.sum() for x in R_list_raw])  ## Total number of entries per ref hist
    R_norm_list = np.array([x*1/x.sum() for x in R_list_raw])  ## Array of normalized ref hists
    R_norm_avg = R_norm_list.mean(axis=0)  ## Average of normalized yields per bin in ref hists

    scaleTol = pow(1 + pow(R_list_raw * tol**2, 2), -0.5)  ## 2D array of scaling factors, nRef x nBins
    #intR_list_tol = intR_list * scaleTol  ## 2D array of scaled number of total reference hist entries
    intR_list_tol = (scaleTol.T * intR_list).T ## to multiply along only the first axis, see https://stackoverflow.com/questions/49435353/numpy-multiply-arbitrary-shape-array-along-first-axis
    R_list_raw_tol = R_list_raw * scaleTol  ## 2D array of scaled number of entries per ref hist bin


    ## Initialize arrays probHi and probLo with same dimensions as D_raw, values all 0
    probHi = np.zeros_like(D_raw)
    probLo = np.zeros_like(D_raw)

    for iRef in range(nRef):
        probHi += sci.betabinom.sf(D_raw-1, intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)
        probLo += sci.betabinom.cdf(D_raw,  intD, R_list_raw_tol[iRef] + 1, intR_list_tol[iRef] - R_list_raw_tol[iRef] + 1)

    ## prob = probHi if (D_norm >= R_norm_avg) else probLo  ## However you do substitution bin-by-bin
    ## bin by bin substitution by declaring the prob = probLo first, then fill with probHi where meets condition 
    prob = probLo
    condition = D_norm >= R_norm_avg
    prob[condition] = probHi[condition]
    
    prob *= (2.0 / nRef)  ## Normalize by the number of reference runs.  Multiply by 2 for 2-sided probability.
    #prob = min(1.0, prob) ## Again bin-by-bin.  Maximum possible probability should be 1.0
    prob[prob > 1] = 1
    
    pull = calcPull(prob)
    
    return pull

def calc_pull_B(D_raw, R_list_raw, tol):
    intD = D_raw.sum()     ## Total number of data entries

    nRef = len(R_list_raw)  ## Number of reference histograms
    sumIntR = (R_list_raw.sum(axis=0)).sum()  ## Total number of entries summed over reference histograms
    avgIntR = sumIntR / nRef  ## Average number of entries per reference histogram
    R_raw_avg = R_list_raw.sum(axis=0) / nRef  ## Reference hist with same stats as average of all hists

    scaleTol = pow(1 + pow(R_raw_avg * tol**2, 2), -0.5)  ## Array of tolerance scaling factors per bin
    intR_tol = R_raw_avg.sum() * scaleTol             ## Scale total number of entries in averaged ref hist
    R_raw_avg_tol = np.multiply(R_raw_avg, scaleTol)  ## Scale number of entries in each bin of ref hist

    probHi = np.zeros_like(D_raw)
    probLo = np.zeros_like(D_raw)

    probHi += sci.betabinom.sf(D_raw-1, intD, R_raw_avg_tol + 1, intR_tol - R_raw_avg_tol + 1)
    probLo += sci.betabinom.cdf(D_raw,  intD, R_raw_avg_tol + 1, intR_tol - R_raw_avg_tol + 1)

    ##  prob = probHi if (D_raw / intD >= R_raw_avg / avgIntR) else probLo  ## However you do substitution bin-by-bin
    ## bin by bin substitution by declaring the prob = probLo first, then fill with probHi where meets condition 
    prob = probLo
    condition = D_raw / intD >= R_raw_avg / avgIntR
    prob[condition] = probHi[condition]
    prob *= 2.0  ## Multiply by 2 for 2-sided probability.
    #prob = min(1.0, prob) ## Again bin-by-bin.  Maximum possible probability should be 1.0
    prob[prob > 1] = 1


    
    
    pull = calcPull(prob)

    
    return pull

def calcPull(prob):
    prob = np.clip(prob, a_min = np.power(0.1,16), a_max=None)## recalc prob so that it's not too small
    pull = np.sqrt(scipy.stats.chi2.ppf(1-prob,1)) ## convert to pull value
    # pull = np.clip(pull, a_min=-25, a_max=25) ## cap the pull value so it doesn't blow up
        
    return pull