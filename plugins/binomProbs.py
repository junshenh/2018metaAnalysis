#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:20:17 2021

@author: chosila
"""

import numpy as np
import scipy
import scipy.stats as sci
import BetaBinomEst

def calc_pull(D_raw, R_list_raw, tol, optAB):
    # intD = D_raw.sum()  ## Total number of data entries

    nRef = len(R_list_raw)  ## Number of reference histograms
    prob = np.zeros_like(D_raw)
    for R_raw in R_list_raw:
        prob += BetaBinomEst.ProbRel(D_raw, R_raw, 'BetaB')
    prob*=1/nRef
    pull = BetaBinomEst.Sigmas(prob)

    return pull
