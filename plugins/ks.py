#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
import numpy as np
#from pullvals import pull, maxPullNorm
from plugins import pullvals
import scipy
import scipy.stats
import time

def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=100000, **kwargs):

    data_hist = histpair.data_hist
    ref_hists_list = [x for x in histpair.ref_hists_list if np.round(x.values()).sum() > 0]

    # check for 1d hists and that refs are not empty
    if "TH1" not in str(type(data_hist)) :
        return None

    data_raw = np.round(np.float64(data_hist.values()))
    ref_list_raw = np.round(np.array([np.float64(x.values()) for x in ref_hists_list]))

    ## num entries
    data_hist_Entries = np.sum(data_raw)
    ref_hist_Entries = ref_list_raw.mean(axis=0).sum()

    is_good = data_hist_Entries > 0

    ## looks like bigger values result in ks test working a little better
    ref_list_norm = np.array(ref_list_raw)#np.array([x*1/x.sum() for x in ref_list_raw])
    ref_norm_avg = ref_list_norm.mean(axis=0)

    if is_good:
        data_norm = data_raw * ref_norm_avg.sum()/data_raw.sum()
    else:
        data_norm = data_raw


    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.mean(axis=0), data_raw))


    if nBinsUsed > 0:
        pulls = pullvals.pull(data_raw, ref_list_raw)
        chi2 = np.square(pulls).sum()/nBinsUsed
        max_pull = pullvals.maxPullNorm(np.amax(pulls), nBinsUsed).max()
    else:
        pulls = np.zeros_like(data_raw)
        chi2 = 0
        max_pull = 0
    nBins = data_hist.values().size

    kslist = []


    for ref_norm in ref_list_norm:
        kslist.append(scipy.stats.kstest(ref_norm, data_norm)[0])
    ks = np.mean(kslist)

    is_outlier = is_good and ks > ks_cut

    canv = None
    artifacts = [pulls]

    histedges = data_hist.to_numpy()[1]

    info = {
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries,
        'KS_Val': ks,
        'Chi_Squared' : chi2,
        'Max_Pull_Val': max_pull,
        'nBins' : nBins,
        'pulls' : (pulls, histedges)
    }

    return PluginResults(
        canv,
        show=is_outlier,
        info=info,
        artifacts=artifacts)
