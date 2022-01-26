#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
import numpy as np
from pullvals import pull, maxPullNorm
import scipy
import scipy.stats
import time

def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=100000, **kwargs):

    data_hist = histpair.data_hist
    ref_hists_list = [x for x in histpair.ref_hists_list if x.values().sum() > 0]

    # check for 1d hists and that refs are not empty
    if "1" not in str(type(data_hist)) :
        return None

    data_raw = np.float64(data_hist.values())
    ref_list_raw = np.array([np.float64(x.values()) for x in ref_hists_list])
        
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

    #print(f'{histpair.data_name=}')
    pulls = pull(data_raw, ref_list_raw)
    
    

    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.mean(axis=0), data_raw)) 
    chi2 = np.square(pulls).sum()/nBinsUsed if nBinsUsed > 0 else 0
    max_pull = maxPullNorm(pulls, nBinsUsed).max()
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

#%%
  

def draw_same(data_hist, data_run, ref_hist, ref_run):
    # Set up canvas
    c = ROOT.TCanvas('c', 'c')
    data_hist = data_hist.Clone()
    ref_hist = ref_hist.Clone()

    # Ensure plot accounts for maximum value
    ref_hist.SetMaximum(
        max(data_hist.GetMaximum(), ref_hist.GetMaximum()) * 1.1)

    ROOT.gStyle.SetOptStat(1)
    ref_hist.SetStats(True)
    data_hist.SetStats(True)

    # Set hist style
    ref_hist.SetLineColor(28)
    ref_hist.SetFillColor(20)
    ref_hist.SetLineWidth(1)
    data_hist.SetLineColor(ROOT.kRed)
    data_hist.SetLineWidth(1)

    # Name histograms
    ref_hist.SetName("Reference")
    data_hist.SetName("Data")

    # Plot hist
    ref_hist.Draw()
    data_hist.Draw("sames hist e")
    c.Update()

    # Modify stats boxes
    r_stats = ref_hist.FindObject("stats")
    f_stats = data_hist.FindObject("stats")

    r_stats.SetY1NDC(0.15)
    r_stats.SetY2NDC(0.30)
    r_stats.SetTextColor(28)
    r_stats.Draw()

    f_stats.SetY1NDC(0.35)
    f_stats.SetY2NDC(0.50)
    f_stats.SetTextColor(ROOT.kRed)
    f_stats.Draw()

    # Text box
    data_text = ROOT.TLatex(.52, .91, "#scale[0.6]{Data: " + str(data_run) + "}")
    ref_text = ROOT.TLatex(.72, .91, "#scale[0.6]{Ref: " + str(ref_run) + "}")
    data_text.SetNDC(ROOT.kTRUE)
    ref_text.SetNDC(ROOT.kTRUE)
    data_text.Draw()
    ref_text.Draw()

    c.Update()
    artifacts = [data_hist, data_text] 
    #artifacts = [data_hist, ref_hist, data_text, ref_text]
    return c, artifacts
