#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
from pullvals import pull
import numpy as np
import scipy.stats

def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=100000, **kwargs):

    data_name = histpair.data_name
    ref_name = histpair.ref_name

    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist


    # Check that the hists are histograms
    if "TH1" not in str(type(data_hist)) or "TH1" not in str(type(ref_hist)):
        return None
    

    # Normalize data_hist by copying histogram and then normalizing (Note declaration of data_hist_Entries & ref_hist_Entries)
    data_hist_norm = np.copy(data_hist.values())
    ref_hist_norm = np.copy(ref_hist.values())

    pull_hist = np.copy(data_hist_norm)

    data_hist_Entries = np.sum(data_hist_norm)
    ref_hist_Entries = np.sum(ref_hist_norm)
    if data_hist_Entries > 0:
        data_hist_norm = data_hist_norm * (ref_hist_Entries / data_hist_Entries)
    

    # Reject empty histograms
    is_good = data_hist_Entries != 0 and data_hist_Entries >= min_entries

    ks = scipy.stats.kstest(ref_hist_norm, data_hist_norm)[0]

    is_outlier = is_good and ks > ks_cut
        

    ref_text = "ref:"+str(histpair.ref_run)
    data_text = "data:"+str(histpair.data_run)
    artifacts = [data_hist_norm, ref_hist_norm, data_text, ref_text]
    

    pull_cap = 25
    ## chi2 and pull vals
    max_pull = 0
    nBins = 0
    nBinsUsed = 0
    chi2 = 0

    # data_hist_norm = np.copy(data_hist.values())
    # ref_hist_norm = np.copy(ref_hist.values())
    data_hist_errs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * data_hist_norm)) / 2 - 1 - data_hist_norm))
    ref_hist_errs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * ref_hist_norm)) / 2 - 1 - ref_hist_norm))
    
    print(data_hist_errs/data_hist_norm)
    print(ref_hist_errs/ref_hist_norm)
    
    for x in range(0, data_hist_norm.shape[0]):
            # Bin 1 data
            bin1 = data_hist_norm[x]

            # Bin 2 data
            bin2 = ref_hist_norm[x]

            # Getting Proper Poisson error
            bin1err, bin2err = data_hist_errs[0, x], ref_hist_errs[1, x]
            if bin1 < bin2:
                bin1err, bin2err = data_hist_errs[1, x], ref_hist_errs[0, x]
            # Count bins for chi2 calculation
            nBins += 1

            # Ensure that divide-by-zero error is not thrown when calculating pull
            if bin1err == 0 and bin2err == 0:
                new_pull = 0
            else:
                new_pull = pull(bin1, bin1err, bin2, bin2err)

            # Sum pulls
            chi2 += new_pull**2

            # Check if max_pull
            max_pull = max(max_pull, abs(new_pull))

            # Clamp the displayed value
            fill_val = max(min(new_pull, pull_cap), -pull_cap)

            # If the input bins were explicitly empty, make this bin white by
            # setting it out of range
            if bin1 == bin2 == 0:
                fill_val = -999

            # Fill Pull Histogram            
            pull_hist[x] = fill_val


    # Compute chi2
    chi2 = (chi2 / nBins)




    info = {
        'Data_Entries': data_hist.values().sum(),
        'Ref_Entries': ref_hist.values().sum(),
        'KS_Val': ks,
        'Chi_Squared' : chi2,
        'Max_Pull_Val': max_pull
    }

    return PluginResults(
        None,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


def pull(bin1, binerr1, bin2, binerr2):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0
    '''
    ## changing to pull with tolerance
    # return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)
    return np.abs(bin1 - bin2)/(np.sqrt(np.power(binerr1,2)+np.power(binerr2,2)+0.01*(bin1+bin2)))

def maxPullNorm(maxPull, nBinsUsed):
    prob = ROOT.TMath.Prob(np.power(maxPull, 2),1)
    probNorm = 1-np.power((1-prob),nBinsUsed)
    ## .9999999999999999 is the max that can go into chi2quantile
    val = (1-probNorm) 
    val = val if val < .9999999999999999 else .9999999999999999
    return np.sqrt(ROOT.TMath.ChisquareQuantile(val,1))


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
