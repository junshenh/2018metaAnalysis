#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
import numpy as np
import time
import root_numpy
import uproot
import scipy
import scipy.stats

def comparators():
    return {
        'pull_values': pullvals
    }


def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=100000, norm_type='all',
             **kwargs):
    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    # ref_hist = histpair.ref_hist
    ref_hists_list = histpair.ref_hists_list
    
    # data_histD = ROOT.TH2D()
    # data_hist.Copy(data_histD)
    # data_hist = data_histD
    # ref_histD = ROOT.TH2D()
    # ref_hist.Copy(ref_histD)
    # ref_hist = ref_histD

    # Check that the hists are histograms
    # if not data_hist.InheritsFrom('TH1') or not ref_hist.InheritsFrom('TH1'):
    #     return None

    # Check that the hists are 2 dimensional
    # if data_hist.GetDimension() != 2 or ref_hist.GetDimension() != 2:
    #     return None
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None

    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
    # ROOT.gStyle.SetNumberContours(255)

    # Get empty clone of reference histogram for pull hist
    # if data_hist.InheritsFrom('TProfile2D'):
    #     pull_hist = ref_hist.ProjectionXY("pull_hist")
    # else:
    #     pull_hist = ref_hist.Clone("pull_hist")
    # pull_hist.Reset()
    data_hist_norm = np.copy(data_hist.values)
    #ref_hist_norm = np .copy(ref_hist.values())
    ref_hists_list_norm = [np.copy(x.values) for x in ref_hists_list]

    ## Clone data_hist array to create pull_hist array to be filled later
    pull_hist = np.copy(data_hist_norm)

    data_hist_Entries = np.sum(data_hist_norm)
    ref_hist_Entries = np.mean(np.sum(ref_hists_list))
    #ref_hist_Entries = np.sum(ref_hist_norm)

    # Reject empty histograms
    is_good = data_hist_Entries != 0 # and data_hist.GetEntries() >= min_entries

    # Normalize data_hist
    # if norm_type == "row":
    #     normalize_rows(data_hist, ref_hist)
    # else:
    #     if data_hist.GetEntries() > 0:
    #         data_hist.Scale(ref_hist.GetSumOfWeights() / data_hist.GetSumOfWeights())
 
    ## normalize all data to integrate to 1
    histscale = 1#ref_hist.GetSumOfWeights()#1
    if data_hist_Entries > 0: 
        data_hist_norm*=histscale/data_hist_norm.sum()
    # if ref_hist.GetEntries() > 0: 
    #     ref_hist.Scale(histscale/ref_hist.GetSumOfWeights())
    for i in ref_hists_list_norm:
        if i.sum() > 0:
            i*=histscale/i.sum()
            
    #Calculate asymmetric error bars 
    data_hist_errs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * data_hist_norm)) / 2 - 1 - data_hist_norm))
            
    # calculate the average of all ref_hists_list 
    ref_hist_arr = np.array(ref_hists_list_norm)
    ref_hist_norm = np.mean(ref_hist_arr, axis=0)
    if ref_hist_norm.sum() > 0: ref_hist_norm*=histscale/ref_hist_norm.sum()
    ref_hist_errs = np.std(ref_hist_arr, axis=0)    


    max_pull = 0
    nBins = 0
    chi2 = 0
    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_hist_norm, data_hist_norm)) 
    
    ## caluclate nBinsUsed 
    # data_arr = root_numpy.hist2array(data_hist)
    #ref_arr = root_numpy.hist2array(ref_hist)
    
    
    
    # pulls = list()
    pulls = np.zeros_like(ref_hist_norm)
    
    ## loop through bins to calculate max pull
    # for x in range(1, data_hist.GetNbinsX() + 1):
    #     for y in range(1, data_hist.GetNbinsY() + 1):
    for x in range(0, data_hist_norm.shape[0]):
        for y in range(0, data_hist_norm.shape[1]):

            # Bin 1 data
            bin1 = data_hist_norm[x,y] #.GetBinContent(x, y)

            # Bin 2 data
            bin2 = ref_hist_norm[x,y]#ref_hist.GetBinContent(x, y)

            # if not (bin1 + bin2 > 0):
            #     pulls.append(0)
            #     continue
            
            # TEMPERARY - Getting Symmetric Error - Need to update with >Proper Poisson error 
            # if data_hist.InheritsFrom('TProfile2D'):
            #     bin1err = data_hist.GetBinError(x, y)
            #     # bin2err = ref_hist.GetBinError(x, y)
            #     bin2err = ref_hist_errs[x-1,y-1] # -1 because root index from 1 apparently
            # else:
            #     # bin1err, bin2err = bin1**(.5), bin2**(.5)
            #     bin1err, bin2err = bin1**(.5), ref_hist_errs[x-1, y-1]
            # # Count bins for chi2 calculation
            # nBins += 1 
            
            ## Getting Proper Poisson error 
            bin1err, bin2err = data_hist_norm[x,y]**0.5, ref_hist_errs[x,y]
            '''bin1err, bin2err = data_hist_errs[0, x, y], ref_hist_errs[x, y]
            if bin1 < bin2:
                bin1err, bin2err = data_hist_errs[1, x, y], ref_hist_errs[x, y]'''

            
            # Ensure that divide-by-zero error is not thrown when calculating pull
            if bin1err == 0 and bin2err == 0:
                new_pull = 0
            else:
                new_pull = pull(bin1, bin1err, bin2, bin2err)
                    
            #pulls.append(new_pull)
            # Sum pulls
            chi2 += new_pull**2

            # Check if max_pull
            max_pull = max(max_pull, abs(new_pull))

            # Clamp the displayed value
            # fill_val = max(min(new_pull, pull_cap), -pull_cap)
            fill_val = max_pull

            # If the input bins were explicitly empty, make this bin white by
            # setting it out of range
            if bin1 == bin2 == 0:
                fill_val = -999

            # Fill Pull Histogram
            # pull_hist.SetBinContent(x, y, fill_val)
            pulls[x,y] = fill_val

    ## 'normalize' all the pulls before passing it back for plotting
    pulls = maxPullNorm(pulls, nBinsUsed)

    ## make normed chi2 and maxpull
    if nBinsUsed > 0:
        chi2 = chi2/nBinsUsed  
        max_pull = maxPullNorm(max_pull, nBinsUsed)
    else:
        chi2 = 0
        max_pull = 0
        
    
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    # Set up canvas
    c = ROOT.TCanvas('c', 'Pull')

    # Plot pull hist
    # pull_hist.GetZaxis().SetRangeUser(-(pull_cap), pull_cap)
    # pull_hist.SetTitle(pull_hist.GetTitle() + " Pull Values")
    # pull_hist.Draw("colz")

    # Text box
    # data_text = ROOT.TLatex(.52, .91,
    #                         "#scale[0.6]{Data: " + str(histpair.data_run) + "}")
    # ref_text = ROOT.TLatex(.72, .91,
    #                        "#scale[0.6]{Ref: " + str(histpair.ref_run) + "}")
    # data_text.SetNDC(ROOT.kTRUE)
    # ref_text.SetNDC(ROOT.kTRUE)
    # data_text.Draw()
    # ref_text.Draw()


    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist_norm, #data_hist_Entries,
        'Ref_Entries': ref_hist_norm, #ref_hist_Entries,
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : pulls # .reshape(refErr)
        
    }

    artifacts = [pulls, 'data_text', 'ref_text']

    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


def pull(bin1, bin1err, bin2, bin2err):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0
    '''
    ## changing to pull with tolerance
    # return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)
    return np.abs(bin1 - bin2)/(np.sqrt(np.power(bin1err,2)+np.power(bin2err,2))+0.01*(bin1+bin2))

def maxPullNorm(maxPull, nBinsUsed):
    probGood = 1-scipy.stats.chi2.cdf(np.power(maxPull,2),1) # will give the same result as ROOT.TMath.Prob
    ## probGood = ROOT.TMath.Prob(np.power(maxPull, 2), 1)
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = np.minimum(probBadNorm, 1 - np.power(0.1,16))
    return np.sqrt(scipy.stats.chi2.ppf(val,1))
    ##return np.sqrt(ROOT.TMath.ChisquareQuantile(val,1))
    

# def normalize_rows(data_hist, ref_hist):

#     for y in range(1, ref_hist.GetNbinsY() + 1):

#         # Stores sum of row elements
#         rrow = 0
#         frow = 0

#         # Sum over row elements
#         for x in range(1, ref_hist.GetNbinsX() + 1):

#             # Bin data
#             rbin = ref_hist.GetBinContent(x, y)
#             fbin = data_hist.GetBinContent(x, y)

#             rrow += rbin
#             frow += fbin

#         # Scaling factors
#         # Prevent divide-by-zero error
#         if frow == 0:
#             frow = 1
#         if frow > 0:
#             sf = float(rrow) / frow
#         else:
#             sf = 1
#         # Prevent scaling everything to zero
#         if sf == 0:
#             sf = 1

#         # Normalization
#         for x in range(1, data_hist.GetNbinsX() + 1):
#             # Bin data
#             fbin = data_hist.GetBinContent(x, y)
#             fbin_err = data_hist.GetBinError(x, y)

#             # Normalize bin
#             data_hist.SetBinContent(x, y, (fbin * sf))
#             data_hist.SetBinError(x, y, (fbin_err * sf))

#     return
