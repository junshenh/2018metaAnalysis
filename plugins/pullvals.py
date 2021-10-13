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
    
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None
    for i in ref_hists_list:
        if i.values.sum() == 0:
            return None

    data_hist_norm = np.copy(data_hist.values)
    #ref_hist_norm = np .copy(ref_hist.values())
    ref_hists_list_norm = [np.copy(x.values) for x in ref_hists_list]

    ## Clone data_hist array to create pull_hist array to be filled later
    pull_hist = np.copy(data_hist_norm)

    data_hist_Entries = np.sum(data_hist_norm)
    ref_hist_Entries = np.mean(np.sum(ref_hists_list))

    # Reject empty histograms
    is_good = data_hist_Entries != 0 # and data_hist.GetEntries() >= min_entries

    # import pickle
    # pickle.dump(data_hist_errs, open(f'pickles/dataErr-data_name.pkl','wb')) 
    
    
    ## calculate the ref arrays 
    ref_hist_arr = np.array(ref_hists_list_norm)
    ref_hist_errs = np.std(ref_hist_arr, axis=0)
    #================================================================
    errlist = np.array([np.sqrt(x.variances) for x in ref_hists_list])
    errpercentlist = np.divide(errlist, ref_hist_arr, out=np.zeros_like(errlist), where=ref_hist_arr!=0)
    errpercent = np.mean(errpercentlist, axis=0)
    #================================================================
    ref_hist_norm = np.mean(ref_hist_arr, axis=0)
    
    
    ref_hist_scale = 1#/ref_hist_norm.sum()
    ref_hist_norm*=ref_hist_scale
    ref_hist_errs*=ref_hist_scale
    #================================================================
    ref_hist_errs = ref_hist_norm*errpercent
    #================================================================
    
    ## data arrays
    data_hist_scale = ref_hist_norm.sum()/data_hist_norm.sum()
    data_hist_norm*=data_hist_scale
    # data_hist_errs = np.sqrt(data_hist_norm*data_hist_scale)
    data_hist_errs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * data_hist_norm)) / 2 - 1 - data_hist_norm))


    max_pull = 0
    nBins = 0
    chi2 = 0
    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_hist_norm, data_hist_norm)) 
    
    # pulls = list()
    pulls = np.zeros_like(ref_hist_norm)

    
    for x in range(0, data_hist_norm.shape[0]):
        for y in range(0, data_hist_norm.shape[1]):

            # Bin 1 data
            bin1 = data_hist_norm[x,y] #.GetBinContent(x, y)

            # Bin 2 data
            bin2 = ref_hist_norm[x,y]#ref_hist.GetBinContent(x, y)

            
            ## Getting Proper Poisson error 
            '''bin1err, bin2err = data_hist_errs[x,y], ref_hist_errs[x,y]'''

            bin1err, bin2err = data_hist_errs[0, x, y], ref_hist_errs[x, y]
            if bin1 < bin2:
                bin1err, bin2err = data_hist_errs[1, x, y], ref_hist_errs[x, y]


            # Ensure that divide-by-zero error is not thrown when calculating pull
            if bin1err == 0 and bin2err == 0:
                new_pull = 0
            else:
                new_pull = pull(bin1, bin1err, bin2, bin2err)

                    
            #pulls.append(new_pull)
            # Sum pulls
            chi2 += new_pull**2

            # Check if max_pull
            #max_pull = max(max_pull, abs(new_pull))

            # Clamp the displayed value
            #fill_val = new_pull#max_pull

            # If the input bins were explicitly empty, make this bin white by
            # setting it out of range
            fill_val = new_pull
            # if bin1 == bin2 == 0:
            #     fill_val = -999

            # Fill Pull Histogram
            pulls[x,y] = fill_val#new_pull
    

    ## make normed chi2 and maxpull
    if nBinsUsed > 0:
        chi2 = chi2/nBinsUsed  
        #max_pull = maxPullNorm(max_pull, nBinsUsed)
        pulls = maxPullNorm(pulls, nBinsUsed)
        max_pull = pulls.max()
    else:
        chi2 = 0
        max_pull = 0
        
    
    
    
    

    #-------------------------------------------------------------------------
    print(data_hist.name)
    for i in ref_hists_list: 
        ref_ratios = np.divide(np.sqrt(i.variances),i.values, out = np.zeros_like(i.values), where=i.values!=0)
        
        print(ref_ratios.sum()/np.count_nonzero(ref_ratios))
        print(ref_ratios.mean())
        print('---')
        
    print('==========================================================')
    
    if histpair.data_name in ['cscDQMOccupancy', 'emtfTrackOccupancy', 'cscLCTOccupancy']:
        import matplotlib.pyplot as plt
        def getBinCenter(arr):
            arrCen = list()
            for i in range(len(arr)-1):
                arrCen.append((arr[i+1]+arr[i])/2)
            return arrCen
    
        histedges = data_hist.edges
        xedges = getBinCenter(histedges[0])
        yedges = getBinCenter(histedges[1])
        
        '''fig, ax = plt.subplots()
        im = ax.pcolormesh(xedges, yedges, data_hist_norm.T, cmap='viridis', shading='auto')
        ax.set_title(f'data_hist_norm ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(xedges, yedges, ref_hist_norm.T, cmap='viridis', shading='auto')
        ax.set_title(f'ref_hist_norm ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(xedges, yedges, ref_hist_errs.T, cmap='viridis', shading='auto')
        ax.set_title(f'ref_hist_errs ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)
        '''
        fig, ax = plt.subplots()
        ratio = np.divide(ref_hist_errs, ref_hist_norm, out=np.zeros_like(ref_hist_errs), where=ref_hist_norm!=0).T
        im = ax.pcolormesh(xedges, yedges, ratio, cmap='viridis', shading='auto')
        ax.set_title(f'ref ratios ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)
    
    
        '''fig, ax = plt.subplots()
        dataErrTmp = data_hist_errs
        top = dataErrTmp[0]
        bot = dataErrTmp[1]
        data_hist_errs = top
        data_hist_errs[ref_hist_norm < data_hist_norm] = bot[ref_hist_norm < data_hist_norm]
        im = ax.pcolormesh(xedges, yedges, data_hist_errs.T, cmap='viridis', shading='auto')
        ax.set_title(f'data_hist_errs ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)'''
    
        '''fig, ax = plt.subplots()
        im = ax.pcolormesh(xedges, yedges, pulls.T, cmap='viridis', shading='auto')
        ax.set_title(f'pulls ({histpair.data_name})')
        fig.colorbar(im)
        plt.show()
        plt.close(fig)'''
        

        
    #--------------------------------------------------------------------------
    
    
    
    
    
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    # Set up canvas
    c = ROOT.TCanvas('c', 'Pull')

    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries,
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : (pulls, data_hist.edges) 
        
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
    

def normalize_rows(data_hist_norm, ref_hist_norm):

    for y in range(0, ref_hist_norm.shape[1]):

        # Stores sum of row elements
        rrow = 0
        frow = 0

        # Sum over row elements
        for x in range(0, ref_hist_norm.shape[0]):

            # Bin data
            rbin = ref_hist_norm[x,y]
            fbin = data_hist_norm[x, y]

            rrow += rbin
            frow += fbin

        # Scaling factors
        # Prevent divide-by-zero error
        if frow == 0:
            frow = 1
        if frow > 0:
            sf = float(rrow) / frow
        else:
            sf = 1
        # Prevent scaling everything to zero
        if sf == 0:
            sf = 1

        # Normalization
        for x in range(0, ref_hist_norm.shape[0]):
            # Bin data
            fbin = data_hist_norm[x, y]
            fbin_err = (fbin)**(.5)

            # Normalize bin

            data_hist_norm[x, y] = (fbin * sf)
    return data_hist_norm
