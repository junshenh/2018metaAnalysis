import ROOT
from plugin_results import PluginResults
import numpy as np
import uproot
import scipy
import scipy.stats
from binomProbs import calc_pull
import time

def comparators():
    return {
        'pull_values': pullvals
    }

def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=100000, norm_type='all',
             **kwargs):
    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    ## if ref hist is empty, don't include it
    ref_hists_list = [x for x in histpair.ref_hists_list if x.values().sum() > 0]


    ## check to make sure histogram is TH2    
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None


    data_raw = np.float64(data_hist.values())
    ref_list_raw = np.array([np.float64(x.values()) for x in ref_hists_list])
        
    ## num entries
    data_hist_Entries = np.sum(data_raw)
    ref_hist_Entries_avg = ref_list_raw.mean(axis=0).sum()
    
    # Reject empty histograms
    is_good = data_hist_Entries != 0

    pulls = pull(data_raw, ref_list_raw)

    #print(f'{histpair.data_name}, {end-start}')

    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.mean(axis=0), data_raw)) 
    chi2 = np.square(pulls).sum()/nBinsUsed if nBinsUsed > 0 else 0
    max_pull = maxPullNorm(pulls, nBinsUsed).max()
    nBins = data_hist.values().size
    #pulls = maxPullNorm(pulls, nBinsUsed)    
    
    histedges = (data_hist.to_numpy()[1], data_hist.to_numpy()[2])

    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries_avg,
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : (pulls, histedges) 
        
    }
    
    artifacts = [pulls, 'data_text', 'ref_text']
    c = ROOT.TCanvas('c', 'Pull') 
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)
    
    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)
    
    
    
def pull(D_raw, R_list_raw):
    ## plan, do all D_norm, R_norm_avg etc. calculations here so that 1d hists also have the same def without need of copy paste
    nRef = len(R_list_raw)
    tol = 0.01
    if D_raw.sum() > 0 and nRef > 0: 
        pullA = calc_pull(D_raw, R_list_raw, tol, 'A')
        ## pullA is used 89% of the time when testing 15 runs. Only pull A will be used to speed upcalc
        # pullB = calc_pull(D_raw, R_list_raw, tol, 'B')

        #value = b if a > 10 else c
        pull = pullA #if (pullA*pullA).sum() < (pullB*pullB).sum() else pullB
        
    else: 
        pull = np.zeros_like( D_raw )
    
    return pull
        
    
def maxPullNorm(maxPull, nBinsUsed):
    sign = np.sign(maxPull)
    probGood = 1-scipy.stats.chi2.cdf(np.power(maxPull,2),1) # will give the same result as ROOT.TMath.Prob
    ## probGood = ROOT.TMath.Prob(np.power(maxPull, 2), 1)
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = np.minimum(probBadNorm, 1 - np.power(0.1,16))
    return np.sqrt(scipy.stats.chi2.ppf(val,1))*sign
    
    
