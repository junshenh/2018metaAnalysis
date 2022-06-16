import ROOT
from plugin_results import PluginResults
import numpy as np
import uproot
import scipy
import scipy.stats
import BetaBinomEst
#from binomProbs import calc_pull
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
    ref_hists_list = [x for x in histpair.ref_hists_list if np.round(x.values()).sum() > 0]


    ## check to make sure histogram is TH2    
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None


    data_raw = np.round(np.float64(data_hist.values()))
    ref_list_raw = np.round(np.array([np.float64(x.values()) for x in ref_hists_list]))

    ## num entries
    data_hist_Entries = np.sum(data_raw)

    nRef = len(ref_hists_list)
    ref_hist_Entries_avg = ref_list_raw.sum(axis=0)/nRef if (nRef > 0) else np.zeros_like(data_raw) ## using np.mean raises too many warnings about mean of empty slice

    # Reject empty histograms
    is_good = data_hist_Entries != 0

    pulls = pull(data_raw, ref_list_raw)

    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.sum(axis=0), data_raw)) 
    chi2 = np.square(pulls).sum()/nBinsUsed if nBinsUsed > 0 else 0
    max_pull = pulls.max() #maxPullNorm(pulls, nBinsUsed).max()
    nBins = data_hist.values().size
    pulls = maxPullNorm(pulls, nBinsUsed)    
    
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
    nRef = len(R_list_raw)
    tol = 0.01
    prob = np.zeros_like(D_raw)
    if (D_raw.sum() <= 0) or (np.array(R_list_raw.flatten()).sum() <= 0):
        pull = np.zeros_like(D_raw)
    else:
        #pull = calc_pull(D_raw, R_list_raw, tol, 'A')
        for R_raw in R_list_raw:
            prob += BetaBinomEst.ProbRel(D_raw, R_raw, 'BetaB')
        prob*=1/nRef
        pull = BetaBinomEst.Sigmas(prob)
        
    return pull

# def calc_pull(D_raw, R_list_raw, tol, optAB):
#     nRef = len(R_list_raw)  ## Number of reference histograms                                                                   
#     prob = np.zeros_like(D_raw)
#     for R_raw in R_list_raw:
#         prob += BetaBinomEst.ProbRel(D_raw, R_raw, 'BetaB')
#     prob*=1/nRef
#     pull = BetaBinomEst.Sigmas(prob)
#     return pull
        
    
def maxPullNorm(maxPull, nBinsUsed):
    sign = np.sign(maxPull)
    probGood = 1-scipy.stats.chi2.cdf(np.power(maxPull,2),1) # will give the same result as ROOT.TMath.Prob
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = np.minimum(probBadNorm, 1 - np.power(0.1,16))
    return np.sqrt(scipy.stats.chi2.ppf(val,1))*sign
    
    
