import ROOT
from plugin_results import PluginResults
import numpy as np
import uproot
import scipy
import scipy.stats
from binomProbs import calc_pull

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
    ref_hists_list = [x for x in histpair.ref_hists_list if x.values.sum() > 0]


    ## check to make sure histogram is TH2    
    if not "2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None


    data_raw = np.float64(data_hist.values)
    ref_list_raw = np.array([np.float64(x.values) for x in ref_hists_list])
        
    ## num entries
    data_hist_Entries = np.sum(data_raw)
    ref_hist_Entries_avg = ref_list_raw.mean(axis=0).sum()
    
    # Reject empty histograms
    is_good = data_hist_Entries != 0


    pulls = pull(data_raw, ref_list_raw)
    
    
    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.mean(axis=0), data_raw)) 
    chi2 = np.square(pulls).sum()/nBinsUsed if nBinsUsed > 0 else 0
    max_pull = maxPullNorm(pulls, nBinsUsed).max()
    nBins = data_hist.numbins
    #pulls = maxPullNorm(pulls, nBinsUsed)    
    
    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries_avg,
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : (pulls, data_hist.edges) 
        
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
        # R_norm_list = np.array([x*1/x.sum() for x in R_list_raw])
        #R_norm_avg = R_norm_list.mean(axis=0)
        # D_norm = D_raw*1/D_raw.sum()
        # intD = D_raw.sum()
        # varR_norm = R_norm_list.var(axis=0)
        # sumR = R_list_raw.sum(axis=0)
        # sumIntR = sumR.sum()
        
        ## create 2 ref norm definition to avoid very high pulls due to ref being mostly zero
        # R_norm_avg_A = R_norm_list.mean(axis=0)
        # R_norm_avg_B = R_list_raw.sum(axis=0) / sumIntR
        
        ## since i have to calculate the num and denom for pull twice, i will make it a function 
        ## to avoid typos
        pullA = calc_pull(D_raw, R_list_raw, tol, 'A') #calc_pull(sumIntR, intD, D_norm, R_norm_avg_A,  varR_norm, sumR, D_raw)
        pullB = calc_pull(D_raw, R_list_raw, tol, 'B') #calc_pull(sumIntR, intD, D_norm, R_norm_avg_B,  varR_norm, sumR, D_raw)
        
        #value = b if a > 10 else c
        pull = pullA if (pullA*pullA).sum() < (pullB*pullB).sum() else pullB
        
        
        
        #------------------------------------------------------------------------
        '''amax = np.argmax(pull) #np.unravel_index(np.argmax(pull), pull.shape) #
        print('amax: ', amax)
        print('pull: ', pull[amax])
        print('D_raw: ', D_raw[amax])
        print('R_raw: ',  R_list_raw[:,amax]) #R_list_raw[:, amax[0], amax[1]]) #
        print('intD: ', D_raw.sum())
        print('sumIntR: ', R_list_raw.sum())
        print('avgIntR: ',  R_list_raw.sum() / R_list_raw.shape[0] )
                
        
        
        
        print('pull: ', pull)
        print('D_raw: ', D_raw)
        print('R_raw: ', R_list_raw)
        print('intD: ', D_raw.sum())
        print('sumIntR: ', R_list_raw.sum())
        print('avgIntR: ',  R_list_raw.sum() / R_list_raw.shape[0] )
        print(R_list_raw.shape)
        print(R_list_raw.sum(axis=-1))'''
        #-----------------------------------------------------------------------
        
        
        
    else: 
        pull = np.zeros_like( D_raw )
    
    return pull
    
'''
def calc_pull(sumIntR, intD, D_norm, R_norm_avg,  varR_norm, sumR, D_raw):
    ## plan - declare array of case sumR_j == 0, then use it as an "out" array for np.divide
    ## to fill in places where the case sumR_j != 0 
    out = D_raw * sumIntR / np.sqrt( intD * (intD + sumIntR) ) 
        
    numerator = D_norm - R_norm_avg
    denom1 = np.divide(R_norm_avg , intD)
    denom2 = varR_norm
    denom3 = np.divide(R_norm_avg*R_norm_avg, sumR, out = np.zeros_like(D_raw),where=sumR>0)
    denom4 = 0.01*0.01*np.square(D_norm + R_norm_avg)

    denominator = np.sqrt( denom1 + denom2 + denom3 + denom4 )
    pull = np.divide( numerator, denominator, out=out, where=R_norm_avg!=0)
    return pull
 '''   
    
    
def maxPullNorm(maxPull, nBinsUsed):
    probGood = 1-scipy.stats.chi2.cdf(np.power(maxPull,2),1) # will give the same result as ROOT.TMath.Prob
    ## probGood = ROOT.TMath.Prob(np.power(maxPull, 2), 1)
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = np.minimum(probBadNorm, 1 - np.power(0.1,16))
    return np.sqrt(scipy.stats.chi2.ppf(val,1))
    
    
