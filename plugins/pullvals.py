
from plugin_results import PluginResults
import numpy as np
import uproot
import scipy.stats as stats
#from plugins import BetaBinomEst
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
    if not "TH2" in str(type(data_hist)): #or not "2" in str(type(ref_hist)):
        return None


    data_raw = np.round(np.float64(data_hist.values()))
    ref_list_raw = np.round(np.array([np.float64(x.values()) for x in ref_hists_list]))

    ## num entries
    data_hist_Entries = np.sum(data_raw)

    nRef = len(ref_hists_list)
    ref_hist_Entries_avg = ref_list_raw.sum(axis=0)/nRef if (nRef > 0) else np.zeros_like(data_raw) ## using np.mean raises too many warnings about mean of empty slice

    # Reject empty histograms
    is_good = (data_hist_Entries != 0)

    ## only fuilled bins used for calculating chi2
    nBinsUsed = np.count_nonzero(np.add(ref_list_raw.sum(axis=0), data_raw))

    if nBinsUsed > 0:
        pulls = pull(data_raw, ref_list_raw)
        chi2 = np.square(pulls).sum()/nBinsUsed if nBinsUsed > 0 else 0
        max_pull = maxPullNorm(np.amax(pulls), nBinsUsed)
    else:
        pulls = np.zeros_like(data_raw)
        chi2 = 0
        max_pull = 0

    nBins = data_hist.values().size

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
    c = None
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
        D_norm = 1.0

    else:
        for R_raw in R_list_raw:
            prob += ProbRel(D_raw, R_raw, 'BetaB')
        prob*=1/nRef
        pull = Sigmas(prob)
        ## get normalized data to get sign on pull
        D_norm = D_raw * R_list_raw.mean(axis=0).sum()/D_raw.sum()

    pull = pull*np.sign(D_norm-R_list_raw.mean(axis=0))

    print(R_list_raw)
    print('------------------')


    return pull

def maxPullNorm(maxPull, nBinsUsed, cutoff=pow(10,-15)):
    sign = np.sign(maxPull)
    ## sf (survival function) better than 1-cdf for large pulls (no precision error)
    probGood = stats.chi2.sf(np.power(min(abs(maxPull), 37), 2), 1)

    ## Use binomial approximation for low probs (accurate within 1%)
    if nBinsUsed * probGood < 0.01:
        probGoodNorm = nBinsUsed * probGood
    else:
        probGoodNorm = 1 - np.power(1 - probGood, nBinsUsed)

    pullNorm = Sigmas(probGoodNorm)

    return pullNorm


## Mean expectation for number of expected data events
def Mean(Data, Ref, func):
    nRef = Ref.sum()
    nData = Data.sum()
    if func == 'Gaus1' or func == 'Gaus2':
        return 1.0*nData*Ref/nRef
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#Moments_and_properties
    ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
    if func == 'BetaB' or func == 'Gamma':
        return 1.0*nData*(Ref+1)/(nRef+2)

    print('\nInside Mean, no valid func = %s. Quitting.\n' % func)
    sys.exit()
## Standard deviation of gaussian and beta-binomial functions
def StdDev(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    mask = Ref > 0.5*nRef
    if func == 'Gaus1':
        ## whole array is calculated using the (Ref <= 0.5*nRef) formula, then the ones where the
        ## conditions are actually failed is replaced using mask with the (Ref > 0.5*nRef) formula
        output = 1.0*nData*np.sqrt(np.clip(Ref, a_min=1, a_max=None))/nRef
        output[mask] = (1.0*nData*np.sqrt(np.clip(nRef-Ref, a_min=1, a_max=None)))[mask]/nRef
    elif func == 'Gaus2':
        ## instead of calculating max(Ref, 1), set the whole array to have a lower limit of 1
        clipped = np.clip(Ref, a_min=1, a_max=None)
        output = 1.0*nData*np.sqrt( clipped/np.square(nRef) + Mean(Data, Ref, nRef, func)/np.square(nData) )
        clipped = np.clip(nRef-Ref, a_min=1, a_max=None)
        output[mask] = (1.0*nData*np.sqrt( clipped/np.square(nRef) + (nData - Mean(Data, Ref, nRef, func))/np.square(nData) ))
    elif (func == 'BetaB') or (func == 'Gamma'):
        output = 1.0*np.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (np.power(nRef+2, 2)*(nRef+3)) )
        #output = 1.0*np.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (np.square(nRef+2)*(nRef+3)) )
    else:
        print('\nInside StdDev, no valid func = %s. Quitting.\n' % func)
        sys.exit()

    return output


## Number of standard devations from the mean in any function
def Pull(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()


    return (Data - Mean(Data, Ref, func)) / StdDev(Data, Ref, func)


## Exact and approximate values for natural log of the Gamma function
def LogGam(z):
    return special.gammaln(z)

## Predicted probability of observing Data / nData given a reference of Ref / nRef
def Prob(Data, nData, Ref, nRef, func, kurt=0):
    tol = 0.01
    scaleTol = np.power(1 + np.power(Ref * tol**2, 2), -0.5)
    nRef_tol = (scaleTol * nRef)
    Ref_tol = Ref * scaleTol

    if func == 'Gaus1' or func == 'Gaus2':
        return stats.norm.pdf( Pull(Data, Ref, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        #return sci.betabinom.pmf(Data, nData, Ref+1, nRef-Ref+1)
        return stats.betabinom.pmf(Data, nData, Ref_tol + 1, nRef_tol - Ref_tol + 1)
    ## Expression for beta-binomial using definition in terms of gamma functions
    ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
    if func == 'Gamma':
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        n_  = nData
        k_  = Data
        a_  = Ref+1
        b_  = nRef-Ref+1
        ab_ = nRef+2
        logProb  = LogGam(n_+1) + LogGam(k_+a_) + LogGam(n_-k_+b_) + LogGam(ab_)
        logProb -= ( LogGam(k_+1) + LogGam(n_-k_+1) + LogGam(n_+ab_) + LogGam(a_) + LogGam(b_) )
        return np.exp(logProb)

    print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
    sys.exit()


## Predicted probability relative to the maximum probability (i.e. at the mean)
def ProbRel(Data, Ref, func, kurt=0):
    nData = Data.sum()
    nRef = Ref.sum()
    ## Find the most likely expected data value
    #exp = np.round(Mean(nData, Ref, 'Gaus1'))
    exp_up = np.clip(np.ceil(Mean(Data, Ref, 'Gaus1')), a_min=None, a_max=nData) # make sure nothing goes above nData
    exp_down = np.clip(np.floor(Mean(Data, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef,func, kurt)
    maxProb_down = Prob(exp_down, nData, Ref, nRef,func, kurt)
    maxProb = np.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, kurt)
    ## Sanity check to not have relative likelihood > 1
    cond = maxProb < thisProb


    #if any(cond.flatten()):
    #    print(f'for ProbRel')
    #    print(f'Data: {Data[cond]}\nnData: {nData}\nRef: {Ref[cond]}\nnRef: {nRef}')

    ## make sure check for thisProb < maxProb*0.001 (account for floating point inaccuracies) and just set the ratio to 1 if that is the case
    ratio = thisProb/maxProb
    ratio[cond] = 1

    return ratio #thisProb / maxProb


## Negative log likelihood
def NLL(prob):
    nllprob = -1.0*np.log(prob, where=(prob>0))
    nllprob[prob==0] = 999
    nllprob[prob < 0] == -999

    return nllprob


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    probRel = np.maximum(probRel, 10E-300)
    return np.sqrt((stats.chi2.isf(probRel,1)))
