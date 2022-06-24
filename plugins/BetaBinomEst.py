import sys
import numpy as np
import scipy.stats as sci
import scipy.special as special
from matplotlib import pyplot as plt


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
        output = 1.0*nData*np.sqrt( clipped/np.square(nRef) + Mean(nData, Ref, nRef, func)/np.square(nData) )
        clipped = np.clip(nRef-Ref, a_min=1, a_max=None)
        output[mask] = (1.0*nData*np.sqrt( clipped/np.square(nRef) + (nData - Mean(nData, Ref, nRef, func))/np.square(nData) ))
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

    ##### !!!!!!!!!!!!!!! TODO !!!!!!!!!!!!!!!!!!!!!!!
    ####### TOLERANCE HERE ######
    tol = 0.01
    scaleTol = np.power(1 + np.power(Ref * tol**2, 2), -0.5)
    intRef_tol = (scaleTol * Ref)
    Ref_tol = Ref * scaleTol


    if func == 'Gaus1' or func == 'Gaus2':
        return sci.norm.pdf( Pull(Data, Ref, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        #return sci.betabinom.pmf(Data, nData, Ref+1, nRef-Ref+1)
        return sci.betabinom.pmf(Data, nData, Ref_tol + 1, intRef_tol - Ref_tol + 1)
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
    exp_up = np.ceil(Mean(nData, Ref, 'Gaus1'))
    exp_down = np.clip(np.floor(Mean(nData, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero

    ## Find the maximum likelihood
    maxProb_up  = Prob(exp_up, nData, Ref, nRef,func, kurt)
    maxProb_down = Prob(exp_down, nData, Ref, nRef,func, kurt)
    maxProb = np.maximum(maxProb_up, maxProb_down)
    thisProb = Prob(Data, nData, Ref, nRef, func, kurt)
    ## Sanity check to not have relative likelihood > 1
    cond = maxProb < thisProb

    if any(cond.flatten()):
        print(f'for ProbRel')
        print(f'Data: {Data[cond]}\nnData: {nData}\nRef: {Ref[cond]}\nnRef: {nRef}')

    ## make sure check for thisProb < maxProb*0.001 (account for floating point inaccuracies) and just set the ratio to 1 if that is the case
    ratio = thisProb/maxProb
    cond = thisProb > maxProb*0.001
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
    return np.sqrt(2.0*NLL(probRel))
