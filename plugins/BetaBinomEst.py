import sys
import math
import datetime
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


## Kurt() function not used anywhere so I mute and did not vectorize
## Kurtosis of beta-binomial function
# def Kurt(nData, Ref, nRef, func):
#     if func == 'BetaB' or func == 'Gamma':
#         ## From https://en.wikipedia.org/wiki/Beta-binomial_distribution#Moments_and_properties
#         ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
#         n_  = nData
#         a_  = Ref+1
#         b_  = nRef-Ref+1
#         ab_ = nRef+2
#         ## Exact computation of kurtosis
#         if n_ > 0:
#             norm = ab_*ab_*(1+ab_) / (n_*a_*b_*(ab_+2)*(ab_+3)*(ab_+n_))
#             kurt = norm * ( ab_*(ab_-1+6*n_) + 3*a_*b_*(n_-2) + 6*n_*n_ - 3*a_*b_*n_*(6-n_)/ab_ - 18*a_*b_*n_*n_/(ab_*ab_) )
#         else:
#             print('\nInside Kurt, n_ = %d, a_ = %d, b_ = %d: n_ is <= 0! Quitting.' % (n_, a_, b_))
#             sys.exit()

#         ## The really useful quantity is "excess kurtosis", i.e. kurtosis - 3
#         return kurt - 3
#     ## End conditional: if func == 'BetaB' or func == 'Gamma'

#     print('\nInside Kurt, no valid func = %s. Quitting.\n' % func)
#     sys.exit()


## Number of standard devations from the mean in any function
def Pull(Data, Ref, func):
    nData = Data.sum()
    nRef = Ref.sum()
    return (Data - Mean(Data, Ref, func)) / StdDev(Data, Ref, func)


## Exact and approximate values for natural log of the Gamma function
def LogGam(z):
    return special.gammaln(z)
    #print('\nInside LogGam, z = %f' % z)
    ## Gamma function not well-defined for negative numbers
    # if z < 0:
    #     print('\nInside LogGam, z = %f, which is less than 0! Quitting.\n' % z)
    #     sys.exit()
    # ## Python's math.gamma function returns 'OverflowError: math range error' for z > 171
    # if z < 171:
    # #if z < 0:
    #     return math.log(math.gamma(z))
    # ## Stirling's / Lanczos approximation: https://en.wikipedia.org/wiki/Gamma_function#Approximations
    # ## Last term from "Order(1/z)" in Stirling's approximation, experimentally optimized to 0.08335
    # else:
    #     return 0.5*math.log(2*math.pi) - 0.5*math.log(z) + z*math.log(z) - z + math.log(1 + (0.08335/z))


## Predicted probability of observing Data / nData given a reference of Ref / nRef
def Prob(Data, nData, Ref, nRef, func, kurt=0):
    if func == 'Gaus1' or func == 'Gaus2':
        return sci.norm.pdf( Pull(Data, Ref, func) )
    if func == 'BetaB':
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        return sci.betabinom.pmf(Data, nData, Ref+1, nRef-Ref+1)
    ## Pearson type VII family from https://en.wikipedia.org/wiki/Kurtosis#The_Pearson_type_VII_family
    if func == 'Pears':
        ## In limit that excess kurtosis is 0, becomes a normal distribution
        if kurt == 0:
            return sci.norm.pdf( Pull(Data, nData, Ref, nRef, 'BetaB') )
        a_ = math.sqrt(2 + (6 / kurt))  ## Scale parameter 'a'
        m_ = 2.5 + (3 / kurt)           ## Shape parameter 'm'
        ## True normalization fails for gamma of very large numbers, i.e. kurt = 0
        ## norm = math.gamma(m_) / (a_ * math.sqrt(math.pi) * math.gamma(m_ - 0.5))
        ## Use normalization of normal distribution instead, for now
        norm = np.power(StdDev(nData, Ref, nRef, 'BetaB') * math.sqrt(2 * math.pi), -1)
        return norm * np.power(1 + np.power(Pull(Data, Ref, 'BetaB') / a_, 2), -1.0*m_)
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
        return math.exp(logProb)

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
    # cond = thisProb > maxProb*0.001
    # ratio[cond] = 1 
        
    return ratio#(thisProb / maxProb)


## Negative log likelihood
def NLL(prob):
    nllprob = -1.0*np.log(prob, where=(prob>0))
    nllprob[prob==0] = 999
    nllprob[prob < 0] == -999

    return nllprob       


## Convert relative probability to number of standard deviations in normal distribution
def Sigmas(probRel):
    return np.sqrt(2.0*NLL(probRel))
