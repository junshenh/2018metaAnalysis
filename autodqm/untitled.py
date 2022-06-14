#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:41:47 2021

@author: si_sutantawibul1
"""
if __name__ == '__main__':
    import sys
    import subprocess as sb
    sys.path.insert(1, sb.check_output(
        'echo $(git rev-parse --show-cdup)',
        shell=True).decode().strip('\n'))



import sys
sys.path.insert(1, '/home/chosila/Projects/metaAnalysis/autodqm')
import importlib.util
#spec = importlib.util.spec_from_file_location('compare_hists', '/home/chosila/Projects/metaAnalysis/autodqm/compare_hists.py')
#compare_hists = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(compare_hists)
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import compare_hists
import pandas as pd
import time 
import json
import argparse 


parser = argparse.ArgumentParser() 
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--loadpkl', dest='loadpkl', type =bool, help='whether to load hist dfs from pickle. This xor --jsonfile flag is required')
group.add_argument('--jsonfile', dest='jsonfile', type=str, help='name of the jsonfile of runs to use')

args = parser.parse_args() 

condition = '' 

hists1ddir = 'pickles/hists1d-goodruns.pkl'
hists2ddir = 'pickles/hists2d-goodruns.pkl'
loadpkl = args.loadpkl

config_dir = '../config'
subsystem = 'EMTF'
data_series = 'Run2018'
data_sample = 'L1T'
ref_series = 'Run2018'
ref_sample = 'L1T'

ynbins = 20
xnbins = 20
props = dict(boxstyle='round', facecolor='white')
plotdir = f'plots/tmp'

def getBinCenter(arr):
    arrCen = list()
    for i in range(len(arr)-1):
        arrCen.append((arr[i+1]+arr[i])/2)
    return arrCen

def generateLabel(df, y, x):
    df = df.sort_values(by=[y], ascending=False)
    for col in df.columns: ## convert to float bc it breaks if int
        if not isinstance(df[col].iloc[0], str):
            df[col] = df[col].astype(float)
    txtstr = '\n'.join((
                        f'{df["histnames"].iloc[0]} :({df[x].iloc[0]:.4}, {df[y].iloc[0]:.4})',
                        f'{df["histnames"].iloc[1]} :({df[x].iloc[1]:.4}, {df[y].iloc[1]:.4})',
                        f'{df["histnames"].iloc[2]} :({df[x].iloc[2]:.4}, {df[y].iloc[2]:.4})',
                        f'{df["histnames"].iloc[3]} :({df[x].iloc[3]:.4}, {df[y].iloc[3]:.4})',
                        f'{df["histnames"].iloc[4]} :({df[x].iloc[4]:.4}, {df[y].iloc[4]:.4})',
                        ))
    return txtstr

def makePlot(histdf, y, x, ybins, xlabel, ylabel, title, plotname):
    '''
    - histdf (dataFrame) - hist1d or hist2d 
    - y (str) - col to be plotted in y
    - x (str) - col to be plot in x
    - ybins (np.array) - array of bin edges
    - xlabel (str)
    - ylabel (str) 
    - figname (str) - name to save the fig as 1
    '''
    
    if y =='ks':
        fig, ax = plt.subplots()
        xbins = np.linspace(0, max(histdf[x]), xnbins)
        if 'nevents' in x:
            xmin, xmax = np.log2(1), np.log2(max(histdf[x]))
        elif 'nbins' in x: 
            xmin, xmax = np.log2(1), np.log2(max(histdf[x]))
        elif 'avg' in x: 
            xmin, xmax = np.log2(2**-5), np.log2(max(histdf[x]))
        # ymin, ymax = np.log2(2**-10), np.log2(max(histdf[y]))
        logxbins = np.logspace(xmin, xmax, xnbins, base=2)
        logybins = ybins#np.logspace(ymin, ymax, xnbins, base=2)

        searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcH\
itPhi', 'rpcHitTheta']
        histdfshrt = histdf[~histdf['histnames'].str.contains('|'.join(searchfor))]
        
        ## clip the bottoms so to include underflow 
        ## this was needed for log-binning but no longer needed but shouldnt affect anything.
        xvals = np.clip(a = histdfshrt[x], a_min = logxbins[0] , a_max = logxbins[-1])
        yvals = np.clip(a = histdfshrt[y], a_min = logybins[0] , a_max = logybins[-1])
        
        counts, _, _ = np.histogram2d(xvals, yvals, bins=(logxbins, logybins))
        ax.pcolormesh(logxbins, logybins, counts.T, norm=mpl.colors.LogNorm(), shading='auto')
        ax.set_xscale('log', base=2)
        # ax.set_yscale('log', base=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(title + condition)
        textstr = generateLabel(df=histdfshrt, y=y, x=x)
        ax.text(1.25*max(histdf[x]), logybins[10], textstr, bbox=props)
        plt.savefig(f'{plotdir}/{plotname}-chi2.png', bbox_inches='tight')
        plt.close(fig)
    else: 
        fig, ax = plt.subplots()
        xbins = np.linspace(0, max(histdf[x]), xnbins)
        if 'nevents' in x:
            xmin, xmax = np.log2(1), np.log2(max(histdf[x]))
        elif 'nbins' in x: 
            xmin, xmax = np.log2(1), np.log2(max(histdf[x]))
        elif 'avg' in x: 
            xmin, xmax = np.log2(2**-5), np.log2(max(histdf[x]))
        #ymin, ymax = np.log2(2**-10), np.log2(max(histdf[y]))
        logxbins = np.logspace(xmin, xmax, xnbins, base=2)
        logybins = ybins#np.logspace(ymin, ymax, xnbins, base=2)
        #maxval = ybins[-2]
        #ytoplot = np.clip(histdf[y], a_min=0, a_max=maxval)
        
        ## remove “cscLCTStrip”, “cscLCTWire”, “cscChamberStrip”, “cscChamberWire”, “rpcChamberPhi”, “rpcChamberTheta”, “rpcHitPhi”, and “rpcHitTheta”
        ## from pull scatter plots
        #if 'pull' in y: 
        searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcHitPhi', 'rpcHitTheta']
        histdfshrt = histdf[~histdf['histnames'].str.contains('|'.join(searchfor))]

        xvals = np.clip(a = histdfshrt[x], a_min = logxbins[0] , a_max = logxbins[-1])
        yvals = np.clip(a = histdfshrt[y], a_min = logybins[0] , a_max = logybins[-1])
        counts, _, _ = np.histogram2d(xvals, yvals, bins=(logxbins, logybins))
        
        ax.pcolormesh(logxbins, logybins, counts.T, norm=mpl.colors.LogNorm(), shading='auto')
        ax.set_xscale('log', base=2)
        # ax.set_yscale('log', base=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title+condition)

        textstr = generateLabel(df=histdfshrt, y=y, x=x)
        ax.text(1.25*max(histdf[x]), logybins[9], textstr, bbox=props)
        plt.savefig(f'{plotdir}/{plotname}-{condition}.png', bbox_inches='tight')
        plt.close(fig)
    #%%


if not loadpkl: 
    ## read the datafile and reffiles from a json 

    datadict = json.load(open(args.jsonfile))

    
    h1d = list()
    h2d = list()
    
    ## for storing items to plot
    hist1dnbins = list()
    hist2dnbins = list()
    maxpulls = list()
    kss = list()
    nevents1ddata = list()
    nevents1dref = list()
    nevents2ddata = list()
    nevents2dref = list()
    chi21d = list()
    chi22d = list()
    maxpull1d = list()
    histnames1d = list()
    histnames2d = list()
    run1d = list()
    run2d = list()
    
    nBinsUsed = list()
    pulls1d = list()
    pulls2d = list()
    
    
    resultslist = []
    datarunlist = []
    start = time.time()
    for data_path in datadict:
        runnum_idx = data_path.find('_R000')+5#data_path[-11:-5]
        data_run = data_path[runnum_idx:runnum_idx+6]
        datarunlist.append(data_run)
        ref_list = datadict[data_path]
        runnum_idxs = [x.find('_R000')+5 for x in ref_list]
        ref_runs_list = [x[i:i+6] for x,i in zip(ref_list,runnum_idxs)]
        ref_path = data_path
        ref_run = data_run
        resultslist.append(compare_hists.process(config_dir, subsystem,
                                                 data_series, data_sample, data_run, data_path,
                                                 ref_series, ref_sample, ref_run, ref_path,
                                                 ref_list, ref_runs_list,
                                                 output_dir='./out/', plugin_dir='/afs/cern.ch/user/c/csutanta/Projects/metaAnalysis/plugins/'))
 


    end = time.time()
    print('time taken: ', end - start)
    
    for i,results in enumerate(resultslist):
        run = datarunlist[i]
        for result in results:
            hists = result['hists']
            for hist in hists:
                if len(hist.shape) == 2:
                    #h2d.append([histarr, histedge])
                    #x = getBinCenter(histedge[0])
                    #y = getBinCenter(histedge[1])
                    histnames2d.append(result['name'])
                    run2d.append(run)
    
                    hist2dnbins.append(hist.shape[0]*hist.shape[1])
                    maxpulls.append(result['info']['Max_Pull_Val'])
                    nevents2ddata.append(result['info']['Data_Entries'])
                    nevents2dref.append(result['info']['Ref_Entries'])
                    chi22d.append(result['info']['Chi_Squared'])
                    nBinsUsed.append(result['info']['nBinsUsed'])
                    pulls2d.append(result['info']['new_pulls'])
        
                elif len(hist.shape) == 1:
                    #histedge = histedge[0]
                    #barval = getBinCenter(histedge)
                    histnames1d.append(result['name'])
                    run1d.append(run)
    
                    hist1dnbins.append(hist.shape[0])
                    kss.append(result['info']['KS_Val'])
                    nevents1ddata.append(result['info']['Data_Entries'])
                    nevents1dref.append(result['info']['Ref_Entries'])
                    chi21d.append(result['info']['Chi_Squared'])
                    maxpull1d.append(result['info']['Max_Pull_Val'])
                    pulls1d.append(result['info']['pulls'])
                    
        
    #------------------------- make pd of info, easy to mannip ------------------
    nevents1ddata = np.array(nevents1ddata)
    hist1dnbins = np.array(hist1dnbins)
    nevents1dref = np.array(nevents1dref)
    hist1dnbins = np.array(hist1dnbins)
    nevents2ddata = np.array(nevents2ddata)
    hist2dnbins = np.array(hist2dnbins)
    nevents2dref = np.array(nevents2dref)
    hist2dnbins = np.array(hist2dnbins)
    
    ## note: i tried to put the pulls array into the df too, but the df coudln't display it
    ## as an array so it was futile. Just use the df to find the correct index and use 
    ## that to find the corresponding pulls in the pulls list
    hists1d = pd.DataFrame(histnames1d)
    hists1d = hists1d.rename(columns={0: 'histnames'})
    hists1d = hists1d.assign(ks = kss)
    hists1d = hists1d.assign(nbins = hist1dnbins)
    hists1d = hists1d.assign(neventsdata = nevents1ddata)
    hists1d = hists1d.assign(neventsref = nevents1dref)
    hists1d = hists1d.assign(chi2 = chi21d)
    hists1d = hists1d.assign(maxpull = maxpull1d)
    hists1d = hists1d.assign(run=run1d)
    hists1d = hists1d.assign(avgdata = np.divide(nevents1ddata, hist1dnbins))
    hists1d = hists1d.assign(avgref = np.divide(nevents1dref,hist1dnbins))
    hists1d = hists1d.assign(chi2 = chi21d)
    
    hists2d = pd.DataFrame(histnames2d)
    hists2d = hists2d.rename(columns={0: 'histnames'})
    hists2d = hists2d.assign(nbins = hist2dnbins)
    hists2d = hists2d.assign(maxpull= maxpulls)
    hists2d = hists2d.assign(neventsdata = nevents2ddata)
    hists2d = hists2d.assign(neventsref = nevents2dref)
    hists2d = hists2d.assign(chi2 = chi22d)
    hists2d = hists2d.assign(run = run2d)
    hists2d = hists2d.assign(avgdata = np.divide(nevents2ddata, hist2dnbins, out = np.zeros_like(nevents2ddata), where=hist2dnbins!=0))
    hists2d = hists2d.assign(avgref = np.divide(nevents2dref, hist2dnbins, out = np.zeros_like(nevents2ddata), where=hist2dnbins!=0))
    hists2d = hists2d.assign(chi2 = chi22d)

else: 
    hists1d = pickle.load(open(hists1ddir, 'rb'))
    hists2d = pickle.load(open(hists2ddir, 'rb'))
    data_run = hists1d['run'][0]


#%% pickles of hist2d 
#pickle.dump(hists2d, open(f'pickles/hists2d-{data_run}.pkl','wb'))
#pickle.dump(hists1d, open(f'pickles/hists1d-{data_run}.pkl','wb'))
hists2d.to_csv(f'csv/hists2d-badrun-EMTF.csv', index=False)
hists1d.to_csv(f'csv/hists1d-badrun-EMTF.csv', index=False)


os.makedirs(plotdir, exist_ok=True)


maxpull1d_max = 10#np.log2(2**3.1)#21
maxpull1d_min = 0#np.log2(2**-10)
maxpull2d_max = 10#np.log2(2**3.1)#10.5
maxpull2d_min = 0#np.log2(2**-10)
chi21d_max = 15#np.log2(2**7)#50
chi21d_min = 0#np.log2(2**-4)
chi22d_max = 15#np.log2(2**7)#100
chi22d_min = 0#np.log2(2**-4)


ksBins = np.linspace(0, 1, ynbins)#np.logspace(np.log2(2**-3), np.log2(1), 20)
maxpull1dBins = np.linspace(maxpull1d_min, maxpull1d_max, ynbins)#np.logspace(maxpull1d_min, maxpull1d_max, ynbins, base=2)
maxpull2dBins = np.linspace(maxpull2d_min, maxpull2d_max, ynbins)#np.logspace(maxpull2d_min, maxpull2d_max, ynbins, base=2)
chi21dBins = np.linspace(chi21d_min, chi22d_max, ynbins)#np.logspace(chi21d_min, chi21d_max, ynbins, base=2)
chi22dBins = np.linspace(chi22d_min, chi22d_max, ynbins)#np.logspace(chi22d_min, chi22d_max, ynbins, base=2)

#%%
#------------------------------ ks/pv/chi2 vs nbins ------------------------------

#%%
makePlot(histdf=hists1d, y='ks', x='nbins', ybins=ksBins, xlabel='nbins', 
         ylabel='ks', title=f'data:{data_run}',
         plotname='ks_nbins')
#%%
makePlot(histdf=hists1d, y='maxpull', x='nbins', ybins=maxpull1dBins, xlabel='nbins', 
         ylabel='Max pull (1D)', title=f'data:{data_run}',
         plotname='maxpull-nbins-1d')

makePlot(histdf=hists2d, y='maxpull', x='nbins', ybins=maxpull2dBins, xlabel='nbins', 
         ylabel='Max pull (2D)', title=f'data:{data_run}',
         plotname='maxpull-nbins-2d')

makePlot(histdf=hists1d, y='chi2', x='nbins', ybins=chi21dBins, xlabel='nbins',
         ylabel='Chi2 (1D)', title=f'data:{data_run}',
         plotname='chi2-nbins-1d')

makePlot(histdf=hists2d, y='chi2', x='nbins', ybins=chi22dBins, xlabel='nbins',
         ylabel='Chi2 (2D)', title=f'data:{data_run}',
         plotname='chi2-nbins-2d')
##----------------------------------------------------------------------------
#%%
#------------------------------ ks/pv vs nevents ------------------------------
makePlot(histdf=hists1d, y='ks', x='neventsdata', ybins=ksBins, xlabel='# of events (data)', 
         ylabel='ks', title=f'data:{data_run}', plotname='data_ks-nevents')

# makePlot(histdf=hists1d, y='ks', x='neventsref', ybins=ksBins, 
#          xlabel='# of events (ref)', ylabel='ks', 
#          title=f'ref:{ref_run}', plotname='ref_ks-nevents')

makePlot(histdf=hists1d, y='maxpull', x='neventsdata', ybins=maxpull1dBins, 
         xlabel='# of events (data)', ylabel='Max pull (1D)', 
         title=f'data:{data_run}', plotname='data_maxpull-nevents-1d')

# makePlot(histdf=hists1d, y='maxpull', x='neventsref', ybins=maxpullBins, 
#          xlabel='# of events (ref)', ylabel='Max pull (1D)', 
#          title=f'ref:{ref_run}', plotname='ref_maxpull-nevents-1d')

makePlot(histdf=hists2d, y='maxpull', x='neventsdata', ybins=maxpull2dBins,
         xlabel='# of events (data)', ylabel='Max pull (2D)',
         title=f'data:{data_run}', plotname='data_maxpull-nevents-2d')

# makePlot(histdf=hists2d, y='maxpull', x='neventsref', ybins=maxpullBins,
#          xlabel='# of events (ref)', ylabel='Max pull (2D)',
#          title=f'ref:{ref_run}', plotname='ref_maxpull-nevents-2d')

makePlot(histdf=hists1d, y='chi2', x='neventsdata', ybins=chi21dBins, 
         xlabel='# of events (data)', ylabel='chi2 (1D)', 
         title=f'data:{data_run}', plotname='data_chi2-nevents-1d')

# makePlot(histdf=hists1d, y='chi2', x='neventsref', ybins=chi21dBins, 
#          xlabel='# of events (ref)', ylabel='Chi2 (1D)', 
#          title=f'ref:{ref_run}', plotname='ref_chi2-nevents-1d')

makePlot(histdf=hists2d, y='chi2', x='neventsdata', ybins=chi22dBins,
         xlabel='# of events (data)', ylabel='Chi2 (2D)', 
         title=f'data:{data_run}', plotname='data_chi2-nevents-2d')

# makePlot(histdf=hists2d, y='chi2', x='neventsref', ybins=chi22dBins,
#          xlabel='# of events (ref)', ylabel='Chi2 (2D)', 
#          title=f'ref:{ref_run}', plotname='ref_chi2-nevents-2d')
#-----------------------------------------------------------------------------


##------------------------------ ks/pv vs avg events per bin ------------------

makePlot(histdf=hists1d, y='ks', x='avgdata', ybins=ksBins, 
         xlabel='average event per bin (data)', ylabel='ks', 
         title=f'data:{data_run}', plotname='data_ks-avg')

# makePlot(histdf=hists1d, y='ks', x='avgref', ybins=ksBins, 
#          xlabel='average event per bin (ref)', ylabel='ks', 
#          title=f'ref:{ref_run}', plotname='ref_ks-avg')

makePlot(histdf=hists1d, y='maxpull', x='avgdata', ybins=maxpull1dBins, 
         xlabel='average event per bin (data)', ylabel='Max pull (1D)', 
         title=f'data:{data_run}', plotname='data_maxpull-avg-1d')

# makePlot(histdf=hists1d, y='maxpull', x='avgref', ybins=maxpullBins, 
#          xlabel='average event per bin (ref)', ylabel='Max pull (1D)', 
#          title=f'ref:{ref_run}', plotname='ref_maxpull-avg-1d')

makePlot(histdf=hists2d, y='maxpull', x='avgdata', ybins=maxpull2dBins, 
         xlabel='average event per bin (data)', ylabel='Max pull (2D)', 
         title=f'data:{data_run}', plotname='data_maxpull-avg-2d')

# makePlot(histdf=hists2d, y='maxpull', x='avgref', ybins=maxpullBins, 
#          xlabel='average event per bin (ref)', ylabel='Max pull (2D)', 
#          title=f'ref:{ref_run}', plotname='ref_maxpull-avg-2d')

makePlot(histdf=hists1d, y='chi2', x='avgdata', ybins=chi21dBins, 
         xlabel='average events per bin (data)', ylabel='Chi2 (1D)', 
         title=f'data:{data_run}', plotname='data_chi2-avg-1d')

# makePlot(histdf=hists1d, y='chi2', x='avgref', ybins=chi22dBins, 
#          xlabel='average event per bin (ref)', ylabel='Chi2 (1D)', 
#          title=f'ref:{ref_run}', plotname='ref_chi2-avg-1d')

makePlot(histdf=hists2d, y='chi2', x='avgdata', ybins=chi22dBins, 
         xlabel='average events per bin (data)', ylabel='Chi2 (2D)', 
         title=f'data:{data_run}', plotname='data_chi2-avg-2d')

# makePlot(histdf=hists2d, y='chi2', x='avgref', ybins=chi22dBins, 
#          xlabel='average events per bin (ref)', ylabel='Chi2 (2D)', 
#          title=f'ref:{ref_run}', plotname='ref_chi2-avg-2d')

#%%

# print(f'maxpull1d: {max(maxpull1d)}, quantile: {np.quantile(maxpull1d, .95)}')
# print(f'chi21d: {max(chi21d)}, quantile: {np.quantile(chi21d, .95)}')
# print(f'maxpull2d: {max(maxpulls)}, quantile: {np.quantile(maxpulls, .95)}')
# print(f'chi22d: {max(chi22d)}, quantile: {np.quantile(chi22d, .95)}')
# 
# maxpull1d.sort()
# maxpulls.sort()
# chi21d.sort()
# chi22d.sort()
# nBinsUsed.sort()




## heatmaps for pulls 
if True:
    maxpullval = 9#8.292361075813595
    minpullval = 0 
    
    # colorbar
    colors = ['#d0e5d2', '#b84323']#['#1e28e9','#d0e5d2', '#b84323'] 
    cmap = mpl.colors.LinearSegmentedColormap.from_list('autodqm scheme', colors, N = 255)
    
    
    
    top5chi2 = hists2d.sort_values(by='chi2',ascending=False).histnames[:5].to_list()
    
    searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcHitPhi', 'rpcHitTheta']
    pullhist2d = hists2d[~hists2d['histnames'].str.contains('|'.join(searchfor))]
    top5maxpull = pullhist2d.sort_values(by='maxpull',ascending=False).histnames[:5].to_list()
    l = ['emtfTrackOccupancy',   'cscLCTTimingBX0', 'cscDQMOccupancy']
    l = top5chi2 + top5maxpull + l
    
    for i,x in enumerate(histnames2d):
        # check if anything in l is in histname2d
        if any(substring in x for substring in l):
            histvals = pulls2d[i][0]
            histedges = pulls2d[i][1]
            xedges = getBinCenter(histedges[0])
            yedges = getBinCenter(histedges[1])
            fig, ax = plt.subplots()
            norm = mpl.colors.Normalize(vmin=minpullval, vmax=maxpullval)
            im = ax.pcolormesh(xedges, yedges, histvals.T, cmap=cmap, shading='auto', norm=norm)
            fig.colorbar(im)
            ax.set_title(x+condition)
            #os.makedirs(f'{plotdir}/pulls2d', exist_ok=True)
            fig.savefig(f'{plotdir}/{x}{condition}.png', bbox_inches='tight') #'pulls2d/{x}-chi2.png', bbox_inches='tight')
            # plt.show()
            plt.close('all')
    
    #%%
    
    top5chi2 = hists1d.sort_values(by='chi2',ascending=False).histnames[:5].to_list()
    
    searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcHitPhi', 'rpcHitTheta']
    pullhist1d = hists1d[~hists1d['histnames'].str.contains('|'.join(searchfor))]
    
    top5maxpull = hists1d.sort_values(by='maxpull',ascending=False).histnames[:5].to_list()
    l = ['emtfTrackEta']
    l = top5chi2 + top5maxpull + l                                                  
                                  
    for i,x in enumerate(histnames1d):
        if any(substring in x for substring in l):
            histvals = pulls1d[i][0]
            histedge = pulls1d[i][1]
            xedges = getBinCenter(histedge)
            width = histedge[1] - histedge[0]
            fig, ax = plt.subplots()
            ax.bar(xedges, histvals, width)
            ax.set_title(x+condition)
            ax.set_ylim([minpullval, maxpullval])
            #os.makedirs(f'{plotdir}/pulls1d', exist_ok=True)
            fig.savefig(f'{plotdir}/{x}{condition}.png', bbox_inches='tight')#pulls1d/{x}-chi2.png', bbox_inches='tight')
            # plt.show()
            plt.close('all')
            
            
    #%%
    
    
    
    
            
    #%%
    
