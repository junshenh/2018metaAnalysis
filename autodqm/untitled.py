#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:41:47 2021

@author: si_sutantawibul1
"""
import sys
sys.path.insert(1, '/home/chosila/Projects/metaAnalysis/autodqm')
import compare_hists
import importlib.util
# spec = importlib.util.spec_from_file_location('compare_hists', '/home/chosila/Projects/metaAnalysis/autodqm/compare_hists.py')
# compare_hists = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(compare_hists)
import pickle
#import root_numpy
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import pandas as pd



plt.tight_layout()
'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OnlineData/original/00032xxxx/0003200xx/'
basedatadir = 'rootfiles/data/'
baserefdir = 'rootfiles/ref/'
datadirs = [basedatadir + i for i in os.listdir(basedatadir)]
refdirs = [baserefdir + i for i in os.listdir(baserefdir)]
#datadirs = [baserefdir + i for i in os.listdir(baserefdir)]
data_path = 'rootfiles/ref/DQM_V0001_L1T_R000320002.root'
ref_path = 'rootfiles/ref/DQM_V0001_L1T_R000320025.root'
config_dir = '../config'
subsystem = 'EMTF'
data_series = 'Run2018'
data_sample = 'L1T'
dataruns = [i[-11:-5] for i in datadirs]
data_run = data_path[-11:-5]
ref_series = 'Run2018'
ref_sample = 'L1T'
refruns = [i[-11:-5] for i in refdirs]
ref_run = ref_path[-11:-5]

ynbins = 22
xnbins = 20
props = dict(boxstyle='round', facecolor='white')
plotdir = f'plots/oldAutoDQM/data{data_run[-2:]}ref{ref_run[-2:]}'

condition = ''
loadpkl = False

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
        
        ## clip the bottoms so to include underflow 
        ## this was needed for log-binning but no longer needed but shouldnt affect anything.
        xvals = np.clip(a = histdf[x], a_min = logxbins[0] , a_max = logxbins[-1])
        yvals = np.clip(a = histdf[y], a_min = logybins[0] , a_max = logybins[-1])
        
        counts, _, _ = np.histogram2d(xvals, yvals, bins=(logxbins, logybins))
        ax.pcolormesh(logxbins, logybins, counts.T, norm=mpl.colors.LogNorm(), shading='auto')
        ax.set_xscale('log', base=2)
        # ax.set_yscale('log', base=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(title + condition)
        textstr = generateLabel(df=histdf, y=y, x=x)
        ax.text(1.25*max(histdf[x]), logybins[10], textstr, bbox=props)
        plt.savefig(f'{plotdir}/{plotname}-chi2.png', bbox_inches='tight')
        plt.show()
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
        if 'pull' in y: 
            searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcHitPhi', 'rpcHitTheta']

            histdf = histdf[~histdf['histnames'].str.contains('|'.join(searchfor))]

        xvals = np.clip(a = histdf[x], a_min = logxbins[0] , a_max = logxbins[-1])
        yvals = np.clip(a = histdf[y], a_min = logybins[0] , a_max = logybins[-1])
        counts, _, _ = np.histogram2d(xvals, yvals, bins=(logxbins, logybins))
        
        ax.pcolormesh(logxbins, logybins, counts.T, norm=mpl.colors.LogNorm(), shading='auto')
        ax.set_xscale('log', base=2)
        # ax.set_yscale('log', base=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title+condition)

        textstr = generateLabel(df=histdf, y=y, x=x)
        ax.text(1.25*max(histdf[x]), logybins[9], textstr, bbox=props)
        plt.savefig(f'{plotdir}/{plotname}-{condition}.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
    #%%
    #%%

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


# if not loadpkl:
#     results = compare_hists.process(config_dir, subsystem,
#                                    data_series, data_sample, data_run, data_path,
#                                    ref_series, ref_sample, ref_run, ref_path,
#                                    output_dir='./out/', plugin_dir='/home/chosila/Projects/AutoDQM-p3/plugins')
#     pickle.dump(results, open('out/results.pkl', 'wb'))

# else:
#     results = pickle.load(open('out/results.pkl', 'rb'))

data_run = data_path[-11:-5]
print(f'ref path: {ref_path}')
print(f'data path: {data_path}')
results = compare_hists.process(config_dir, subsystem,
                                data_series, data_sample, data_run, data_path,
                                ref_series, ref_sample, ref_run, ref_path,
                                output_dir='./out/', plugin_dir='/home/chosila/Projects/metaAnalysis/plugins')

for result in results:
        hists = result['hists']
        for hist in hists:
            #histarr, histedge = hists# root_numpy.hist2array(hist, return_edges=True)#, include_overflow=True)
            if len(hist.shape) == 2:#hist.InheritsFrom('TH2'):
                #h2d.append([histarr, histedge])
                #x = getBinCenter(histedge[0])
                #y = getBinCenter(histedge[1])
                histnames2d.append(result['name'])
                #run2d.append(run)
    
                #------------------ pull values vs nbins ------------------
    
                hist2dnbins.append(hist.shape[0]*hist.shape[1])
                maxpulls.append(result['info']['Max_Pull_Val'])
    
                #-------------------------------------------------------------
    
                #--------------------- pv vs nevents ------------------------------
    
                nevents2ddata.append(result['info']['Data_Entries'])
                nevents2dref.append(result['info']['Ref_Entries'])
    
                #-------------------------------------------------------------
    
    
                #------------------------ chi2 ------------------------------------
    
                chi22d.append(result['info']['Chi_Squared'])
    
                #----------------------------------------------------------------
                
                #--------------------------- nbinsUsed ----------------------------
                #nBinsUsed.append(result['info']['nBinsUsed'])
                #pulls2d.append(result['info']['new_pulls'])
                #------------------------------------------------------------------
                
    
            elif len(hist.shape) == 1: #hist.InheritsFrom('TH1'):
                ## have to make this plot both the data and ref
                ## looks like it returns the data first, but autodqm plots it second. this
                ## doesn't really matter, just documenting
                
                #histedge = histedge[0]
                #barval = getBinCenter(histedge)
                histnames1d.append(result['name'])
                #run1d.append(run)
    
    
                #------------------------ ks vs nbins ------------------------
    
                hist1dnbins.append(hist.shape[0])
                kss.append(result['info']['KS_Val'])
    
                #-------------------------------------------------------------
    
                #------------------------ ks vs nevents ------------------------
    
                nevents1ddata.append(result['info']['Data_Entries'])
                nevents1dref.append(result['info']['Ref_Entries'])
    
                #-------------------------------------------------------------
    
                #--------------------------- chi2& max pull ----------------------
    
                chi21d.append(result['info']['Chi_Squared'])
                maxpull1d.append(result['info']['Max_Pull_Val'])
    
                #----------------------------------------------------------------
    
                #-----------------------pulls------------------------------------
                #pulls1d.append(result['info']['pulls'])
                #-------------------------------------------------


    
#------------------------- make pd of info, easy to mannip ------------------

hists1d = pd.DataFrame(histnames1d)
hists1d = hists1d.rename(columns={0: 'histnames'})
hists1d = hists1d.assign(ks = kss)
hists1d = hists1d.assign(nbins = hist1dnbins)
hists1d = hists1d.assign(neventsdata = nevents1ddata)
hists1d = hists1d.assign(neventsref = nevents1dref)
hists1d = hists1d.assign(chi2 = chi21d)
hists1d = hists1d.assign(maxpull = maxpull1d)
#hists1d = hists1d.assign(run=run1d)
hists1d = hists1d.assign(avgdata = np.array(nevents1ddata)/np.array(hist1dnbins))
hists1d = hists1d.assign(avgref = np.array(nevents1dref)/np.array(hist1dnbins))
hists1d = hists1d.assign(chi2 = chi21d)

hists2d = pd.DataFrame(histnames2d)
hists2d = hists2d.rename(columns={0: 'histnames'})
hists2d = hists2d.assign(nbins = hist2dnbins)
hists2d = hists2d.assign(maxpull= maxpulls)
hists2d = hists2d.assign(neventsdata = nevents2ddata)
hists2d = hists2d.assign(neventsref = nevents2dref)
hists2d = hists2d.assign(chi2 = chi22d)
#hists2d = hists2d.assign(run = run2d)
hists2d = hists2d.assign(avgdata = np.array(nevents2ddata)/np.array(hist2dnbins))
hists2d = hists2d.assign(avgref = np.array(nevents2dref)/np.array(hist2dnbins))
hists2d = hists2d.assign(chi2 = chi22d)


import os

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


makePlot(histdf=hists1d, y='ks', x='nbins', ybins=ksBins, xlabel='nbins', 
         ylabel='ks', title=f'data:{data_run}, ref:{ref_run}',
         plotname='ks_nbins')

makePlot(histdf=hists1d, y='maxpull', x='nbins', ybins=maxpull1dBins, xlabel='nbins', 
         ylabel='Max pull (1D)', title=f'data:{data_run}, ref:{ref_run}',
         plotname='maxpull-nbins-1d')

makePlot(histdf=hists2d, y='maxpull', x='nbins', ybins=maxpull2dBins, xlabel='nbins', 
         ylabel='Max pull (2D)', title=f'data:{data_run}, ref:{ref_run}',
         plotname='maxpull-nbins-2d')

makePlot(histdf=hists1d, y='chi2', x='nbins', ybins=chi21dBins, xlabel='nbins',
         ylabel='Chi2 (1D)', title=f'data:{data_run}, ref:{ref_run}',
         plotname='chi2-nbins-1d')

makePlot(histdf=hists2d, y='chi2', x='nbins', ybins=chi22dBins, xlabel='nbins',
         ylabel='Chi2 (2D)', title=f'data:{data_run}, ref:{ref_run}',
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
print(f'maxpull1d: {max(maxpull1d)}, quantile: {np.quantile(maxpull1d, .95)}')
print(f'chi21d: {max(chi21d)}, quantile: {np.quantile(chi21d, .95)}')
print(f'maxpull2d: {max(maxpulls)}, quantile: {np.quantile(maxpulls, .95)}')
print(f'chi22d: {max(chi22d)}, quantile: {np.quantile(chi22d, .95)}')

maxpull1d.sort()
maxpulls.sort()
chi21d.sort()
chi22d.sort()
