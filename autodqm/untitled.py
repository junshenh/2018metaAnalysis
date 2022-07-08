#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:41:47 2021

@author: si_sutantawibul1
"""

import sys
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
group.add_argument('--jsonfile', dest='jsonfile', type=str, help='name of the jsonfile of runs to use')

args = parser.parse_args()

condition = ''

config_dir = '../config'
subsystem = 'L1T_online'
data_series = 'Run2018'
data_sample = 'L1T'
ref_series = 'Run2018'
ref_sample = 'L1T'

plotdir = f'plots/betaEst'

def getBinCenter(arr):
    arrCen = list()
    for i in range(len(arr)-1):
        arrCen.append((arr[i+1]+arr[i])/2)
    return arrCen

### ------ start of script ------- ###
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

## run autodqm backend
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
                                             output_dir='./out/', plugin_dir='./../plugins/'))



end = time.time()
print('time taken: ', end - start)

## get results from autodqm into a list then into a df
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


#------------------------- make pd of results from autodqm ------------------
nevents1ddata = np.array(nevents1ddata, dtype=object)
hist1dnbins = np.array(hist1dnbins, dtype=object)
nevents1dref = np.array(nevents1dref, dtype=object)
hist1dnbins = np.array(hist1dnbins, dtype=object)
nevents2ddata = np.array(nevents2ddata, dtype=object)
hist2dnbins = np.array(hist2dnbins, dtype=object)
nevents2dref = np.array(nevents2dref, dtype=object)
hist2dnbins = np.array(hist2dnbins, dtype=object)

## note: i tried to put the pulls array into the df too, but the df coudln't display it
## as an array so it was futile. Just use the df to find the correct index and use
## that to find the corresponding pulls in the pulls list
hists1d = pd.DataFrame(histnames1d)
hists1d = hists1d.rename(columns={0: 'histnames'})
hists1d = hists1d.assign(ks = kss)
hists1d = hists1d.assign(chi2 = chi21d)
hists1d = hists1d.assign(maxpull = maxpull1d)
hists1d = hists1d.assign(run=run1d)
hists1d = hists1d.assign(chi2 = chi21d)

hists2d = pd.DataFrame(histnames2d)
hists2d = hists2d.rename(columns={0: 'histnames'})
hists2d = hists2d.assign(maxpull= maxpulls)
hists2d = hists2d.assign(chi2 = chi22d)
hists2d = hists2d.assign(run = run2d)
hists2d = hists2d.assign(chi2 = chi22d)

## get result into csv format
os.makedirs('csv', exist_ok=True)
jsonname = args.jsonfile.split(".")[0]
hists2d.to_csv(f'csv/hists2d_{jsonname}_{subsystem}.csv', index=False)
hists1d.to_csv(f'csv/hists1d_{jsonname}_{subsystem}.csv', index=False)
print('csv files made')
import sys
sys.exit()
os.makedirs(plotdir, exist_ok=True)

## heatmaps for pulls
maxpullval = 40
minpullval = -40
figsize = (12,7)

# colorbar
colors = ['#1e28e9','#d0e5d2', '#b84323']##['#d0e5d2', '#b84323']#
cmap = mpl.colors.LinearSegmentedColormap.from_list('autodqm scheme', colors, N = 255)

top5chi2 = hists2d.sort_values(by='chi2',ascending=False).histnames[:5].to_list()

searchfor = ['cscLCTStrip', 'cscLCTWire', 'cscChamberStrip', 'cscChamberWire', 'rpcChamberPhi', 'rpcChamberTheta', 'rpcHitPhi', 'rpcHitTheta']
pullhist2d = hists2d[~hists2d['histnames'].str.contains('|'.join(searchfor))]
top5maxpull = pullhist2d.sort_values(by='maxpull',ascending=False).histnames[:5].to_list()
l = ['emtfTrackOccupancy',   'cscLCTTimingBX0', 'cscDQMOccupancy']
l = top5chi2 + top5maxpull + l

l = hists2d.histnames

for i,x in enumerate(histnames2d):
    # check if anything in l is in histname2d
    if any(substring in x for substring in l):
        histvals = pulls2d[i][0]
        histedges = pulls2d[i][1]
        xedges = getBinCenter(histedges[0])
        yedges = getBinCenter(histedges[1])
        fig, ax = plt.subplots(figsize=figsize)
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

l = hists1d.histnames

for i,x in enumerate(histnames1d):
    if any(substring in x for substring in l):
        histvals = pulls1d[i][0]
        histedge = pulls1d[i][1]

        xedges = getBinCenter(histedge)
        width = histedge[1] - histedge[0]
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(xedges, histvals, width)
        ax.set_title(x+condition)
        ax.set_ylim([minpullval, maxpullval])
        #os.makedirs(f'{plotdir}/pulls1d', exist_ok=True)
        fig.savefig(f'{plotdir}/{x}{condition}.png', bbox_inches='tight')#pulls1d/{x}-chi2.png', bbox_inches='tight')
        # plt.show()
        plt.close('all')

