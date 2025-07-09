# Written by Natasha Toghramadjian (natasha_toghramadjian@g.harvard.edu), December 2024, for Seattle nodal seismic network analysis, modified from NoisePy functionalities.

import os
import sys
import glob
import obspy
import scipy
import pyasdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
import pandas as pd
import h5py

sta_file = '/mnt/data1/SEATTLE_June-2021_studies/station.txt'
stn_df=pd.read_csv(sta_file)
stns = np.unique(stn_df['station'].values) # array of all stations

sfiles = glob.glob('/mnt/data1/SEATTLE_June-2021_studies/STACK_CCFs--S2-output_one-bit--August-2022--rotated-RTZ--Feb-2023/*/*h5')

freqmin = 1.
freqmax = 2.
ccomps = ['ZZ','TT','RR','RT','RZ','TR','TZ','ZR','ZT']

# distance bins:
dist_bin_mins = np.arange(0,5.8,0.05) # 0.,5.8.,0.1 -- 5.9 km is max interstation distance bound; 0.05 km, i.e. bin every 50m
dist_bin_size = int((dist_bin_mins[1]-dist_bin_mins[0])*1000) # convert to meters for filename ease of reading

## LAG TIMES FOR PLOTTING DISPLAY:
disp_lag = 15
amp_reduce_factor = 15. # value by which to divide waveforms for visualization, to make plotting clearer
dtype='Allstack_linear'
stack_method = dtype.split('_')[-1]

stack_mean_or_median = 'mean' ## set as mean or median
h5_files_output_dir = f'/mnt/data1/SEATTLE_June-2021_studies/CCFs-one-bit-RTZ--stacked-by-50m-distance-bins/stacked-using-{stack_mean_or_median}'


for ccomp in ccomps:

    for dist_bin_min in dist_bin_mins:
        waveforms = []
        dist_bin_max = dist_bin_min+0.05 # each bin is 50m, 0.05 km
        
        # extract common variables
        try:
            ds    = pyasdf.ASDFDataSet(sfiles[0],mode='r')
            dt    = ds.auxiliary_data[dtype][ccomp].parameters['dt']
            maxlag= ds.auxiliary_data[dtype][ccomp].parameters['maxlag']
        except Exception:
            print("exit! cannot open %s to read"%sfiles[0]);sys.exit()

        # lags for display   
        tt = np.arange(-int(disp_lag),int(disp_lag)+dt,dt)

        indx1 = int((maxlag-disp_lag)/dt)
        indx2 = indx1+2*int(disp_lag/dt)+1

        for ii in range(len(sfiles)):
            
            sfile = sfiles[ii]

            ## skip single-station correlations:
            stn1 = sfile.split('/')[-1].split('_')[0]
            stn2 = sfile.split('/')[-1].split('_')[-1].split('.h5')[0]
            if stn1 == stn2:
                continue

            ds = pyasdf.ASDFDataSet(sfile,mode='r')

            # load data to variables
            dist = ds.auxiliary_data[dtype][ccomp].parameters['dist']
            ngood= ds.auxiliary_data[dtype][ccomp].parameters['ngood']
            tdata  = ds.auxiliary_data[dtype][ccomp].data[indx1:indx2]

            # plot only CCFs within the desired range if interstation distance:
            if dist_bin_max < dist or dist_bin_min > dist: 
                continue

            tdata = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            waveforms.append(tdata)

        print(f'{len(waveforms)} waveforms in distance bin')
        if len(waveforms)==0: # don't move forward to plot if no waveforms found in distance bin
            continue

        waveforms=np.array(waveforms)

        if stack_mean_or_median == 'mean':
            stacked_wave = np.mean(waveforms,axis=0) # stack by mean
        if stack_mean_or_median == 'median':
            stacked_wave = np.median(waveforms,axis=0) # stack by median

        stacked_wave /= np.max(np.abs(stacked_wave)) # normalize by max amp

        ## save waveform as output h5 file:
        h5_filename = h5_files_output_dir+f'/{ccomp}_{freqmin}-{freqmax}-Hz--stacked-by-{stack_mean_or_median}--{int(dist_bin_min*1000)}-{int(dist_bin_max*1000)}m-interstation-dist.h5'
        h5f = h5py.File(h5_filename, 'w')
        h5f.create_dataset('stacked-waveform', data=stacked_wave)
        h5f.close()

        print('stacked wave generated')
        
        stacked_wave /= float(amp_reduce_factor) # reduce amplitude for visualization; not included in the saved h5 file.

        bin_avg_dist = dist_bin_max-0.025
        print(f'avg bin dist is {np.round(bin_avg_dist,3)}')
        plt.grid()
        plt.plot(tt, stacked_wave + bin_avg_dist,'k',linewidth=0.6)
        plt.xlim(-disp_lag, disp_lag)  
        plt.ylim(0, max(dist_bin_mins)+dist_bin_size/1000) 
        plt.margins(0)  
        plt.tight_layout()

    # titles and axis labels, and text display of information on waveform binning:
    # plt.text(disp_lag+0.4, bin_avg_dist, f'{len(waveforms)} waveforms in distance bin',fontsize=3)
    plt.title(f'{ccomp}, bandpassed {freqmin}-{freqmax} Hz, stacked by {stack_method}, {stack_mean_or_median}, within {dist_bin_size}m distance bins, downscaled to 1/{str(int(amp_reduce_factor))} amplitude',fontsize=8)
    plt.xlabel('Time (s)')
    plt.ylabel('Interstation distance (km)')
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.grid()
    outdir = f'/mnt/data1/SEATTLE_June-2021_studies/figures/ALL-STATIONS-COMBINED-moveouts--{dist_bin_size}m-bins--{stack_method}-stack-by-{stack_mean_or_median}'
    if not os.path.isdir(outdir):os.mkdir(outdir)
    plt.savefig(outdir+f'/{ccomp}-moveout_1D_ALL-stn-pairs_{stack_method}--binned-by-{dist_bin_size}m--stack-by-{stack_mean_or_median}--{freqmin}-{freqmax}-Hz--amp-downscaled-by-{str(int(amp_reduce_factor))}.png', format='png', dpi=500)
    plt.cla()
