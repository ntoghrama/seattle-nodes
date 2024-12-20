from scipy.fftpack.helper import next_fast_len
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
# from pysurf96 import surf96
from numba import jit
import pandas as pd
import numpy as np 
import pyasdf
import obspy
import scipy
import time
import glob
import os
import sys

'''
this script finds all station-pairs within a small region and use the cross-correlation functions 
for all these pairs resulted from 1-year daily stacking of Kanto data to perform integral transformation.
it then extracts the dispersion info from the local minimum

by Chengxin Jiang (chengxin_jiang@fas.harvard.edu) @ Aug/2019
'''

@jit(nopython = True)
def integral_transform(freqVec,cc,dist,spec):
    '''
    try git compiler to speed up the process

    PARAMETERS:
    -----------------
    freqVec: frequency vector representing targeted range
    cc: phase velocity vector 
    dist: distance vector from observational data
    spec: spectra matrix of observed data

    RETURNS:
    -----------------
    disp_array: 2D integral image
    '''        
    tNfft = len(freqVec)
    nc = len(cc)
    nfiles = len(dist)
    disp_array = np.zeros(shape=(nc,tNfft),dtype=np.complex64)
    pi2 = 2*np.pi

    #----3 iterative loops-----
    for ifreq in range(tNfft):
        freq = freqVec[ifreq]
        for ic in range(nc):
            tc = cc[ic]
            for idis in range(nfiles):
                disp_array[ic][ifreq] +=  spec[idis][ifreq]*np.exp(1j*dist[idis]/tc*pi2*freq)

    return np.abs(disp_array)



#######################################
########## PARAMETER SECTION ##########
#######################################

# absolute path for stacked data
# rootpath = '/Users/chengxin/Documents/Kanto_basin/stacked'
rootpath = '/mnt/data1/SEATTLE_June-2021_studies/STACK_CCFs--S2-output_one-bit--August-2022--rotated-RTZ--Feb-2023'

## could plot these over the dispersion figures to see if the manual picks are ~correct (files are named like: manual-picks_ZZ_0.5-4_Hz.txt):
manual_picks_directory = '/mnt/data1/SEATTLE_June-2021_studies/dispersion-curves-picks--April-2023'
manual_pick_files = glob.glob(os.path.join(manual_picks_directory,'*.txt'))

# loop through each station
sta_file = '/mnt/data1/SEATTLE_June-2021_studies/station.txt'
locs = pd.read_csv(sta_file)
net  = locs['network']
sta  = locs['station']
lon  = locs['longitude']
lat  = locs['latitude']

# different components
dtype = 'Allstack_linear'
ccomp  = ['ZZ','RR','TT','ZT','ZR','RT','RZ','TR','TZ'] 
# ccomp  = ['ZZ','RR','TT','ZR','RZ'] # for dispersion picking, No ZT/TZ/RT/TR
plot_disp_picks = True

pindx  = [1,0,1,1,0,1,0,2,0]
onelag = False
norm   = False
bdpass = False
savenp = False
# stack_method = dtype.split('0')[-1]
stack_method = dtype.split('_')[-1] # natasha

# basic parameters for loading data
freqmin  = 0.5
freqmax  = 4
maxdist  = 6.
minnpair = 10
lag      = 15

# loop through each source station
for ista in range(16,17):
    t0=time.time()

    # find all stations within maxdist
    sta_list = []
    sta_lon = []
    sta_lat = []

    for ii in range(len(sta)):
        dist,_,_ = obspy.geodetics.base.gps2dist_azimuth(lat[ista],lon[ista],lat[ii],lon[ii])
        if dist/1000 < maxdist:
            sta_list.append(net[ii]+'.'+str(sta[ii]))
            sta_lon.append(lon[ii])
            sta_lat.append(lat[ii])
    nsta = len(sta_list)

    # construct station pairs from the found stations
    allfiles = []
    for ii in range(nsta-1):
        for jj in range(ii+1,nsta):
            tfile1 = rootpath+'/'+sta_list[ii]+'/'+sta_list[ii]+'_'+sta_list[jj]+'.h5'
            tfile2 = rootpath+'/'+sta_list[ii]+'/'+sta_list[jj]+'_'+sta_list[ii]+'.h5'
            if os.path.isfile(tfile1):
                allfiles.append(tfile1)
            elif os.path.isfile(tfile2):
                allfiles.append(tfile2)
    nfiles = len(allfiles)
    # print(allfiles);sys.exit()

    t1=time.time()
    print('finding all stations takes %6.3fs'%(t1-t0))

    # give it another chance for larger region
    if nfiles<minnpair:
        print('station %s has no enough pairs'%sta[ista])
        continue

    # extract common variables from ASDF file
    try:
        ds    = pyasdf.ASDFDataSet(allfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
    except Exception:
        #raise ValueError('cannot open %s to read'%allfiles[0])
        print('cannot open %s to read'%allfiles[0])
        continue

    # plt.figure(figsize=(4,3))
    for path in ccomp:
        
        ####################################
        ########## LOAD CCFS DATA ##########
        ####################################

        # initialize array
        anpts = int(maxlag/dt)+1
        tnpts = int(lag/dt)+1
        Nfft = int(next_fast_len(tnpts))
        dist = np.zeros(nfiles,dtype=np.float32)
        cc_array = np.zeros(shape=(nfiles,tnpts),dtype=np.float32)
        spec = np.zeros(shape=(nfiles,Nfft//2),dtype=np.complex64)
        flag = np.zeros(nfiles,dtype=np.int16)

        t2=time.time()
        # loop through each cc file
        icc=0
        for ii in range(nfiles):
            tmp = allfiles[ii]

            # load the data into memory
            with pyasdf.ASDFDataSet(tmp,mode='r') as ds:
                print(ds)
                sys.exit()
                try:
                    dist[icc] = ds.auxiliary_data[dtype][path].parameters['dist']
                    tdata    = ds.auxiliary_data[dtype][path].data[:]
                    data = tdata[anpts-1:anpts+tnpts-1]*0.5+np.flip(tdata[anpts-tnpts:anpts],axis=0)*0.5 # here he converts the waveform from points to time 
                except Exception:
                    print("continue! cannot read %s "%tmp)
                    print(f'comp is {path}')
                    # print(ds.auxiliary_data[dtype][path].parameters)
                    continue  
            
            cc_array[icc] = data
            if bdpass:
                cc_array[icc] = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            if norm:
                cc_array[icc] = cc_array[icc]/max(np.abs(cc_array[icc]))
            spec[icc] = scipy.fftpack.fft(cc_array[icc],Nfft)[:Nfft//2]
            icc+=1
        t3=time.time()
        print('loading ccfs takes %6.3fs'%(t3-t2))

        # remove bad ones
        cc_array = cc_array[:icc]
        dist     = dist[:icc]
        spec     = spec[:icc]
        nfiles1   = icc

        #####################################
        ############ FK TRANSFORM ###########
        #####################################

        #------common variables------
        ### these are the min and max values for wave period (x-axis) and phase velocity (y-axis) that will be shown in plotting - natasha
        # p = period; c = phase velocity
        # p1 = 0.5 # natasha - commenting out Chengxin's preset
        p1 = 1/freqmax # natasha -- changing to lower value to include shorter period wave data; should be less than or = 1/freqmax
        p2 = 9.0
        pp = np.arange(p1,p2,0.02)
        c1 = 0.1
        c2 = 3.8
        cc = np.arange(c1,c2,0.02)
        ncc = np.arange(c1,c2,0.002)
        nc = len(cc)

        #------2D dispersion array-------
        freqVec = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft//2]
        indx = np.where((freqVec<=freqmax) & (freqVec>=freqmin))[0]
        freqVec = freqVec[indx]
        spec    = spec[:,indx]
        tNfft = len(freqVec)
        disp_array = np.zeros(shape=(nc,tNfft),dtype=np.complex64)

        # perform integral transformation
        disp_array = integral_transform(freqVec,cc,dist,spec)

        t4=time.time()
        print('performing integral transform takes %6.3fs'%(t4-t3))

        # convert from freq-c domain to period-c domain by interpolation
        fc = scipy.interpolate.interp2d(1/freqVec,cc,np.abs(disp_array),kind='cubic')
        ampo_new = fc(pp,ncc)
        for ii in range(len(pp)):
            ampo_new[:,ii] /= np.max(ampo_new[:,ii])

        # save the data into numpy 
        if savenp:
            tname = rootpath+'/figures/fk_analysis_'+str(maxdist)+'km/data_numpy/'+str(sta[ista])+'_'+path+'_'+str(maxdist)+'km_'+str(nfiles1)+'_pairs'
            np.save(tname,ampo_new)

        ##############################################
        ############ PLOTTING FK RESULTS #############
        ##############################################

        #---plot the 2D dispersion image-----
        # tmpt = '13'+str(ccomp.index(path)+1)
        # plt.subplot(tmpt)

        extent = [pp[0],pp[-1],cc[0],cc[-1]]
        print(extent)
        cx=plt.imshow(ampo_new,cmap='jet',interpolation='bicubic',extent=extent,origin='lower',aspect='auto') ## this is the plotting of the actual data
        # if ccomp.index(path) == 1:
        plt.title(f'{path}, {freqmin}-{freqmax} Hz, %s with %d pairs in %d km'%(sta[ista],nfiles1,maxdist))
        plt.grid()
        plt.xlabel('period [s]')
        plt.ylabel('phase velocity [km/s]')
        # plt.colorbar(cx) # don't need colorbar for now, it just shows power which is intuitive
        font = {'family': 'serif', 'color':  'black', 'weight': 'bold','size': 16}
        # plt.text(max(pp)*0.8,cc[-1]*0.8,path,fontdict=font)
        # plt.text((1/freqmin)*0.8,cc[-1]*0.8,path,fontdict=font)
        # plt.ylim(cc[0],cc[-1])

        plt.text((1/freqmin)*0.8,1.6,path,fontdict=font)
        # plt.xticks(ticks=np.arange(1/freqmax,1/freqmin,0.1),fontsize=9, rotation=90)
        # plt.xticks(ticks=np.arange(0,2.,0.05),fontsize=4, rotation=90)
        plt.xticks(ticks=np.arange(0,2.,0.2),fontsize=4, rotation=90)
        plt.xlim(1/freqmax,1/freqmin)

        # plt.yticks(ticks=np.arange(0,2,0.05), fontsize=4)
        plt.yticks(ticks=np.arange(0,2,0.2), fontsize=4)
        plt.ylim(0.1,2)

        ## plot our manual picks:
        # ccomp_manual_picks = [element for element in manual_pick_files if path in element.split('/')[-1]]
        if plot_disp_picks:
            if len(ccomp_manual_picks) > 1: ## if plotting both fundamental and a higher mode:
                df1=pd.read_csv(ccomp_manual_picks[0])
                plt.scatter(df1['period'].values,df1['phase velocity'].values,c='w',s=3)
                df2=pd.read_csv(ccomp_manual_picks[1])
                plt.scatter(df2['period'].values,df2['phase velocity'].values,c='w',s=3)

            else:
                df=pd.read_csv(ccomp_manual_picks[0])
                plt.scatter(df['period'].values,df['phase velocity'].values,c='w',s=3)
        
            outfname = '/mnt/data1/SEATTLE_June-2021_studies/'+f'figures/dispersion--figure6/figure6_{path}_{freqmin}-{freqmax}-Hz_{stack_method}_{maxdist}_km-maxdist--with-manual-picks--new-grid.png'
            plt.savefig(outfname, format='png', dpi=300)
            plt.cla()

        else: # not plotting dispersion picks
            plt.gcf().subplots_adjust(bottom=0.15) # add buffer so x-axis label not cutoff - natasha
            plt.tight_layout()
            outfname = '/mnt/data1/SEATTLE_June-2021_studies/'+f'figures/dispersion--figure6/figure6_{path}_{freqmin}-{freqmax}-Hz_{stack_method}_{maxdist}_km-maxdist--new-grid.png'
            plt.savefig(outfname, format='png', dpi=300)
            plt.cla()


    # output figure to pdf files
    if not os.path.isdir(os.path.join(rootpath,'figures')):
        os.mkdir(os.path.join(rootpath,'figures'))

    outfname = rootpath+'/figures/figure6_'+str(sta[ista])+'_'+stack_method+'_'+str(maxdist)+'km.pdf'
    plt.savefig(outfname, format='pdf', dpi=300)
    plt.close()

## to make a directory: if not os.path.isdir(tmp):os.mkdir(tmp)
