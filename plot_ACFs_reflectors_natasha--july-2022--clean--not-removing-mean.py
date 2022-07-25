### STATION FILE NEEDS TO BE BUILT FROM THE STACK FILES THAT ARE OUTPUT
# not just written from array.
## you must remove from the station.txt file any stations that did not have stacks output.

import obspy
from obspy import UTCDateTime
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyasdf
from obspy import Trace
from obspy import read
from obspy.signal.filter import bandpass

## get unique stacked output station list:

##acf_stacks_dir = '/mnt/data1/SEATTLE_June-2021_studies/NoisePy-master-June2021/laura-clustering-scripts/seattle-2--Oct-2021/output--March-2022/stacked--RECLASSIFIED--non-whitened_ACFs--July-2021--dist-from-hour--ZZ-only--whitened--April-2022/'

## include smoothN values:
#acf_stacks_dir = '/mnt/data1/SEATTLE_June-2021_studies/NoisePy-master-June2021/laura-clustering-scripts/seattle-2--Oct-2021/output--March-2022/stacked--RECLASSIFIED--non-whitened_ACFs--July-2021--dist-from-hour--ZZ-only--whitened--April-2022_smoothN=20/'
acf_stacks_dir = '/mnt/data1/SEATTLE_June-2021_studies/NoisePy-master-June2021/laura-clustering-scripts/seattle-2--Oct-2021/output--March-2022/stacked--RECLASSIFIED--non-whitened_ACFs--July-2021--dist-from-hour--ZZ-only--whitened--July-2022/'
cluster_files = glob.glob(acf_stacks_dir+'Z6.*.h5')

# station_list=[]
# for ii in range(len(cluster_files)):
#     station=cluster_files[ii].split('/')[-1].split('..')[0].split('Z6.')[-1]
#     station_list.append(station)
# stns=np.unique(station_list) ## list of all stations for which stacked ACF output actually exists


### universal params: remove mean, then renormalize the waveforms
## scheme 1: same freq band, and component pair. vary the study area/or transect line
## this string, and the save_dir and plt.savefig lines below are what you need to change per scheme.
scheme = "scheme_1--compare-study-area"  

# mean_setting="demeaned-renormalized"                          
mean_setting="not-demeaned" 
amplifier = 5  # multiply waveform amplitudes by this value when plotting
a_value=0.1  # parameter for plotting infilled curves
x_axis=np.linspace(-100,100,8001)   # lin space for number of data points per stack file (8001)

#study_areas = ['All_stations','West','East','transect_1','transect_2','transect_3','transect_4']
study_areas = ['All_stations']

# frequency bands are the first two elements of each row; the number of seconds to plot of the ACF is the third:
freqs_disp_seconds = np.array([[0.1,1.0,3],[1.0,10.0,1],[0.1,0.2,20],[0.2,0.5,8],[0.5,1.0,6],[1.0,2.0,4],[2.0,4.0,2],[4.0,8.0,0.5]])
cluster_labels = ['cl2','cl1','cl-4']  ## cl1=day, cl2=night, cl-4=else
cluster_filters = ['0.1-1.0Hz','1.0-10.0Hz','0.1-0.2Hz','0.2-0.5Hz','0.5-1.0Hz','1.0-2.0Hz','2.0-4.0Hz','4.0-8.0Hz']
#freqs_disp_seconds = np.array([[0.1,1.0,3]])
#cluster_labels = ['cl2']
#cluster_filters = ['0.1-1.0Hz']
channel = 'ZZ'
#figs_dir = '/mnt/data1/SEATTLE_June-2021_studies/figures/Seattle-ACF-plots/stacked_clustered_whitening-corrected_ACFs/'  ## original version
figs_dir = '/mnt/data1/SEATTLE_June-2021_studies/figures/Seattle-ACF-plots/mean-not-removed/stacked_clustered_whitening-corrected_ACFs/'  ## for mean not removed

if not os.path.isdir(figs_dir):os.makedirs(figs_dir)

for study_area in study_areas:

    station_file='/mnt/data1/SEATTLE_June-2021_studies/STATION_TEXTFILES--study_areas/station-'+study_area+'-MISSING-REMOVED.txt'
    station_file=pd.read_csv(station_file)       
    #station_file['station']==str()
    station_file_df=pd.DataFrame(station_file)
    stations=station_file_df.station.values

    station_file_df=station_file_df.drop(columns=['network','channel']) ## drop the network column so you can convert entire Dataframe to floats.
    station_file_df.latitude.apply(lambda x: float(x)) # convert pandas objects into floats
    station_file_df.longitude.apply(lambda x: float(x)) # convert pandas objects into floats

    # make array of the range of lats
    lats_array = station_file_df.latitude.values
    lat_range = (lats_array-np.min(lats_array)) / (np.max(lats_array) - np.min(lats_array))  #(array - array.min()) / (array.max() - array.min()) 
    lat_max=np.max(lats_array)
    lat_min=np.min(lats_array)
    # array of range of longitudes
    longs_array = station_file_df.longitude.values
    long_range = (longs_array-np.min(longs_array)) / (np.max(longs_array) - np.min(longs_array))
    long_min = np.min(longs_array)
    long_max = np.max(longs_array)
    
   # all_files = []

    # for stn in stations:
    #     files=glob.glob(acf_stacks_dir+'Z6*'+str(stn)+'*h5')
    #     all_files.append(files)
    #     print(len(all_files))
    # print(all_files[55:61])
    # sys.exit()
    
    for bp in cluster_filters: ## bp = string of filtering freq band. plot by freq band.
        #files_one_comp_one_bp=glob.glob(os.path.join(acf_stacks_dir,study_area+'/*DP'+channel[0]+'*DP'+channel[1]+'*'+bp+'*.h5*'))               
        for cl in cluster_labels:

            if cl=='cl1':
                cluster_name='daytime'
            if cl=='cl2':
                cluster_name='nighttime'
            if cl=='cl-4':
                cluster_name='other'

            # files of one freq band, one cluster label:
            files = glob.glob(os.path.join(acf_stacks_dir,'*'+bp+'*'+cl+'.h5'))
            #files=glob.glob(os.path.join(acf_stacks_dir,'*'+'1.0-10.0Hz*'))
            #print(len(files))
            #sys.exit()
            #cmap_lin = np.linspace(0,1,len(files))  # make a colormap linspace from 0 to 1 for the number of files
            #cmap = cm.get_cmap(name='seismic')  # monotonic cmaps: viridis, plasma, magma

            time_series=np.zeros(shape=(len(files),8001))
            d_norm=np.zeros(shape=(len(files),8001))
            #d_mean = np.zeros(shape=8001) # mean of all traces
            lats = np.zeros(shape=len(files))
            longs = np.zeros(shape=len(files))

            plt.cla()
            plt.close()
            plt.figure(figsize=(20, 12))   

            for i, ff in enumerate(files):    # looping through all station pairs (each .h5 file being one station pair)
                
                stn_name_int=int(ff.split('.')[1]) 

                if stn_name_int not in stations:
                    print("Station ",str(stn_name_int),' is not in study area--skipping.')
                    continue
                print("Station ",str(stn_name_int),' IS in study area ',study_area,'--processing.')

                stn_1=ff.split('/')[-1].split('..')[0] 
                stn_2=ff.split('/')[-1].split('..')[1].split('--')[-1] 

                qq_index=np.where(station_file_df.station.values==stn_name_int)
                lats[i]=station_file_df.latitude.values[qq_index][0]
                longs[i]=station_file_df.longitude.values[qq_index][0]
                
                # now access the correlation function:
                data_h5=h5py.File(ff)
                try:
                    time_series[i] = data_h5['corr_windows']['data'][0][0:]
                    #timestamps=data_h5['corr_windows']['timestamps'][0:][0]
                except: continue

                #time_series[i]=aux.data[0:]    # raw waveform data (length 8001)
                tr=Trace(time_series[i])
                tr.stats.sampling_rate=40.0    # set the sampling rate in the metadata
                time_series[i]=tr 
                #print(time_series[i].shape)
                #sys.exit()
                time_series[i] /= np.max(np.abs(time_series[i]))  # normalize the waveform

            # d_mean = np.mean(time_series,axis=0) # take the average of all normalized waveforms, considering each row as a bulk

            for i, ff in enumerate(files):      
                stn_name_int=int(ff.split('.')[1])               
                if stn_name_int not in stations:
                    #print("Station ",str(stn_name_int),' is not in study area--skipping.')
                    continue
                #print("Station ",str(stn_name_int),' IS in study area--processing.')
                #print(station_file_df.loc[station_file_df['station']==stn_name_int])  ## get the whole row of that station from the station txt file

                ## only analyze component pairs we are interested (set in parameters at beginning of script):
                first_comp=ff.split('DP')[1][0] ## get first channel from file name
                second_comp=ff.split('DP')[-1][0]  ## get second channel from file name
                chanchan=first_comp+second_comp  ## combine strings to get comp pair
                if chanchan != channel:
                    print('CHANNEL MISMATCH--should be analyzing '+channel+' but this file is '+chanchan)
                    sys.exit()

                ## extract the frequency band from the file name:    
                cluster_bandpassed=ff.split('_')[-2] # outputs 'bp0' through 'bp5'
                print('working freq band is: '+cluster_bandpassed)
                
                if cluster_bandpassed==cluster_filters[0]: ## here
                    freqmin=freqs_disp_seconds[0][0]
                    freqmax=freqs_disp_seconds[0][1]
                    display_seconds=freqs_disp_seconds[0][2]
                if cluster_bandpassed==cluster_filters[1]:
                    freqmin=freqs_disp_seconds[1][0]
                    freqmax=freqs_disp_seconds[1][1]
                    display_seconds=freqs_disp_seconds[1][2]
                if cluster_bandpassed==cluster_filters[2]:
                    freqmin=freqs_disp_seconds[2][0]
                    freqmax=freqs_disp_seconds[2][1]
                    display_seconds=freqs_disp_seconds[2][2]
                if cluster_bandpassed==cluster_filters[3]:
                    freqmin=freqs_disp_seconds[3][0]
                    freqmax=freqs_disp_seconds[3][1]
                    display_seconds=freqs_disp_seconds[3][2]
                if cluster_bandpassed==cluster_filters[4]:
                    freqmin=freqs_disp_seconds[4][0]
                    freqmax=freqs_disp_seconds[4][1]
                    display_seconds=freqs_disp_seconds[4][2]
                if cluster_bandpassed==cluster_filters[5]:
                    freqmin=freqs_disp_seconds[5][0]
                    freqmax=freqs_disp_seconds[5][1]
                    display_seconds=freqs_disp_seconds[5][2]
                if cluster_bandpassed==cluster_filters[6]:
                    freqmin=freqs_disp_seconds[6][0]
                    freqmax=freqs_disp_seconds[6][1]
                    display_seconds=freqs_disp_seconds[6][2]
                if cluster_bandpassed==cluster_filters[7]:
                    freqmin=freqs_disp_seconds[7][0]
                    freqmax=freqs_disp_seconds[7][1]
                    display_seconds=freqs_disp_seconds[7][2]

                # demean and renormalize the waveform
                # demeaned=time_series[i]-d_mean               
                # demeaned_norm = demeaned/np.max(np.abs(demeaned)) # first, normalize demeaned waveforms again

                print(str(stn_name_int),channel,freqmin,freqmax,cluster_bandpassed)

                # plt.fill_between(amplifier*(demeaned_norm[4000:])+((lats[i]-47.5)*1112.5),x_axis[4000:],color='k',alpha=a_value) # plotting demeaned
                plt.fill_between(amplifier*(time_series[i][4000:])+((lats[i]-47.5)*1112.5),x_axis[4000:],color='k',alpha=a_value) # plotting not demenaed
                # plt.title("Linear stack ACFs, "+str(study_area)+', '+str(channel)+" component, mean removed, renormalized, "+str(freqmin)+'-'+str(freqmax)+' Hz, cluster '+str(cl.split('l')[-1])+', '+cluster_name+', amplified x'+str(amplifier)) # for demeaned and renormalized
                plt.title("Linear stack ACFs, "+str(study_area)+', '+str(channel)+" component, mean not removed, "+str(freqmin)+'-'+str(freqmax)+' Hz, cluster '+str(cl.split('l')[-1])+', '+cluster_name+', amplified x'+str(amplifier)) # for not demeaned
                plt.annotate(s='SOUTH',xy=[68,0],size=20);plt.annotate('NORTH',xy=[129,0],size=20) ## label cross-section with north and south                                                   
                plt.ylim(display_seconds,0)   # number of seconds of the correlation function to plot
                plt.grid(True); plt.ylabel("Time (s)"); plt.xlabel("Scaled N-S Position")
                fig_to_save=plt.gcf()  

                #save_dir = figs_dir+scheme+'/'+str(freqmin)+'-'+str(freqmax)+'-Hz/'+channel+'/amplified_x'+str(amplifier)+'/'
                save_dir = figs_dir+scheme+'/'+str(freqmin)+'-'+str(freqmax)+'-Hz/'+channel+'/'+cl+'_'+cluster_name+'/'
                if not os.path.isdir(save_dir):os.makedirs(save_dir)
                print(save_dir)
                ## save in specific sub-dirs:
                # plt.savefig(save_dir+'ACFs_plot_'+str(freqmin)+'-'+str(freqmax)+'-Hz_'+mean_setting+'_'+str(channel)+'_'+study_area+'_'+cl+'_'+cluster_name+'_'+str(display_seconds)+'-sec_x'+str(amplifier)+'.png',orientation='landscape')
                ## save all in a single directory, figs_dir as defined above:
                plt.savefig(figs_dir+'ACFs_plot_'+str(freqmin)+'-'+str(freqmax)+'-Hz_'+mean_setting+'_'+str(channel)+'_'+study_area+'_'+cl+'_'+cluster_name+'_'+str(display_seconds)+'-sec_x'+str(amplifier)+'.png',orientation='landscape')
                #plt.show()
                #sys.exit()
            plt.cla()
            #sys.exit()

