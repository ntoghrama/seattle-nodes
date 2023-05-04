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
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import pyasdf
from obspy import Trace
from obspy import read
from obspy.signal.filter import bandpass

acf_stacks_dir = '/mnt/data1/SEATTLE_June-2021_studies/NoisePy-master-June2021/laura-clustering-scripts/seattle-2--Oct-2021/output--March-2022/stacked--RECLASSIFIED--non-whitened_ACFs--July-2021--dist-from-hour--ZZ-only--FIXED-whitening--August-2022/'
cluster_files = glob.glob(acf_stacks_dir+'Z6.*.h5')
scheme = "scheme_1--compare-study-area"  
mean_setting="demeaned-renormalized"                          
amplifier = 5  # multiply waveform amplitudes by this value when plotting
a_value=0.1  # parameter for plotting infilled curves
x_axis=np.linspace(-100,100,8001)   # lin space for number of data points per stack file (8001)
#study_areas = ['All_stations','West','East','transect_1','transect_2','transect_3','transect_4']
study_areas = ['All_stations']
freqs_disp_seconds = np.array([[0.1,1.0,10],[1.0,10.0,1],[0.1,0.2,20],[0.2,0.5,8],[0.5,1.0,6],[1.0,2.0,4],[2.0,4.0,2],[4.0,8.0,0.5]])
# cluster_filters = ['0.1-1.0Hz','1.0-10.0Hz','0.1-0.2Hz','0.2-0.5Hz','0.5-1.0Hz','1.0-2.0Hz','2.0-4.0Hz','4.0-8.0Hz']
cluster_filters = ['1.0-10.0Hz','0.1-0.2Hz','0.2-0.5Hz','0.5-1.0Hz','1.0-2.0Hz','2.0-4.0Hz','4.0-8.0Hz']

cluster_labels = ['cl1','cl2','cl-4']  ## cl1=day, cl2=night, cl-4=else
channel = 'ZZ'
figs_dir = '/mnt/data1/SEATTLE_June-2021_studies/figures/Seattle-ACF-plots/stacked_clustered_whitening-corrected_ACFs--August-2022/' 
if not os.path.isdir(figs_dir):os.makedirs(figs_dir)

## to mark latitudes of SFZ surface features on plot:
SFZ_deformation_front_stn = 76
SFZ_main_fault_trace_stn = 930

for study_area in study_areas:

    station_file='/mnt/data1/SEATTLE_June-2021_studies/STATION_TEXTFILES--study_areas/station.txt'
    station_file=pd.read_csv(station_file)       
    station_file_df=pd.DataFrame(station_file)
    stations=station_file_df.station.values

    station_file_df=station_file_df.drop(columns=['network','channel']) ## drop the network column so you can convert entire Dataframe to floats.
    station_file_df.latitude.apply(lambda x: float(x)) # convert pandas objects into floats
    station_file_df.longitude.apply(lambda x: float(x)) # convert pandas objects into floats

    # make arrays of the ranges of the lat and long values:
    lats_array = station_file_df.latitude.values

    lat_range = (lats_array-np.min(lats_array)) / (np.max(lats_array) - np.min(lats_array))  #(array - array.min()) / (array.max() - array.min()) 
    lat_max=np.max(lats_array)
    lat_min=np.min(lats_array)
    longs_array = station_file_df.longitude.values
    long_range = (longs_array-np.min(longs_array)) / (np.max(longs_array) - np.min(longs_array))
    long_min = np.min(longs_array)
    long_max = np.max(longs_array)
       
    for bp in cluster_filters: ## bp = string of filtering freq band. plot by freq band.
        for cl in cluster_labels:

            if cl=='cl1':
                cluster_name='daytime'
            if cl=='cl2':
                cluster_name='nighttime'
            if cl=='cl-4':
                cluster_name='other'

            files = glob.glob(os.path.join(acf_stacks_dir,'*'+bp+'*'+cl+'.h5'))   # files of one freq band, one cluster label:      
            time_series=[] ## instead of an array, initiate a list.

            d_norm=np.zeros(shape=(len(files),8001))
            d_mean = np.zeros(shape=8001) # mean of all traces
            lats=[]
            longs=[]

            stations_for_pcolor=[]

            plt.cla()
            plt.close()
            # plt.figure(figsize=(15,20))
            plt.figure(figsize=(8,20))
            plt.tight_layout()

            for i, ff in enumerate(files):    # looping through all station pairs (each .h5 file being one station pair)
                # now access the correlation function:
                try:
                    data_h5=h5py.File(ff,mode='r')
                    corr_windows=data_h5['corr_windows']['data'][0][0:]
                except: continue
                
                stn_name_int=int(ff.split('.')[1]) 

                if stn_name_int not in stations:
                    print("Station ",str(stn_name_int),' is not in study area--skipping.')
                    continue
                
                # # for plotting SFZ surface features:
                # if stn_name_int==SFZ_deformation_front_stn:

                # if stn_name_int==SFZ_main_fault_trace_stn:

                stn_1=ff.split('/')[-1].split('..')[0] 
                stn_2=ff.split('/')[-1].split('..')[1].split('--')[-1] 

                qq_index=np.where(station_file_df.station.values==stn_name_int)
                lats.append(station_file_df.latitude.values[qq_index][0])
                longs.append(station_file_df.latitude.values[qq_index][0])

                stations_for_pcolor.append(int(stn_1.split('.')[-1]))

                tr = corr_windows  # set all indexing to be based on 8001 points; if option in next line, it's 4001 points.
                tr /= np.max(np.abs(tr[0:]))  # normalize the waveform
                
                time_series.append(tr)
                if tr[4000]!=1:
                    print('maximum amplitude of ACF is not at midpoint. exiting')
                    sys.exit()
                #print(time_series[i][4000]) # test
                #plt.plot(tr); plt.show(); sys.exit() # test

            time_series=np.array(time_series) # convert list to array
            d_mean = np.mean(time_series,axis=0) # take the average of all normalized waveforms, considering each row as a bulk
            
            #print(d_mean[4000]==1) # amplitude of the ACF at the center of the vector should = 1 -- i.e. the maximum amplitude should be at the center. output should be True
            if d_mean[4000]!=1:
                print('maximum amplitude of ACF is not at midpoint. exiting');sys.exit()

            plot_list=[]  # for using pcolormesh
            
            for i, ff in enumerate(files):      
                stn_name_int=int(ff.split('.')[1])               
                if stn_name_int not in stations:
                    #print("Station ",str(stn_name_int),' is not in study area--skipping.')
                    continue
                ## only analyze component pairs we are interested (set in parameters at beginning of script):
                first_comp=ff.split('DP')[1][0] ## get first channel from file name
                second_comp=ff.split('DP')[-1][0]  ## get second channel from file name
                chanchan=first_comp+second_comp  ## combine strings to get comp pair
                if chanchan != channel: # check that channel is correct:
                    print('CHANNEL MISMATCH--should be analyzing '+channel+' but this file is '+chanchan);sys.exit()
                cluster_bandpassed=ff.split('_')[-2] ## extract the frequency band from the file name: outputs 'bp0' through 'bp5'
                
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
                try:
                    demeaned=time_series[i]-d_mean       
                except: continue
   
                demeaned_norm = demeaned/np.max(np.abs(demeaned)) # first, normalize demeaned waveforms again
                
                plot_list.append(demeaned_norm[4000:])


            plot_list=np.array(plot_list) ## this is the list of waveforms sorted by file. same sort as array 'lats'        


            ## sort list of data by a list of latitudes, and then sort the latitudes themselves after:
            waveforms_by_lat=[p for _,p in sorted(zip(lats,plot_list))]   ## this is the array of DATA sorted by lat
            lats_sorted=sorted(lats) ## use sorted latitudes as x-axis
            indices_lats=np.arange(lats_sorted[0],lats_sorted[-1],0.02)
            station_indices = np.arange(len(stations_for_pcolor))

            station_grid, lag_grid = np.meshgrid(station_indices,x_axis[4000:],indexing='ij')
            plt.pcolormesh(station_grid, lag_grid, waveforms_by_lat, cmap='seismic') 

            ## to plot SFZ surface features along x-axis -- find the INDEX of the station that aligns with each feature:
            SFZ_deformation_front_stn = 76
            SFZ_main_fault_trace_stn = 930
            x_def_front=47.596494
            x_fault_trace=47.577676
            def_front_idx = lats_sorted.index(x_def_front)
            fault_trace_idx = lats_sorted.index(x_fault_trace)
            # print(def_front_idx)
            # print(fault_trace_idx)
            # sys.exit()

            plt.plot(def_front_idx, 1, marker="D", markersize=12, markerfacecolor='yellow',markeredgecolor='y')
            # plt.plot(fault_trace_idx, 0, marker="D", markersize=8, color='limegreen')

            # print(np.min(lats_sorted),np.max(lats_sorted));sys.exit()

            # to have lag time axis instead of points:
            # lag_time = points / sampling rate
            num_data_points = len(plot_list[0])
            sampling_rate=40.
            # lag_time_yaxis=(num_data_points-1)/sampling_rate

            ax=plt.gca()
            ax.invert_yaxis()
            ax.set_title(str(channel)+" autocorrelation functions sorted by station latitude, "+str(freqmin)+'-'+str(freqmax)+' Hz', pad=14,fontsize=16) ## add spacing between plot title and plot
            print(cl+'_'+cluster_name)
            print(str(channel)+" autocorrelation functions sorted by station latitude, "+str(freqmin)+'-'+str(freqmax)+' Hz')
            print(np.min(lats_sorted),np.max(lats_sorted))
            print('mid array lat is: ')
            print((np.max(lats_sorted)+np.min(lats_sorted))/2)
            # ttl = ax.title
            # ttl.set_position([.5, 1.05])
            # plt.title(str(channel)+" autocorrelation functions sorted by station latitude, "+str(freqmin)+'-'+str(freqmax)+' Hz',fontsize=16)


            # print(np.arange(np.min(lats),np.max(lats),0.01))
            # plt.gcf().subplots_adjust(bottom=0.15) # add buffer so x-axis label not cutoff - natasha

            # plt.xticks(ticks=np.arange(np.min(lats),np.max(lats),0.01),fontsize=4) ### wrong
            # plt.xticks(np.arange(np.min(lats),np.max(lats),0.1),visible=True) ### wrong
            plt.xlabel('Station latitude',fontsize=12)

            plt.yticks(ticks=np.arange(0,num_data_points/sampling_rate,10),fontsize=12)
            plt.ylabel('Lag time (s)',fontsize=12)
            # plt.annotate('NORTH',xy=[86,0],size=20)
            # plt.annotate('SOUTH',xy=[0,0],size=20)
            # plt.title("Linear stack ACFs, "+str(study_area)+', '+str(channel)+" component, mean removed, renormalized, "+str(freqmin)+'-'+str(freqmax)+' Hz, cluster '+str(cl.split('l')[-1])+', '+cluster_name)
            # plt.title("Autocorrelation functions sorted by station latitude, "+str(channel)+", "+str(freqmin)+'-'+str(freqmax)+' Hz, '+cluster_name+' cluster ',fontsize=14)
            # plt.title(str(channel)+" autocorrelation functions sorted by station latitude, "+str(freqmin)+'-'+str(freqmax)+' Hz',fontsize=16)
            # plt.savefig('/home/natasha/Downloads/test-SFZ-figs/ACFs_plot_'+str(freqmin)+'-'+str(freqmax)+'-Hz_'+mean_setting+'_'+str(channel)+'_'+study_area+'_'+cl+'_'+cluster_name+'_'+str(display_seconds)+'-sec.png',orientation='landscape')
            plt.savefig('/mnt/data1/SEATTLE_June-2021_studies/figures/Seattle-ACF-plots/stacked_clustered_whitening-corrected_ACFs--August-2022/heatmap-ACF-plots/ACFs_plot_'+str(freqmin)+'-'+str(freqmax)+'-Hz_'+mean_setting+'_'+str(channel)+'_'+study_area+'_'+cl+'_'+cluster_name+'_'+'with-deformation-front.png',orientation='landscape')
            # plt.show()
            # sys.exit()
            plt.cla()
