import sys
import glob
import os,gc
import obspy
import time
import pyasdf
import numpy as np
import noise_module
import pandas as pd
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('export HDF5_USE_FILE=FALSE')

'''
this script helps clean the sac/mseed files stored on your local machine in order to be connected 
with the NoisePy package. it is similar to the script of S0A in essence. 

by Chengxin Jiang, Marine Denolle (Jul.30.2019)
Modified by Natasha Toghramadjian (natasha_toghramadjian@g.harvard.edu), December 2024, for Seattle nodal seismic network analysis.

NOTE: 
    0. MOST occasions you just need to change parameters followed with detailed explanations to run the script. 
    1. In this script, the station of the same name but of different channels are treated as different stations.
    2. The bandpass function from obspy will output data in float64 format in default.
    3. For flexibilty to handle data of messy structures, the code loops through all sub-directory in RAWDATA and collects the
    starttime and endtime info. this enables us to find all data pieces located in each targeted time window. However, this process
    significantly slows down the code, particuarly for data of a big station list. we recommend to prepare a csv file (L48) that contains 
    all sac/mseed file names with full path and their associated starttime/endtime info if possible. based on tests, this improves the
    efficiency of the code by 2-3 orders of magnitude.
'''

#######################################################
################PARAMETER SECTION######################
#######################################################
tt0=time.time()

# data/file paths:
rootpath = '/mnt/data1/SEATTLE_June-2021_studies/'
RAWDATA='/mnt/data0/SEATTLE_NODES_DATA--2019/DATA/NODES_IRIS_MSEED' # dir where mseed/SAC files are located
DATADIR='/mnt/data0/SEATTLE_NODES_DATA/JUNE_2021_studies/NODES_asdf_data_40_Hz' # dir where cleaned data in ASDF format are going to be outputted
locations = os.path.join(rootpath,'station.txt') # station info including network,station,channel,latitude,longitude,elevation

if not os.path.isfile(locations): 
    raise ValueError('Abort! station info is needed for this script')
locs = pd.read_csv(locations)
nsta = len(locs)

# useful parameters for cleaning the data
input_fmt = 'mseed'                                                      # input file format between 'sac' and 'mseed' 
samp_freq = 40                                                          # targeted sampling rate
stationxml= False                                                       # station.XML file exists or not
rm_resp   = 'no'                                                        # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
respdir   = '/mnt/data0/SEATTLE_NODES_DATA/NODE_RESP'
freqmin   = 0.02                                                        # pre filtering frequency bandwidth
freqmax   = 10                                                           # note this cannot exceed Nquist freq
flag      = True                                                       # print intermediate variables and computing time
ncomp     = 3

# having this file saves a tons of time: see L95-126 for why
wiki_file = os.path.join(DATADIR,'allfiles_time.txt')    #ALWAYS DELETE THIS FILE AFTER EVERY RUN!!!       # file containing the path+name for all sac/mseed files and its start-end time      
allfiles_path = os.path.join(RAWDATA,'*'+'mseed')                   # make sure all sac/mseed files can be found through this format
#print(allfiles_path)
messydata = False                                                       # set this to False when daily noise data is well sorted 

# targeted time range
start_date = ['2019_07_11_0_0_0']                                       # start date of local data
end_date   = ['2019_08_14_0_0_0']                                       # end date of local data
inc_hours  = 24                                                         # sac/mseed file length for a continous recording

# get rough estimate of memory needs to ensure it now below up in S1
cc_len    = 1200                                                        # basic unit of data length for fft (s)
step      = 600                                                         # overlapping between each cc_len (s)
MAX_MEM   = 8.0                                                         # maximum memory allowed per core in GB

##################################################
# we expect no parameters need to be changed below

# assemble parameters for data pre-processing
prepro_para = {'RAWDATA':RAWDATA,'wiki_file':wiki_file,'messydata':messydata,'input_fmt':input_fmt,'stationxml':stationxml,\
    'rm_resp':rm_resp,'respdir':respdir,'freqmin':freqmin,'freqmax':freqmax,'samp_freq':samp_freq,'inc_hours':inc_hours,\
    'start_date':start_date,'end_date':end_date,'allfiles_path':allfiles_path,'cc_len':cc_len,'step':step,'MAX_MEM':MAX_MEM}
metadata = os.path.join(DATADIR,'download_info.txt') 

##########################################################
#################PROCESSING SECTION#######################
##########################################################

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    # make directory
    if not os.path.isdir(DATADIR):os.mkdir(DATADIR)
    if not os.path.isdir(RAWDATA):raise ValueError('Abort! no path of %s exists for RAWDATA'%RAWDATA)

    # output parameter info
    fout = open(metadata,'w')
    fout.write(str(prepro_para));fout.close()

    # assemble timestamp info
    allfiles = glob.glob(allfiles_path)
    all_stimes = noise_module.make_timestamps(prepro_para)

    # all time chunk for output: loop for MPI
    all_chunk = noise_module.get_event_list(start_date[0],end_date[0],inc_hours)   
    splits     = len(all_chunk)-1
    if splits<1:raise ValueError('Abort! no chunk found between %s-%s with inc %s'%(start_date[0],end_date[0],inc_hours))

    # rough estimation on memory needs needed in S1 (assume float32 dtype)
    nsec_chunk = inc_hours/24*86400
    nseg_chunk = int(np.floor((nsec_chunk-cc_len)/step))+1
    npts_chunk = int(nseg_chunk*cc_len*samp_freq)
    memory_size = nsta*npts_chunk*4/1024**3
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (memory_size,MAX_MEM))
else:
    splits,all_chunk,all_stimes,allfiles = [None for _ in range(4)]

# broadcast the variables
splits     = comm.bcast(splits,root=0)
all_chunk = comm.bcast(all_chunk,root=0)
all_stimes = comm.bcast(all_stimes,root=0)
allfiles   = comm.bcast(allfiles,root=0)

# MPI: loop through each time-chunk
for ick in range(rank,splits,size):
    t0=time.time()

    # time window defining the time-chunk
    s1=obspy.UTCDateTime(all_chunk[ick])
    s2=obspy.UTCDateTime(all_chunk[ick+1]) 
    date_info = {'starttime':s1,'endtime':s2}
    time1=s1-obspy.UTCDateTime(1970,1,1)
    time2=s2-obspy.UTCDateTime(1970,1,1) 

    # find all data pieces having data of the time-chunk
    indx1 = np.where((time1>=all_stimes[:,0]) & (time1<all_stimes[:,1]))[0]
    indx2 = np.where((time2>all_stimes[:,0]) & (time2<=all_stimes[:,1]))[0]
    indx3 = np.where((time1<=all_stimes[:,0]) & (time2>=all_stimes[:,1]))[0]
    indx4 = np.concatenate((indx1,indx2,indx3))
    indx  = np.unique(indx4)
    if not len(indx): print('continue! no data found between %s-%s'%(s1,s2));continue

    # trim down the sac/mseed file list with time in time-chunk
    tfiles = []
    if not len(indx): print('continue! no data found between %s-%s'%(s1,s2));continue
    for ii in indx:
        tfiles.append(allfiles[ii])

    # keep a track of the channels already exists
    num_records = np.zeros(nsta,dtype=np.int16)

    # filename of the ASDF file
    ff=os.path.join(DATADIR,all_chunk[ick]+'T'+all_chunk[ick+1]+'.h5')
    if not os.path.isfile(ff):
        with pyasdf.ASDFDataSet(ff,mpi=False,compression="gzip-3",mode='w') as ds:
            pass
    else:
        with pyasdf.ASDFDataSet(ff,mpi=False,mode='r') as rds:
            alist = rds.waveforms.list()
            for ista in range(nsta):
                net = locs.iloc[ista]['network']
                sta = str(locs.iloc[ista]['station'])
                print(net,sta,ista,nsta)
                tname = net+'.'+sta
                if tname in alist:
                    num_records[ista] = len(rds.waveforms[tname].get_waveform_tags())


    # loop through station
    nsta = len(locs)
    for ista in range(nsta):

        if num_records[ista] == ncomp:
            continue

        # the station info:
        network = locs.iloc[ista]['network']
        station = str(locs.iloc[ista]['station'])
        comp    = locs.iloc[ista]['channel']
        if flag: print("working on station %s channel %s" % (station,comp)) 

        # narrow down file list by using sta/net info in the file name
        ttfiles  = [ifile for ifile in tfiles if station in ifile] 
        if not len(ttfiles): continue 
        tttfiles = [ifile for ifile in ttfiles if comp in ifile]
        if not len(tttfiles): continue

        source = obspy.Stream()
        for ifile in tttfiles:
        #for ifile in tfiles:
            try:
                tr = obspy.read(ifile)
                for ttr in tr:
                    source.append(ttr)
            except Exception as inst:
                print(inst);continue
            
        # jump if no good data left
        if not len(source):continue

        # make inventory to save into ASDF file
        t1=time.time()
        inv1   = noise_module.stats2inv(source[0].stats,prepro_para,locs=locs)      
        tr = noise_module.preprocess_raw(source,inv1,prepro_para,date_info)

        if not len(tr):continue

        t2 = time.time()
        if flag:print('pre-processing takes %6.2fs'%(t2-t1))

        # appending when file exists
        with pyasdf.ASDFDataSet(ff,mpi=False,compression="gzip-3",mode='a') as ds:
            # add the inventory for all components + all time of this station         
            try:ds.add_stationxml(inv1) 
            except Exception: pass 

            tlocation = str('00')        
            new_tags = '{0:s}_{1:s}'.format(comp.lower(),tlocation.lower())
            ds.add_waveforms(tr,tag=new_tags)    
    
    t3=time.time()
    print('it takes '+str(t3-t0)+' s to process '+str(inc_hours)+'h length in step 0B')

tt1=time.time()
print('step0B takes '+str(tt1-tt0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
