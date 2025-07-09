#!/usr/bin/env python3
import os
import gc
import pickle
import glob
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read
import pandas as pd
from obspy.signal import PPSD
import numpy as np

client = Client("IRISPH5")

# Define date range for processing data
start_date = UTCDateTime("2019-07-14")
end_date = UTCDateTime("2019-08-10")

data_dir = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/DATA/NODES_IRIS_MSEED"
file_pattern = f"{data_dir}/IRISPH5-Z6.*..*.mseed"
files = sorted(glob.glob(file_pattern))

# frequencies of interest
freqs = [3.0, 5.0, 13.0, 10.0, 2.0, 3.5, 4.0, 12.0, 15.0]

# downsample to 40 Hz
desired_sampling_rate = 40.0

# processed PPSD objects in files
ppsd_pickle_file = "ppsd_dict.pkl"
processed_files_file = "processed_files.pkl"

# load saved PPSD objects and list of processed files if available
if os.path.exists(ppsd_pickle_file) and os.path.exists(processed_files_file):
    with open(ppsd_pickle_file, "rb") as f:
        ppsd_dict = pickle.load(f)
    with open(processed_files_file, "rb") as f:
        processed_files = pickle.load(f)
    print(f"Loaded saved PPSD objects from {ppsd_pickle_file}.")
    print(f"Loaded list of processed files from {processed_files_file}.")
else:
    ppsd_dict = {}
    processed_files = []

# filter files by date and remove those already processed
file_list = []
for f in files:
    try:
        file_date_str = f.split(".")[-2][:10]
        file_date = UTCDateTime(file_date_str)
    except Exception as e:
        print(f"Skipping {f} due to date parsing error: {e}")
        continue
    if start_date <= file_date <= end_date and f not in processed_files:
        file_list.append((f, file_date))

print(f"Total new files to process: {len(file_list)}")

# process each file and add its trace to the appropriate aggregated PPSD
for idx, (f, file_date) in enumerate(file_list):
    try:
        st = read(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

    # downsample:
    if st[0].stats.sampling_rate > desired_sampling_rate:
        factor = int(st[0].stats.sampling_rate / desired_sampling_rate)
        st.decimate(factor, no_filter=False)
        new_rate = st[0].stats.sampling_rate
        print(f"Downsampled {f} by factor {factor}, new sampling rate ~{new_rate:.2f} Hz")

    # extract metadata from the first trace
    sta = st[0].stats.station
    net = st[0].stats.network
    cha = st[0].stats.channel
    key = (sta, cha)
    
    if key not in ppsd_dict:
        try:
            inv = client.get_stations(network=net, station=sta,
                                      starttime=file_date, endtime=file_date + 86400.*40,
                                      level="response")
        except Exception as e:
            print(f"Error fetching metadata for {sta}: {e}")
            del st
            continue

        # create a new PPSD object and add the trace
        ppsd_obj = PPSD(st[0].stats, ppsd_length=300., metadata=inv)
        ppsd_obj.add(st)
        # save station metadata
        lat = inv.networks[0].stations[0].latitude
        lon = inv.networks[0].stations[0].longitude
        ppsd_dict[key] = {"ppsd": ppsd_obj, "lat": lat, "lon": lon}
    else:
        # add trace to the existing PPSD object
        ppsd_dict[key]["ppsd"].add(st)
    
    del st
    processed_files.append(f)  # mark file as processed

    # every 50 files, print a checkpoint and save progress
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx+1} out of {len(file_list)} new files.")
        gc.collect()
        # save current PPSD objects and processed file list
        with open(ppsd_pickle_file, "wb") as f_out:
            pickle.dump(ppsd_dict, f_out)
        with open(processed_files_file, "wb") as f_out:
            pickle.dump(processed_files, f_out)
        print("Intermediate results saved to disk.")

print("Finished processing all new files and building PPSD objects.")

# save final PPSD dictionary and processed file list
with open(ppsd_pickle_file, "wb") as f_out:
    pickle.dump(ppsd_dict, f_out)
with open(processed_files_file, "wb") as f_out:
    pickle.dump(processed_files, f_out)
print("Final PPSD objects and processed file list saved.")

# for each frequency, compute the median PSD for every station/component and save as .csv
date_range_str = f"{start_date.datetime.strftime('%Y-%m-%d')}_to_{end_date.datetime.strftime('%Y-%m-%d')}"
for freq in freqs:
    results = []
    period = 1.0 / freq
    print(f"\nProcessing frequency {freq} Hz (period = {period:.3f} s)...")
    for (sta, cha), data in ppsd_dict.items():
        ppsd_obj = data["ppsd"]
        psd_values, period_min, _, period_max = ppsd_obj.extract_psd_values(period)
        median_psd = np.median(psd_values)
        print(f"Station {sta} {cha} at {freq}Hz: median PSD = {median_psd}")
        result = {
            "station": sta,
            "component": cha,
            "lat": data["lat"],
            "lon": data["lon"],
            "median_psd": median_psd
        }
        results.append(result)
    
    # create df and save results for this freq:
    df = pd.DataFrame(results)
    output_csv = f"lat_lon_psdlevels--freq-{freq}Hz--{date_range_str}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved results for frequency {freq}Hz to {output_csv}")
    gc.collect()

print("\nAll frequency results processed and saved.")
