#!/usr/bin/env python3
import obspy
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from obspy.signal.filter import envelope  
from obspy.geodetics import gps2dist_azimuth  

#  yes/no envelope plotting
plot_envelope = False 

data_dir = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/NODES_mseed_data_downsampled_20_Hz/"
stations_file = "station.txt"
event_date = "2019-08-03"
components = ["DP2", "DP1", "DPZ"]
minfreq = 0.5
maxfreq = 8.0

display_sec_start = 8.0  # seconds
display_sec_end = 26.0
event_time_utc = UTCDateTime("2019-08-03T22:02:03")  # UTC event time

# S wave arrival window
en_window_start, en_window_end = 14, 17 # UW SP2 S wave pick is at 14.5 sec

# station info
stations_df = pd.read_csv(stations_file, delimiter=",", dtype={"station": str})
stations_df = stations_df[stations_df["channel"] == "DP1"]
stations_df = stations_df.sort_values(by="latitude", ascending=False).reset_index(drop=True)

# latitude scaling, amplitude scaling
lat_min = stations_df["latitude"].min()
lat_max = stations_df["latitude"].max()
lat_range = lat_max - lat_min
lat_scale = 800              # need vertical spacing for visual separation
amplitude_scale = lat_range * 10  # for visibility
conversion_factor = 111.1    # approx. km per degree latitude

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0, 1], [0, 0], color='orange', lw=2, linestyle='--', label='SFZ deformation front'),
    Line2D([0, 1], [0, 0], color='red', lw=2, linestyle='--', label='SFZ frontal fault'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
           markeredgecolor='navy', markeredgewidth=1, markersize=6,
           label=f'S-wave peak amplitude ({en_window_start}–{en_window_end} s)')]

# plotting:
fig, axs = plt.subplots(
    nrows=1, ncols=3,
    figsize=(12,7),  
    sharex=True, sharey=True)

for i, component in enumerate(components):
    ax = axs[i]
    for idx, row in stations_df.iterrows():
        station = row["station"].lstrip("0")
        lat = row["latitude"]
        files = glob.glob(f"{data_dir}/**/IRISPH5-Z6.{station}.{component}..{event_date}T*.mseed", recursive=True)
        if not files:
            print(f"Skipping {station} (no file found for {event_date})")
            continue
        try:
            st = read(files[0])
            st.trim(starttime=event_time_utc - 10, endtime=event_time_utc + 60)
            st.detrend("demean")
            st.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, corners=4, zerophase=True)

            if len(st) > 0:
                trace = st[0]
                times = trace.times(reftime=event_time_utc)
                data_env = envelope(trace.data)
                norm_factor = np.percentile(np.abs(trace.data), 99)
                if norm_factor > 0:
                    normalized_trace = (trace.data / norm_factor) * amplitude_scale
                    normalized_env = (data_env / norm_factor) * amplitude_scale

                    # plot raw trace
                    ax.plot(times,
                            normalized_trace + lat * lat_scale + 10,
                            alpha=0.7, color="black", linewidth=0.3)
                    if plot_envelope:
                        ax.plot(times,
                                normalized_env + lat * lat_scale + 10,
                                alpha=0.7, color="red", linewidth=0.2)

                    # envelope peak markers
                    if component in ["DP2", "DP1"]:
                        window_indices = np.where((times >= en_window_start) & (times <= en_window_end))[0]
                        if window_indices.size > 0:
                            window_vals = normalized_env[window_indices]
                            chosen_idx = window_indices[np.argmax(window_vals)]
                            ax.plot(times[chosen_idx],
                                    normalized_env[chosen_idx] + lat * lat_scale + 10,
                                    'o', markersize=4, color='cyan',
                                    markeredgecolor='navy', markeredgewidth=.6,
                                    zorder=10)
                    # mark SFZ:
                    if 47.5776 <= lat <= 47.578:
                        ax.hlines(y=lat * lat_scale + 10,
                                  xmin=display_sec_start,
                                  xmax=display_sec_end,
                                  colors='red', linestyles='--', lw=1.5, zorder=5)
                    elif 47.5997 <= lat <= 47.6005:
                        ax.hlines(y=lat * lat_scale + 10,
                                  xmin=display_sec_start,
                                  xmax=display_sec_end,
                                  colors='orange', linestyles='--', lw=1.5, zorder=5)

        except Exception as e:
            print(f"Error processing {station}: {e}")
            continue

    ax.set_xlim(display_sec_start, display_sec_end)
    ax.set_xticks(np.arange(display_sec_start, display_sec_end + 1, 4))

    # leftmost subplot - latitude y axis
    if i == 0:
        ax.set_ylabel("Latitude (°N)", fontsize=12,labelpad=8)
        y_min = lat_min * lat_scale + 7
        y_max = lat_max * lat_scale + 12
        ax.set_ylim(y_min, y_max)
        yticks = ax.get_yticks()
        new_yticks = [f"{(yt - 10)/lat_scale:.3f}" for yt in yticks]
        ax.set_yticklabels(new_yticks, fontsize=9)
    else:
        ax.tick_params(labelleft=False)

    # rightmost subplot - distance y axis
    if i == 2:
        forward = lambda y: (((y - 10) / lat_scale) - lat_min) * conversion_factor
        inverse = lambda d: ((d / conversion_factor + lat_min) * lat_scale + 10)
        secax = ax.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel("Distance from southernmost station (km)", fontsize=12,labelpad=10)
        secax.tick_params(labelsize=10)

    ax.set_box_aspect(1.5)

    # annotate component
    annotation_text = "Z" if component == "DPZ" else "N" if component == "DP1" else "E"
    ax.text(0.95, 0.95, annotation_text,
            transform=ax.transAxes,
            fontsize=25, fontweight='bold',
            verticalalignment='top', horizontalalignment='right')

# compute backazimuth:
if "longitude" in stations_df.columns:
    avg_lat = stations_df["latitude"].mean()
    avg_lon = stations_df["longitude"].mean()
    eq_lat, eq_lon = 47.599, -121.775
    _, _, baz = gps2dist_azimuth(avg_lat, avg_lon, eq_lat, eq_lon)
    approx_baz = round(baz)  
    arrow_angle = (baz + 180) % 360
    # convert compass angle to standard math angle
    theta = np.deg2rad(90 - arrow_angle)
    inset_ax = fig.add_axes([0.81, 0.92, 0.12, 0.12])
    inset_ax.set_aspect('equal')
    arrow_length = 0.4  
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    inset_ax.annotate('', xy=(0.5 + dx, 0.5 + dy), xytext=(0.5, 0.5),
                  arrowprops=dict(arrowstyle='-|>', color='k', lw=3))
    inset_ax.text(0.5, 0.25, f"Backazimuth: ~{approx_baz}°", ha='center', va='center', fontsize=10)
    inset_ax.set_xlim(0, 1)
    inset_ax.set_ylim(0, 1)
    inset_ax.axis('off')
    inset_ax.set_clip_on(False)

fig.tight_layout(rect=[0, 0.13, 1, 0.98])
fig.subplots_adjust(wspace=0.02)
fig.suptitle("M2.8 Snoqualmie Earthquake", fontsize=15, fontweight='bold', y=0.99)
fig.supxlabel("Time after earthquake (seconds)", fontsize=12, y=0.12)

fig.legend(
    handles=legend_elements,loc='lower center',bbox_to_anchor=(0.5, 0.05),
    fontsize=11,framealpha=0.97,ncol=3)

env_tag = "plot-envelope" if plot_envelope else "no-env"
filename = (f"M2.8--arrival-peaks-maps-1x3--{minfreq}-{maxfreq}-Hz--"
            f"{display_sec_start}-{display_sec_end}-sec-EN-window-"
            f"{en_window_start}-{en_window_end}-sec--peaks-marked--{env_tag}.png")
fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close()