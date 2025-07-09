#!/usr/bin/env python3
import glob
import obspy
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal import PPSD
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pytz
import numpy as np

data_dir = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/DATA/NODES_IRIS_MSEED/"
stations = ['79', '86']
component = 'DPZ'
network = "Z6"
freq_min = 0.1
freq_max = 50.0
start_date = UTCDateTime("2019-07-16")
end_date = UTCDateTime("2019-08-11")
client = Client('IRISPH5')

# PPSD settings
ppsd_length = 300
overlap = 0.5
period_limits = (1/freq_max, 1/freq_min)

# percentiles for colorbar
lower_percentile, upper_percentile = 1, 99  # for cbar range

# -------- time axis tick formatter -- set ticks at 5 AM PST --------
TICK_WEEKDAY = 0
TICK_HOUR = 12
local_tz = pytz.timezone("America/Los_Angeles")
def tick_formatter(x, pos=0):
    dt_utc = mdates.num2date(x).replace(tzinfo=pytz.utc)
    dt_local = dt_utc.astimezone(local_tz)
    if dt_local.weekday() == TICK_WEEKDAY:
        return dt_local.strftime("%A\n%m-%d\n%-I %p PDT")
    else:
        return ""

# -------- Load data and compute PSDs for all stations --------
spec_matrices = []
freq_edges_list = []
time_edges_list = []
titles = []
all_psd_vals = []

import time       
t0_total = time.perf_counter()   

for station in stations:
    file_pattern = f"{data_dir}/IRISPH5-Z6.{station}.{component}..*.mseed"
    file_list = sorted(glob.glob(file_pattern))
    st = obspy.Stream()
    for file in file_list:
        st += read(file)
    if len(st) == 0:
        print(f"No data for station {station}. Skipping...")
        spec_matrices.append(None)
        freq_edges_list.append(None)
        time_edges_list.append(None)
        titles.append(f"Station {station}, {component} component")
        all_psd_vals.append(np.array([]))
        continue

    st.merge(method=1, interpolation_samples=0)
    # Downsample if needed
    target_sr = 110.0
    for tr in st:
        if tr.stats.sampling_rate > target_sr:
            dec_factor = int(tr.stats.sampling_rate // target_sr)
            tr.filter('lowpass', freq=target_sr/2.5, corners=4, zerophase=True)
            tr.decimate(dec_factor, no_filter=True)
            print(f"{tr.id}: Downsampled to {tr.stats.sampling_rate} Hz")

    # Get and attach response
    sta_inv = client.get_stations(network=network,
                                  station=station,
                                  location="*",
                                  starttime=start_date,
                                  endtime=end_date,
                                  level="response")
    tr0 = st[0]
    tr0.attach_response(sta_inv)

    # Compute PPSD
    ppsd = PPSD(tr0.stats, metadata=sta_inv,
                ppsd_length=ppsd_length, overlap=overlap,
                period_limits=period_limits)
    ppsd.add(st)

    psd_list = ppsd.psd_values
    if not psd_list or len(psd_list) == 0:
        print(f"No PPSD data available for plotting for station {station}")
        spec_matrices.append(None)
        freq_edges_list.append(None)
        time_edges_list.append(None)
        titles.append(f"Station {station}, {component} component")
        all_psd_vals.append(np.array([]))
        continue

    # PSD matrix (n_bins, n_segments)
    psd_matrix = np.array(psd_list).T
    # Frequency bins for PSD (convert period_bin_centers to Hz)
    freq_bins = 1.0 / ppsd.period_bin_centers
    order = np.argsort(freq_bins)
    freq_bins = freq_bins[order]
    psd_matrix = psd_matrix[order, :]
    # Mask freq range if needed
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)
    freq_bins = freq_bins[freq_mask]
    psd_matrix = psd_matrix[freq_mask, :]
    masked_spec = np.ma.masked_invalid(psd_matrix)
    # Time axis
    times = ppsd.times_processed
    time_nums = mdates.date2num([t.datetime for t in times])

    # --- Calculate frequency bin edges (log scale, geometric mean) ---
    freq_edges = np.sqrt(freq_bins[:-1] * freq_bins[1:])
    freq_edges = np.concatenate((
        [freq_bins[0] * freq_bins[0] / freq_edges[0]],
        freq_edges,
        [freq_bins[-1] * freq_bins[-1] / freq_edges[-1]]
    ))

    # --- Calculate time bin edges ---
    if len(time_nums) > 1:
        dt = np.diff(time_nums).mean()
    else:
        dt = 1  # arbitrary, won't be used if only one segment
    time_edges = np.concatenate((
        [time_nums[0] - dt/2],
        (time_nums[:-1] + time_nums[1:]) / 2,
        [time_nums[-1] + dt/2]
    ))

    spec_matrices.append(masked_spec)
    freq_edges_list.append(freq_edges)
    time_edges_list.append(time_edges)
    titles.append(f"Station {station}, {component} component")
    all_psd_vals.append(masked_spec.compressed())

# -------- Determine color limits --------
all_vals = np.concatenate([arr for arr in all_psd_vals if arr.size > 0])
shared_vmin, shared_vmax = np.percentile(all_vals, [lower_percentile, upper_percentile])
indiv_vmin = [np.percentile(vals, lower_percentile) if vals.size > 0 else None for vals in all_psd_vals]
indiv_vmax = [np.percentile(vals, upper_percentile) if vals.size > 0 else None for vals in all_psd_vals]

# --------- SHARED COLORBAR FIGURE ---------
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharey=True, sharex=True)
for ax in axs:
    ax.tick_params(labelbottom=True)

pcm_list = []
for idx, ax in enumerate(axs):
    if spec_matrices[idx] is None:
        ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center', fontsize=16, color='red')
        continue
    pcm = ax.pcolormesh(
        time_edges_list[idx], freq_edges_list[idx], spec_matrices[idx],
        cmap=plt.cm.viridis, vmin=shared_vmin, vmax=shared_vmax, shading='auto'
    )
    pcm_list.append(pcm)
    ax.set_yscale("log")
    ax.set_ylim(freq_min, freq_max)
    ax.grid(True, which='both', linestyle='--', linewidth=0.45, color='black')
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[TICK_HOUR]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_formatter))
    ax.set_xlim(time_edges_list[idx][0], time_edges_list[idx][-1])
    padval = 8 
    ax.set_title(titles[idx], fontsize=16, pad=padval)
    label_letter = "A" if idx == 0 else "B"
    ax.text(0.01, 0.08, label_letter, transform=ax.transAxes,
            fontsize=24, color="white", fontweight="bold")

# ── layout the two spectrogram axes ───────────────────────────────────────────
fig.subplots_adjust(hspace=0.31,right=0.88)
top    = axs[0].get_position().y1      # top of upper subplot
bottom = axs[-1].get_position().y0     # bottom of lower subplot
height = top - bottom                  # combined height

cbar_ax = fig.add_axes([0.90, bottom,   # left, bottom
                        0.025, height]) # width, height
cbar = fig.colorbar(pcm_list[-1], cax=cbar_ax)
cbar.set_label("Power (dB re 1 (m/s)²/Hz)", fontsize=19)   
fig.supxlabel("Time (days)", fontsize=19)   

# ── single y-axis label ───────────────────────────────
fig.text(0.07, 0.55, "Frequency (Hz)",
         rotation='vertical', fontsize=19,
         ha='center', va='center')

spectrogram_path_shared = (
    f"spectrogram_combined_PPSD_{component}_{freq_min}-{freq_max}-Hz_"
    f"shared-cbar--{lower_percentile}-{upper_percentile}--"
    f"{tr.stats.sampling_rate}-samp-rate.png"
)
fig.savefig(spectrogram_path_shared, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved (shared colorbar): {spectrogram_path_shared}")


# --------- INDIVIDUAL COLORBARS FIGURE ---------
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10), sharey=True)
for idx, ax in enumerate(axs2):
    if spec_matrices[idx] is None:
        ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center', fontsize=16, color='red')
        continue
    pcm = ax.pcolormesh(
        time_edges_list[idx], freq_edges_list[idx], spec_matrices[idx],
        cmap=plt.cm.viridis, vmin=indiv_vmin[idx], vmax=indiv_vmax[idx], shading='auto'
    )
    ax.set_yscale("log")
    ax.set_ylim(freq_min, freq_max)
    ax.grid(True, which='both', linestyle='--', linewidth=0.45, color='black')
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[TICK_HOUR]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_formatter))
    ax.set_xlim(time_edges_list[idx][0], time_edges_list[idx][-1])
    padval = 8 if idx == 0 else 30
    ax.set_title(titles[idx], fontsize=16, pad=padval)
    label_letter = "A" if idx == 0 else "B"
    ax.text(0.01, 0.08, label_letter, transform=ax.transAxes,
            fontsize=24, color="white", fontweight="bold")
    fig2.colorbar(pcm, ax=ax, orientation='vertical',
                  label="Power (dB re 1 (m/s)²/Hz)", pad=0.02, fraction=0.025)

fig2.text(0.07, 0.55, "Frequency (Hz)", rotation='vertical',
          fontsize=19, ha='center', va='center')
fig2.subplots_adjust(hspace=0.31)
spectrogram_path_indiv = f"spectrogram_combined_PPSD_{component}_{freq_min}-{freq_max}-Hz_indiv-cbar--{lower_percentile}-{upper_percentile}-percentile--{tr.stats.sampling_rate}-samp-rate.png"
fig2.savefig(spectrogram_path_indiv, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved (individual colorbars): {spectrogram_path_indiv}")

total_elapsed = time.perf_counter() - t0_total       # 
print(f"\nAll stations finished in {total_elapsed/60:.1f} min")