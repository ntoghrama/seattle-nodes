#!/usr/bin/env python3
import glob
import os
import obspy
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import osmnx as ox
import matplotlib.colors as colors
import geopandas as gpd
from shapely.geometry import box
from obspy.signal.filter import envelope
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def main():
    quake = "M2.8" # this variable is "M2.8" or "M3.0"

    data_dir = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/NODES_mseed_data_downsampled_20_Hz/"
    stations_file = '/mnt/data1/SEATTLE_June-2021_studies/station.txt'
    if quake == "M2.8":
        event_date = "2019-08-03"
        event_time_utc = UTCDateTime("2019-08-03T22:02:03")
        window_start, window_end = 13, 15.5 # UW.SP2 S wave pick is at 14.5 sec; P wave is between 9-10 sec
        
    elif quake == "M3.0":
        event_date = "2019-07-17"
        event_time_utc = UTCDateTime("2019-07-17T22:02:03")
        window_start, window_end = 15, 17.5 

    components = ["DP2", "DP1", "DPZ"]
    minfreq = 0.5
    maxfreq = 8.0
    cbar_cap = 11   # which value to cap the colorbar with
    capped_lower = "natural"   
    NORM_PCTL = 0   # which percentile do you want to normalize the amplification by? set to 0 if you want to norm by the minimum

    # --- READ STATION INFORMATION ---
    stations_df = pd.read_csv(stations_file, delimiter=",", dtype={"station": str})
    stations_df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    # Use one record per station (using DP1 channel as a proxy for location)
    stations_df = stations_df[stations_df["channel"] == "DP1"].copy()
    stations_df = stations_df.sort_values(by="lat", ascending=False).reset_index(drop=True)

    # --- PREPARE CONTAINER FOR PEAK VALUES ---
    results = {
        "DP2": {"lon": [], "lat": [], "peak": []},  # E component
        "DP1": {"lon": [], "lat": [], "peak": []},  # N component
        "DPZ": {"lon": [], "lat": [], "peak": []}}   # Z component

    # --- LOOP OVER STATIONS & COMPONENTS TO PICK PEAK AMPLITUDES ---
    for idx, row in stations_df.iterrows():
        station = row["station"].lstrip("0")
        lat = row["lat"]
        lon = row["lon"]
        for comp in components:
            pattern = f"{data_dir}/**/IRISPH5-Z6.{station}.{comp}..{event_date}T*.mseed"
            files = glob.glob(pattern, recursive=True)
            if not files:
                print(f"Skipping station {station} for component {comp} (no file found)")
                continue
            try:
                st = read(files[0])
                st.trim(starttime=event_time_utc - 10, endtime=event_time_utc + 60)
                st.detrend("demean")
                st.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, corners=4, zerophase=True)
                if len(st) == 0:
                    continue
                trace = st[0]
                times = trace.times(reftime=event_time_utc)
                # normalize trace by its 99th percentile amplitude to reduce outlier influence
                norm_factor = np.percentile(np.abs(trace.data), 99)
                if norm_factor == 0:
                    continue
                env_data = envelope(trace.data)               # compute amplitude envelope of the trace
                normalized_env = env_data / norm_factor       # normalize envelope by 99th percentile

                # find peak amplitude within the S-wave arrival window
                window_indices = np.where((times >= window_start) & (times <= window_end))[0]
                if window_indices.size > 0:
                    window_vals = normalized_env[window_indices]
                    peak_amp = np.max(window_vals)            # maximum envelope amplitude in window
                    energy = peak_amp ** 2                    # square it to represent value proportional to energy 
                    results[comp]["lon"].append(lon)
                    results[comp]["lat"].append(lat)
                    results[comp]["peak"].append(energy)
            except Exception as e:
                print(f"Error processing station {station}, component {comp}: {e}")
                continue

    # --- SAVE NORMALIZED PEAK AMPLITUDES TO CSV ---
    csv_rows = []
    for comp in results:
        comp_peaks = np.array(results[comp]["peak"])
        if comp_peaks.size == 0:
            continue
        ref = np.nanpercentile(comp_peaks, NORM_PCTL)
        print(f"{comp}: {NORM_PCTL}th percentile energy = {ref:.4g}")
        norm_peaks = comp_peaks if (not np.isfinite(ref) or ref == 0) else comp_peaks / ref

        for lon_val, lat_val, energy_val, norm_val in zip(results[comp]["lon"], results[comp]["lat"], comp_peaks, norm_peaks):
            csv_rows.append({
                "component": comp,
                "lon": lon_val,
                "lat": lat_val,
                "peak_energy": energy_val,
                "normalized_peak_energy": norm_val})
    csv_df = pd.DataFrame(csv_rows)
    csv_filename = f"{quake}--normalized_peak_amps.csv"
    csv_df.to_csv(csv_filename, index=False)
    print(f"Normalized peak amplitudes saved to {csv_filename}")

    # --- SET UP INTERPOLATION GRID ---
    lons_all = stations_df["lon"].values
    lats_all = stations_df["lat"].values
    lon_min, lon_max = lons_all.min(), lons_all.max()
    lat_min, lat_max = lats_all.min(), lats_all.max()
    lon_margin = 0.1 * (lon_max - lon_min)
    lat_margin = 0.1 * (lat_max - lat_min)
    grid_lon_vals = np.linspace(lon_min - lon_margin, lon_max + lon_margin, 100)
    grid_lat_vals = np.linspace(lat_min - lat_margin, lat_max + lat_margin, 100)
    grid_lon, grid_lat = np.meshgrid(grid_lon_vals, grid_lat_vals)

    # --- GET OSM ROAD EDGES FOR BACKGROUND MAP ---
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    distance_meters = 5000
    G = ox.graph_from_point((center_lat, center_lon), dist=distance_meters, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)

    # --- LOAD AND CLIP SFZ SHAPEFILES ---
    shp_file1 = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/SHP/Qfaults_US_Database.shp"
    shp_file2  = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/sfz-deformation-front.shp"
    try:
        gdf1 = gpd.read_file(shp_file1)
        gdf2 = gpd.read_file(shp_file2)
        if gdf1.crs is None or gdf1.crs.to_string() != "EPSG:4326":
            gdf1 = gdf1.to_crs(epsg=4326)
        if gdf2.crs is None or gdf2.crs.to_string() != "EPSG:4326":
            gdf2 = gdf2.to_crs(epsg=4326)
        clip_box = box(grid_lon_vals[0], grid_lat_vals[0], grid_lon_vals[-1], grid_lat_vals[-1])
        clipped_gdf1 = gdf1[gdf1.intersects(clip_box)]
        clipped_gdf2 = gdf2[gdf2.intersects(clip_box)]
    except Exception as e:
        print(f"Error loading shapefiles: {e}")
        clipped_gdf1 = None
        clipped_gdf2 = None

    # --- PLOT COMBINED MAPS FOR EACH COLORBAR OPTION ---
    maps_order = ["DP2", "DP1", "DPZ"]  # plot order: E N Z
    for cbar_option in ["capped", "natural"]:
        if cbar_option == "capped":
            cap_value = cbar_cap
            title_suffix = "(Capped Colorbar)"
        else:
            cap_value = None
            title_suffix = "(Natural Colorbar Range)"
        fig, axs = plt.subplots(1, len(maps_order), figsize=(4 * len(maps_order), 9),
                                 sharex=True, sharey=True, constrained_layout=(cbar_option == "natural"))
        if cbar_option == "capped":
            shared_norm   = colors.Normalize(vmin=1.0, vmax=cbar_cap, clip=True)
            shared_levels = np.linspace(1.0, cbar_cap, 101)
        fig.suptitle(f"Ratio of peak S-wave amplitudes from {quake} event arrivals, normalized by {NORM_PCTL} percentile",
                     fontsize=14, y=0.97)
        for ax, key in zip(axs, maps_order):
            comp_label = "Z" if key == "DPZ" else ("N" if key == "DP1" else ("E" if key == "DP2" else key))
            lons = np.array(results[key]["lon"])
            lats = np.array(results[key]["lat"])
            peaks = np.array(results[key]["peak"])
            # filter out NaNs or missing values
            valid = ~np.isnan(peaks)
            if valid.sum() == 0:
                ax.set_title(f"{comp_label}: no data")
                continue
            lons = lons[valid]
            lats = lats[valid]
            peaks = peaks[valid]
            ref = np.nanpercentile(peaks, NORM_PCTL)
            norm_peaks = peaks if (not np.isfinite(ref) or ref == 0) else peaks / ref

            # define color scale range
            # grid interpolation for contour map
            grid_peaks = griddata((lons, lats), norm_peaks, (grid_lon, grid_lat), method='linear')
            if cbar_option == "capped":
                grid_peaks_masked = np.ma.masked_less(grid_peaks, 1.0)
                contour = ax.contourf(
                    grid_lon, grid_lat, grid_peaks_masked,
                    levels=shared_levels, cmap="plasma", norm=shared_norm, zorder=1)
            else:
                # NATURAL full range of cbar: auto range per subplot
                vmin = np.nanmin(grid_peaks)
                vmax = np.nanmax(grid_peaks)
                levels = np.linspace(vmin, vmax, 101)
                contour = ax.contourf(
                    grid_lon, grid_lat, grid_peaks,
                    levels=levels, cmap="plasma", zorder=1)
                cb = fig.colorbar(contour, ax=ax, orientation='vertical',shrink=0.70, pad=0.02)
                cb.set_label("Ratio of peak S-wave amplitudes", fontsize=10)
                cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                cb.update_ticks()

            # Plot roads and SFZ
            edges.plot(ax=ax, color='black', linewidth=0.7, alpha=1.0, zorder=2)
            if clipped_gdf1 is not None and not clipped_gdf1.empty:
                clipped_gdf1.plot(ax=ax, facecolor="none", edgecolor='black', linewidth=3, zorder=2.5)
            if clipped_gdf2 is not None and not clipped_gdf2.empty:
                clipped_gdf2.plot(ax=ax, facecolor="none", edgecolor='black', linewidth=3, zorder=2.5)
            # plot stations 
            ax.scatter(lons, lats, color='cyan', edgecolors='black', linewidths=0.3,
                       marker='o', s=22, zorder=3)
            ax.set_xlim(grid_lon_vals[0], grid_lon_vals[-1])
            ax.set_ylim(grid_lat_vals[0], grid_lat_vals[-1])
            ax.text(
                0.04, 0.97,             
                comp_label,              
                transform=ax.transAxes,  
                fontsize=34,              
                fontweight='bold',       
                va='top', ha='left')     
            ax.tick_params(labelsize=10)
        fig.supxlabel("Longitude", fontsize=15, y=0.03)
        fig.supylabel("Latitude", fontsize=15,x=0.03)

        for ax in axs:
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            new_labels = [f"{t:.2f}" if i % 2 == 1 else "" for i, t in enumerate(ticks)]
            ax.set_xticklabels(new_labels)
            ax.tick_params(axis="x", labelsize=10, rotation=0)

        # build filename & shared capped colorbar 
        if cbar_option == "capped":
            fig.subplots_adjust(right=0.82, bottom=0.12)
            cax = fig.add_axes([0.87, 0.09, 0.03, 0.75])
            
            sm = plt.cm.ScalarMappable(cmap="plasma", norm=shared_norm)  # 1 → cbar_cap
            sm.set_array([])
            shared_cb = fig.colorbar(sm, cax=cax, orientation='vertical', fraction=0.03, pad=0.05)
            shared_cb.set_label("Ratio of peak S-wave amplitudes", fontsize=12)
            shared_cb.ax.tick_params(labelsize=14)

            # integer ticks from 1 to cbar_cap
            start = int(np.ceil(shared_norm.vmin))   # 1
            end   = int(np.floor(shared_norm.vmax))  # cbar_cap
            n_steps = 5
            ticks = np.linspace(start, end, num=n_steps, dtype=int) if end > start else [start]
            shared_cb.set_ticks(ticks)
            shared_cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            combined_filename = f"figure-7--{quake}--normalized-S-wave-amplitude-maps_{NORM_PCTL}-pctl--cbar-capped-{cbar_cap}--window-{window_start}-{window_end}-s.png"
            print(combined_filename)
            fig.tight_layout(rect=[0.04, 0.02, 0.86, 0.91], w_pad=0.6)
        else:
            combined_filename = f"figure-7--{quake}--normalized-S-wave-amplitude-maps_{NORM_PCTL}-pctl--natural-cbars--window-{window_start}-{window_end}-s.png"
        fig.savefig(combined_filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved combined map: {combined_filename}")

if __name__ == '__main__':
    main()
