#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import osmnx as ox
import matplotlib.colors as colors
import matplotlib.ticker as ticker 
from math import ceil, floor
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

def main():
    csv_file = "lat_lon_psdlevels--all-stns--all-comp--full-time-duration--all-freqs--MEDIAN.csv"
    
    components = ["DP2", "DP1", "DPZ"]
    component_label_map = {
        "DP2": "E",
        "DP1": "N",
        "DPZ": "Z"}
    
    df = pd.read_csv(csv_file)
    
    noise_cols = [col for col in df.columns if col.startswith("medianpsd_")]
    if not noise_cols:
        print("No noise columns found in the CSV.")
        return

    noise_cols = sorted(noise_cols, key=lambda x: float(x.split('_')[1].replace('Hz','')))
    df_all = df[df["component"].isin(components)]
    if df_all.empty:
        print("No data found for specified components.")
        return
    
    lons_all = df_all["lon"].values
    lats_all = df_all["lat"].values
    lon_min, lon_max = lons_all.min(), lons_all.max()
    lat_min, lat_max = lats_all.min(), lats_all.max()
    lon_margin = 0.1 * (lon_max - lon_min)
    lat_margin = 0.1 * (lat_max - lat_min)
    grid_lon_vals = np.linspace(lon_min - lon_margin, lon_max + lon_margin, 100)
    grid_lat_vals = np.linspace(lat_min - lat_margin, lat_max + lat_margin, 100)
    grid_lon, grid_lat = np.meshgrid(grid_lon_vals, grid_lat_vals)
    
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    distance_meters = 5000
    G = ox.graph_from_point((center_lat, center_lon), dist=distance_meters, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    
    for noise_col in noise_cols:
        freq = noise_col.split('_')[1].replace('Hz','')
        print(f"Processing frequency: {freq} Hz")
        fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.08, right=0.88, top=0.98, bottom=0.09, wspace=0.002)
        
        all_noise_values = []
        for comp in components:
            comp_df = df[df["component"] == comp]
            if not comp_df.empty:
                all_noise_values.extend(comp_df[noise_col].values)
        
        if not all_noise_values:
            print(f"No noise data available for frequency {freq} Hz.")
            continue
        
        norm = colors.Normalize(vmin=min(all_noise_values), vmax=max(all_noise_values))
        levels = np.linspace(min(all_noise_values), max(all_noise_values), 101)
        
        for i, comp in enumerate(components):
            ax = axs[i]
            comp_df = df[df["component"] == comp]
            print(f"Component: {comp}, Data points: {len(comp_df)}")
            if comp_df.empty:
                print(f"No stations found for component {comp} at frequency {freq} Hz.")
                ax.set_title(f"{comp}: No data")
                continue

            noise = comp_df[noise_col].values
            coords = (comp_df["lon"].values, comp_df["lat"].values)
            print(f"Coords count: {len(coords[0])}, Noise count: {len(noise)}")
            
            grid_noise = griddata(
                coords, noise, (grid_lon, grid_lat), method='linear')
            
            contour = ax.contourf(
                grid_lon, grid_lat, grid_noise,
                levels=levels,
                cmap='viridis', norm=norm, zorder=1)
            
            edges.plot(ax=ax, color='black', linewidth=0.7, alpha=1.0, zorder=2)
            
            special_stations = comp_df[comp_df["station"].astype(str).str.strip().isin(["79", "86"])]
            normal_stations = comp_df[~comp_df["station"].astype(str).str.strip().isin(["79", "86"])]
            
            if not normal_stations.empty:
                ax.scatter(
                    normal_stations["lon"].values, normal_stations["lat"].values,
                    color='orange', edgecolors='black', linewidths=0.3,
                    marker='o', s=16, zorder=3)
            
            if not special_stations.empty:
                ax.scatter(
                    special_stations["lon"].values, special_stations["lat"].values,
                    color='white', marker='*', s=280, edgecolors='brown', zorder=4)
                for idx, row in special_stations.iterrows():
                    ax.text(
                        row["lon"], row["lat"], row["station"],
                        fontsize=4, color='black', ha='center', va='center', zorder=5)
            
            ax.text(    # annotate component
                0.04, 0.985,
                component_label_map[comp],
                transform=ax.transAxes,
                fontsize=28,
                fontweight='bold',
                ha='left',
                va='top',
                zorder=10)

            ax.set_xlim(grid_lon_vals[0], grid_lon_vals[-1])
            ax.set_ylim(grid_lat_vals[0], grid_lat_vals[-1])

            if i == 0:
                ax.set_ylabel("Latitude", fontsize=12)
            if i == 1:
                ax.set_xlabel("Longitude", fontsize=12)
            ax.tick_params(labelsize=8)
        
        cbar_ax = fig.add_axes([0.885, 0.09, 0.025, 0.89])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Median PSD (dB (m/s)²/Hz)", fontsize=10)  # VELOCITY units assumed
        cbar.ax.tick_params(labelsize=8)
        
        # colorbar tick labels -- whole numbers
        vmin, vmax = cbar.mappable.get_clim() # get bounds
        lower = int(ceil(vmin))
        upper = int(floor(vmax))
        ticks = np.arange(lower, upper + 1, 1)
        cbar.set_ticks(ticks)

        def every_third(x, pos): # label every third tick
            return f"{int(x)}" if ((int(x) - lower) % 3 == 0) else ''

        formatter = FuncFormatter(every_third)
        cbar.ax.yaxis.set_major_formatter(formatter)

        # fig.suptitle(f"Three-Component Noise Power Maps for {freq} Hz", fontsize=10, y=0.98)
        output = f"noise_map_median_{freq}_Hz--no-cbar-values-set--full-time-duration.png"
        plt.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

if __name__ == '__main__':
    main()
