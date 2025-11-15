#!/usr/bin/env python3

import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
import matplotlib.cm as cm
from matplotlib.lines import Line2D


###### paths ######
DATA_DIR   = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/NODES_mseed_data_downsampled_20_Hz/"
STATIONS   = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figure--Vs-calculation-from-p-wave-polarization--ray-parameter--incidence-angle/station.txt"
GEO_TIF    = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figure--Vs-calculation-from-p-wave-polarization--ray-parameter--incidence-angle/2005-Troost-Seattle-map--node-area_WGS84.tif"
SHP_FAULTS = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/SHP/Qfaults_US_Database.shp"
SHP_DEF    = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/sfz-deformation-front.shp"

# ────────────── USER SETTINGS ─────────────────────────────────────────

reference_station = "south"        # which station do you want to compute the corr coeff relative to? set to "north" or "south" 
model = TauPyModel('iasp91')     # choose TauP v model
KM_PER_DEG = 111.19

# M7.2 earthquake info: https://earthquake.usgs.gov/earthquakes/eventpage/us70004jyv/executive
evt = dict(time=UTCDateTime("2019-07-14T09:10:51"),
           lat=-0.586, lon=128.034, depth=19.0, name="M7.2")

# figure/nodal array region bounds
XLIM = (-122.32, -122.27)
YLIM = ( 47.56 ,  47.625)
MEAN_LAT   = 0.5*(YLIM[0] + YLIM[1])
BOX_ASPECT = (YLIM[1] - YLIM[0]) / ((XLIM[1] - XLIM[0]) * np.cos(np.deg2rad(MEAN_LAT)))
# axes height / width so that degrees map to meters locally (Plate Carrée → cos(lat))

MAP_OPACITY      = 0.75  # geoTIFF transparency
default_font = plt.rcParams['font.family'][0]

# frequency bands
CORR_FREQMIN, CORR_FREQMAX = 0.02, 0.3    # freq band to filter to before computing correlation coefficient

# define the correlation window offsets -- these are based on Taupy arrival predictions which will be printed
pwave_corr_window = [830,1330]
swave_corr_window = [1475,1590] 
all_arrivals_corr_window = [830,3200]

USE_ABS_CORR = True  # treat correlation as similarity regardless of sign? if False, it can include negative corr. coeff. values
ABS_TAG = "_ABS-value_corr-coeff" if USE_ABS_CORR else "_SIGNED-corr-coeff" # Suffix for filenames

# which component to use for the full arrivals time range correlation:
FULL_WIN_COMP = "DPZ"   # "DP1"=N, "DP2"=E, "DPZ"=Z
COMP_SHORT = {"DPZ":"Z", "DP1":"N", "DP2":"E"}

# --- marker/legend sizing ---
NODE_SIZE = 170              # nodes with data
MISSING_NODE_SIZE = 170      # nodes without data
REF_NODE_SIZE = 180          # reference station
NODE_EDGEWIDTH = 1.4
REF_EDGEWIDTH = 2.8
LEGEND_MARKERSIZE = 9.5
LEGEND_EDGEWIDTH = 1.6
plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12})
WSPACE = 0.0005 # subplot spacing   

# ────────────── helper functions ───────────────────────────────────────────────

def truncate_colormap(cmap, minval=0.08, maxval=0.98, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", new_colors)

# great-circle distance in degrees    
def gc_deg(lat1,lon1,lat2,lon2): 
    φ1,φ2 = map(math.radians,(lat1,lat2))
    dφ    = math.radians(lat2-lat1)
    dλ    = math.radians(lon2-lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return math.degrees(2*math.asin(math.sqrt(a)))

def load_merged_trace(sta, cha, t_evt):
    days = [t_evt - 86400, t_evt, t_evt + 86400]
    fns = []
    for d in days:
        day = d.strftime("%Y-%m-%d")
        fns.extend(glob.glob(os.path.join(DATA_DIR, f"*Z6.{sta}.{cha}..{day}T*.mseed")))
    if not fns:
        return None
    st = None
    for fn in sorted(fns):
        try:
            tr = read(fn)
            st = tr if st is None else (st + tr)
        except Exception:
            pass
    if st is None or len(st) == 0:
        return None
    st.merge(method=1, fill_value=0)
    sel = st.select(channel=cha)
    return sel[0] if len(sel) else st[0]

def stations_with_data(comp, win):
    """
    Return only stations whose merged trace actually covers the [start,end] window.
    """
    start_off, end_off = win
    c_df = stn[stn.channel == comp].reset_index(drop=True)

    keep_rows = []
    for _, s in c_df.iterrows():
        tr = load_merged_trace(s.station, comp, evt['time'])
        if tr is None:
            continue
        # require coverage of the full window
        if tr.stats.starttime <= evt['time'] + start_off and tr.stats.endtime >= evt['time'] + end_off:
            keep_rows.append(s)
    return pd.DataFrame(keep_rows).reset_index(drop=True)

def scalebar(ax, km=1, fontsize=12, fontweight='bold'):
    (x0,x1),(y0,y1) = ax.get_xlim(), ax.get_ylim()
    midlat = 0.5*(y0+y1)
    deg = km / (111.320*math.cos(math.radians(midlat)))
    xb = x0 + 0.05*(x1-x0)
    xe = xb + deg
    yb = y0 + 0.02*(y1-y0)  # shift scalebar up or down with coefficient
    ax.plot([xb, xe], [yb, yb], color='k', lw=5, zorder=99)
    ax.text(
        (xb+xe)/2, yb+0.006*(y1-y0),
        f"{km} km",
        ha='center', va='bottom',
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=99)

def plot_base_map(ax, show_xlabel=True):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_axisbelow(True)
    ax.grid(True, ls='--', color='lightgray', zorder=1)
    if geo_img is not None:
        ax.imshow(geo_img, extent=geo_ext, alpha=MAP_OPACITY, zorder=0)
    if shp_faults is not None:
        shp_faults.plot(ax=ax, color='black', edgecolor='k', lw=2, zorder=2)
    if shp_def is not None:
        shp_def.plot(ax=ax, color='black', edgecolor='k', lw=2, zorder=2)
    if show_xlabel:
        ax.set_xlabel("Longitude", fontsize=16, labelpad=4)
    else:
        ax.set_xlabel("")  # suppress on this panel
    ax.tick_params(which="both", labelsize=13)

def compute_corr_map(comp, win):
    """
    Build station list ONLY from those with data in the window, pick ref (north/south),
    and compute corr coeffs against the ref.
    """
    start_off, end_off = win

    c_df = stations_with_data(comp, win)
    if c_df.empty:
        print(f"[INFO] No stations cover window {win} for {comp}.")
        return pd.DataFrame(), pd.DataFrame(), np.array([]), np.nan, np.nan, None
    # pick reference from list of deployed stations only:
    if reference_station.lower() == "south":
        ref = c_df.loc[c_df['latitude'].idxmin()]
    else:
        ref = c_df.loc[c_df['latitude'].idxmax()]
    # prep reference data
    ref_tr = load_merged_trace(ref.station, comp, evt['time'])
    ref_tr.trim(starttime=evt['time'] - 200, endtime=evt['time'] + end_off + 200)
    ref_tr.detrend("linear")  
    ref_tr.detrend("demean")    
    ref_tr.taper(max_percentage=0.05, type="cosine")
    ref_tr.filter('bandpass', freqmin=CORR_FREQMIN, freqmax=CORR_FREQMAX,corners=4, zerophase=True)
    ref_seg = ref_tr.copy().trim(starttime=evt['time'] + start_off, endtime=evt['time'] + end_off)
    if ref_seg.data.size == 0:
        print(f"[INFO] Reference {ref.station} has no samples in window {win}.")
        return pd.DataFrame(), pd.DataFrame(), np.array([]), np.nan, np.nan, None
    print(f"check # points in each window: {comp} window {start_off}-{end_off}s | ref={ref.station} | npts={ref_seg.stats.npts} | dt={ref_seg.stats.delta:.3f}s")
    ref_data = ref_seg.data.copy()
    ok_pts, miss_pts, cc_vals = [], [], []
    for _, s in c_df.iterrows():
        try:
            if s.station == ref.station:
                ok_pts.append({"station": s.station, "longitude": s.longitude, "latitude": s.latitude})
                cc_vals.append(1.0)
                continue
            tr = load_merged_trace(s.station, comp, evt['time'])
            tr.trim(starttime=evt['time'] - 200, endtime=evt['time'] + end_off + 200)
            tr.detrend("linear")
            tr.detrend("demean")
            tr.taper(max_percentage=0.05, type="cosine")
            tr.filter('bandpass', freqmin=CORR_FREQMIN, freqmax=CORR_FREQMAX, corners=4, zerophase=True)
            seg = tr.copy().trim(starttime=evt['time'] + start_off, endtime=evt['time'] + end_off)
            if seg.data.size == 0:
                raise RuntimeError("empty window")
            val = np.corrcoef(seg.data, ref_data)[0, 1]
            if USE_ABS_CORR:
                val = abs(val)
            ok_pts.append({"station": s.station, "longitude": s.longitude, "latitude": s.latitude})
            cc_vals.append(val)
        except Exception: # stns that were not deployed at time of EQ:
            miss_pts.append({"station": s.station, "longitude": s.longitude, "latitude": s.latitude})
    return (pd.DataFrame(ok_pts),
            pd.DataFrame(miss_pts),
            np.array(cc_vals),
            ref.longitude, ref.latitude, ref.station)

def plot_corr_panel(ax, ok_df, miss_df, cc, ref_lon, ref_lat, title, vmin_override=None, vmax_override=None,show_xlabel=True):
    plot_base_map(ax, show_xlabel=show_xlabel)
    cmap_corr = truncate_colormap(plt.get_cmap("hsv"), 0.08, 0.98)
    vmin = float(np.nanmin(cc)) if len(cc) else 0.0
    vmax = float(np.nanmax(cc)) if len(cc) else 1.0
    if vmin_override is not None: vmin = float(vmin_override) # can force cbar bounds if desired
    if vmax_override is not None: vmax = float(vmax_override)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        ok_df.longitude if not ok_df.empty else [],
        ok_df.latitude  if not ok_df.empty else [],
        c=cc, cmap=cmap_corr, norm=norm,
        s=NODE_SIZE, edgecolors='k', linewidths=NODE_EDGEWIDTH, zorder=4)
    if not miss_df.empty:
        ax.scatter(miss_df.longitude, miss_df.latitude,
                   facecolors='none', edgecolors='k', s=NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)
    if np.isfinite(ref_lon) and np.isfinite(ref_lat):
        ax.scatter([ref_lon], [ref_lat],
                   c=[vmax], cmap=cmap_corr, norm=norm,
                   s=REF_NODE_SIZE, edgecolors='k', linewidths=REF_EDGEWIDTH, zorder=6)
    ax.set_title(title, pad=10, fontsize=10)
    scalebar(ax, km=1)
    return sc, norm, cmap_corr

def read_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        l, b, r, t = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            aff, w, h = calculate_default_transform(src.crs, "EPSG:4326",
                                                   src.width, src.height, *src.bounds)
            rep = np.empty((img.shape[0], h, w), img.dtype)
            for i in range(img.shape[0]):
                reproject(src.read(i+1), rep[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=aff, dst_crs="EPSG:4326",
                          resampling=Resampling.nearest)
            img = rep; l, b = aff.c, aff.f + aff.e * h; r, t = aff.c + aff.a * w, aff.f
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.moveaxis(img, 0, -1)
        return img, (l, r, b, t)
    return None, None

def safe_shp(path):
    try:
        g = gpd.read_file(path).to_crs(epsg=4326)
        g = g[g.intersects(box(XLIM[0]-0.01, YLIM[0]-0.01, XLIM[1]+0.01, YLIM[1]+0.01))]
        return g
    except Exception:
        return None

def missing_nodes_for_window(comp, ok_df):
    # use ALL stations in station.txt, regardless of channel
    all_nodes = stn[["station","longitude","latitude"]].drop_duplicates("station").copy()
    all_nodes["station"] = all_nodes["station"].astype(str)
    if ok_df is None or ok_df.empty:
        return all_nodes
    return all_nodes[~all_nodes.station.isin(ok_df.station.astype(str))]

_TMP_CMAP = truncate_colormap(plt.get_cmap("hsv"), 0.08, 0.98)
REF_FACE_COLOR = _TMP_CMAP(1.0)

legend_handles_nodes = [
    Line2D([0], [0],
           marker='o', linestyle='None',
           markerfacecolor='blue', markeredgecolor='k',
           markersize=LEGEND_MARKERSIZE, markeredgewidth=LEGEND_EDGEWIDTH, label='Deployed node'),
    Line2D([0], [0],
           marker='o', linestyle='None',
           markerfacecolor='none', markeredgecolor='k',
           markersize=LEGEND_MARKERSIZE, markeredgewidth=LEGEND_EDGEWIDTH, label='Node not deployed'),
    Line2D([0], [0],
           marker='o', linestyle='None',
           markerfacecolor=REF_FACE_COLOR, markeredgecolor='k',
           markersize=int(LEGEND_MARKERSIZE*1.2), markeredgewidth=REF_EDGEWIDTH,
           label='Reference node')]

# ────────────── STATIC DATA LOAD ──────────────────────────────────────

stn = pd.read_csv(STATIONS)
stn['station'] = stn['station'].astype(str)

all_stations = stn[stn.channel == "DPZ"].reset_index(drop=True)
all_stations['station'] = all_stations['station'].astype(str)

# ─── compute representative distance & print all TauP predicted arrivals ─────────────────
rep_deg = all_stations.apply(
    lambda s: gc_deg(evt['lat'], evt['lon'], s.latitude, s.longitude),
    axis=1).median()
print(f"Representative distance (median over Z stations): {rep_deg:.2f}° (~{rep_deg*KM_PER_DEG:.0f} km)")
arrivals = model.get_travel_times(source_depth_in_km=evt['depth'],distance_in_degree=rep_deg)
print(arrivals)

# ────────────── CORRELATION COEFFICIENT MAPS ──────────────
# load tifs and shp files for plotting:
geo_img, geo_ext = read_tif(GEO_TIF)
shp_faults = safe_shp(SHP_FAULTS)
shp_def    = safe_shp(SHP_DEF)

# figure 1a: P waves, Z component; S waves, ENZ components
okP,  missP,  ccP,  rlonP, rlatP, _ = compute_corr_map("DPZ", pwave_corr_window)   # P waves, Z component
okSE, missSE, ccSE, rlonE, rlatE, _ = compute_corr_map("DP2", swave_corr_window)   # S waves, E comp
okSN, missSN, ccSN, rlonN, rlatN, _ = compute_corr_map("DP1", swave_corr_window)   # S waves, N comp
okSZ, missSZ, ccSZ, rlonZ, rlatZ, _ = compute_corr_map("DPZ", swave_corr_window)   # S waves, Z comp

fig1, (axP, axSE, axSN, axSZ) = plt.subplots(1, 4, figsize=(28, 12), sharey=True, dpi=300)
for ax in (axP, axSE, axSN, axSZ):
    ax.set_box_aspect(BOX_ASPECT)
for ax in (axP, axSE, axSN, axSZ):
    ax.xaxis.labelpad = 12   
    ax.tick_params(axis='x', pad=4)
fig1.subplots_adjust(left=0.045, right=0.985, bottom=0.1, top=0.94, wspace=WSPACE)
axP.set_ylabel("Latitude", fontsize=12, labelpad=4)
for ax in (axSN, axSE, axSZ):
    ax.tick_params(labelleft=False)

# subplot 1: P waves, Z comp
scP, normP, cmapP = plot_corr_panel(axP, okP, missP, ccP, rlonP, rlatP,f"Z component | P waves {pwave_corr_window[0]}–{pwave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz")
fig1.colorbar(cm.ScalarMappable(norm=normP, cmap=cmapP), ax=axP, fraction=0.06, pad=0.016).set_label("Correlation coefficient")
# subplot 2: S waves, E comp
scE, normE, cmapE = plot_corr_panel(axSE, okSE, missSE, ccSE, rlonE, rlatE,f"E component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz")
fig1.colorbar(cm.ScalarMappable(norm=normE, cmap=cmapE), ax=axSE, fraction=0.06, pad=0.016).set_label("Correlation coefficient")
# subplot 3: S waves, N comp
scN, normN, cmapN = plot_corr_panel(axSN, okSN, missSN, ccSN, rlonN, rlatN,f"N component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz")
fig1.colorbar(cm.ScalarMappable(norm=normN, cmap=cmapN), ax=axSN, fraction=0.06, pad=0.016).set_label("Correlation coefficient")
# subplot 4: S waves, Z comp
scZ, normZ, cmapZ = plot_corr_panel(axSZ, okSZ, missSZ, ccSZ, rlonZ, rlatZ,f"Z component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz")
fig1.colorbar(cm.ScalarMappable(norm=normZ, cmap=cmapZ), ax=axSZ, fraction=0.06, pad=0.016).set_label("Correlation coefficient")

# plot nodes not yet deployed as empty circles
missNodesP  = missing_nodes_for_window("DPZ", okP)
missNodesSN = missing_nodes_for_window("DP1", okSN)
missNodesSE = missing_nodes_for_window("DP2", okSE)
missNodesSZ = missing_nodes_for_window("DPZ", okSZ)
if not missNodesP.empty:
    axP.scatter(missNodesP.longitude, missNodesP.latitude, facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)
if not missNodesSN.empty:
    axSN.scatter(missNodesSN.longitude, missNodesSN.latitude, facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)
if not missNodesSE.empty:
    axSE.scatter(missNodesSE.longitude, missNodesSE.latitude, facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)
if not missNodesSZ.empty:
    axSZ.scatter(missNodesSZ.longitude, missNodesSZ.latitude, facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)

# legend
axSZ.legend(handles=legend_handles_nodes,
            loc='lower right', frameon=True, framealpha=0.90)
# annotate panel letters A–D
for ax, lbl in zip((axP, axSE, axSN, axSZ), ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(0.015, 0.985, lbl,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=36, fontweight="bold", color="black",
            zorder=1000)

out1a = os.path.join(
    f"figure-5--1a--{evt['name']}_corrmaps_P-wave-Zcomp_{pwave_corr_window[0]}-{pwave_corr_window[1]}s_"
    f"S-wave-E-N-Z-comps_S{swave_corr_window[0]}-{swave_corr_window[1]}_{reference_station}-ref{ABS_TAG}.png").replace(' ', '_')
fig1.savefig(out1a, dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig1)
print("saved →", out1a)

# ==== figure 1b: same four plots, now sharing a single colorbar for comparison ====
_ccs = [ccP, ccSE, ccSN, ccSZ]
_ccs = [c for c in _ccs if len(c)]
if USE_ABS_CORR:
    vmin_shared, vmax_shared = 0.0, 1.0
else:
    vmin_shared = min(np.nanmin(c) for c in _ccs) if _ccs else 0.0
    vmax_shared = max(np.nanmax(c) for c in _ccs) if _ccs else 1.0

fig1b, (axP2, axSE2, axSN2, axSZ2) = plt.subplots(
    1, 4, figsize=(24, 10), sharey=True, dpi=300)
for ax in (axP2, axSE2, axSN2, axSZ2):
    ax.set_box_aspect(BOX_ASPECT)
for ax in (axP2, axSE2, axSN2, axSZ2):
    ax.tick_params(axis='x', pad=6)

# fig1b.subplots_adjust(left=0.05, right=0.88, bottom=0.12, top=0.95)
fig1b.subplots_adjust(left=0.05, right=0.87, bottom=0.13, top=0.95, wspace=0.003)
# center xlabel:
sp = fig1b.subplotpars
xc = 0.5 * (sp.left + sp.right)
fig1b.text(xc, sp.bottom - 0.05, 'Longitude', ha='center', va='top', fontsize=18)

axP2.set_ylabel("Latitude", fontsize=18, labelpad=9)
for ax in (axSE2, axSN2, axSZ2):
    ax.tick_params(labelleft=False)
scP2, _, cmapShared = plot_corr_panel(
    axP2, okP,  missP,  ccP,  rlonP, rlatP,
    f"Z component | P waves {pwave_corr_window[0]}–{pwave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
    vmin_override=vmin_shared, vmax_override=vmax_shared, show_xlabel=False)
plot_corr_panel(
    axSE2, okSE, missSE, ccSE, rlonE, rlatE,
    f"E component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
    vmin_override=vmin_shared, vmax_override=vmax_shared, show_xlabel=False)
plot_corr_panel(
    axSN2, okSN, missSN, ccSN, rlonN, rlatN,
    f"N component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
    vmin_override=vmin_shared, vmax_override=vmax_shared, show_xlabel=False)
plot_corr_panel(
    axSZ2, okSZ, missSZ, ccSZ, rlonZ, rlatZ,
    f"Z component | S waves {swave_corr_window[0]}–{swave_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
    vmin_override=vmin_shared, vmax_override=vmax_shared, show_xlabel=False)
OVERLAY_Z  = 3        
OVERLAY_LW = 1.5       # thicker edge to stand out
if not missNodesP.empty:
    axP2.scatter(missNodesP.longitude,  missNodesP.latitude,
                 facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE,
                 linewidths=OVERLAY_LW, zorder=OVERLAY_Z)
if not missNodesSE.empty:
    axSE2.scatter(missNodesSE.longitude, missNodesSE.latitude,
                  facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE,
                  linewidths=OVERLAY_LW, zorder=OVERLAY_Z)
if not missNodesSN.empty:
    axSN2.scatter(missNodesSN.longitude, missNodesSN.latitude,
                  facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE,
                  linewidths=OVERLAY_LW, zorder=OVERLAY_Z)
if not missNodesSZ.empty:
    axSZ2.scatter(missNodesSZ.longitude, missNodesSZ.latitude,
                  facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE,
                  linewidths=OVERLAY_LW, zorder=OVERLAY_Z)

# show every other longitude tick
xt = axP2.get_xticks()
for ax in (axP2, axSE2, axSN2, axSZ2):
    if len(xt) >= 2:
        ax.set_xticks(xt[::2])
# one shared x-label
for ax in (axP2, axSN2, axSE2, axSZ2):
    ax.set_xlabel('')  # remove per-axis xlabels

# shared colorbar:
sm  = cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin_shared, vmax=vmax_shared), cmap=cmapShared)
cax = fig1b.add_axes([0.868, 0.17, 0.015, 0.75])  # x, y, width, height 
cb  = fig1b.colorbar(sm, cax=cax)
cb.set_label("Correlation coefficient", fontsize=18, labelpad=10)
cb.ax.tick_params(labelsize=16)
# legend
axSZ2.legend(handles=legend_handles_nodes,
             loc='lower right', frameon=True, framealpha=0.90,fontsize=10)
# annotate panel letters ABCD
for ax, lbl in zip((axP2, axSE2, axSN2, axSZ2), ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(0.015, 0.985, lbl, transform=ax.transAxes,
            ha="left", va="top", fontsize=36, fontweight="bold", color="black", zorder=1000)
# save
out1b = os.path.join(
    f"figure-5--1b--{evt['name']}_corrmaps_P-wave-Zcomp_{pwave_corr_window[0]}-{pwave_corr_window[1]}s_"
    f"S-wave-E-N-Z-comps_S{swave_corr_window[0]}-{swave_corr_window[1]}_{reference_station}-ref_ONE-CBAR{ABS_TAG}.png").replace(' ', '_')
fig1b.savefig(out1b, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig1b)
print("saved →", out1b)


# Figure 2: full arrivals window correlations
okA, missA, ccA, rlonA, rlatA, _ = compute_corr_map(FULL_WIN_COMP, all_arrivals_corr_window)
# shared colorbar range
if USE_ABS_CORR:
    vminFO, vmaxFO = 0.0, 1.0
else:
    vminFO = float(np.nanmin(ccA)) if len(ccA) else 0.0
    vmaxFO = float(np.nanmax(ccA)) if len(ccA) else 1.0

fig2, ax = plt.subplots(1, 1, figsize=(7, 8), dpi=300, constrained_layout=True)
ax.set_ylabel("Latitude", fontsize=12, labelpad=4)
scA, normA, cmapA = plot_corr_panel(
    ax, okA, missA, ccA, rlonA, rlatA,
    f"{COMP_SHORT[FULL_WIN_COMP]}-comp | Full window {all_arrivals_corr_window[0]}–{all_arrivals_corr_window[1]} s, {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
    vmin_override=vminFO, vmax_override=vmaxFO)
cbarA = fig2.colorbar(cm.ScalarMappable(norm=normA, cmap=cmapA), ax=ax, fraction=0.046, pad=0.04)
cbarA.set_label("Correlation coefficient")
ticks = np.linspace(vminFO, vmaxFO, 5); cbarA.set_ticks(ticks)
missNodesA = missing_nodes_for_window(FULL_WIN_COMP, okA) # empty circles for nodes without data
if not missNodesA.empty:
    ax.scatter(missNodesA.longitude, missNodesA.latitude,facecolors='none', edgecolors='k', s=MISSING_NODE_SIZE, lw=NODE_EDGEWIDTH, zorder=4)
out2 = os.path.join(
    f"figure-5--2--{evt['name']}_corrmap_{COMP_SHORT[FULL_WIN_COMP]}_{reference_station}-ref_full-{all_arrivals_corr_window[0]}-{all_arrivals_corr_window[1]}s{ABS_TAG}.png").replace(' ', '_')
fig2.savefig(out2, dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig2)
print("saved →", out2)