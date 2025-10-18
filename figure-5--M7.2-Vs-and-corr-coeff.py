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
from obspy.geodetics import gps2dist_azimuth
from adjustText import adjust_text
import matplotlib.cm as cm

SELECTED_V_MODEL = 'iasp91'      # choose from v_models list (obspy TauP)
# v_models = ['iasp91','1066a','1066b','ak135','ak135f_no_mud','herrin','jb','prem','pwdk','sp6']

# map region bounds
XLIM = (-122.32, -122.27)
YLIM = ( 47.56 ,  47.625)
MAP_OPACITY      = 0.75          # geologic map transparency

# frequency bands
FREQMIN, FREQMAX           = 1.0 , 4.0 # polarization
CORR_FREQMIN, CORR_FREQMAX = 0.02, 0.3    # correlation

# paths 
DATA_DIR   = "/mnt/data0/Seattle-work-2019/SEATTLE_NODES_DATA--2019/NODES_mseed_data_downsampled_20_Hz/"
STATIONS   = "station.txt"
GEO_TIF    = "2005-Troost-Seattle-map--node-area_WGS84.tif"
USGS_VS30  = "USGS-McPhillips-2020--vs30data.csv"
SHP_FAULTS = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/SHP/Qfaults_US_Database.shp"
SHP_DEF    = "/mnt/data1/Seattle-sfz-nodes-2025-SRL-paper/000-sfz-seattle-SRL-paper-2025/figures--station maps/usgs-qfaults/sfz-deformation-front.shp"

# define corr coeff time range
window_to_corrcoeff_begin = 1000   
window_to_corrcoeff_end   = 2000 
diffwindow = window_to_corrcoeff_end - window_to_corrcoeff_begin  

default_font = plt.rcParams['font.family'][0] 

earth_radius_km = 6371.
KM_PER_DEG = (np.pi * earth_radius_km) / 180.
VS_CMAP = mcolors.LinearSegmentedColormap.from_list("Vs_abs", [(0,"green"),(.5,"yellow"),(1,"red")])

# functions
def truncate_colormap(cmap, minval=0.08, maxval=0.98, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", new_colors)

def p_s_per_km(arr):
    if hasattr(arr,"ray_param_sec_per_deg"):
        return arr.ray_param_sec_per_deg / KM_PER_DEG
    return arr.ray_param / earth_radius_km

def gc_deg(lat1,lon1,lat2,lon2):
    # great-circle distance in degrees
    φ1,φ2 = map(math.radians,(lat1,lat2))
    dφ    = math.radians(lat2-lat1)
    dλ    = math.radians(lon2-lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return math.degrees(2*math.asin(math.sqrt(a)))

def load_trace(sta, cha, t_evt):
    day = t_evt.strftime("%Y-%m-%d")
    pat = os.path.join(DATA_DIR, f"*Z6.{sta}.{cha}..{day}T*.mseed")
    fns = glob.glob(pat)
    if not fns:
        raise RuntimeError("no mseed")
    st  = read(fns[0]); st.merge(method=1)
    for tr in st:
        if tr.stats.starttime <= t_evt <= tr.stats.endtime:
            return tr
    raise RuntimeError("trace does not cover")

def polarization(trz, trn, tre, t0, azi):
    WIN = 2.0  # 2 sec window
    t1, t2 = t0 - WIN/2, t0 + WIN/2

    # slice each trace to the 2 s window around t0 and use the sliced copy for filtering
    tz = trz.slice(t1, t2)
    tn = trn.slice(t1, t2)
    te = tre.slice(t1, t2)
    for tr in (tz, tn, te):
        tr.detrend('linear')
        tr.taper(max_percentage=0.05)
        tr.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, corners=4, zerophase=True)
    # use the sliced and filtered data for polarization analysis
    z = tz.data - tz.data.mean()
    n = tn.data - tn.data.mean()
    e = te.data - te.data.mean()
    azr = math.radians(azi)
    r = n * math.cos(azr) + e * math.sin(azr)
    t = -n * math.sin(azr) + e * math.cos(azr)
    M = np.vstack((z, r, t))
    vals, vecs = np.linalg.eigh(np.cov(M))
    k = np.argmax(vals); v = vecs[:, k]
    if v[0] < 0:
        v = -v
    inc = math.degrees(math.acos(v[0]))
    plan = vals[k] / vals.sum()
    # compute SNR using the vertical component in an equal-length noise window before t0
    noise_win = trz.slice(t1 - WIN, t1)
    noise_win.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, corners=4, zerophase=True)
    rms_signal = np.sqrt(np.mean(z**2))
    rms_noise  = np.sqrt(np.mean((noise_win.data - noise_win.data.mean())**2))
    snr = rms_signal / rms_noise if rms_noise > 0 else np.nan
    return inc, plan, snr

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

# load data:
stn = pd.read_csv(STATIONS)
vs_stations = stn[stn.channel == "DPZ"].reset_index(drop=True)
print(f"{len(vs_stations)} stations used for Vs inversion (Z only)")

try:
    stars = pd.read_csv(USGS_VS30)
    stars['Vs30_meters_per_sec'] = pd.to_numeric(stars['Vs30_meters_per_sec'])
    stars = stars.dropna(subset=['Longitude', 'Latitude', 'Vs30_meters_per_sec'])
    stars = stars[(stars.Longitude.between(*XLIM)) & (stars.Latitude.between(*YLIM))]
except Exception:
    stars = pd.DataFrame(columns=['Longitude', 'Latitude', 'Vs30_meters_per_sec'])

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

geo_img, geo_ext = read_tif(GEO_TIF)

def safe_shp(path):
    try:
        g = gpd.read_file(path).to_crs(epsg=4326)
        g = g[g.intersects(box(XLIM[0]-0.01, YLIM[0]-0.01, XLIM[1]+0.01, YLIM[1]+0.01))]
        return g
    except Exception:
        return None

shp_faults = safe_shp(SHP_FAULTS)
shp_def    = safe_shp(SHP_DEF)

# ────────────── specify earthquake and model ─────────────────────────────────────────

evt = dict(time=UTCDateTime("2019-07-14T09:10:51"),
           lat=-0.586, lon=128.034, depth=19.0, name="M7.2")
model = TauPyModel(SELECTED_V_MODEL)

# ────────────── Vs inversion ─────────────────────────────────────

rows = []
for _, s in vs_stations.iterrows():
    try:
        dist = gc_deg(evt['lat'], evt['lon'], s.latitude, s.longitude)
        azi  = gps2dist_azimuth(evt['lat'], evt['lon'], s.latitude, s.longitude)[1]
        arrs = model.get_travel_times(evt['depth'], dist, phase_list=["P", "Pdiff"])
        if not arrs:
            continue
        arr = next((a for a in arrs if a.name.upper() == "P"), arrs[0])
        t0  = evt['time'] + arr.time
        pkm = p_s_per_km(arr)
        trz = load_trace(s.station, 'DPZ', t0)
        trn = load_trace(s.station, 'DP1', t0)
        tre = load_trace(s.station, 'DP2', t0)
        inc, plan, snr = polarization(trz, trn, tre, t0, azi)
        Vs = math.sin(math.radians(inc/2)) / pkm * 1000
        rows.append(dict(station=s.station, lon=s.longitude, lat=s.latitude, Vs=Vs))
    except Exception:
        pass

df = pd.DataFrame(rows)
if df.empty:
    print("No stations produced Vs results for model", SELECTED_V_MODEL)
else:
    avg_vs = df['Vs'].mean()
    print(f"Computed Vs for {len(df)} stations (model {SELECTED_V_MODEL}). Average Vs = {avg_vs:.1f} m/s")
    out_csv = f"vs_results_{evt['name']}--vmodel-{SELECTED_V_MODEL}--v3c.csv"
    df.to_csv(out_csv, index=False)
    print("Saved Vs results to", out_csv)


# ─── dual colorbar ranges ────────────────────────────────────────────
if not df.empty:
    vmin = df.Vs.min()
    vmax = df.Vs.max()
    if not stars.empty:
        star_min = stars.Vs30_meters_per_sec.min()
        star_max = stars.Vs30_meters_per_sec.max()
        vmin_incl = min(vmin, star_min)
        vmax_incl = max(vmax, star_max)
    else:
        vmin_incl = vmin
        vmax_incl = vmax
    # always want the "nodes-only" range
    vmin_excl = vmin
    vmax_excl = vmax
else:
    vmin = vmax = None
    vmin_incl = vmax_incl = None
    vmin_excl = vmax_excl = None

# ────────────── COMPONENT LOOP ───────────────────────────────────────

for comp in ("DPZ", "DP1", "DP2"):
    do_vs = (comp == "DPZ" and not df.empty)
    compL = {"DPZ": "Z", "DP1": "N", "DP2": "E"}[comp]
    # Only include stations that have Vs (the deployed nodes)
    c_df = stn[
        (stn.channel == comp) &
        (stn.station.isin(df['station']))
    ].reset_index(drop=True)

    # ── PICK SOUTHERNMOST STATION AS REFERENCE ────────────────────────
    southernmost = c_df.loc[c_df['latitude'].idxmin()]
    ref_sta      = southernmost.station
    ref_lon      = southernmost.longitude 
    ref_lat      = southernmost.latitude  
    # load & preprocess its trace once
    ref_tr = load_trace(ref_sta, comp, evt['time'])
    ref_tr.trim(starttime=evt['time'] - 200,
                endtime=evt['time'] + 3000)
    ref_tr.detrend("demean")
    ref_tr.filter('bandpass',
                  freqmin=CORR_FREQMIN,
                  freqmax=CORR_FREQMAX,
                  corners=4, zerophase=True)
    # isolate the correlation‐window
    ref_seg  = ref_tr.copy().trim(
                  starttime=evt['time'] + window_to_corrcoeff_begin,
                  endtime=evt['time'] + window_to_corrcoeff_end)
    ref_data = ref_seg.data.copy()

    # Absolute times for correlation window
    t1 = evt['time'] + window_to_corrcoeff_begin
    t2 = evt['time'] + window_to_corrcoeff_end

    # Representative distance (median over stations in c_df)
    rep_deg = c_df.apply(
        lambda s: gc_deg(evt['lat'], evt['lon'], s.latitude, s.longitude),
        axis=1).median()

    ok, miss, cc = [], [], []

    for _, s in c_df.iterrows():
        try:
            # if this is the reference station, give it corr=1 and skip
            if s.station == ref_sta:
                ok.append({"longitude": s.longitude, "latitude": s.latitude})
                cc.append(1.0)
                continue

            # otherwise load, trim & filter exactly as ref was done
            tr = load_trace(s.station, comp, evt['time'])
            tr.trim(starttime=evt['time'] - 200,
                    endtime=evt['time'] + 3000)
            tr.detrend("demean")
            tr.filter('bandpass',
                      freqmin=CORR_FREQMIN,
                      freqmax=CORR_FREQMAX,
                      corners=4, zerophase=True)

            seg = tr.copy().trim(
                      starttime=evt['time'] + window_to_corrcoeff_begin,
                      endtime=evt['time'] + window_to_corrcoeff_end
                  )
            data = seg.data
            if data.size == 0:
                raise RuntimeError("No data in the correlation window")

            cc_val = np.corrcoef(data, ref_data)[0, 1]
            cc.append(cc_val)
            ok.append({"longitude": s.longitude, "latitude": s.latitude})

        except Exception:
            miss.append({"longitude": s.longitude, "latitude": s.latitude})


    cc = np.array(cc)
    if cc.size > 0:
        print(f"Corr coeff (ref=southernmost)  min={np.nanmin(cc):.3f},  max={np.nanmax(cc):.3f}")
    else:
        print("No correlation coefficients computed for comp", comp)

    ok_df   = pd.DataFrame(ok)
    miss_df = pd.DataFrame(miss)

    # ─── plotting ─────────────────────────────────────────────────────
    for suffix, (v0, v1) in [
        ("withStars", (vmin_incl, vmax_incl)),
        ("noStars",   (vmin_excl, vmax_excl))
    ]:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 8), sharey=True, dpi=300)
        axL.set_ylabel("Latitude", fontsize=14,labelpad=11)
        fig.subplots_adjust(left=0.05, right=0.72, top=0.93, bottom=0.08, wspace=0.1) 

        # legend background box -- numbers are in figure‐fraction coords:
        x0, y0    = 0.8, 0.645    # lower‐left corner of box
        width     = 0.17          # width of box
        height    = 0.25          # height of box
        boxstyle  = "square,pad=0.02"
        rect      = mpatches.FancyBboxPatch(
            (x0, y0),
            width, height,
            transform=fig.transFigure,
            boxstyle=boxstyle,
            facecolor="white",
            edgecolor="black",
            linewidth=2,
            zorder=1)
        fig.patches.append(rect)

        # geologic legend in the right margin
        fig.text(
            0.79, 0.88,              # x, y in figure‐fraction coords
            "GEOLOGIC UNITS",
            ha="left", va="bottom", 
            fontsize=11, fontfamily=default_font,
            fontweight="bold")

        # make geologic legend contents:
        y = 0.86
        for col, label in [
            ("xkcd:dark beige", "Lake deposits (non-glacial)"),
            ("xkcd:squash", "Recessional Vashon glacial deposits"),
            ("xkcd:deep lavender", "Vashon subglacial till"),
            ("xkcd:sky", "Vashon advance outwash deposits"),
            ("xkcd:leaf", "Pre-Fraser glaciation deposits"),
            ("xkcd:chartreuse", "Olympia beds"),
            ("xkcd:greeny grey", "Pre-Olympia age deposits"),
            ("xkcd:dark sky blue", "Pre-Olympia age glacial deposits")]:
            fig.text(0.79, y, "\u25A0", color=col,
                    fontsize=16, ha='left', va='center')
            fig.text(0.79 + 0.02, y, label, color='black',
                    fontsize=9, fontfamily=default_font,
                    ha='left', va='center')
            y -= 0.03
            
        # map background and shapefiles:
        for ax in (axL, axR):
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
            ax.set_axisbelow(True)
            ax.grid(True, ls='--', color='lightgray', zorder=1)
            if geo_img is not None:
                ax.imshow(geo_img, extent=geo_ext, alpha=MAP_OPACITY, zorder=0)
            if shp_faults is not None:
                shp_faults.plot(ax=ax, color='black', edgecolor='k', lw=2, zorder=2)
            if shp_def is not None:
                shp_def.plot(ax=ax, color='black', edgecolor='k', lw=2, zorder=2)
            ax.set_xlabel("Longitude", fontsize=14, labelpad=11)

        if do_vs:

            threshold = 50.0 # min acceptable Vs in m/s. there is only one single station below this, it makes colorbar less meaningful
            df = df[df.Vs >= threshold]

            # recompute node‐only vmin/vmax
            vmin_excl = df.Vs.min()   
            vmax_excl = df.Vs.max()
            vmin_incl = min(vmin_excl, stars.Vs30_meters_per_sec.min()) if not stars.empty else vmin_excl
            vmax_incl = max(vmax_excl, stars.Vs30_meters_per_sec.max()) if not stars.empty else vmax_excl

            if suffix == "withStars":
                curr_vmin, curr_vmax = vmin_incl, vmax_incl
            else:  # USGS measurements (stars) plotted but not used in cbar range
                curr_vmin, curr_vmax = vmin_excl, vmax_excl

            # plot Vs scatter (nodes)
            sc_vs = axL.scatter(
                df.lon, df.lat,
                c=df.Vs, cmap=VS_CMAP,
                vmin=v0, vmax=v1,
                s=85, edgecolors='k', zorder=6,
                label="Node (Vs)")
            # plot nodes not yet deployed as empty circles
            missing_vs = stn[~stn.station.isin(df.station)]
            axL.scatter(
                missing_vs.longitude, missing_vs.latitude,
                facecolors='none', edgecolors='k',
                s=85, linewidths=1.2, zorder=4,
                label='Node not yet deployed')
            axR.scatter(
                missing_vs.longitude, missing_vs.latitude,
                facecolors='none', edgecolors='k',
                s=85, linewidths=1.2, zorder=4,
                label='Node not yet deployed')
            # plot USGS Vs30 stars
            if not stars.empty:
                sc_star = axL.scatter(
                    stars.Longitude, stars.Latitude,
                    c=(stars.Vs30_meters_per_sec if suffix == "withStars" else 'dodgerblue'),
                    cmap=(VS_CMAP if suffix == "withStars" else None),
                    vmin=(v0 if suffix == "withStars" else None),
                    vmax=(v1 if suffix == "withStars" else None),
                    marker='*', s=260, edgecolors='k', lw=.6, zorder=5)
                # add Vs30 labels
                for i, r in stars.reset_index().iterrows():
                    dx = 0.0015 if i % 2 == 0 else -0.0015
                    dy = 0.0015 if i % 2 == 0 else -0.0015
                    axL.text(
                        r.Longitude + dx, r.Latitude + dy,
                        str(int(r.Vs30_meters_per_sec)),
                        fontsize=9, weight='bold', color='navy',
                        ha='center', va='center', zorder=6,
                        bbox=dict(fc='white', ec='none', pad=.1, alpha=.7))
            # Vs colorbar 
            cbar_vs_ax = fig.add_axes([0.35, 0.10, 0.022, 0.82]) # x, y, width, height

            norm_vs = mcolors.Normalize(vmin=curr_vmin, vmax=curr_vmax)
            cbar_vs = fig.colorbar(
                cm.ScalarMappable(norm=norm_vs, cmap=VS_CMAP),
                cax=cbar_vs_ax,label='Vs (m/s)')
            cbar_vs.set_label('Vs (m/s)', fontsize=13,labelpad=10)
            ticks = np.linspace(curr_vmin, curr_vmax, 5)
            cbar_vs.set_ticks(ticks)
            cbar_vs.ax.set_yticklabels([f"{t:.0f}" for t in ticks])

        else:
            # No Vs inversion for horizontal components
            axL.set_facecolor("whitesmoke")
            axL.set_xticks([]); axL.set_yticks([])
            axL.text(0.5, 0.5, "no Vs inversion\nfor this comp",
                     ha="center", va="center", transform=axL.transAxes,
                     color="gray", fontsize=8)

        # correlation coefficient map (right panel)
        cmap_corr = truncate_colormap(plt.get_cmap("hsv"), 0.08, 0.98)
        norm_corr = mcolors.Normalize(
            vmin=np.nanmin(cc) if cc.size > 0 else 0,
            vmax=np.nanmax(cc) if cc.size > 0 else 1)
        sc_c = axR.scatter(
            ok_df.longitude, ok_df.latitude,
            c=cc, cmap=cmap_corr, norm=norm_corr,
            s=85, edgecolors='k', zorder=4,
            label='Node (corr)')
        # mark reference station with corr = 1.0
        if not ok_df.empty:
            axR.scatter(
                [ref_lon], [ref_lat],
                c=[1.0],               
                cmap=cmap_corr,
                norm=norm_corr,
                s=120,                  
                edgecolors='k',
                linewidths=2.5,           # extra thick border
                marker='o',
                zorder=6,
                label='_nolegend_')     

        if not miss_df.empty:
            axR.scatter(
                miss_df.longitude, miss_df.latitude,
                facecolors='none', edgecolors='k',
                s=85, lw=1.2, label='No data', zorder=4)

        # corr coeff cbar
        cbar_cc_ax = fig.add_axes([0.7, 0.10, 0.022, 0.82])
        cbar_cc = fig.colorbar(
            cm.ScalarMappable(norm=norm_corr, cmap=cmap_corr),
            cax=cbar_cc_ax)
        cbar_cc.set_label('Correlation coefficient relative to southernmost station', fontsize=12,labelpad=10)

        from matplotlib.lines import Line2D
        legend_handles_Vs = [
            Line2D([0], [0], marker='o', color='yellow', markeredgecolor='k',
                   linestyle='None', markersize=8, label='Deployed node'),
            Line2D([0], [0], marker='o', color='none', markeredgecolor='k',
                   linestyle='None', markersize=8, label='Node not yet deployed'),
            Line2D([0], [0], marker='*', color='dodgerblue', markeredgecolor='k',
                    linestyle='None', markersize=10, label='Vs30 measurement (USGS)')]
        legend_handles_cc = [
            Line2D([0], [0], marker='o', color='yellow', markeredgecolor='k',
                   linestyle='None', markersize=8, label='Deployed node'),
            Line2D([0], [0], marker='o', color='none', markeredgecolor='k',
                   linestyle='None', markersize=8, label='Node not yet deployed')]
        ref_handle = Line2D(
            [0], [0],
            marker='o',
            color='red',          
            markeredgecolor='k',
            markersize=8,
            markeredgewidth=2,   
            linestyle='None',
            label='Ref. station (corr=1.0)')
        axL.legend(handles=legend_handles_Vs, loc='lower right', fontsize=7)
        axR.legend(handles=legend_handles_cc + [ref_handle], loc='lower right', fontsize=7)

        axL.set_title(f"Vs computed from M7.2 P-wave polarization, filtered {FREQMIN}-{FREQMAX} Hz",
                      pad=12, fontsize=8)
        axR.set_title(f"Corr. coeff. re. southernmost station, {comp} component, {window_to_corrcoeff_begin}-{window_to_corrcoeff_end} s after event, filtered {CORR_FREQMIN}-{CORR_FREQMAX} Hz",
                      pad=12, fontsize=8)

        # add 1 km scale bar to each subplot
        scalebar(axL, km=1)
        scalebar(axR, km=1)

        # save fig
        out_png = f"{evt['name']}_Vs-corr_{compL}--{suffix}--southRef--corr-window-{window_to_corrcoeff_begin}-{window_to_corrcoeff_end}-sec--v3c.png".replace(' ', '_')
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print("saved →", out_png)
