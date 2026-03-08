"""
Antenna Downtilt Calculator — Streamlit
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import zipfile, io, math

# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Antenna Downtilt Calculator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background:#f0f4f8; color:#1e293b; }
.stApp { background:#f0f4f8; }
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #dde3ec; }
section[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }

.metric-card {
  background:#fff; border:1px solid #dde3ec; border-radius:10px;
  padding:14px 16px; margin-bottom:4px;
}
.metric-label { font-size:0.62rem; font-weight:600; text-transform:uppercase;
  letter-spacing:1.1px; color:#64748b; margin-bottom:5px; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.35rem;
  font-weight:700; color:#0d9488; line-height:1.1; margin-bottom:3px; }
.metric-sub { font-size:0.7rem; color:#64748b; margin-top:2px; }

.status-ok  { background:#fff; border:1px solid #dde3ec; border-left:3px solid #22c55e;
  border-radius:6px; padding:8px 13px; font-family:'JetBrains Mono',monospace;
  font-size:0.7rem; color:#64748b; margin-bottom:12px; }
.status-err { background:#fff; border:1px solid #dde3ec; border-left:3px solid #ef4444;
  border-radius:6px; padding:8px 13px; font-family:'JetBrains Mono',monospace;
  font-size:0.7rem; color:#64748b; margin-bottom:12px; }
.status-idle{ background:#fff; border:1px solid #dde3ec; border-left:3px solid #94a3b8;
  border-radius:6px; padding:8px 13px; font-family:'JetBrains Mono',monospace;
  font-size:0.7rem; color:#64748b; margin-bottom:12px; }

.tstat-row { display:flex; gap:9px; margin:8px 0 12px; flex-wrap:wrap; }
.tstat { flex:1; min-width:80px; background:#fff; border:1px solid #dde3ec;
  border-radius:7px; padding:10px; text-align:center; }
.tstat-val { font-family:'JetBrains Mono',monospace; font-size:0.92rem;
  font-weight:700; color:#d97706; }
.tstat-lbl { font-size:0.6rem; text-transform:uppercase; letter-spacing:.7px;
  color:#94a3b8; margin-top:3px; }

.assume { font-size:0.72rem; color:#64748b; background:#f8fafc; border:1px solid #dde3ec;
  border-radius:6px; padding:8px 12px; margin-bottom:14px; line-height:1.55; }
.assume code { background:#eff6ff; color:#0284c7; padding:1px 5px; border-radius:3px;
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; }

.sec-hdr { font-size:0.95rem; font-weight:700; color:#1e293b;
  margin:14px 0 8px; border-bottom:1px solid #dde3ec; padding-bottom:6px; }

/* Map legend */
.map-legend-box { background:#fff; border:1px solid #dde3ec; border-radius:8px;
  padding:12px 14px; margin-top:8px; }
.map-legend-title { font-size:0.7rem; font-weight:600; text-transform:uppercase;
  letter-spacing:.9px; color:#64748b; margin-bottom:8px; }
.mleg-row { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:10px; }
.mleg { display:flex; align-items:center; gap:6px; font-size:0.74rem; color:#64748b; font-weight:500; }
.mleg-box { width:14px; height:14px; border-radius:3px; flex-shrink:0; border:1.5px solid; }
.mleg-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.sm-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:6px; }
.sm { background:#f8fafc; border:1px solid #dde3ec; border-radius:6px; padding:9px 12px; }
.sm-lbl { font-size:0.63rem; text-transform:uppercase; letter-spacing:.8px; color:#94a3b8; margin-bottom:3px; }
.sm-val { font-family:'JetBrains Mono',monospace; font-size:0.88rem; font-weight:700; color:#1e293b; }

/* Click coordinate card */
.click-card { background:#fff; border:1px solid #dde3ec; border-radius:8px;
  padding:11px 14px; margin-top:8px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.click-icon { width:30px; height:30px; background:#7c3aed; border-radius:7px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-size:14px; color:#fff; }
.click-body { flex:1; min-width:200px; }
.click-title { font-size:0.65rem; font-weight:600; text-transform:uppercase;
  letter-spacing:1px; color:#64748b; margin-bottom:6px; }
.click-fields { display:flex; gap:20px; flex-wrap:wrap; }
.click-field span { font-size:0.62rem; font-weight:500; color:#94a3b8;
  text-transform:uppercase; letter-spacing:.7px; display:block; margin-bottom:2px; }
.click-field { font-family:'JetBrains Mono',monospace; font-size:1rem;
  font-weight:700; color:#7c3aed; }
.click-hint { font-size:0.68rem; color:#94a3b8; margin-top:5px; }
.click-idle { background:#f8fafc; border:1px dashed #dde3ec; border-radius:8px;
  padding:9px 14px; margin-top:8px; font-size:0.74rem; color:#94a3b8;
  display:flex; align-items:center; gap:7px; }

#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:1.6rem; }
label { color:#64748b !important; font-size:0.82rem !important; font-weight:500 !important; }
.stNumberInput input { background:#f8fafc !important; border:1px solid #dde3ec !important;
  color:#1e293b !important; font-family:'JetBrains Mono',monospace !important;
  font-size:0.82rem !important; border-radius:6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MATHS
# ─────────────────────────────────────────────────────
def to_m(v, units):      return v if units == "Metric (m, km)" else v * 0.3048
def fmt_d(m, units, dec=2):
    if units == "Metric (m, km)":
        return f"{m/1000:.{dec}f} km" if m >= 1000 else f"{m:.0f} m"
    ft = m / 0.3048
    return f"{ft/5280:.{dec}f} mi" if ft >= 5280 else f"{ft:.0f} ft"

def flat_geom(h, dt, vbw):
    d, half = math.radians(dt), math.radians(vbw/2)
    main_d = h / math.tan(d)
    near_d = h / math.tan(d + half)
    fa = max(0.0002, d - half)
    return main_d, h / math.tan(fa), near_d  # returns main, far, near

def flat_geom_full(h, dt, vbw):
    d, half = math.radians(dt), math.radians(vbw/2)
    main_d = h / math.tan(d)
    near_d = h / math.tan(d + half)
    far_d  = h / math.tan(max(0.0002, d - half))
    return main_d, near_d, far_d

def ray_intersect(h, site_elev, angle_deg, dem_d, dem_elev):
    dt = math.radians(angle_deg)
    for i in range(1, len(dem_d)):
        d0, e0, d1, e1 = dem_d[i-1], dem_elev[i-1], dem_d[i], dem_elev[i]
        r0 = site_elev + h - d0 * math.tan(dt)
        r1 = site_elev + h - d1 * math.tan(dt)
        if r0 >= e0 and r1 <= e1:
            denom = (r0 - r1) + (e1 - e0)
            t = (r0 - e0) / denom if denom > 1e-6 else 0
            return d0 + t * (d1 - d0)
    return None

def gc_dest(lat, lon, bearing, dist_m):
    R = 6_371_000
    b, dR = math.radians(bearing), dist_m / R
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1)*math.cos(dR) + math.cos(la1)*math.sin(dR)*math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b)*math.sin(dR)*math.cos(la1),
                            math.cos(dR) - math.sin(la1)*math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)

# ─────────────────────────────────────────────────────
# DEM FETCH
# ─────────────────────────────────────────────────────
# ── SRTM local elevation lookup ──────────────────────────────────────────────
# Uses the `srtm` package which auto-downloads and caches HGT tile files
# locally (~3 MB per 1°×1° tile). Egypt spans ~120 tiles total.
# Tiles are stored in ~/.cache/srtm/ and never need re-downloading.
# No API key, no rate limit, works fully offline after first tile download.
#
# Install once:  pip install srtm.py
#
# SRTM coverage for Egypt:
#   Latitude  22 N → 32 N
#   Longitude 25 E → 37 E
# Resolution: ~90 m (SRTM3) — accurate enough for RF terrain profiling.

import srtm as _srtm

@st.cache_resource(show_spinner=False)
def _get_srtm():
    """Load SRTM data object once and reuse across all calls."""
    return _srtm.get_data()

def _srtm_elevation(srtm_data, lat, lon):
    """Return elevation in metres for a single point, 0.0 if tile missing."""
    v = srtm_data.get_elevation(lat, lon)
    return float(v) if v is not None else 0.0

@st.cache_data(show_spinner=False)
def fetch_dem(lat, lon, az, dist_m, n=100):
    """
    Build an elevation profile of n points along azimuth `az` from (lat,lon)
    out to dist_m metres, using locally cached SRTM tiles.

    On first call for a new tile the srtm package downloads that HGT file
    (~3 MB) automatically; subsequent calls for the same area are instant.
    """
    srtm_data = _get_srtm()

    distances = np.linspace(0, dist_m, n)
    elevations = []
    for d in distances:
        p_lat, p_lon = gc_dest(lat, lon, az, d)
        elevations.append(_srtm_elevation(srtm_data, p_lat, p_lon))

    elev = np.array(elevations, dtype=float)

    # Sanity check — srtm returns 0 for ocean/missing; warn if all zeros
    if np.all(elev == 0.0):
        raise ValueError(
            "All elevation values are zero — tile may be missing or "
            "location is outside SRTM coverage (lat 60S–60N)."
        )

    return distances, elev

# ─────────────────────────────────────────────────────
# PLOTLY CHART  — matches rfuniverse style exactly
# ─────────────────────────────────────────────────────
def build_chart(h_m, dt_deg, vbw_deg, dist_m, main_d, near_d, far_d,
                dem_d, dem_elev, slider_d, units):

    N = 400
    xs = np.linspace(0, dist_m, N)
    has_dem   = dem_d is not None and len(dem_d) > 1
    site_elev = float(dem_elev[0]) if has_dem else 0.0
    far_angle = max(0.05, dt_deg - vbw_deg / 2)

    def elev_at(d):
        return float(np.interp(d, dem_d, dem_elev)) if has_dem else site_elev

    def ray_z(d, ang):
        return site_elev + h_m - d * math.tan(math.radians(ang))

    terrain_y = np.array([elev_at(d) for d in xs])
    main_ray   = np.array([ray_z(d, dt_deg) for d in xs])
    near_ray   = np.array([ray_z(d, dt_deg + vbw_deg/2) for d in xs])
    far_ray    = np.array([ray_z(d, far_angle) for d in xs])

    y_min = float(np.min(terrain_y)) - 20
    y_max = site_elev + h_m + 50

    ant_z = site_elev + h_m  # antenna absolute elevation (MSL)

    # ── LOS shadow detection ──────────────────────────────────────────────────
    # A point at distance d is shadowed if ANY terrain sample between 0 and d
    # rises ABOVE the straight line-of-sight from the antenna to that point.
    # This is true terrain shadowing — not a flat-ray comparison.
    def compute_los_clear(distances, terrain):
        n = len(distances)
        los_clear = np.ones(n, dtype=bool)
        for i in range(1, n):
            d_target = distances[i]
            z_target = terrain[i]
            if d_target <= 0:
                continue
            # LOS line height at each previous sample point
            los_line = ant_z + (z_target - ant_z) * (distances[:i] / d_target)
            # Shadowed if any intermediate terrain exceeds LOS line
            if np.any(terrain[:i] > los_line + 0.1):   # 0.1 m tolerance
                los_clear[i] = False
        return los_clear

    if has_dem:
        # Compute LOS on dense xs grid by interpolating DEM
        los_clear = compute_los_clear(xs, terrain_y)
    else:
        los_clear = np.ones(len(xs), dtype=bool)   # flat earth → no shadows

    # ── Signal classification for each x ──────────────────────────────────────
    in_foot = (xs >= near_d) & (xs <= far_d)
    strong  = los_clear  & ~in_foot    # LOS clear, outside footprint  → green
    shadow  = ~los_clear & ~in_foot    # LOS blocked, outside footprint → red
    # in_foot → orange regardless of LOS (it is the coverage zone)

    # Segment arrays: green/red/orange on terrain surface
    def make_seg(mask):
        return np.where(mask, terrain_y, np.nan)

    seg_green  = make_seg(strong)
    seg_red    = make_seg(shadow)
    seg_orange = make_seg(in_foot)

    # ── Vertical band fills ──
    band_strong = np.where(strong,  y_max, np.nan)
    band_foot   = np.where(in_foot, y_max, np.nan)
    band_shadow = np.where(shadow,  y_max, np.nan)

    fig = go.Figure()

    # ── Background vertical bands ──
    for band_y, col, name in [
        (band_strong, 'rgba(187,247,208,0.20)', '_bstrong'),
        (band_foot,   'rgba(254,215,170,0.28)', '_bfoot'),
        (band_shadow, 'rgba(254,202,202,0.25)', '_bshadow'),
    ]:
        fig.add_trace(go.Scatter(
            x=xs, y=band_y,
            fill='tozeroy', fillcolor=col,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name=name,
            connectgaps=False, hoverinfo='skip'
        ))

    # ── Terrain base fill (light teal) ──
    fig.add_trace(go.Scatter(
        x=xs, y=terrain_y,
        fill='tozeroy',
        fillcolor='rgba(186,230,253,0.40)',
        line=dict(color='#94a3b8', width=1.2),
        name='Terrain', showlegend=False,
        hovertemplate='Dist: %{x:.0f} m<br>Elev: %{y:.1f} m MSL<extra></extra>'
    ))

    # ── Coloured terrain line segments ──
    for seg_y, col, lw, leg in [
        (seg_green,  '#16a34a', 2.5, 'Strong signal'),
        (seg_red,    '#dc2626', 2.5, 'Shadowed'),
        (seg_orange, '#f97316', 3.0, 'Footprint'),
    ]:
        fig.add_trace(go.Scatter(
            x=xs, y=seg_y,
            line=dict(color=col, width=lw),
            showlegend=False, name=leg,
            connectgaps=False, hoverinfo='skip'
        ))

    # ── Main lobe ray (solid green) ──
    fig.add_trace(go.Scatter(
        x=xs, y=main_ray,
        line=dict(color='#16a34a', width=2),
        name='Main lobe ray',
        hovertemplate='Main lobe: %{y:.1f} m<extra></extra>'
    ))

    # ── Footprint lobe limits (dashed light blue) ──
    fig.add_trace(go.Scatter(
        x=xs, y=near_ray,
        line=dict(color='#7dd3fc', width=1.5, dash='dash'),
        name='Footprint lobe limits',
        hovertemplate='Near limit: %{y:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=far_ray,
        line=dict(color='#7dd3fc', width=1.5, dash='dash'),
        showlegend=False, name='_farlimit',
        hovertemplate='Far limit: %{y:.1f} m<extra></extra>'
    ))

    # ── Slider vertical line (red dashed) ──
    sl_elev = elev_at(slider_d)
    fig.add_trace(go.Scatter(
        x=[slider_d, slider_d],
        y=[y_min, y_max],
        line=dict(color='#dc2626', width=1.8, dash='dot'),
        name='Selected distance',
        hovertemplate=f'Selected: {fmt_d(slider_d, units)}<br>Terrain: {sl_elev:.1f} m<extra></extra>'
    ))

    # ── Antenna marker ──
    ant_elev = site_elev + h_m
    fig.add_trace(go.Scatter(
        x=[0], y=[ant_elev],
        mode='markers+text',
        marker=dict(size=11, color='#0ea5e9', symbol='circle',
                    line=dict(color='white', width=2)),
        text=[f'Antenna {h_m:.0f} m AGL'],
        textposition='top right',
        textfont=dict(family='Inter', size=10, color='#0ea5e9'),
        showlegend=False, name='Antenna',
        hovertemplate=f'Antenna<br>AGL: {h_m:.0f} m<br>MSL: {ant_elev:.1f} m<extra></extra>'
    ))

    # ── Main lobe hit marker ──
    hit_elev = elev_at(main_d)
    fig.add_trace(go.Scatter(
        x=[main_d], y=[hit_elev],
        mode='markers+text',
        marker=dict(size=10, color='#0d9488', symbol='circle',
                    line=dict(color='white', width=2)),
        text=[f'Main hit: {fmt_d(main_d, units)}'],
        textposition='top right',
        textfont=dict(family='Inter', size=10, color='#0d9488'),
        showlegend=False, name='Main lobe hit',
        hovertemplate=f'Main hit: {fmt_d(main_d, units)}<extra></extra>'
    ))

    # ── Antenna vertical marker line ──
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[site_elev, ant_elev],
        line=dict(color='#0ea5e9', width=3),
        showlegend=False, name='_antline', hoverinfo='skip'
    ))

    # Layout — LIGHT background like rfuniverse
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter', color='#64748b', size=11),
        margin=dict(l=55, r=20, t=40, b=50),
        height=320,
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.28,
            xanchor='left', x=0,
            font=dict(size=11, family='Inter', color='#64748b'),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
        ),
        xaxis=dict(
            title=dict(text='Distance (m)', font=dict(size=11, color='#94a3b8', family='Inter')),
            gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.1)',
            tickfont=dict(size=10, color='#94a3b8', family='JetBrains Mono'),
            range=[0, dist_m], showline=True, linecolor='#dde3ec',
        ),
        yaxis=dict(
            title=dict(text='Elevation (m)', font=dict(size=11, color='#94a3b8', family='Inter')),
            gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.1)',
            tickfont=dict(size=10, color='#94a3b8', family='JetBrains Mono'),
            range=[y_min, y_max], showline=True, linecolor='#dde3ec',
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#fff', bordercolor='#dde3ec',
                        font=dict(family='JetBrains Mono', size=11, color='#1e293b')),
        # Top annotation
        annotations=[
            dict(x=0.02, y=1.04, xref='paper', yref='paper', showarrow=False,
                 text=f'Main hit: {fmt_d(main_d, units)}',
                 font=dict(size=10, color='#64748b', family='Inter'), xanchor='left'),
            dict(x=0.38, y=1.04, xref='paper', yref='paper', showarrow=False,
                 text=f'Downtilt {dt_deg} deg | V-BW {vbw_deg} deg',
                 font=dict(size=10, color='#64748b', family='Inter'), xanchor='center'),
            dict(x=1.0, y=1.04, xref='paper', yref='paper', showarrow=False,
                 text='Green=strong, yellow=weak, red=shadowed by terrain',
                 font=dict(size=10, color='#64748b', family='Inter'), xanchor='right'),
        ]
    )

    return fig

# ─────────────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────────────
def build_map(lat, lon, az, hbw, main_d, near_d, far_d, dem_d, dist_m):
    m = folium.Map(location=[lat, lon], zoom_start=13,
                   tiles='OpenStreetMap', control_scale=True)
    half, S = hbw/2, 72
    outer = [gc_dest(lat, lon, az - half + i*hbw/S, far_d)  for i in range(S+1)]
    inner = [gc_dest(lat, lon, az - half + i*hbw/S, near_d) for i in range(S, -1, -1)]
    path  = [gc_dest(lat, lon, az, d) for d in (dem_d if dem_d is not None else np.linspace(0,dist_m,60))]

    folium.Polygon([(lat,lon)]+outer+[(lat,lon)],
        color='#0ea5e9', weight=1.5, fill=True, fill_color='#0ea5e9', fill_opacity=0.06).add_to(m)
    folium.Polygon(outer+inner,
        color='#4ade80', weight=2, fill=True, fill_color='#4ade80', fill_opacity=0.20).add_to(m)
    folium.PolyLine(path, color='#f97316', weight=2.2, opacity=0.8, dash_array='7 5').add_to(m)
    folium.PolyLine([(lat,lon), gc_dest(lat,lon,az,main_d)],
        color='#16a34a', weight=1.5, opacity=0.7).add_to(m)
    folium.CircleMarker([lat,lon], radius=9, color='#0ea5e9', fill=True,
        fill_color='#0ea5e9', fill_opacity=1, weight=2,
        popup=folium.Popup(f'<b>Antenna Site</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Az: {az}°')).add_to(m)
    hit = gc_dest(lat, lon, az, main_d)
    folium.CircleMarker(hit, radius=8, color='#0d9488', fill=True,
        fill_color='#0d9488', fill_opacity=1, weight=2,
        popup=folium.Popup(f'<b>Main Lobe Hit</b><br>{fmt_d(main_d,"Metric (m, km)")} from site')).add_to(m)
    # Clicking anywhere on the map shows a popup with lat/lon
    # and returns the coordinates to Streamlit via st_folium
    m.add_child(folium.LatLngPopup())
    return m

# ─────────────────────────────────────────────────────
# KMZ
# ─────────────────────────────────────────────────────
def build_kmz(lat, lon, az, hbw, main_d, near_d, far_d):
    half, S = hbw/2, 60
    def c(b, d):
        p = gc_dest(lat, lon, b, d); return f"{p[1]},{p[0]},0"
    outer = " ".join(c(az-half+i*hbw/S, far_d)  for i in range(S+1))
    inner = " ".join(c(az-half+i*hbw/S, near_d) for i in range(S,-1,-1))
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <n>Sector Az{az}</n>
  <Style id="s"><LineStyle><color>ffe9a50e</color><width>2</width></LineStyle>
    <PolyStyle><color>1ae9a50e</color></PolyStyle></Style>
  <Style id="f"><LineStyle><color>ff80de4a</color><width>2</width></LineStyle>
    <PolyStyle><color>3380de4a</color></PolyStyle></Style>
  <Placemark><n>Sector</n><styleUrl>#s</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {lon},{lat},0 {outer} {lon},{lat},0
    </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
  <Placemark><n>Footprint</n><styleUrl>#f</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {outer} {inner}
    </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
  <Placemark><n>Site</n>
    <Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>
</Document></kml>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return buf.getvalue()

# ─────────────────────────────────────────────────────
# SESSION STATE  — store ONLY raw DEM data, nothing derived
# ─────────────────────────────────────────────────────
for k, v in [('dem_d', None), ('dem_elev', None),
             ('dem_status', None), ('dem_msg', '')]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #dde3ec;padding-bottom:14px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">
    <div style="width:30px;height:30px;background:#0ea5e9;border-radius:7px;display:flex;
                align-items:center;justify-content:center;font-size:15px;color:#fff;">📡</div>
    <span style="font-size:1.2rem;font-weight:700;color:#1e293b;">Antenna Downtilt Calculator</span>
  </div>
  <div style="padding-left:40px;font-size:0.8rem;color:#64748b;">
    Calculate main lobe impact and ground footprint with an interactive RF visual.
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
                'letter-spacing:1.5px;color:#94a3b8;margin-bottom:12px;">Inputs</div>',
                unsafe_allow_html=True)
    st.caption("Set basic RF parameters:")

    units  = st.radio("Unit System", ["Metric (m, km)", "USA (ft, mi)"], horizontal=True)
    h_lbl  = "Antenna Height (m)" if units == "Metric (m, km)" else "Antenna Height (ft)"
    d_lbl  = "Distance (m)"       if units == "Metric (m, km)" else "Distance (ft)"

    h_raw  = st.number_input(h_lbl,  min_value=1.0,   value=30.0,  step=1.0,  format="%.1f")
    dt_deg = st.number_input("Downtilt (deg)",           min_value=0.1, max_value=89.0, value=6.0,  step=0.5)
    vbw    = st.number_input("Vertical Beamwidth (deg)", min_value=1.0, max_value=90.0, value=6.0,  step=0.5)
    hbw    = st.number_input("Horizontal Beamwidth (deg)", min_value=1.0, max_value=360.0, value=65.0, step=1.0)
    d_raw  = st.number_input(d_lbl,  min_value=100.0,  value=1000.0, step=100.0, format="%.0f")

    h_m    = to_m(h_raw, units)
    dist_m = to_m(d_raw, units)
    st.markdown(f"<div style='font-size:.72rem;color:#0284c7;font-family:JetBrains Mono,monospace;"
                f"margin-top:-6px;margin-bottom:8px;'>→ {fmt_d(dist_m, units)}</div>",
                unsafe_allow_html=True)
    st.divider()

    terrain_on = st.checkbox("Include elevation profile", value=True,
                             help="Uses site coordinates + azimuth and distance as profile length.")
    if terrain_on:
        site_lat = st.number_input("Site Latitude (deg)",   value=30.0028686, format="%.7f", step=0.0001)
        site_lon = st.number_input("Site Longitude (deg)",  value=31.0719953, format="%.7f", step=0.0001)
        az_deg   = st.number_input("Antenna Azimuth (deg)", min_value=0.0, max_value=360.0, value=80.0, step=1.0)

        if st.button("↻ Fetch Elevation Profile", type="primary", use_container_width=True):
            with st.spinner("Fetching elevation from Open-Meteo DEM…"):
                try:
                    d_arr, e_arr = fetch_dem(site_lat, site_lon, az_deg, dist_m, n=100)
                    # ── Store ONLY raw elevation data. All stats computed live below. ──
                    st.session_state.dem_d      = d_arr
                    st.session_state.dem_elev   = e_arr
                    st.session_state.dem_status = "ok"
                    st.session_state.dem_msg    = f"Elevation profile loaded — {len(d_arr)} DEM samples. Source: Open-Meteo DEM."
                    st.rerun()
                except Exception as ex:
                    st.session_state.dem_status = "error"
                    st.session_state.dem_msg    = f"Fetch failed: {ex}"
                    st.session_state.dem_d      = None
                    st.session_state.dem_elev   = None
                    st.rerun()
    else:
        site_lat, site_lon, az_deg = 30.0028686, 31.0719953, 80.0
        # Clear DEM when terrain toggled off
        st.session_state.dem_d     = None
        st.session_state.dem_elev  = None
        st.session_state.dem_status = None
        st.session_state.dem_msg   = ""

    st.divider()
    if st.button("↺ Reset", use_container_width=True):
        for k in ['dem_d', 'dem_elev', 'dem_status', 'dem_msg']:
            st.session_state[k] = "" if k == "dem_msg" else None
        st.rerun()

# ─────────────────────────────────────────────────────
# COMPUTE GEOMETRY  (runs on every rerender with current inputs)
# ─────────────────────────────────────────────────────
dem_d    = st.session_state.dem_d
dem_elev = st.session_state.dem_elev
has_dem  = dem_d is not None and dem_elev is not None and len(dem_d) > 1

if has_dem and terrain_on:
    se     = float(dem_elev[0])
    main_d = ray_intersect(h_m, se, dt_deg,                   dem_d, dem_elev)
    near_d = ray_intersect(h_m, se, dt_deg + vbw / 2,         dem_d, dem_elev)
    far_d  = ray_intersect(h_m, se, max(0.05, dt_deg - vbw/2), dem_d, dem_elev)
    fb     = flat_geom_full(h_m, dt_deg, vbw)
    if not main_d: main_d = fb[0]
    if not near_d: near_d = fb[1]
    if not far_d:  far_d  = fb[2]
    note   = "Terrain-adjusted"
else:
    main_d, near_d, far_d = flat_geom_full(h_m, dt_deg, vbw)
    note   = "Flat-earth model"

# ─────────────────────────────────────────────────────
# COMPUTE TERRAIN STATS LIVE  (recalculates on every input change)
# Uses stored dem_d / dem_elev  +  current h_m, dt_deg
# ─────────────────────────────────────────────────────
live_stats = None
if has_dem and terrain_on:
    se_stats  = float(dem_elev[0])
    ant_z_stats = se_stats + h_m
    # LOS-based shadow detection on actual DEM samples
    n_dem = len(dem_d)
    los_arr = np.ones(n_dem, dtype=bool)
    for i in range(1, n_dem):
        d_t = dem_d[i]
        if d_t <= 0:
            continue
        los_line = ant_z_stats + (dem_elev[i] - ant_z_stats) * (dem_d[:i] / d_t)
        if np.any(dem_elev[:i] > los_line + 0.1):
            los_arr[i] = False
    n_above   = int(np.sum(los_arr))
    avg_sig   = round(n_above / n_dem * 100)
    shadow    = 100 - avg_sig
    blocked   = ~los_arr
    first_obs = float(dem_d[np.argmax(blocked)]) if np.any(blocked) else None
    live_stats = dict(
        avg       = avg_sig,
        shadow    = shadow,
        first_obs = first_obs,
        n         = len(dem_d),
    )

# ─────────────────────────────────────────────────────
# STATUS BAR  (also live — shows current stats in message)
# ─────────────────────────────────────────────────────
s = st.session_state.dem_status
if s == "ok" and live_stats:
    obs_str = fmt_d(live_stats['first_obs'], units) if live_stats['first_obs'] else "None"
    live_msg = (f"Elevation loaded ({live_stats['n']} DEM samples) · "
                f"Avg signal: {live_stats['avg']}% · "
                f"Shadowed: {live_stats['shadow']}% · "
                f"1st obstruction: {obs_str}")
    st.markdown(f'<div class="status-ok">✓ {live_msg}</div>', unsafe_allow_html=True)
elif s == "error":
    st.markdown(f'<div class="status-err">✕ {st.session_state.dem_msg}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="status-idle">Flat model · h={fmt_d(h_m, units)} AGL · '
        f'Main lobe={fmt_d(main_d, units)}</div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Main Lobe Impact Point</div>
      <div class="metric-value">{fmt_d(main_d, units)}</div>
      <div class="metric-sub">Ground intersection of central lobe ({note}).</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Footprint Coverage</div>
      <div class="metric-value" style="font-size:1.05rem;">{fmt_d(near_d, units)} to {fmt_d(far_d, units)}</div>
      <div class="metric-sub">{note} · VBW {vbw:.1f}° · Width: {fmt_d(far_d - near_d, units)}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# ASSUMPTION NOTE
# ─────────────────────────────────────────────────────
u_lbl = "Metric (m/km)" if units == "Metric (m, km)" else "USA (ft/mi)"
st.markdown(f"""<div class="assume">
  <strong>Assumption:</strong> Vertical beamwidth defaults to <code>{vbw:.1f} deg</code>,
  but you can modify it. Unit system: {u_lbl}.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TERRAIN STATS ROW  — live, updates on every input change
# ─────────────────────────────────────────────────────
if live_stats:
    obs_str = fmt_d(live_stats['first_obs'], units) if live_stats['first_obs'] else "None"
    st.markdown(f"""<div class="tstat-row">
      <div class="tstat">
        <div class="tstat-val">{obs_str}</div>
        <div class="tstat-lbl">1st Obstruction</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# VISUAL PROFILE CHART + SLIDER
# ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Visual Profile</div>', unsafe_allow_html=True)

slider_d = st.slider(
    "Footprint vs selected distance",
    min_value=0.0, max_value=float(dist_m),
    value=float(dist_m * 0.25),
    step=max(1.0, dist_m / 500), format="%.0f m"
)

sl_elev = float(np.interp(slider_d, dem_d, dem_elev)) if has_dem else 0.0
in_foot = near_d <= slider_d <= far_d
ci1, ci2, ci3 = st.columns(3)
ci1.metric("📍 Position",     fmt_d(slider_d, units))
ci2.metric("⛰ Terrain Elev", f"{sl_elev:.1f} m MSL" if has_dem else "—")
ci3.metric("📶 Footprint",    "✓ Inside" if in_foot else "✗ Outside")

fig = build_chart(h_m, dt_deg, vbw, dist_m, main_d, near_d, far_d,
                  dem_d, dem_elev, slider_d, units)
st.plotly_chart(fig, use_container_width=True,
                config={'displayModeBar': True, 'displaylogo': False})

# ─────────────────────────────────────────────────────
# SECTOR MAP
# ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Sector Map (OSM)</div>', unsafe_allow_html=True)
st.caption("Sector built from site coordinates, azimuth, horizontal beamwidth, and auto-adjusted distance.")

map_obj  = build_map(site_lat, site_lon, az_deg, hbw, main_d, near_d, far_d, dem_d, dist_m)
map_data = st_folium(map_obj, width="100%", height=380,
                     returned_objects=["last_clicked"],
                     key="sector_map")

# ── Clicked-point display ──────────────────────────────
_clicked = (map_data or {}).get("last_clicked")
if _clicked and _clicked.get("lat") is not None:
    _clat, _clng = _clicked["lat"], _clicked["lng"]
    # Great-circle distance from antenna site to clicked point
    _R = 6_371_000.0
    _dlat = math.radians(_clat - site_lat)
    _dlon = math.radians(_clng - site_lon)
    _a   = (math.sin(_dlat/2)**2
            + math.cos(math.radians(site_lat))
            * math.cos(math.radians(_clat))
            * math.sin(_dlon/2)**2)
    _click_dist = _R * 2 * math.atan2(math.sqrt(_a), math.sqrt(1 - _a))
    st.markdown(f"""
<div class="click-card">
  <div class="click-icon">📍</div>
  <div class="click-body">
    <div class="click-title">Selected Point — Map Click</div>
    <div class="click-fields">
      <div class="click-field"><span>Latitude</span>{_clat:.6f}°</div>
      <div class="click-field"><span>Longitude</span>{_clng:.6f}°</div>
      <div class="click-field"><span>Distance from Site</span>{fmt_d(_click_dist, units)}</div>
    </div>
    <div class="click-hint">Click anywhere on the map to update</div>
  </div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="click-idle">🖱 Click anywhere on the map to see its coordinates and distance from the antenna site</div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MAP LEGEND — Sector Metrics (also live)
# ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="map-legend-box">
  <div class="map-legend-title">Sector Metrics</div>
  <div class="mleg-row">
    <div class="mleg"><div class="mleg-box" style="background:rgba(14,165,233,0.10);border-color:#0ea5e9;"></div>Sector outline</div>
    <div class="mleg"><div class="mleg-box" style="background:rgba(74,222,128,0.22);border-color:#4ade80;"></div>Footprint zone</div>
    <div class="mleg"><div class="mleg-dot" style="background:#0ea5e9;"></div>Antenna Site</div>
    <div class="mleg"><div class="mleg-dot" style="background:#0d9488;"></div>Main Lobe Hit</div>
  </div>
  <div class="sm-row">
    <div class="sm"><div class="sm-lbl">Main Lobe</div><div class="sm-val">{fmt_d(main_d, units)}</div></div>
    <div class="sm"><div class="sm-lbl">Footprint</div>
      <div class="sm-val" style="font-size:0.8rem;">{fmt_d(near_d, units)} to {fmt_d(far_d, units)}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MAP FOOTER + KMZ EXPORT
# ─────────────────────────────────────────────────────
col_meta, col_kmz = st.columns([3, 1])
with col_meta:
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#94a3b8;margin-top:8px;'>"
        f"Site {site_lat:.6f}, {site_lon:.6f} · Az {az_deg:.0f}° · "
        f"H-BW {hbw:.0f}° · Radius {fmt_d(far_d, units)}.</div>",
        unsafe_allow_html=True)
with col_kmz:
    kmz = build_kmz(site_lat, site_lon, az_deg, hbw, main_d, near_d, far_d)
    st.download_button("⬇ Export Sector KMZ", data=kmz,
                       file_name=f"antenna_az{az_deg:.0f}.kmz",
                       mime="application/vnd.google-earth.kmz",
                       use_container_width=True)