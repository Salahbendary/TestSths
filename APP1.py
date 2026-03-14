"""
Antenna Downtilt Calculator — Streamlit
Elevation: Copernicus DEM 30m via Open-Meteo (primary), Open-Elevation fallback
"""

import streamlit as st
import numpy as np
import requests
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import zipfile, io, math
import time

# ── Copernicus DEM 30m via Open-Meteo API (primary source) ──────────────────
def _elev_open_meteo(lats, lons):
    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={','.join(str(x) for x in lats)}"
        f"&longitude={','.join(str(x) for x in lons)}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()["elevation"]

def _elev_open_elevation(lats, lons):
    """Second cloud fallback — open-elevation.com"""
    locations = [{"latitude": la, "longitude": lo}
                 for la, lo in zip(lats, lons)]
    r = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        json={"locations": locations}, timeout=20,
    )
    r.raise_for_status()
    return [pt["elevation"] for pt in r.json()["results"]]

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

/* Elevation source badge */
.src-badge { display:inline-block; font-size:0.64rem; font-weight:600;
  font-family:'JetBrains Mono',monospace; padding:2px 8px; border-radius:4px;
  margin-left:6px; }
.src-local  { background:#dcfce7; color:#15803d; }
.src-cloud  { background:#fef3c7; color:#92400e; }

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
    return main_d, h / math.tan(fa), near_d

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
# ELEVATION FETCH  — Copernicus DEM 30m primary, Open-Elevation fallback
# ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_dem(lat, lon, az, dist_m, n=100):
    """
    Fetch elevation profile along a great-circle path.

    Priority:
      1. Open-Meteo elevation API — uses Copernicus DEM GLO-30 (30m resolution)
         cloud-based, 600 req/min free tier.
      2. Open-Elevation API (cloud, free, slower) — fallback.

    Returns (distances_m array, elevations_m array, source_label)
    """
    lats, lons = [], []
    for i in range(n):
        p = gc_dest(lat, lon, az, i / (n - 1) * dist_m)
        lats.append(round(p[0], 7))
        lons.append(round(p[1], 7))

    # ── 1. Copernicus DEM 30m via Open-Meteo API ────────────────────────────
    CHUNK = 25
    elevs = []
    source = "Copernicus DEM 30m"
    for start in range(0, n, CHUNK):
        lat_c = lats[start: start + CHUNK]
        lon_c = lons[start: start + CHUNK]
        try:
            elevs.extend(_elev_open_meteo(lat_c, lon_c))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(2.0)
                try:
                    elevs.extend(_elev_open_meteo(lat_c, lon_c))
                except Exception:
                    # ── 2. Open-Elevation fallback ────────────────────────
                    elevs.extend(_elev_open_elevation(lat_c, lon_c))
                    source = "Open-Elevation (cloud)"
            else:
                raise
        except Exception:
            # ── 2. Open-Elevation fallback ────────────────────────────────
            elevs.extend(_elev_open_elevation(lat_c, lon_c))
            source = "Open-Elevation (cloud)"
        if start + CHUNK < n:
            time.sleep(0.15)

    return np.linspace(0, dist_m, n), np.array(elevs, dtype=float), source


# ─────────────────────────────────────────────────────
# LOBE HELPERS
# ─────────────────────────────────────────────────────
def _get_clipped_ray(xs, terrain_y, site_elev, h_m, angle_deg):
    """Return (xs_c, ys_c) clipped at first terrain intersection (with interpolated hit point)."""
    tan_a  = math.tan(math.radians(angle_deg))
    ray_y  = site_elev + h_m - xs * tan_a
    xs_c, ys_c = [], []
    for i in range(len(xs)):
        if ray_y[i] >= terrain_y[i]:
            xs_c.append(float(xs[i]))
            ys_c.append(float(ray_y[i]))
        else:
            if i > 0:
                r0, r1 = ray_y[i-1], ray_y[i]
                t0, t1 = terrain_y[i-1], terrain_y[i]
                denom = (r0 - r1) - (t0 - t1)
                if abs(denom) > 1e-6:
                    frac   = (r0 - t0) / denom
                    xi     = float(xs[i-1]) + frac * float(xs[i] - xs[i-1])
                    yi     = t0 + frac * (t1 - t0)
                    xs_c.append(xi); ys_c.append(yi)
            break
    if not xs_c:
        xs_c, ys_c = [float(xs[0])], [float(site_elev + h_m)]
    return np.array(xs_c), np.array(ys_c)


def _lobe_polygon(xs_ray, ys_ray, xs_ray2, ys_ray2, xs_all, terrain_all):
    """
    Build closed polygon between two clipped rays, bridged by terrain at hit points.
    ray1 must extend FARTHER than ray2 (i.e. xs_ray[-1] >= xs_ray2[-1]).
    """
    x1, x2 = xs_ray[-1], xs_ray2[-1]
    # terrain bridge from x2 → x1 (reversed so polygon winds correctly)
    mask   = (xs_all >= x2) & (xs_all <= x1)
    br_x   = xs_all[mask][::-1]
    br_y   = terrain_all[mask][::-1]
    poly_x = list(xs_ray)  + list(br_x)  + list(xs_ray2[::-1])
    poly_y = list(ys_ray)  + list(br_y)  + list(ys_ray2[::-1])
    return poly_x, poly_y


def _lower_polygon(xs_near, ys_near, xs_all, terrain_all):
    """Polygon for lower lobe: near_ray forward, then terrain back to x=0."""
    x_hit  = xs_near[-1]
    mask   = xs_all <= x_hit
    br_x   = xs_all[mask][::-1]
    br_y   = terrain_all[mask][::-1]
    return list(xs_near) + list(br_x), list(ys_near) + list(br_y)


# ─────────────────────────────────────────────────────
# PLOTLY CHART
# ─────────────────────────────────────────────────────
def build_chart(h_m, dt_deg, vbw_deg, dist_m, main_d, near_d, far_d,
                dem_d, dem_elev, slider_d, units, az_deg=0.0):

    N = 400
    xs = np.linspace(0, dist_m, N)
    has_dem   = dem_d is not None and len(dem_d) > 1
    site_elev = float(dem_elev[0]) if has_dem else 0.0
    far_angle = max(0.05, dt_deg - vbw_deg / 2)

    def elev_at(d):
        return float(np.interp(d, dem_d, dem_elev)) if has_dem else site_elev

    terrain_y = np.array([elev_at(d) for d in xs])
    y_min     = float(np.min(terrain_y)) - 20
    y_max     = site_elev + h_m + 50

    # ── Clipped ray arrays ────────────────────────────────────────────────
    xs_far,  ys_far  = _get_clipped_ray(xs, terrain_y, site_elev, h_m, far_angle)
    xs_main, ys_main = _get_clipped_ray(xs, terrain_y, site_elev, h_m, dt_deg)
    xs_near, ys_near = _get_clipped_ray(xs, terrain_y, site_elev, h_m, dt_deg + vbw_deg/2)

    # ── Intersection distances on terrain ────────────────────────────────
    far_hit_x  = float(xs_far[-1])
    main_hit_x = float(xs_main[-1])
    near_hit_x = float(xs_near[-1])

    # ── Lobe colours ─────────────────────────────────────────────────────
    C_UPPER_FILL = 'rgba(59,130,246,0.28)'
    C_UPPER_LINE = '#3b82f6'
    C_MAIN_FILL  = 'rgba(248,113,113,0.28)'
    C_MAIN_LINE  = '#f87171'
    C_LOWER_FILL = 'rgba(253,224,71,0.28)'
    C_LOWER_LINE = '#fde047'

    fig = go.Figure()

    # ── Terrain fill (light blue) ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=terrain_y,
        fill='tozeroy',
        fillcolor='rgba(125,211,252,0.30)',
        line=dict(color='#7dd3fc', width=1.5),
        name='Terrain',
        hovertemplate='Dist: %{x:.0f} m<br>Elev: %{y:.1f} m MSL<extra></extra>'
    ))

    # ── Lower lobe fill (light yellow) ────────────────────────────────────
    lo_px, lo_py = _lower_polygon(xs_near, ys_near, xs, terrain_y)
    fig.add_trace(go.Scatter(
        x=lo_px, y=lo_py,
        fill='toself', fillcolor=C_LOWER_FILL,
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_lower_fill', hoverinfo='skip'
    ))

    # ── Main lobe fill (light red) ────────────────────────────────────────
    # Ensure ordering: main extends farther than near
    if main_hit_x >= near_hit_x:
        ma_px, ma_py = _lobe_polygon(xs_main, ys_main, xs_near, ys_near, xs, terrain_y)
    else:
        ma_px, ma_py = _lobe_polygon(xs_near, ys_near, xs_main, ys_main, xs, terrain_y)
    fig.add_trace(go.Scatter(
        x=ma_px, y=ma_py,
        fill='toself', fillcolor=C_MAIN_FILL,
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_main_fill', hoverinfo='skip'
    ))

    # ── Upper lobe fill (blue) ────────────────────────────────────────────
    if far_hit_x >= main_hit_x:
        up_px, up_py = _lobe_polygon(xs_far, ys_far, xs_main, ys_main, xs, terrain_y)
    else:
        up_px, up_py = _lobe_polygon(xs_main, ys_main, xs_far, ys_far, xs, terrain_y)
    fig.add_trace(go.Scatter(
        x=up_px, y=up_py,
        fill='toself', fillcolor=C_UPPER_FILL,
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_upper_fill', hoverinfo='skip'
    ))

    # ── Lobe ray lines (clipped) ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs_far, y=ys_far,
        line=dict(color=C_UPPER_LINE, width=2, dash='solid'),
        name=f'Upper Lobe ({fmt_d(far_hit_x, units)})',
        connectgaps=False,
        hovertemplate='Upper: %{y:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xs_main, y=ys_main,
        line=dict(color=C_MAIN_LINE, width=2, dash='solid'),
        name=f'Main Lobe ({fmt_d(main_hit_x, units)})',
        connectgaps=False,
        hovertemplate='Main: %{y:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xs_near, y=ys_near,
        line=dict(color=C_LOWER_LINE, width=2, dash='solid'),
        name=f'Lower Lobe ({fmt_d(near_hit_x, units)})',
        connectgaps=False,
        hovertemplate='Lower: %{y:.1f} m<extra></extra>'
    ))

    # ── Intersection markers ──────────────────────────────────────────────
    for hx, col in [(far_hit_x, C_UPPER_LINE), (main_hit_x, C_MAIN_LINE), (near_hit_x, C_LOWER_LINE)]:
        fig.add_trace(go.Scatter(
            x=[hx], y=[elev_at(hx)],
            mode='markers',
            marker=dict(size=9, color=col, symbol='x', line=dict(color='white', width=1.5)),
            showlegend=False, name='_hit',
            hovertemplate=f'Hit: {fmt_d(hx, units)}<extra></extra>'
        ))

    # ── Slider line ───────────────────────────────────────────────────────
    sl_elev = elev_at(slider_d)
    fig.add_trace(go.Scatter(
        x=[slider_d, slider_d], y=[y_min, y_max],
        line=dict(color='#dc2626', width=1.8, dash='dot'),
        name='Selected distance',
        hovertemplate=f'Selected: {fmt_d(slider_d, units)}<br>Terrain: {sl_elev:.1f} m<extra></extra>'
    ))

    # ── Antenna marker ────────────────────────────────────────────────────
    ant_elev = site_elev + h_m
    fig.add_trace(go.Scatter(
        x=[0], y=[ant_elev],
        mode='markers+text',
        marker=dict(size=11, color='#ef4444', symbol='diamond',
                    line=dict(color='white', width=2)),
        text=[f'Antenna {h_m:.0f} m AGL'], textposition='top right',
        textfont=dict(family='Inter', size=10, color='#ef4444'),
        showlegend=False, name='Antenna',
        hovertemplate=f'Antenna<br>AGL: {h_m:.0f} m<br>MSL: {ant_elev:.1f} m<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[site_elev, ant_elev],
        line=dict(color='#ef4444', width=3),
        showlegend=False, name='_antline', hoverinfo='skip'
    ))

    # ── Layout ────────────────────────────────────────────────────────────
    title_txt = (f'Terrain Profile + Lobe Projection | '
                 f'Az: {az_deg:.2f}° | Tilt: {dt_deg:.2f}° | '
                 f'VB: {vbw_deg:.2f}° | Profile: {fmt_d(dist_m, units)}')

    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(family='Inter', color='#94a3b8', size=11),
        margin=dict(l=60, r=20, t=55, b=55),
        height=340,
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.28,
            xanchor='left', x=0,
            font=dict(size=11, family='Inter', color='#cbd5e1'),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
        ),
        xaxis=dict(
            title=dict(text='Distance in meters', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[0, dist_m], showline=True, linecolor='#334155',
        ),
        yaxis=dict(
            title=dict(text='Elevation (m MSL)', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[y_min, y_max], showline=True, linecolor='#334155',
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155',
                        font=dict(family='JetBrains Mono', size=11, color='#e2e8f0')),
        annotations=[
            dict(x=0.5, y=1.06, xref='paper', yref='paper', showarrow=False,
                 text=title_txt,
                 font=dict(size=11, color='#cbd5e1', family='Inter', weight=600),
                 xanchor='center'),
        ]
    )
    return fig


# ─────────────────────────────────────────────────────
# 2D LOBE PROJECTION CHART (dark theme)
# ─────────────────────────────────────────────────────
def build_lobe_chart(h_m, dt_deg, vbw_deg):
    """Flat-earth 2D antenna lobe projection — triangular filled areas."""
    far_angle = max(0.05, dt_deg - vbw_deg / 2)
    far_d_f   = h_m / math.tan(math.radians(far_angle))
    main_d_f  = h_m / math.tan(math.radians(dt_deg))
    near_d_f  = h_m / math.tan(math.radians(dt_deg + vbw_deg / 2))
    x_max     = far_d_f * 1.08

    fig = go.Figure()

    # Upper Lobe — blue (widest, rendered first/behind)
    fig.add_trace(go.Scatter(
        x=[0, far_d_f, main_d_f, 0],
        y=[h_m, 0, 0, h_m],
        fill='toself',
        fillcolor='rgba(59,130,246,0.55)',
        line=dict(color='rgba(59,130,246,0.85)', width=1.5),
        name='Upper Lobe',
        hoverinfo='skip'
    ))

    # Main Lobe — light red
    fig.add_trace(go.Scatter(
        x=[0, main_d_f, near_d_f, 0],
        y=[h_m, 0, 0, h_m],
        fill='toself',
        fillcolor='rgba(248,113,113,0.55)',
        line=dict(color='rgba(248,113,113,0.85)', width=1.5),
        name='Main Lobe',
        hoverinfo='skip'
    ))

    # Lower Lobe — light yellow (narrowest, rendered last/front)
    fig.add_trace(go.Scatter(
        x=[0, near_d_f, 0, 0],
        y=[h_m, 0, 0, h_m],
        fill='toself',
        fillcolor='rgba(253,224,71,0.55)',
        line=dict(color='rgba(253,224,71,0.85)', width=1.5),
        name='Lower Lobe',
        hoverinfo='skip'
    ))

    # Antenna marker (red diamond)
    fig.add_trace(go.Scatter(
        x=[0], y=[h_m],
        mode='markers',
        marker=dict(size=12, color='#ef4444', symbol='diamond',
                    line=dict(color='white', width=1.5)),
        name='Antenna',
        hovertemplate=f'Antenna: {h_m:.0f} m AGL<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='Lobe Distance Projection',
            font=dict(size=13, color='#e2e8f0', family='Inter', weight=700),
            x=0.03, xanchor='left'
        ),
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(family='Inter', color='#94a3b8', size=11),
        margin=dict(l=60, r=20, t=65, b=55),
        height=380,
        legend=dict(
            orientation='v', yanchor='top', y=0.95, xanchor='right', x=0.98,
            font=dict(size=11, family='Inter', color='#cbd5e1'),
            bgcolor='rgba(0,0,0,0.35)', borderwidth=0,
        ),
        xaxis=dict(
            title=dict(text='Distance (meters)', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[0, x_max], showline=True, linecolor='#334155',
        ),
        yaxis=dict(
            title=dict(text='Antenna Height (m)', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[0, h_m * 1.18], showline=True, linecolor='#334155',
        ),
        annotations=[
            dict(x=0.03, y=1.10, xref='paper', yref='paper', showarrow=False,
                 text='Theoretical lobe distances relative to the antenna',
                 font=dict(size=10, color='#64748b', family='Inter'), xanchor='left'),
        ]
    )
    return fig


# ─────────────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────────────
def build_map(lat, lon, az, hbw, main_d, near_d, far_d, dem_d, dist_m):
    m = folium.Map(
        location=[lat, lon], zoom_start=13, control_scale=True,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )
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
# SESSION STATE
# ─────────────────────────────────────────────────────
for k, v in [('dem_d', None), ('dem_elev', None),
             ('dem_status', None), ('dem_msg', ''), ('dem_source', '')]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────

# Show Copernicus DEM availability banner once
_copernicus_badge = '<span class="src-badge src-local">🌍 Copernicus DEM 30m — primary elevation source</span>'

st.markdown(f"""
<div style="border-bottom:1px solid #dde3ec;padding-bottom:14px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">
    <div style="width:30px;height:30px;background:#0ea5e9;border-radius:7px;display:flex;
                align-items:center;justify-content:center;font-size:15px;color:#fff;">📡</div>
    <span style="font-size:1.2rem;font-weight:700;color:#1e293b;">Antenna Downtilt Calculator</span>
    {_copernicus_badge}
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

        # Show which source will be used
        st.markdown(
            "<div style='font-size:0.68rem;color:#15803d;background:#dcfce7;"
            "border-radius:5px;padding:5px 10px;margin-bottom:4px;'>"
            "🌍 Will use Copernicus DEM 30m via Open-Meteo API</div>",
            unsafe_allow_html=True)

        if st.button("↻ Fetch Elevation Profile", type="primary", use_container_width=True):
            with st.spinner("Loading Copernicus DEM 30m elevation…"):
                try:
                    d_arr, e_arr, source = fetch_dem(site_lat, site_lon, az_deg, dist_m, n=100)
                    st.session_state.dem_d      = d_arr
                    st.session_state.dem_elev   = e_arr
                    st.session_state.dem_status = "ok"
                    st.session_state.dem_source = source
                    st.session_state.dem_msg    = (
                        f"Elevation loaded — {len(d_arr)} samples · Source: {source}")
                    st.rerun()
                except Exception as ex:
                    st.session_state.dem_status = "error"
                    st.session_state.dem_msg    = f"Fetch failed: {ex}"
                    st.session_state.dem_d      = None
                    st.session_state.dem_elev   = None
                    st.session_state.dem_source = ""
                    st.rerun()
    else:
        site_lat, site_lon, az_deg = 30.0028686, 31.0719953, 80.0
        st.session_state.dem_d      = None
        st.session_state.dem_elev   = None
        st.session_state.dem_status = None
        st.session_state.dem_msg    = ""
        st.session_state.dem_source = ""

    st.divider()
    if st.button("↺ Reset", use_container_width=True):
        for k in ['dem_d', 'dem_elev', 'dem_status', 'dem_msg', 'dem_source']:
            st.session_state[k] = "" if k in ("dem_msg", "dem_source") else None
        st.rerun()

# ─────────────────────────────────────────────────────
# COMPUTE GEOMETRY
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
# LIVE TERRAIN STATS
# ─────────────────────────────────────────────────────
live_stats = None
if has_dem and terrain_on:
    se_stats  = float(dem_elev[0])
    ray_z     = se_stats + h_m - dem_d * np.tan(np.radians(dt_deg))
    above_arr = ray_z >= dem_elev
    n_above   = int(np.sum(above_arr))
    avg_sig   = round(n_above / len(dem_d) * 100)
    shadow    = 100 - avg_sig
    blocked   = ~above_arr
    first_obs = float(dem_d[np.argmax(blocked)]) if np.any(blocked) else None
    live_stats = dict(
        avg       = avg_sig,
        shadow    = shadow,
        first_obs = first_obs,
        n         = len(dem_d),
    )

# ─────────────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────────────
s      = st.session_state.dem_status
src    = st.session_state.dem_source
if src == "Copernicus DEM 30m":
    src_html = '<span class="src-badge src-local">Copernicus DEM 30m</span>'
elif src:
    src_html = '<span class="src-badge src-cloud">cloud API</span>'
else:
    src_html = ''

if s == "ok" and live_stats:
    obs_str  = fmt_d(live_stats['first_obs'], units) if live_stats['first_obs'] else "None"
    live_msg = (f"Elevation loaded ({live_stats['n']} samples) · "
                f"Avg signal: {live_stats['avg']}% · "
                f"Shadowed: {live_stats['shadow']}% · "
                f"1st obstruction: {obs_str}")
    st.markdown(f'<div class="status-ok">✓ {live_msg} {src_html}</div>', unsafe_allow_html=True)
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

u_lbl = "Metric (m/km)" if units == "Metric (m, km)" else "USA (ft/mi)"
st.markdown(f"""<div class="assume">
  <strong>Assumption:</strong> Vertical beamwidth defaults to <code>{vbw:.1f} deg</code>,
  but you can modify it. Unit system: {u_lbl}.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TERRAIN STATS ROW
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
                  dem_d, dem_elev, slider_d, units, az_deg=az_deg)
st.plotly_chart(fig, use_container_width=True,
                config={'displayModeBar': True, 'displaylogo': False})

# ─────────────────────────────────────────────────────
# GRID ROW: Sector Map (Esri Satellite) + Lobe Projection
# ─────────────────────────────────────────────────────
map_col, lobe_col = st.columns(2, gap="medium")

with map_col:
    st.markdown('<div class="sec-hdr">① Sector Map (Esri Satellite)</div>', unsafe_allow_html=True)
    st.caption("Sector built from site coordinates, azimuth, horizontal beamwidth.")
    map_obj  = build_map(site_lat, site_lon, az_deg, hbw, main_d, near_d, far_d, dem_d, dist_m)
    map_data = st_folium(map_obj, width="100%", height=380,
                         returned_objects=["last_clicked"],
                         key="sector_map")

with lobe_col:
    st.markdown('<div class="sec-hdr">② Lobe Projection</div>', unsafe_allow_html=True)
    st.caption("Theoretical lobe distances relative to the antenna.")
    lobe_fig = build_lobe_chart(h_m, dt_deg, vbw)
    st.plotly_chart(lobe_fig, use_container_width=True,
                    config={'displayModeBar': False, 'displaylogo': False})

_clicked = (map_data or {}).get("last_clicked")
if _clicked and _clicked.get("lat") is not None:
    _clat, _clng = _clicked["lat"], _clicked["lng"]
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
# MAP LEGEND
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
