"""
Antenna Downtilt Calculator — Streamlit
Elevation: Copernicus DEM GLO-30 (30 m) — offline tiles, free, no API key
           Auto-downloads from AWS public S3 on first use → ~/.copdem30/
Fallback:  SRTM1 (srtm.py) → Open-Elevation cloud
"""

import streamlit as st
import numpy as np
import requests
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import zipfile, io, math, pathlib
import time

# ─────────────────────────────────────────────────────────────────────────────
# Copernicus DEM GLO-30  (primary — 30 m, free, no API key, offline after 1st
# download)
#
# Tiles are Cloud-Optimised GeoTIFFs hosted on the public AWS bucket:
#   s3://copernicus-dem-30m/  (anonymous access — no credentials)
# Each 1°×1° tile covers Egypt perfectly.  Tile size: ~5–15 MB.
# After the first profile fetch, tiles live in ~/.copdem30/ and work offline.
#
# Requirements:  pip install rasterio
# ─────────────────────────────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.windows import Window as _RioWin
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

_COPDEM_CACHE = pathlib.Path.home() / ".copdem30"

def _copdem_name(lat_f: int, lon_f: int) -> str:
    ns  = "N" if lat_f >= 0 else "S"
    ew  = "E" if lon_f >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat_f):02d}_00"
            f"_{ew}{abs(lon_f):03d}_00_DEM")

def _copdem_url(lat_f: int, lon_f: int) -> str:
    name = _copdem_name(lat_f, lon_f)
    return (f"https://copernicus-dem-30m.s3.amazonaws.com"
            f"/{name}/{name}_DEM.tif")

def _copdem_path(lat_f: int, lon_f: int) -> pathlib.Path:
    return _COPDEM_CACHE / f"{_copdem_name(lat_f, lon_f)}.tif"

def _copdem_download_tile(lat_f: int, lon_f: int) -> pathlib.Path:
    """Download one Copernicus DEM GLO-30 tile (~5-15 MB) from public AWS S3."""
    _COPDEM_CACHE.mkdir(parents=True, exist_ok=True)
    path = _copdem_path(lat_f, lon_f)
    url  = _copdem_url(lat_f, lon_f)
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    tmp.rename(path)
    return path

def _elev_copdem30(lats, lons):
    """
    Query Copernicus DEM GLO-30 (30 m) elevation.
    Tiles auto-download from public AWS S3 on first call, then serve offline.
    Requires: pip install rasterio
    """
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio not available — pip install rasterio")

    result     = []
    open_tiles = {}          # (lat_floor, lon_floor) → open rasterio.DatasetReader

    try:
        for lat, lon in zip(lats, lons):
            lat_f = int(math.floor(lat))
            lon_f = int(math.floor(lon))
            key   = (lat_f, lon_f)

            if key not in open_tiles:
                tpath = _copdem_path(lat_f, lon_f)
                if not tpath.exists():
                    _copdem_download_tile(lat_f, lon_f)
                open_tiles[key] = rasterio.open(tpath)

            ds  = open_tiles[key]
            row, col = ds.index(lon, lat)
            row = max(0, min(row, ds.height - 1))
            col = max(0, min(col, ds.width  - 1))
            val = float(ds.read(1, window=_RioWin(col, row, 1, 1))[0, 0])
            nd  = ds.nodata if ds.nodata is not None else -9999.0
            result.append(val if (val != nd and not math.isnan(val)) else 0.0)
    finally:
        for ds in open_tiles.values():
            try: ds.close()
            except Exception: pass

    return result

# ─────────────────────────────────────────────────────────────────────────────
# SRTM1 30m  (secondary offline fallback — used when rasterio is not installed)
# pip install srtm.py
# ─────────────────────────────────────────────────────────────────────────────
try:
    import srtm as _srtm_lib
    _SRTM_AVAILABLE = True
except ImportError:
    _SRTM_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def _load_srtm1():
    if not _SRTM_AVAILABLE:
        return None
    return _srtm_lib.get_data(srtm1=True)

def _elev_srtm1(lats, lons):
    data = _load_srtm1()
    if data is None:
        raise RuntimeError("srtm library not available")
    result = []
    for lat, lon in zip(lats, lons):
        e = data.get_elevation(lat, lon)
        result.append(float(e) if e is not None else 0.0)
    return result

# ── Open-Elevation — free cloud last-resort (no API key) ─────────────────────
def _elev_open_elevation(lats, lons):
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
# ELEVATION FETCH
# Priority: 1. Copernicus DEM GLO-30 (rasterio, offline)
#           2. SRTM1 30m (srtm.py, offline fallback)
#           3. Open-Elevation cloud (last resort)
# ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_dem(lat, lon, az, dist_m, n=100):
    """
    Fetch elevation profile along a great-circle path.

    Source priority:
      1. Copernicus DEM GLO-30 (rasterio) — 30 m, free, no API key.
         1°×1° tiles auto-download from AWS public S3 (~5-15 MB each) on
         first call and are cached in ~/.copdem30/.  Fully offline after.
         Best accuracy for Egypt. Requires: pip install rasterio
      2. SRTM1 30m (srtm.py) — 30 m, free, offline-capable fallback.
         Requires: pip install srtm.py
      3. Open-Elevation REST API — free cloud API, no key needed.

    Returns (distances_m array, elevations_m array, source_label)
    """
    lats, lons = [], []
    for i in range(n):
        p = gc_dest(lat, lon, az, i / (n - 1) * dist_m)
        lats.append(round(p[0], 7))
        lons.append(round(p[1], 7))

    # ── 1. Copernicus DEM GLO-30 offline tiles (best accuracy) ───────────
    if _RASTERIO_AVAILABLE:
        try:
            elevs = _elev_copdem30(lats, lons)
            return (np.linspace(0, dist_m, n),
                    np.array(elevs, dtype=float),
                    "Copernicus DEM GLO-30 (local)")
        except Exception:
            pass   # fall through

    # ── 2. SRTM1 30m offline tiles (fallback if rasterio not installed) ──
    if _SRTM_AVAILABLE:
        try:
            elevs = _elev_srtm1(lats, lons)
            return (np.linspace(0, dist_m, n),
                    np.array(elevs, dtype=float),
                    "SRTM1 30m (local)")
        except Exception:
            pass   # fall through

    # ── 3. Open-Elevation free cloud API (last resort) ────────────────────
    CHUNK  = 25
    elevs  = []
    source = "Open-Elevation (cloud)"
    for start in range(0, n, CHUNK):
        lat_c = lats[start: start + CHUNK]
        lon_c = lons[start: start + CHUNK]
        try:
            elevs.extend(_elev_open_elevation(lat_c, lon_c))
        except Exception:
            elevs.extend([0.0] * len(lat_c))
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
        showlegend=False, name='Antenna'
       
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
        margin=dict(l=60, r=20, t=55, b=100),
        height=380,
        legend=dict(
            orientation='h', yanchor='top', y=-0.22,
            xanchor='left', x=0,
            font=dict(size=11, family='Inter', color='#cbd5e1'),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
            tracegroupgap=0,
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
# 2D LOBE PROJECTION CHART  — terrain-aware / dynamic
# ─────────────────────────────────────────────────────
def build_lobe_chart(h_m, dt_deg, vbw_deg,
                     dem_d=None, dem_elev=None,
                     units="Metric (m, km)", az_deg=0.0):
    """
    Lobe projection chart.

    Flat-earth mode (no DEM):  classic triangular lobes, Y = relative height.
    Terrain mode (DEM loaded): lobes clipped at real ground using the same
                               _get_clipped_ray / _lobe_polygon helpers as
                               build_chart.  Y-axis = height relative to site
                               ground (0 = site ground, h_m = antenna tip).
    """
    has_dem   = dem_d is not None and dem_elev is not None and len(dem_d) > 1
    far_angle = max(0.05, dt_deg - vbw_deg / 2)

    # ── Flat-earth reference distances (always computed) ─────────────────────
    far_d_f   = h_m / math.tan(math.radians(far_angle))
    main_d_f  = h_m / math.tan(math.radians(dt_deg))
    near_d_f  = h_m / math.tan(math.radians(dt_deg + vbw_deg / 2))

    # ── Lobe colours (same as build_chart) ───────────────────────────────────
    C_UPPER_FILL = 'rgba(59,130,246,0.28)'
    C_UPPER_LINE = '#3b82f6'
    C_MAIN_FILL  = 'rgba(248,113,113,0.28)'
    C_MAIN_LINE  = '#f87171'
    C_LOWER_FILL = 'rgba(253,224,71,0.28)'
    C_LOWER_LINE = '#fde047'

    fig = go.Figure()

    # ══════════════════════════════════════════════════════════════════════════
    # TERRAIN MODE  — reuses _get_clipped_ray / _lobe_polygon / _lower_polygon
    # ══════════════════════════════════════════════════════════════════════════
    if has_dem:
        site_elev = float(dem_elev[0])
        N         = 400
        xs        = np.linspace(0, float(dem_d[-1]), N)

        # Interpolate terrain in relative coords (0 = site ground)
        terrain_abs = np.interp(xs, dem_d, dem_elev)
        terrain_rel = terrain_abs - site_elev        # relative height

        # _get_clipped_ray expects absolute elevations; pass relative by
        # treating site_elev=0 and terrain as relative values
        xs_far,  ys_far  = _get_clipped_ray(xs, terrain_rel, 0.0, h_m, far_angle)
        xs_main, ys_main = _get_clipped_ray(xs, terrain_rel, 0.0, h_m, dt_deg)
        xs_near, ys_near = _get_clipped_ray(xs, terrain_rel, 0.0, h_m,
                                             dt_deg + vbw_deg / 2)

        far_hit_x  = float(xs_far[-1])
        main_hit_x = float(xs_main[-1])
        near_hit_x = float(xs_near[-1])
        far_hit_y  = float(np.interp(far_hit_x,  xs, terrain_rel))
        main_hit_y = float(np.interp(main_hit_x, xs, terrain_rel))
        near_hit_y = float(np.interp(near_hit_x, xs, terrain_rel))

        y_min  = float(np.min(terrain_rel)) - 10
        y_max  = h_m + 30
        x_max  = float(dem_d[-1])

        # ── Terrain fill ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=xs, y=terrain_rel,
            fill='tozeroy',
            fillcolor='rgba(125,211,252,0.25)',
            line=dict(color='#7dd3fc', width=1.5),
            name='Terrain',
            hovertemplate=(
                'Dist: %{x:.0f} m<br>'
                'Rel. height: %{y:.1f} m'
                '<extra></extra>'
            ),
        ))

        # ── Lower lobe fill ───────────────────────────────────────────────────
        lo_px, lo_py = _lower_polygon(xs_near, ys_near, xs, terrain_rel)
        fig.add_trace(go.Scatter(
            x=lo_px, y=lo_py, fill='toself', fillcolor=C_LOWER_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_lf', hoverinfo='skip',
        ))

        # ── Main lobe fill ────────────────────────────────────────────────────
        if main_hit_x >= near_hit_x:
            ma_px, ma_py = _lobe_polygon(xs_main, ys_main, xs_near, ys_near,
                                          xs, terrain_rel)
        else:
            ma_px, ma_py = _lobe_polygon(xs_near, ys_near, xs_main, ys_main,
                                          xs, terrain_rel)
        fig.add_trace(go.Scatter(
            x=ma_px, y=ma_py, fill='toself', fillcolor=C_MAIN_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_mf', hoverinfo='skip',
        ))

        # ── Upper lobe fill ───────────────────────────────────────────────────
        if far_hit_x >= main_hit_x:
            up_px, up_py = _lobe_polygon(xs_far, ys_far, xs_main, ys_main,
                                          xs, terrain_rel)
        else:
            up_px, up_py = _lobe_polygon(xs_main, ys_main, xs_far, ys_far,
                                          xs, terrain_rel)
        fig.add_trace(go.Scatter(
            x=up_px, y=up_py, fill='toself', fillcolor=C_UPPER_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_uf', hoverinfo='skip',
        ))

        # ── Clipped lobe ray lines with hover ────────────────────────────────
        for xs_r, ys_r, col, lbl, angle_lbl, hit_x, hit_y, emoji in [
            (xs_far,  ys_far,  C_UPPER_LINE,
             f'Upper Lobe ({fmt_d(far_hit_x, units)})',
             f'{far_angle:.2f}°', far_hit_x, far_hit_y, '🔵'),
            (xs_main, ys_main, C_MAIN_LINE,
             f'Main Lobe ({fmt_d(main_hit_x, units)})',
             f'{dt_deg:.2f}°', main_hit_x, main_hit_y, '🔴'),
            (xs_near, ys_near, C_LOWER_LINE,
             f'Lower Lobe ({fmt_d(near_hit_x, units)})',
             f'{dt_deg + vbw_deg/2:.2f}°', near_hit_x, near_hit_y, '🟡'),
        ]:
            fig.add_trace(go.Scatter(
                x=xs_r, y=ys_r,
                mode='lines',
                line=dict(color=col, width=2),
                name=lbl,
                hovertemplate=(
                    f'<b>{emoji} {lbl.split("(")[0].strip()}</b><br>'
                    'Distance: %{x:.0f} m<br>'
                    'Rel. height: %{y:.1f} m<br>'
                    f'Ground hit: {fmt_d(hit_x, units)} '
                    f'(elev +{hit_y:.1f} m)<br>'
                    f'Lobe angle: {angle_lbl}'
                    '<extra></extra>'
                ),
            ))

        # ── Terrain-hit intersection markers ─────────────────────────────────
        for hx, hy, col, emoji in [
            (far_hit_x,  far_hit_y,  C_UPPER_LINE, '🔵'),
            (main_hit_x, main_hit_y, C_MAIN_LINE,  '🔴'),
            (near_hit_x, near_hit_y, C_LOWER_LINE, '🟡'),
        ]:
            fig.add_trace(go.Scatter(
                x=[hx], y=[hy],
                mode='markers',
                marker=dict(size=9, color=col, symbol='x',
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hovertemplate=(
                    f'<b>{emoji} Ground intersection</b><br>'
                    f'Distance: {fmt_d(hx, units)}<br>'
                    f'Rel. elevation: {hy:.1f} m'
                    '<extra></extra>'
                ),
            ))

        # ── Antenna stem & marker ─────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, h_m],
            line=dict(color='#ef4444', width=3),
            showlegend=False, name='_stem', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[h_m],
            mode='markers+text',
            marker=dict(size=11, color='#ef4444', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f'  {h_m:.0f} m AGL'], textposition='middle right',
            textfont=dict(family='Inter', size=10, color='#ef4444'),
            name='Antenna',

        ))

        title_txt  = (f'Lobe Projection | Az: {az_deg:.2f}° | '
                      f'Tilt: {dt_deg:.2f}° | VB: {vbw_deg:.2f}° | '
                      f'Profile: {fmt_d(x_max, units)}')
        subtitle   = 'Terrain-adjusted — lobes clipped at real ground intersections'
        y_axis_lbl = 'Relative Height (m)'

    # ══════════════════════════════════════════════════════════════════════════
    # FLAT-EARTH MODE  — classic triangular lobes
    # ══════════════════════════════════════════════════════════════════════════
    else:
        x_max  = far_d_f * 1.08
        y_min  = -2.0
        y_max  = h_m * 1.18

        def _fe_ys(xs_e, angle_deg):
            return h_m - xs_e * math.tan(math.radians(angle_deg))

        xs_hover = np.linspace(0, far_d_f, 60)

        # ── Flat terrain line ────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, far_d_f * 1.08], y=[0, 0],
            fill='tozeroy',
            fillcolor='rgba(125,211,252,0.18)',
            line=dict(color='#7dd3fc', width=1.2),
            showlegend=False, name='_flat_gnd', hoverinfo='skip',
        ))

        # ── Filled lobe triangles ────────────────────────────────────────────
        for poly_x, poly_y, fill_col, lbl in [
            ([0, far_d_f,  main_d_f, 0], [h_m, 0, 0, h_m],
             'rgba(59,130,246,0.45)',  'Upper Lobe'),
            ([0, main_d_f, near_d_f, 0], [h_m, 0, 0, h_m],
             'rgba(248,113,113,0.45)', 'Main Lobe'),
            ([0, near_d_f, 0,        0], [h_m, 0, 0, h_m],
             'rgba(253,224,71,0.45)', 'Lower Lobe'),
        ]:
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                fill='toself', fillcolor=fill_col,
                line=dict(color=fill_col.replace('0.45', '0.85'), width=1.5),
                name=lbl, hoverinfo='skip',
            ))

        # ── Invisible hover lines ────────────────────────────────────────────
        for angle, d_hit, col, lbl, emoji in [
            (far_angle,           far_d_f,  C_UPPER_LINE, 'Upper Lobe', '🔵'),
            (dt_deg,              main_d_f, C_MAIN_LINE,  'Main Lobe',  '🔴'),
            (dt_deg + vbw_deg/2,  near_d_f, C_LOWER_LINE, 'Lower Lobe', '🟡'),
        ]:
            xs_e = np.linspace(0, d_hit, 40)
            ys_e = _fe_ys(xs_e, angle)
            fig.add_trace(go.Scatter(
                x=xs_e, y=ys_e,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=10),
                showlegend=False,
                hovertemplate=(
                    f'<b>{emoji} {lbl}</b><br>'
                    'Distance: %{x:.0f} m<br>'
                    'Height: %{y:.1f} m<br>'
                    f'Ground hit: {fmt_d(d_hit, units)}<br>'
                    f'Angle: {angle:.2f}°'
                    '<extra></extra>'
                ),
            ))

        # ── Ground hit markers ───────────────────────────────────────────────
        for d_hit, col, lbl, emoji in [
            (far_d_f,  C_UPPER_LINE, f'Upper: {fmt_d(far_d_f, units)}',  '🔵'),
            (main_d_f, C_MAIN_LINE,  f'Main: {fmt_d(main_d_f, units)}',  '🔴'),
            (near_d_f, C_LOWER_LINE, f'Lower: {fmt_d(near_d_f, units)}', '🟡'),
        ]:
            fig.add_trace(go.Scatter(
                x=[d_hit], y=[0],
                mode='markers',
                marker=dict(size=9, color=col, symbol='circle',
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hovertemplate=(
                    f'<b>{emoji} Ground hit</b><br>'
                    f'{lbl}'
                    '<extra></extra>'
                ),
            ))

        # ── Antenna ──────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, h_m],
            line=dict(color='#ef4444', width=3),
            showlegend=False, name='_stem', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[h_m],
            mode='markers+text',
            marker=dict(size=11, color='#ef4444', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f'  {h_m:.0f} m AGL'], textposition='middle right',
            textfont=dict(family='Inter', size=10, color='#ef4444'),
            name='Antenna',
            hovertemplate=(
                '<b>📡 Antenna</b><br>'
                f'Height AGL: {h_m:.0f} m<br>'
                f'Downtilt: {dt_deg:.2f}°  VBW: {vbw_deg:.2f}°'
                '<extra></extra>'
            ),
        ))

        title_txt  = 'Lobe Distance Projection'
        subtitle   = 'Theoretical flat-earth distances — load terrain profile for real intersections'
        y_axis_lbl = 'Antenna Height (m)'

    # ── Shared layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(family='Inter', color='#94a3b8', size=11),
        margin=dict(l=60, r=20, t=70, b=55),
        height=380,
        legend=dict(
            orientation='v', yanchor='top', y=0.97, xanchor='right', x=0.98,
            font=dict(size=10, family='Inter', color='#cbd5e1'),
            bgcolor='rgba(15,23,42,0.65)', bordercolor='#334155', borderwidth=1,
            tracegroupgap=2,
        ),
        xaxis=dict(
            title=dict(text='Distance in meters',
                       font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[0, x_max], showline=True, linecolor='#334155',
        ),
        yaxis=dict(
            title=dict(text=y_axis_lbl,
                       font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[y_min, y_max], showline=True, linecolor='#334155',
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(family='JetBrains Mono', size=11, color='#e2e8f0'),
            align='left',
        ),
        annotations=[
            dict(x=0.5, y=1.07, xref='paper', yref='paper', showarrow=False,
                 text=title_txt,
                 font=dict(size=12, color='#e2e8f0', family='Inter', weight=700),
                 xanchor='center'),
            dict(x=0.5, y=1.01, xref='paper', yref='paper', showarrow=False,
                 text=subtitle,
                 font=dict(size=9, color='#64748b', family='Inter'),
                 xanchor='center'),
        ],
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

# Show elevation source banner once
if _RASTERIO_AVAILABLE:
    _copernicus_badge = (
        '<span class="src-badge src-local">'
        '🌍 Copernicus DEM GLO-30 — offline tiles, 30 m, no API key'
        '</span>'
    )
elif _SRTM_AVAILABLE:
    _copernicus_badge = (
        '<span class="src-badge src-local">'
        '⚡ SRTM1 30m local — install rasterio for Copernicus DEM'
        '</span>'
    )
else:
    _copernicus_badge = (
        '<span class="src-badge src-cloud">'
        '☁ Cloud elevation — pip install rasterio for offline Copernicus DEM'
        '</span>'
    )

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
        if _RASTERIO_AVAILABLE:
            st.markdown(
                "<div style='font-size:0.68rem;color:#15803d;background:#dcfce7;"
                "border-radius:5px;padding:5px 10px;margin-bottom:4px;'>"
                "🌍 <b>Copernicus DEM GLO-30</b> — 30 m offline tiles<br>"
                "Tiles auto-download to <code>~/.copdem30/</code> on first fetch</div>",
                unsafe_allow_html=True)
        elif _SRTM_AVAILABLE:
            st.markdown(
                "<div style='font-size:0.68rem;color:#15803d;background:#dcfce7;"
                "border-radius:5px;padding:5px 10px;margin-bottom:4px;'>"
                "⚡ SRTM1 30m tiles — <code>pip install rasterio</code> for Copernicus DEM</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size:0.68rem;color:#92400e;background:#fef3c7;"
                "border-radius:5px;padding:5px 10px;margin-bottom:4px;'>"
                "☁ Will use Open-Elevation cloud API<br>"
                "<code>pip install rasterio</code> for offline Copernicus DEM GLO-30</div>",
                unsafe_allow_html=True)

        if st.button("↻ Fetch Elevation Profile", type="primary", use_container_width=True):
            if _RASTERIO_AVAILABLE:
                src_label = "Copernicus DEM GLO-30 (local)"
            elif _SRTM_AVAILABLE:
                src_label = "SRTM1 30m (local)"
            else:
                src_label = "Open-Elevation (cloud)"
            with st.spinner(f"Loading elevation via {src_label}…"):
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
if src == "Copernicus DEM GLO-30 (local)":
    src_html = '<span class="src-badge src-local">Copernicus DEM GLO-30</span>'
elif src == "SRTM1 30m (local)":
    src_html = '<span class="src-badge src-local">SRTM1 30m local</span>'
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
# SECTOR METRICS  (placed here — above the map/lobe grid)
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
# GRID ROW: Sector Map (Esri Satellite) + Lobe Projection
# ─────────────────────────────────────────────────────
map_col, lobe_col = st.columns(2, gap="medium")

with map_col:
    st.markdown('<div class="sec-hdr">① Sector Map (Esri Satellite)</div>', unsafe_allow_html=True)
    st.caption("Sector built from site coordinates, azimuth, horizontal beamwidth, and auto-adjusted distance.")
    map_obj  = build_map(site_lat, site_lon, az_deg, hbw, main_d, near_d, far_d, dem_d, dist_m)
    map_data = st_folium(map_obj, width="100%", height=380,
                         returned_objects=["last_clicked"],
                         key="sector_map")

with lobe_col:
    st.markdown('<div class="sec-hdr">② Lobe Projection</div>', unsafe_allow_html=True)
    if has_dem:
        st.caption("Terrain-adjusted — lobes clipped at real ground intersections.")
    else:
        st.caption("Flat-earth model — load terrain profile for terrain-aware lobes.")
    lobe_fig = build_lobe_chart(
        h_m, dt_deg, vbw,
        dem_d=dem_d, dem_elev=dem_elev,
        units=units, az_deg=az_deg,
    )
    st.plotly_chart(lobe_fig, use_container_width=True,
                    config={'displayModeBar': True, 'displaylogo': False})

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
