"""
Antenna Downtilt Calculator — Streamlit
Elevation: Copernicus DEM GLO-30 (30 m) — offline tiles, free, no API key
           Auto-downloads from AWS public S3 on first use → ~/.copdem30/
Fallback:  SRTM1 (srtm.py) → Open-Elevation cloud
"""

# ── Standard & third-party imports ───────────────────────────────────────────
import streamlit as st                      # Main Streamlit framework for building the web UI
import numpy as np                          # Numerical array operations (linspace, interp, etc.)
import requests                             # HTTP requests for downloading DEM tiles and cloud elevation
import plotly.graph_objects as go           # Plotly figures for interactive terrain/lobe charts
import folium                               # Leaflet-based interactive map rendering
from streamlit_folium import st_folium      # Embeds Folium maps inside Streamlit
import zipfile, io, math, pathlib           # Standard lib: archives, byte buffers, math helpers, paths
import time                                 # Used for rate-limiting batched API requests

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

# Attempt to import rasterio; flag availability for conditional use later
try:
    import rasterio                                                 # GeoTIFF reader for Copernicus DEM tiles
    from rasterio.windows import Window as _RioWindow              # Pixel-window helper for reading a single cell
    _RASTERIO_AVAILABLE = True                                      # Signal that rasterio is installed and usable
except ImportError:
    _RASTERIO_AVAILABLE = False                                     # Graceful fallback — rasterio not installed

# Local filesystem cache directory for downloaded Copernicus DEM tiles
_COPDEM_CACHE = pathlib.Path.home() / ".copdem30"                  # Resolves to ~/.copdem30/


def _copdemName(latF: int, lonF: int) -> str:
    """Return the official Copernicus GLO-30 base name for a given 1°×1° tile cell."""
    nsDir = "N" if latF >= 0 else "S"                              # Hemisphere prefix: North or South
    ewDir = "E" if lonF >= 0 else "W"                              # Hemisphere prefix: East or West
    # Build the standardised Copernicus naming convention string
    return (f"Copernicus_DSM_COG_10_{nsDir}{abs(latF):02d}_00"
            f"_{ewDir}{abs(lonF):03d}_00_DEM")


def _copdemUrl(latF: int, lonF: int) -> str:
    """Construct the public AWS S3 HTTPS URL for a specific Copernicus DEM GLO-30 GeoTIFF tile."""
    tileName = _copdemName(latF, lonF)                             # Get the standardised tile base name
    # Build full URL pointing to the tile on the public Copernicus S3 bucket
    return (f"https://copernicus-dem-30m.s3.amazonaws.com"
            f"/{tileName}/{tileName}_DEM.tif")


def _copdemPath(latF: int, lonF: int) -> pathlib.Path:
    """Return the local filesystem path where a given DEM tile is (or will be) cached."""
    return _COPDEM_CACHE / f"{_copdemName(latF, lonF)}.tif"        # Compose cache dir + tile filename


def _copdemDownloadTile(latF: int, lonF: int) -> pathlib.Path:
    """Download one Copernicus DEM GLO-30 tile (~5-15 MB) from public AWS S3."""
    _COPDEM_CACHE.mkdir(parents=True, exist_ok=True)               # Ensure the cache directory exists
    tilePath = _copdemPath(latF, lonF)                             # Destination path for the cached tile file
    tileUrl  = _copdemUrl(latF, lonF)                              # Source URL on the public S3 bucket
    response = requests.get(tileUrl, timeout=60, stream=True)      # Start a streaming HTTP GET request
    response.raise_for_status()                                    # Raise an exception on HTTP 4xx/5xx status
    tmpPath = tilePath.with_suffix(".tmp")                         # Write to .tmp first to avoid partial reads
    with open(tmpPath, "wb") as fileHandle:                        # Open temp file in binary write mode
        for chunk in response.iter_content(65536):                 # Stream in 64 KB chunks to limit memory use
            fileHandle.write(chunk)                                # Write each chunk to disk immediately
    tmpPath.rename(tilePath)                                       # Atomically rename .tmp → .tif on success
    return tilePath                                                # Return the finalised cached tile path


def _elevCopdem30(lats, lons):
    """
    Query Copernicus DEM GLO-30 (30 m) elevation for a list of lat/lon points.
    Tiles auto-download from public AWS S3 on first call, then serve offline.
    Requires: pip install rasterio
    """
    if not _RASTERIO_AVAILABLE:                                    # Guard: rasterio must be installed
        raise RuntimeError("rasterio not available — pip install rasterio")

    elevResults = []                                               # Accumulator for per-point elevation values
    openTiles   = {}                                               # Cache of open rasterio datasets keyed by tile (latF, lonF)

    try:
        for lat, lon in zip(lats, lons):                           # Iterate over each query coordinate pair
            latF    = int(math.floor(lat))                         # Floor to get the tile's SW-corner latitude
            lonF    = int(math.floor(lon))                         # Floor to get the tile's SW-corner longitude
            tileKey = (latF, lonF)                                 # Unique key identifying this 1°×1° tile

            if tileKey not in openTiles:                           # Open the tile dataset only if not already cached
                tilePath = _copdemPath(latF, lonF)                 # Resolve local path for this tile
                if not tilePath.exists():                          # Download the tile if it is not yet on disk
                    _copdemDownloadTile(latF, lonF)
                openTiles[tileKey] = rasterio.open(tilePath)       # Open tile file and store the dataset handle

            dataset   = openTiles[tileKey]                         # Retrieve the already-open dataset handle
            row, col  = dataset.index(lon, lat)                    # Convert geographic coords → pixel row/col
            row       = max(0, min(row, dataset.height - 1))       # Clamp row to valid pixel range
            col       = max(0, min(col, dataset.width  - 1))       # Clamp col to valid pixel range
            pixelVal  = float(dataset.read(1, window=_RioWindow(col, row, 1, 1))[0, 0])  # Read single-pixel value
            noDataVal = dataset.nodata if dataset.nodata is not None else -9999.0         # Use dataset nodata or fallback sentinel
            # Append valid elevation value; substitute 0.0 for nodata or NaN pixels
            elevResults.append(pixelVal if (pixelVal != noDataVal and not math.isnan(pixelVal)) else 0.0)
    finally:
        for dataset in openTiles.values():                         # Always close all open tile datasets
            try: dataset.close()                                   # Suppress any errors during cleanup
            except Exception: pass

    return elevResults                                             # Return list of elevation values in metres


# ─────────────────────────────────────────────────────────────────────────────
# SRTM1 30m  (secondary offline fallback — used when rasterio is not installed)
# pip install srtm.py
# ─────────────────────────────────────────────────────────────────────────────

# Attempt to import the srtm library; flag availability for conditional use
try:
    import srtm as _srtm_lib        # SRTM elevation data library (offline-capable fallback)
    _SRTM_AVAILABLE = True          # Signal that srtm.py is installed and usable
except ImportError:
    _SRTM_AVAILABLE = False         # Graceful fallback — srtm.py not installed


@st.cache_resource(show_spinner=False)      # Cache the loaded SRTM dataset across Streamlit reruns
def _loadSrtm1():
    """Load and cache the SRTM1 (1 arc-second, ~30 m) elevation dataset."""
    if not _SRTM_AVAILABLE:                 # Guard: srtm library must be installed
        return None
    return _srtm_lib.get_data(srtm1=True)   # Load SRTM1 data (downloads tiles on demand)


def _elevSrtm1(lats, lons):
    """Query SRTM1 elevation for a list of lat/lon points using the offline srtm.py library."""
    srtmData = _loadSrtm1()                                        # Retrieve the cached SRTM dataset
    if srtmData is None:                                           # Guard: dataset must be available
        raise RuntimeError("srtm library not available")
    elevResults = []                                               # Accumulator for elevation values
    for lat, lon in zip(lats, lons):                               # Iterate each coordinate pair
        elev = srtmData.get_elevation(lat, lon)                    # Query SRTM1 elevation at this point
        elevResults.append(float(elev) if elev is not None else 0.0)  # Fallback to 0 m if data is missing
    return elevResults                                             # Return list of elevation values in metres


# ── Open-Elevation — free cloud last-resort (no API key) ─────────────────────
def _elevOpenElevation(lats, lons):
    """Query elevation via the Open-Elevation free REST API (cloud fallback, no API key needed)."""
    locationsList = [{"latitude": la, "longitude": lo}
                     for la, lo in zip(lats, lons)]                # Build JSON-serialisable list of coordinate dicts
    response = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        json={"locations": locationsList}, timeout=20,             # POST all points in a single request
    )
    response.raise_for_status()                                    # Raise an exception on HTTP error response
    return [pt["elevation"] for pt in response.json()["results"]]  # Extract elevation value from each result dict


# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Antenna Downtilt Calculator",               # Browser tab title
    page_icon="📡",                                         # Browser tab favicon
    layout="wide",                                         # Use full browser width for the layout
    initial_sidebar_state="expanded",                      # Show the sidebar open on page load
)

# ── Custom CSS injected once at app start ─────────────────────────────────────
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
""", unsafe_allow_html=True)       # unsafe_allow_html required to inject raw CSS into Streamlit


# ─────────────────────────────────────────────────────
# MATHS HELPERS
# ─────────────────────────────────────────────────────

def toM(v, units):
    """Convert a value from the selected unit system to metres."""
    return v if units == "Metric (m, km)" else v * 0.3048          # Feet → metres conversion factor (1 ft = 0.3048 m)


def fmtD(m, units, dec=2):
    """Format a distance in metres as a human-readable string in the selected unit system."""
    if units == "Metric (m, km)":
        return f"{m/1000:.{dec}f} km" if m >= 1000 else f"{m:.0f} m"   # Show km if ≥1000 m, else plain metres
    feetVal = m / 0.3048                                               # Convert metres → feet
    return f"{feetVal/5280:.{dec}f} mi" if feetVal >= 5280 else f"{feetVal:.0f} ft"  # Show mi if ≥1 mile, else feet


def flatGeom(h, dt, vbw):
    """Calculate main lobe, far-edge, and near-edge ground distances assuming a flat Earth."""
    dtRad    = math.radians(dt)                                        # Downtilt angle converted to radians
    halfRad  = math.radians(vbw / 2)                                  # Half vertical beamwidth in radians
    mainD    = h / math.tan(dtRad)                                     # Horizontal distance to main lobe centre
    nearD    = h / math.tan(dtRad + halfRad)                           # Distance to the near (lower) lobe edge
    farAngle = max(0.0002, dtRad - halfRad)                            # Far-lobe angle clamped to avoid zero/negative
    return mainD, h / math.tan(farAngle), nearD                        # Return (main, far, near) ground distances


def flatGeomFull(h, dt, vbw):
    """Calculate main, near, and far lobe ground distances on a flat Earth (all three returned)."""
    dtRad   = math.radians(dt)                                         # Downtilt angle in radians
    halfRad = math.radians(vbw / 2)                                    # Half vertical beamwidth in radians
    mainD   = h / math.tan(dtRad)                                      # Distance to main lobe centre
    nearD   = h / math.tan(dtRad + halfRad)                            # Distance to near (lower) lobe edge
    farD    = h / math.tan(max(0.0002, dtRad - halfRad))               # Distance to far lobe edge (angle clamped)
    return mainD, nearD, farD                                          # Return all three footprint boundaries


def rayIntersect(h, siteElev, angleDeg, demD, demElev):
    """Find the ground distance where a downward ray at angleDeg intersects the DEM terrain profile."""
    dtRad = math.radians(angleDeg)                                     # Depression angle in radians
    for i in range(1, len(demD)):                                      # Walk consecutive DEM profile segments
        d0, e0, d1, e1 = demD[i-1], demElev[i-1], demD[i], demElev[i]  # Segment start (d0,e0) and end (d1,e1)
        ray0 = siteElev + h - d0 * math.tan(dtRad)                    # Ray height at segment start distance
        ray1 = siteElev + h - d1 * math.tan(dtRad)                    # Ray height at segment end distance
        if ray0 >= e0 and ray1 <= e1:                                  # Ray crosses the terrain between these two samples
            denominator = (ray0 - ray1) + (e1 - e0)                   # Linear interpolation denominator
            fracT = (ray0 - e0) / denominator if denominator > 1e-6 else 0  # Parametric intersection position
            return d0 + fracT * (d1 - d0)                             # Interpolated ground distance of terrain hit
    return None                                                        # No intersection found within profile extent


def gcDest(lat, lon, bearing, distM):
    """Compute destination lat/lon given an origin, bearing (degrees), and distance (metres)."""
    R          = 6_371_000                                             # Earth mean radius in metres
    bearingRad = math.radians(bearing)                                 # Convert bearing to radians
    distRad    = distM / R                                             # Angular distance along Earth's surface
    lat1Rad    = math.radians(lat)                                     # Origin latitude in radians
    lon1Rad    = math.radians(lon)                                     # Origin longitude in radians
    lat2Rad    = math.asin(                                            # Destination latitude via spherical law of cosines
        math.sin(lat1Rad) * math.cos(distRad) +
        math.cos(lat1Rad) * math.sin(distRad) * math.cos(bearingRad)
    )
    lon2Rad    = lon1Rad + math.atan2(                                 # Destination longitude via atan2 (handles all quadrants)
        math.sin(bearingRad) * math.sin(distRad) * math.cos(lat1Rad),
        math.cos(distRad) - math.sin(lat1Rad) * math.sin(lat2Rad)
    )
    return math.degrees(lat2Rad), math.degrees(lon2Rad)                # Return destination as (lat°, lon°)


# ─────────────────────────────────────────────────────
# ELEVATION FETCH
# Priority: 1. Copernicus DEM GLO-30 (rasterio, offline)
#           2. SRTM1 30m (srtm.py, offline fallback)
#           3. Open-Elevation cloud (last resort)
# ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)      # Cache results keyed on all inputs; avoids redundant network calls on rerun
def fetchDem(lat, lon, az, distM, n=100):
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
    latsList, lonsList = [], []                                        # Initialise empty lat/lon accumulators
    for i in range(n):                                                 # Sample n evenly-spaced points along the profile
        point = gcDest(lat, lon, az, i / (n - 1) * distM)             # Geographic position at fractional distance
        latsList.append(round(point[0], 7))                            # Latitude rounded to ~1 cm precision
        lonsList.append(round(point[1], 7))                            # Longitude rounded to ~1 cm precision

    # ── 1. Copernicus DEM GLO-30 offline tiles (best accuracy) ───────────
    if _RASTERIO_AVAILABLE:                                            # Only attempt if rasterio is installed
        try:
            elevsList = _elevCopdem30(latsList, lonsList)              # Query Copernicus tiles (auto-downloads if needed)
            return (np.linspace(0, distM, n),                          # Evenly-spaced distance array (m)
                    np.array(elevsList, dtype=float),                  # Elevation array (m MSL)
                    "Copernicus DEM GLO-30 (local)")                   # Human-readable source label for the UI
        except Exception:
            pass                                                       # Silently fall through to the next source

    # ── 2. SRTM1 30m offline tiles (fallback if rasterio not installed) ──
    if _SRTM_AVAILABLE:                                                # Only attempt if srtm.py is installed
        try:
            elevsList = _elevSrtm1(latsList, lonsList)                 # Query SRTM1 offline tiles
            return (np.linspace(0, distM, n),
                    np.array(elevsList, dtype=float),
                    "SRTM1 30m (local)")
        except Exception:
            pass                                                       # Silently fall through to cloud API

    # ── 3. Open-Elevation free cloud API (last resort) ────────────────────
    CHUNK      = 25                                                    # Max points per API request to avoid timeouts
    elevsList  = []                                                    # Accumulator for all elevation values
    sourceName = "Open-Elevation (cloud)"                              # Label shown in the UI status bar
    for chunkStart in range(0, n, CHUNK):                              # Split the full profile into CHUNK-sized batches
        latChunk = latsList[chunkStart: chunkStart + CHUNK]            # Latitude sublist for this batch
        lonChunk = lonsList[chunkStart: chunkStart + CHUNK]            # Longitude sublist for this batch
        try:
            elevsList.extend(_elevOpenElevation(latChunk, lonChunk))   # Fetch batch from the cloud API
        except Exception:
            elevsList.extend([0.0] * len(latChunk))                    # Fill with 0 m on any API error
        if chunkStart + CHUNK < n:                                     # Avoid sleeping after the final batch
            time.sleep(0.15)                                           # Rate-limit: 150 ms between successive requests

    return np.linspace(0, distM, n), np.array(elevsList, dtype=float), sourceName


# ─────────────────────────────────────────────────────
# LOBE GEOMETRY HELPERS
# ─────────────────────────────────────────────────────

def _getClippedRay(xs, terrainY, siteElev, hM, angleDeg):
    """Return (xs, ys) arrays for a lobe ray clipped at its first terrain intersection."""
    tanAngle          = math.tan(math.radians(angleDeg))               # Precompute tangent of the depression angle
    rayY              = siteElev + hM - xs * tanAngle                  # Ray absolute height at each sample distance
    xsClipped, ysClipped = [], []                                      # Output arrays for the visible (above-terrain) ray

    for i in range(len(xs)):                                           # Walk each sample along the DEM profile
        if rayY[i] >= terrainY[i]:                                     # Ray is still above terrain — keep this point
            xsClipped.append(float(xs[i]))
            ysClipped.append(float(rayY[i]))
        else:                                                          # Ray has dipped below terrain — find exact hit
            if i > 0:                                                  # Need at least two points to interpolate
                ray0, ray1         = rayY[i-1],     rayY[i]           # Ray heights at the previous and current sample
                terrain0, terrain1 = terrainY[i-1], terrainY[i]       # Terrain heights at the same two samples
                denominator        = (ray0 - ray1) - (terrain0 - terrain1)  # Slope-difference denominator
                if abs(denominator) > 1e-6:                            # Avoid division by near-zero denominator
                    fracT      = (ray0 - terrain0) / denominator       # Parametric position of exact intersection
                    xIntersect = float(xs[i-1]) + fracT * float(xs[i] - xs[i-1])  # Interpolated hit x distance
                    yIntersect = terrain0 + fracT * (terrain1 - terrain0)          # Interpolated hit elevation
                    xsClipped.append(xIntersect)                       # Append exact terrain-hit point
                    ysClipped.append(yIntersect)
            break                                                      # Stop — ray is terrain-blocked beyond here

    if not xsClipped:                                                  # Edge case: ray is blocked from the very first sample
        xsClipped, ysClipped = [float(xs[0])], [float(siteElev + hM)] # Return single antenna-tip point as fallback
    return np.array(xsClipped), np.array(ysClipped)                    # Return as NumPy arrays for Plotly


def _lobePolygon(xsRay, ysRay, xsRay2, ysRay2, xsAll, terrainAll):
    """
    Build a closed polygon between two clipped rays, bridged by terrain at their hit points.
    xsRay must extend FARTHER than xsRay2 (i.e. xsRay[-1] >= xsRay2[-1]).
    """
    xFar, xNear = xsRay[-1], xsRay2[-1]                               # Terminus x of the far and near rays
    bridgeMask  = (xsAll >= xNear) & (xsAll <= xFar)                  # Boolean mask for terrain bridge points
    bridgeX     = xsAll[bridgeMask][::-1]                             # Terrain x reversed (far→near) for correct polygon winding
    bridgeY     = terrainAll[bridgeMask][::-1]                        # Corresponding terrain heights for the bridge
    polyX = list(xsRay) + list(bridgeX) + list(xsRay2[::-1])          # Concatenate: far ray + terrain bridge + near ray reversed
    polyY = list(ysRay) + list(bridgeY) + list(ysRay2[::-1])          # Corresponding heights for each x vertex
    return polyX, polyY                                                # Return closed polygon vertex lists


def _lowerPolygon(xsNear, ysNear, xsAll, terrainAll):
    """Build the lower-lobe polygon: near ray forward, then terrain back to x=0 (site)."""
    xHit        = xsNear[-1]                                          # Terminus of the near-lobe ray (closest ground hit)
    terrainMask = xsAll <= xHit                                        # Select terrain points from site to near hit
    bridgeX     = xsAll[terrainMask][::-1]                            # Terrain x reversed (hit → site) for correct winding
    bridgeY     = terrainAll[terrainMask][::-1]                       # Corresponding terrain heights
    return list(xsNear) + list(bridgeX), list(ysNear) + list(bridgeY) # Return closed polygon vertex lists


# ─────────────────────────────────────────────────────
# PLOTLY TERRAIN PROFILE + LOBE CHART
# ─────────────────────────────────────────────────────

def buildChart(hM, dtDeg, vbwDeg, distM, mainD, nearD, farD,
               demD, demElev, sliderD, units, azDeg=0.0):
    """Build the Plotly terrain profile chart with overlaid lobe projections and a slider position marker."""
    N        = 400                                                     # Number of x-samples for smooth curve rendering
    xSamples = np.linspace(0, distM, N)                               # Evenly-spaced distance array across the full profile
    hasDem   = demD is not None and len(demD) > 1                     # True when a valid DEM profile has been loaded
    siteElev = float(demElev[0]) if hasDem else 0.0                   # Site ground elevation (MSL) or flat 0 fallback
    farAngle = max(0.05, dtDeg - vbwDeg / 2)                          # Far-lobe depression angle (clamped to avoid 0)

    def elevAt(d):
        """Interpolate terrain elevation (m MSL) at any distance d along the profile."""
        return float(np.interp(d, demD, demElev)) if hasDem else siteElev   # Interp or flat fallback

    terrainY = np.array([elevAt(d) for d in xSamples])                # Terrain elevation at every sample point
    yMin     = float(np.min(terrainY)) - 20                           # Chart y-min: 20 m below the lowest terrain point
    yMax     = siteElev + hM + 50                                     # Chart y-max: 50 m above the antenna tip

    # ── Clipped ray arrays ────────────────────────────────────────────────
    xsFar,  ysFar  = _getClippedRay(xSamples, terrainY, siteElev, hM, farAngle)          # Upper (far) lobe ray
    xsMain, ysMain = _getClippedRay(xSamples, terrainY, siteElev, hM, dtDeg)             # Main lobe ray
    xsNear, ysNear = _getClippedRay(xSamples, terrainY, siteElev, hM, dtDeg + vbwDeg/2) # Lower (near) lobe ray

    # ── Terrain intersection distances ────────────────────────────────────
    farHitX  = float(xsFar[-1])                                       # Ground distance where far lobe ray hits terrain
    mainHitX = float(xsMain[-1])                                      # Ground distance where main lobe ray hits terrain
    nearHitX = float(xsNear[-1])                                      # Ground distance where near lobe ray hits terrain

    # ── Lobe colour constants ─────────────────────────────────────────────
    C_UPPER_FILL = 'rgba(59,130,246,0.28)'                            # Translucent blue fill for upper (far) lobe
    C_UPPER_LINE = '#3b82f6'                                          # Solid blue line for upper lobe
    C_MAIN_FILL  = 'rgba(248,113,113,0.28)'                           # Translucent red fill for main lobe
    C_MAIN_LINE  = '#f87171'                                          # Solid red line for main lobe
    C_LOWER_FILL = 'rgba(253,224,71,0.28)'                            # Translucent yellow fill for lower (near) lobe
    C_LOWER_LINE = '#fde047'                                          # Solid yellow line for lower lobe

    fig = go.Figure()                                                  # Initialise an empty Plotly figure

    # ── Terrain fill — split at main lobe ground hit ─────────────────────
    # Everything up to mainHitX is the illuminated zone (light blue).
    # Everything after mainHitX is the shadow zone (dark fill + red line).
    splitIdx = int(np.searchsorted(xSamples, mainHitX))              # First sample index at or beyond the main-lobe hit
    splitIdx = max(1, min(splitIdx, N - 1))                           # Clamp to a valid interior index

    # Segment 1 — illuminated terrain (antenna → main-lobe hit): light blue
    fig.add_trace(go.Scatter(
        x=xSamples[:splitIdx + 1], y=terrainY[:splitIdx + 1],
        fill='tozeroy',                                               # Fill down to y=0
        fillcolor='rgba(125,211,252,0.30)',                           # Translucent sky-blue fill
        line=dict(color='#7dd3fc', width=1.5),                       # Solid light-blue outline
        name='Terrain',
        hovertemplate='Dist: %{x:.0f} m<br>Elev: %{y:.1f} m MSL<extra></extra>'
    ))

    # Segment 2 — shadow terrain (main-lobe hit → profile end): dark fill + red line
    fig.add_trace(go.Scatter(
        x=xSamples[splitIdx:], y=terrainY[splitIdx:],
        fill='tozeroy',                                               # Fill down to y=0
        fillcolor='rgba(15,5,5,0.72)',                                # Dark shadow fill
        line=dict(color='#ef4444', width=1.5),                       # Red terrain outline in shadow zone
        showlegend=False, name='_terrain_shadow',
        hovertemplate='Dist: %{x:.0f} m<br>Elev: %{y:.1f} m MSL<extra></extra>'
    ))

    # ── Lower lobe fill (light yellow) ────────────────────────────────────
    loPx, loPy = _lowerPolygon(xsNear, ysNear, xSamples, terrainY)   # Build the lower-lobe closed polygon
    fig.add_trace(go.Scatter(
        x=loPx, y=loPy,
        fill='toself', fillcolor=C_LOWER_FILL,                        # Closed polygon with yellow fill
        line=dict(color='rgba(0,0,0,0)', width=0),                   # Invisible border — fill only
        showlegend=False, name='_lower_fill', hoverinfo='skip'
    ))

    # ── Main lobe fill (light red) ─────────────────────────────────────────
    # Ensure the farther-hitting ray is always passed as the first argument to _lobePolygon
    if mainHitX >= nearHitX:
        maPx, maPy = _lobePolygon(xsMain, ysMain, xsNear, ysNear, xSamples, terrainY)
    else:
        maPx, maPy = _lobePolygon(xsNear, ysNear, xsMain, ysMain, xSamples, terrainY)
    fig.add_trace(go.Scatter(
        x=maPx, y=maPy,
        fill='toself', fillcolor=C_MAIN_FILL,                         # Closed polygon with red fill
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_main_fill', hoverinfo='skip'
    ))

    # ── Upper lobe fill (blue) ────────────────────────────────────────────
    if farHitX >= mainHitX:
        upPx, upPy = _lobePolygon(xsFar, ysFar, xsMain, ysMain, xSamples, terrainY)
    else:
        upPx, upPy = _lobePolygon(xsMain, ysMain, xsFar, ysFar, xSamples, terrainY)
    fig.add_trace(go.Scatter(
        x=upPx, y=upPy,
        fill='toself', fillcolor=C_UPPER_FILL,                        # Closed polygon with blue fill
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_upper_fill', hoverinfo='skip'
    ))

    # ── Lobe ray lines (clipped at first terrain intersection) ────────────
    fig.add_trace(go.Scatter(
        x=xsFar, y=ysFar,
        line=dict(color=C_UPPER_LINE, width=2, dash='solid'),
        name=f'Upper Lobe ({fmtD(farHitX, units)})',                  # Legend label includes terrain hit distance
        connectgaps=False,
        hovertemplate='Upper: %{y:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xsMain, y=ysMain,
        line=dict(color=C_MAIN_LINE, width=2, dash='solid'),
        name=f'Main Lobe ({fmtD(mainHitX, units)})',
        connectgaps=False,
        hovertemplate='Main: %{y:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=xsNear, y=ysNear,
        line=dict(color=C_LOWER_LINE, width=2, dash='solid'),
        name=f'Lower Lobe ({fmtD(nearHitX, units)})',
        connectgaps=False,
        hovertemplate='Lower: %{y:.1f} m<extra></extra>'
    ))

    # ── Terrain intersection markers (X symbols at each hit point) ────────
    for hitX, lineColor in [(farHitX, C_UPPER_LINE), (mainHitX, C_MAIN_LINE), (nearHitX, C_LOWER_LINE)]:
        fig.add_trace(go.Scatter(
            x=[hitX], y=[elevAt(hitX)],
            mode='markers',
            marker=dict(size=9, color=lineColor, symbol='x', line=dict(color='white', width=1.5)),
            showlegend=False, name='_hit',
            hovertemplate=f'Hit: {fmtD(hitX, units)}<extra></extra>'
        ))

    # ── Slider position vertical line ─────────────────────────────────────
    sliderElev = elevAt(sliderD)                                       # Terrain elevation directly at the slider position
    fig.add_trace(go.Scatter(
        x=[sliderD, sliderD], y=[yMin, yMax],                         # Vertical dotted line spanning the full y-range
        line=dict(color='#dc2626', width=1.8, dash='dot'),
        name='Selected distance',
        hovertemplate=f'Selected: {fmtD(sliderD, units)}<br>Terrain: {sliderElev:.1f} m<extra></extra>'
    ))

    # ── Antenna tower marker ──────────────────────────────────────────────
    antElev = siteElev + hM                                            # Absolute elevation of the antenna tip (MSL)
    fig.add_trace(go.Scatter(
        x=[0], y=[antElev],
        mode='markers+text',
        marker=dict(size=11, color='#ef4444', symbol='diamond',
                    line=dict(color='white', width=2)),
        text=[f'Antenna {hM:.0f} m AGL'], textposition='top right',
        textfont=dict(family='Inter', size=10, color='#ef4444'),
        showlegend=False, name='Antenna',
        hovertemplate=f'Antenna<br>AGL: {hM:.0f} m<br>MSL: {antElev:.1f} m<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[siteElev, antElev],                              # Vertical red line representing the antenna mast
        line=dict(color='#ef4444', width=3),
        showlegend=False, name='_antline', hoverinfo='skip'
    ))

    # ── Chart layout ──────────────────────────────────────────────────────
    titleText = (f'Terrain Profile + Lobe Projection | '
                 f'Az: {azDeg:.2f}° | Tilt: {dtDeg:.2f}° | '
                 f'VB: {vbwDeg:.2f}° | Profile: {fmtD(distM, units)}')

    fig.update_layout(
        plot_bgcolor='#0f172a',                                        # Dark chart background colour
        paper_bgcolor='#1e293b',                                       # Slightly lighter figure frame colour
        font=dict(family='Inter', color='#94a3b8', size=11),
        margin=dict(l=60, r=20, t=55, b=100),                         # Extra bottom margin for the horizontal legend
        height=380,
        legend=dict(
            orientation='h', yanchor='top', y=-0.22,                  # Horizontal legend positioned below the chart
            xanchor='left', x=0,
            font=dict(size=11, family='Inter', color='#cbd5e1'),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
            tracegroupgap=0,
        ),
        xaxis=dict(
            title=dict(text='Distance in meters', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[0, distM], showline=True, linecolor='#334155',
        ),
        yaxis=dict(
            title=dict(text='Elevation (m MSL)', font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[yMin, yMax], showline=True, linecolor='#334155',
        ),
        hovermode='x unified',                                         # Show unified tooltip across all traces at x position
        hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155',
                        font=dict(family='JetBrains Mono', size=11, color='#e2e8f0')),
        annotations=[
            dict(x=0.5, y=1.04, xref='paper', yref='paper', showarrow=False,
                 text=titleText,
                 font=dict(size=11, color='#cbd5e1', family='Inter', weight=600),
                 xanchor='center'),
        ]
    )
    return fig                                                         # Return the fully configured Plotly figure


# ─────────────────────────────────────────────────────
# 2D LOBE PROJECTION CHART  — terrain-aware / dynamic
# ─────────────────────────────────────────────────────

def buildLobeChart(hM, dtDeg, vbwDeg,
                   demD=None, demElev=None,
                   units="Metric (m, km)", azDeg=0.0):
    """
    Build the 2D lobe projection chart.

    Flat-earth mode (no DEM):  classic triangular lobes, Y = relative height.
    Terrain mode (DEM loaded): lobes clipped at real ground using the same
                               _getClippedRay / _lobePolygon helpers as
                               buildChart. Y-axis = height relative to site
                               ground (0 = site ground, hM = antenna tip).
    """
    hasDem   = demD is not None and demElev is not None and len(demD) > 1   # True when a valid DEM is loaded
    farAngle = max(0.05, dtDeg - vbwDeg / 2)                               # Far-lobe depression angle (clamped)

    # ── Flat-earth reference distances (always computed as fallback) ─────
    farDF  = hM / math.tan(math.radians(farAngle))                         # Flat-earth far-edge ground distance
    mainDF = hM / math.tan(math.radians(dtDeg))                            # Flat-earth main-lobe ground distance
    nearDF = hM / math.tan(math.radians(dtDeg + vbwDeg / 2))               # Flat-earth near-edge ground distance

    # ── Lobe colour constants ─────────────────────────────────────────────
    C_UPPER_FILL = 'rgba(59,130,246,0.28)'
    C_UPPER_LINE = '#3b82f6'
    C_MAIN_FILL  = 'rgba(248,113,113,0.28)'
    C_MAIN_LINE  = '#f87171'
    C_LOWER_FILL = 'rgba(253,224,71,0.28)'
    C_LOWER_LINE = '#fde047'

    fig = go.Figure()                                                       # Initialise an empty Plotly figure

    # ══════════════════════════════════════════════════════════════════════
    # TERRAIN MODE — reuses _getClippedRay / _lobePolygon / _lowerPolygon
    # ══════════════════════════════════════════════════════════════════════
    if hasDem:
        siteElev = float(demElev[0])                                        # Absolute site ground elevation (MSL)
        N        = 400                                                      # Sample count for smooth curve rendering
        xs       = np.linspace(0, float(demD[-1]), N)                      # Distance array spanning the full DEM profile

        terrainAbs = np.interp(xs, demD, demElev)                          # Absolute terrain elevation at each sample
        terrainRel = terrainAbs - siteElev                                  # Height relative to site ground (0 = site level)

        # Compute clipped rays in relative-height space (siteElev=0, terrain=terrainRel)
        xsFar,  ysFar  = _getClippedRay(xs, terrainRel, 0.0, hM, farAngle)
        xsMain, ysMain = _getClippedRay(xs, terrainRel, 0.0, hM, dtDeg)
        xsNear, ysNear = _getClippedRay(xs, terrainRel, 0.0, hM, dtDeg + vbwDeg / 2)

        farHitX  = float(xsFar[-1])                                        # Terrain hit distance of far lobe ray
        mainHitX = float(xsMain[-1])                                       # Terrain hit distance of main lobe ray
        nearHitX = float(xsNear[-1])                                       # Terrain hit distance of near lobe ray
        farHitY  = float(np.interp(farHitX,  xs, terrainRel))              # Relative elevation at far hit
        mainHitY = float(np.interp(mainHitX, xs, terrainRel))              # Relative elevation at main hit
        nearHitY = float(np.interp(nearHitX, xs, terrainRel))              # Relative elevation at near hit

        yMin = float(np.min(terrainRel)) - 10                              # Chart y-min: 10 m below lowest terrain
        yMax = hM + 30                                                     # Chart y-max: 30 m headroom above antenna tip
        xMax = float(demD[-1])                                             # Chart x-max: end of the DEM profile

        # ── Terrain fill ──────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=xs, y=terrainRel,
            fill='tozeroy',                                                # Fill terrain area down to y=0 (site level)
            fillcolor='rgba(125,211,252,0.25)',
            line=dict(color='#7dd3fc', width=1.5),
            name='Terrain',
            hovertemplate=(
                'Dist: %{x:.0f} m<br>'
                'Rel. height: %{y:.1f} m'
                '<extra></extra>'
            ),
        ))

        # ── Lower lobe fill ───────────────────────────────────────────────
        loPx, loPy = _lowerPolygon(xsNear, ysNear, xs, terrainRel)        # Build polygon from near ray to terrain
        fig.add_trace(go.Scatter(
            x=loPx, y=loPy, fill='toself', fillcolor=C_LOWER_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_lf', hoverinfo='skip',
        ))

        # ── Main lobe fill ────────────────────────────────────────────────
        if mainHitX >= nearHitX:                                           # Ensure farther ray is passed first
            maPx, maPy = _lobePolygon(xsMain, ysMain, xsNear, ysNear, xs, terrainRel)
        else:
            maPx, maPy = _lobePolygon(xsNear, ysNear, xsMain, ysMain, xs, terrainRel)
        fig.add_trace(go.Scatter(
            x=maPx, y=maPy, fill='toself', fillcolor=C_MAIN_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_mf', hoverinfo='skip',
        ))

        # ── Upper lobe fill ───────────────────────────────────────────────
        if farHitX >= mainHitX:                                            # Ensure farther ray is passed first
            upPx, upPy = _lobePolygon(xsFar, ysFar, xsMain, ysMain, xs, terrainRel)
        else:
            upPx, upPy = _lobePolygon(xsMain, ysMain, xsFar, ysFar, xs, terrainRel)
        fig.add_trace(go.Scatter(
            x=upPx, y=upPy, fill='toself', fillcolor=C_UPPER_FILL,
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False, name='_uf', hoverinfo='skip',
        ))

        # ── Clipped lobe ray lines with terrain-clearance hover ───────────
        for xsR, ysR, lineColor, lbl, angleLbl, hitX, hitY, emoji in [
            (xsFar,  ysFar,  C_UPPER_LINE,
             f'Upper Lobe ({fmtD(farHitX, units)})',
             f'{farAngle:.2f}°', farHitX, farHitY, '🔵'),
            (xsMain, ysMain, C_MAIN_LINE,
             f'Main Lobe ({fmtD(mainHitX, units)})',
             f'{dtDeg:.2f}°', mainHitX, mainHitY, '🔴'),
            (xsNear, ysNear, C_LOWER_LINE,
             f'Lower Lobe ({fmtD(nearHitX, units)})',
             f'{dtDeg + vbwDeg/2:.2f}°', nearHitX, nearHitY, '🟡'),
        ]:
            terrainAtRay = np.interp(xsR, xs, terrainRel)                 # Terrain height directly below each ray sample
            diff         = ysR - terrainAtRay                             # Ray clearance above terrain at each sample
            fig.add_trace(go.Scatter(
                x=xsR, y=ysR,
                mode='lines',
                line=dict(color=lineColor, width=2),
                name=lbl,
                customdata=np.column_stack([diff]),                        # Pass clearance values as custom hover data
                hovertemplate=(
                    f'<b>{emoji} {lbl.split("(")[0].strip()}</b><br>'
                    'Rel. Difference: %{customdata[0]:.1f} m<br>'
                    f'Ground hit: {fmtD(hitX, units)}'
                    '<extra></extra>'
                ),
            ))

        # ── Terrain-hit intersection markers ──────────────────────────────
        for hitX, hitY, lineColor, emoji, lobeName in [
            (farHitX,  farHitY,  C_UPPER_LINE, '🔵', 'Upper Lobe'),
            (mainHitX, mainHitY, C_MAIN_LINE,  '🔴', 'Main Lobe'),
            (nearHitX, nearHitY, C_LOWER_LINE, '🟡', 'Lower Lobe'),
        ]:
            fig.add_trace(go.Scatter(
                x=[hitX], y=[hitY],
                mode='markers',
                marker=dict(size=10, color='white', symbol='x',
                            line=dict(color=lineColor, width=2)),
                showlegend=False,
                hovertemplate=(
                    f'<b>{emoji} {"Ground Intersection"}</b><br>'
                    f'Distance: {fmtD(hitX, units)}<br>'
                    f'Rel. elevation: {hitY:.1f} m'
                    '<extra></extra>'
                ),
            ))

        # ── Antenna stem and tip marker ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, hM],                                           # Vertical red line from site ground to antenna tip
            line=dict(color='#ef4444', width=3),
            showlegend=False, name='_stem', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[hM],                                                 # Diamond marker at the antenna tip
            mode='markers+text',
            marker=dict(size=11, color='#ef4444', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f'  {hM:.0f} m AGL'], textposition='middle right',
            textfont=dict(family='Inter', size=10, color='#ef4444'),
            name='Antenna',
            hoverinfo='skip',
        ))

        titleText  = 'Terrain-adjusted'                                    # Chart annotation title for terrain mode
        yAxisLabel = 'Antenna Height (m)'                                  # Y-axis label (relative to site ground)

    # ══════════════════════════════════════════════════════════════════════
    # FLAT-EARTH MODE  — classic triangular lobe polygons
    # ══════════════════════════════════════════════════════════════════════
    else:
        xMax = farDF * 1.08                                                # 8% horizontal padding beyond the far-lobe edge
        yMin = -2.0                                                        # Slight below-ground margin
        yMax = hM * 1.18                                                   # 18% headroom above the antenna height

        def _feYs(xsE, angleDeg):
            """Compute ray height above flat ground at each distance sample."""
            return hM - xsE * math.tan(math.radians(angleDeg))            # Linear height decay with distance

        xsHover = np.linspace(0, farDF, 60)                               # Reduced-density x array for invisible hover traces

        # ── Flat terrain base line ────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, farDF * 1.08], y=[0, 0],
            fill='tozeroy',                                                # Light-blue flat ground fill area
            fillcolor='rgba(125,211,252,0.18)',
            line=dict(color='#7dd3fc', width=1.2),
            showlegend=False, name='_flat_gnd', hoverinfo='skip',
        ))

        # ── Filled triangular lobe polygons ───────────────────────────────
        for polyX, polyY, fillCol, lbl in [
            ([0, farDF,  mainDF, 0], [hM, 0, 0, hM],
             'rgba(59,130,246,0.45)',  'Upper Lobe'),                      # site→far edge→main edge→site
            ([0, mainDF, nearDF, 0], [hM, 0, 0, hM],
             'rgba(248,113,113,0.45)', 'Main Lobe'),                       # site→main edge→near edge→site
            ([0, nearDF, 0,     0], [hM, 0, 0, hM],
             'rgba(253,224,71,0.45)', 'Lower Lobe'),                       # site→near edge→site
        ]:
            fig.add_trace(go.Scatter(
                x=polyX, y=polyY,
                fill='toself', fillcolor=fillCol,
                line=dict(color=fillCol.replace('0.45', '0.85'), width=1.5),  # Slightly opaque polygon border
                name=lbl, hoverinfo='skip',
            ))

        # ── Invisible hover-only lines (for tooltip interactivity) ────────
        for angleDeg, dHit, lineColor, lbl, emoji in [
            (farAngle,           farDF,  C_UPPER_LINE, 'Upper Lobe', '🔵'),
            (dtDeg,              mainDF, C_MAIN_LINE,  'Main Lobe',  '🔴'),
            (dtDeg + vbwDeg/2,   nearDF, C_LOWER_LINE, 'Lower Lobe', '🟡'),
        ]:
            xsE = np.linspace(0, dHit, 40)                                # 40-point distance array to the hit point
            ysE = _feYs(xsE, angleDeg)                                    # Corresponding ray heights
            fig.add_trace(go.Scatter(
                x=xsE, y=ysE,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=10),               # Fully transparent — hover capture area only
                showlegend=False,
                hovertemplate=(
                    f'<b>{emoji} {lbl}</b><br>'
                    'Height: %{y:.1f} m<br>'
                    f'Ground hit: {fmtD(dHit, units)}'
                    '<extra></extra>'
                ),
            ))

        # ── Ground hit markers ────────────────────────────────────────────
        for dHit, lineColor, lbl, emoji in [
            (farDF,  C_UPPER_LINE, f'Upper: {fmtD(farDF, units)}',  '🔵'),
            (mainDF, C_MAIN_LINE,  f'Main: {fmtD(mainDF, units)}',  '🔴'),
            (nearDF, C_LOWER_LINE, f'Lower: {fmtD(nearDF, units)}', '🟡'),
        ]:
            fig.add_trace(go.Scatter(
                x=[dHit], y=[0],                                           # Marker positioned at ground level
                mode='markers',
                marker=dict(size=9, color=lineColor, symbol='circle',
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hovertemplate=f'{lbl}<extra></extra>',
            ))

        # ── Antenna stem and tip marker ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, hM],                                           # Vertical red antenna mast line
            line=dict(color='#ef4444', width=3),
            showlegend=False, name='_stem', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[hM],
            mode='markers+text',
            marker=dict(size=11, color='#ef4444', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f'  {hM:.0f} m AGL'], textposition='middle right',
            textfont=dict(family='Inter', size=10, color='#ef4444'),
            name='Antenna',
            hoverinfo='skip',
        ))

        titleText  = 'Lobe Distance Projection'                            # Chart title for flat-earth mode
        subtitle   = 'Theoretical flat-earth distances'                    # Subtitle annotation text
        yAxisLabel = 'Antenna Height (m)'                                  # Y-axis label

    # ── Shared layout (applied to both terrain and flat-earth modes) ──────
    fig.update_layout(
        plot_bgcolor='#0f172a',                                            # Dark chart background colour
        paper_bgcolor='#1e293b',                                           # Slightly lighter figure paper colour
        font=dict(family='Inter', color='#94a3b8', size=11),
        margin=dict(l=60, r=20, t=70, b=55),
        height=380,
        legend=dict(
            orientation='v', yanchor='top', y=0.97, xanchor='right', x=0.98,  # Top-right vertical legend
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
            range=[0, xMax], showline=True, linecolor='#334155',
        ),
        yaxis=dict(
            title=dict(text=yAxisLabel,
                       font=dict(size=11, color='#64748b', family='Inter')),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.10)',
            tickfont=dict(size=10, color='#64748b', family='JetBrains Mono'),
            range=[yMin, yMax], showline=True, linecolor='#334155',
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(family='JetBrains Mono', size=11, color='#e2e8f0'),
            align='left',
        ),
        annotations=[
            dict(x=0.5, y=1.04, xref='paper', yref='paper', showarrow=False,
                 text=titleText,
                 font=dict(size=12, color='#e2e8f0', family='Inter', weight=700),
                 xanchor='center'),
        ],
    )
    return fig                                                             # Return the fully configured lobe chart


# ─────────────────────────────────────────────────────
# FOLIUM SECTOR MAP
# ─────────────────────────────────────────────────────

def buildMap(lat, lon, az, hbw, mainD, nearD, farD, demD, distM):
    """Build a Folium satellite map showing the antenna sector, footprint zone, and DEM profile path."""
    mapObj = folium.Map(
        location=[lat, lon], zoom_start=13, control_scale=True,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'                                          # Esri satellite imagery basemap
    )
    halfAngle   = hbw / 2                                                 # Half the horizontal beamwidth in degrees
    numSegments = 72                                                       # Arc resolution: 72 segments ≈ 5° per step
    # Build outer arc of sector at far-edge distance
    outerRing = [gcDest(lat, lon, az - halfAngle + i * hbw / numSegments, farD)
                 for i in range(numSegments + 1)]
    # Build inner arc of sector at near-edge distance, reversed for correct polygon winding
    innerRing = [gcDest(lat, lon, az - halfAngle + i * hbw / numSegments, nearD)
                 for i in range(numSegments, -1, -1)]
    # Build the DEM elevation profile path (or a default straight-line profile)
    profilePath = [gcDest(lat, lon, az, d)
                   for d in (demD if demD is not None else np.linspace(0, distM, 60))]

    # ── Sector outline polygon (faint blue fill) ──────────────────────────
    folium.Polygon([(lat, lon)] + outerRing + [(lat, lon)],
        color='#0ea5e9', weight=1.5, fill=True,
        fill_color='#0ea5e9', fill_opacity=0.06).add_to(mapObj)
    # ── Footprint zone polygon (green fill between inner and outer arcs) ──
    folium.Polygon(outerRing + innerRing,
        color='#4ade80', weight=2, fill=True,
        fill_color='#4ade80', fill_opacity=0.20).add_to(mapObj)
    # ── DEM elevation profile path (dashed orange line) ───────────────────
    folium.PolyLine(profilePath, color='#f97316', weight=2.2,
                    opacity=0.8, dash_array='7 5').add_to(mapObj)
    # ── Main lobe direction line (solid green from site to main hit) ──────
    folium.PolyLine([(lat, lon), gcDest(lat, lon, az, mainD)],
        color='#16a34a', weight=1.5, opacity=0.7).add_to(mapObj)
    # ── Antenna site circle marker ────────────────────────────────────────
    folium.CircleMarker([lat, lon], radius=9, color='#0ea5e9', fill=True,
        fill_color='#0ea5e9', fill_opacity=1, weight=2,
        popup=folium.Popup(
            f'<b>Antenna Site</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Az: {az}°'
        )).add_to(mapObj)
    # ── Main lobe ground-hit circle marker ───────────────────────────────
    mainLobeHit = gcDest(lat, lon, az, mainD)                             # Geographic coordinates of the main lobe impact point
    folium.CircleMarker(mainLobeHit, radius=8, color='#0d9488', fill=True,
        fill_color='#0d9488', fill_opacity=1, weight=2,
        popup=folium.Popup(
            f'<b>Main Lobe Hit</b><br>{fmtD(mainD, "Metric (m, km)")} from site'
        )).add_to(mapObj)
    mapObj.add_child(folium.LatLngPopup())                                # Enable click-to-show-coordinates popup on the map
    return mapObj                                                         # Return the fully configured Folium map object


# ─────────────────────────────────────────────────────
# KMZ EXPORT
# ─────────────────────────────────────────────────────

def buildKmz(lat, lon, az, hbw, mainD, nearD, farD):
    """Build a KMZ file (zipped KML) with sector and footprint polygons for Google Earth."""
    halfAngle   = hbw / 2                                                 # Half horizontal beamwidth in degrees
    numSegments = 60                                                       # Arc resolution for KML polygon vertices

    def coordStr(bearing, dist):
        """Return a KML coordinate string 'lon,lat,0' for a destination point."""
        point = gcDest(lat, lon, bearing, dist)                           # Compute geographic destination
        return f"{point[1]},{point[0]},0"                                 # KML uses lon,lat order (reversed from standard)

    # Build space-separated KML coordinate strings for outer and inner arcs
    outerCoords = " ".join(
        coordStr(az - halfAngle + i * hbw / numSegments, farD)
        for i in range(numSegments + 1)
    )
    innerCoords = " ".join(
        coordStr(az - halfAngle + i * hbw / numSegments, nearD)
        for i in range(numSegments, -1, -1)
    )

    kmlContent = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <n>Sector Az{az}</n>
  <Style id="s"><LineStyle><color>ffe9a50e</color><width>2</width></LineStyle>
    <PolyStyle><color>1ae9a50e</color></PolyStyle></Style>
  <Style id="f"><LineStyle><color>ff80de4a</color><width>2</width></LineStyle>
    <PolyStyle><color>3380de4a</color></PolyStyle></Style>
  <Placemark><n>Sector</n><styleUrl>#s</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {lon},{lat},0 {outerCoords} {lon},{lat},0
    </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
  <Placemark><n>Footprint</n><styleUrl>#f</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {outerCoords} {innerCoords}
    </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
  <Placemark><n>Site</n>
    <Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>
</Document></kml>"""                                                       # Full KML markup defining sector, footprint, and site

    kmzBuffer = io.BytesIO()                                              # In-memory byte buffer for the ZIP archive
    with zipfile.ZipFile(kmzBuffer, "w", zipfile.ZIP_DEFLATED) as zipFile:  # Create a compressed ZIP archive (KMZ format)
        zipFile.writestr("doc.kml", kmlContent)                           # Add KML string as "doc.kml" inside the archive
    return kmzBuffer.getvalue()                                           # Return raw KMZ bytes ready for download


# ─────────────────────────────────────────────────────
# SESSION STATE — initialise keys on first run only
# ─────────────────────────────────────────────────────
for k, v in [('dem_d', None), ('dem_elev', None),
             ('dem_status', None), ('dem_msg', ''), ('dem_source', '')]:
    if k not in st.session_state:                                         # Only set the key if it doesn't already exist
        st.session_state[k] = v                                           # Prevents overwriting live values on rerun


# ─────────────────────────────────────────────────────
# PAGE HEADER — elevation source badge
# ─────────────────────────────────────────────────────

# Build badge HTML based on which elevation library is currently installed
if _RASTERIO_AVAILABLE:
    _copernicusBadge = (                                                  # Green badge: best offline source is available
        '<span class="src-badge src-local">'
        '🌍 Copernicus DEM GLO-30 — offline tiles, 30 m, no API key'
        '</span>'
    )
elif _SRTM_AVAILABLE:
    _copernicusBadge = (                                                  # Green badge: SRTM offline fallback available
        '<span class="src-badge src-local">'
        '⚡ SRTM1 30m local'
        '</span>'
    )
else:
    _copernicusBadge = (                                                  # Yellow badge: cloud-only fallback active
        '<span class="src-badge src-cloud">'
        '☁ Cloud elevation — pip install rasterio for offline Copernicus DEM'
        '</span>'
    )

# Render the main page header with app title and active elevation source badge
st.markdown(f"""
<div style="border-bottom:1px solid #dde3ec;padding-bottom:14px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">
    <div style="width:30px;height:30px;background:#0ea5e9;border-radius:7px;display:flex;
                align-items:center;justify-content:center;font-size:15px;color:#fff;">📡</div>
    <span style="font-size:1.2rem;font-weight:700;color:#1e293b;">Antenna Downtilt Calculator</span>
    {_copernicusBadge}
  </div>
  <div style="padding-left:40px;font-size:0.8rem;color:#64748b;">
    Calculate main lobe impact and ground footprint with an interactive RF visual.
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
                'letter-spacing:1.5px;color:#94a3b8;margin-bottom:12px;">Inputs</div>',
                unsafe_allow_html=True)                                    # Section heading in uppercase monospace style
    st.caption("Set basic RF parameters:")                                 # Helper text beneath the section heading

    units  = st.radio("Unit System", ["Metric (m, km)", "USA (ft, mi)"], horizontal=True)  # Toggle between metric and imperial
    hLabel = "Antenna Height (m)" if units == "Metric (m, km)" else "Antenna Height (ft)"  # Dynamic input label
    dLabel = "Distance (m)"       if units == "Metric (m, km)" else "Distance (ft)"        # Dynamic input label

    hRaw  = st.number_input(hLabel,  min_value=1.0,   value=30.0,  step=1.0,  format="%.1f")  # Raw antenna height in chosen units
    dtDeg = st.number_input("Downtilt (deg)",           min_value=0.1, max_value=89.0, value=6.0,  step=0.5)  # Mechanical/electrical downtilt
    vbw   = st.number_input("Vertical Beamwidth (deg)", min_value=1.0, max_value=90.0, value=6.0,  step=0.5)  # Vertical half-power beamwidth
    hbw   = st.number_input("Horizontal Beamwidth (deg)", min_value=1.0, max_value=360.0, value=65.0, step=1.0)  # Horizontal half-power beamwidth
    dRaw  = st.number_input(dLabel,  min_value=100.0, value=1000.0, step=100.0, format="%.0f")    # Raw profile distance in chosen units

    hM    = toM(hRaw, units)                                               # Convert antenna height to metres
    distM = toM(dRaw, units)                                               # Convert profile distance to metres
    st.markdown(f"<div style='font-size:.72rem;color:#0284c7;font-family:JetBrains Mono,monospace;"
                f"margin-top:-6px;margin-bottom:8px;'>→ {fmtD(distM, units)}</div>",
                unsafe_allow_html=True)                                    # Show the converted distance in metres below the input
    st.divider()                                                           # Visual separator between parameter groups

    terrainOn = st.checkbox("Include elevation profile", value=True,
                            help="Uses site coordinates + azimuth and distance as profile length.")  # Toggle DEM fetch on/off
    if terrainOn:
        siteLat = st.number_input("Site Latitude (deg)",   value=30.0028686, format="%.7f", step=0.0001)   # Site latitude in WGS84 degrees
        siteLon = st.number_input("Site Longitude (deg)",  value=31.0719953, format="%.7f", step=0.0001)   # Site longitude in WGS84 degrees
        azDeg   = st.number_input("Antenna Azimuth (deg)", min_value=0.0, max_value=360.0, value=80.0, step=1.0)  # Bearing for the profile path

        # Inform the user which elevation source will be used based on installed libraries
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
                "⚡ SRTM1 30m tiles </div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size:0.68rem;color:#92400e;background:#fef3c7;"
                "border-radius:5px;padding:5px 10px;margin-bottom:4px;'>"
                "☁ Will use Open-Elevation cloud API<br>"
                "<code>pip install rasterio</code> for offline Copernicus DEM GLO-30</div>",
                unsafe_allow_html=True)

        if st.button("↻ Fetch Elevation Profile", type="primary", use_container_width=True):
            # Determine the source label used in the loading spinner message
            if _RASTERIO_AVAILABLE:
                srcLabel = "Copernicus DEM GLO-30 (local)"
            elif _SRTM_AVAILABLE:
                srcLabel = "SRTM1 30m (local)"
            else:
                srcLabel = "Open-Elevation (cloud)"
            with st.spinner(f"Loading elevation via {srcLabel}…"):        # Display spinner while fetching
                try:
                    dArr, eArr, sourceName = fetchDem(siteLat, siteLon, azDeg, distM, n=100)  # Fetch DEM profile data
                    st.session_state.dem_d      = dArr                    # Persist distance array in session state
                    st.session_state.dem_elev   = eArr                    # Persist elevation array in session state
                    st.session_state.dem_status = "ok"                    # Mark fetch status as successful
                    st.session_state.dem_source = sourceName              # Record which elevation source was used
                    st.session_state.dem_msg    = (
                        f"Elevation loaded — {len(dArr)} samples · Source: {sourceName}")
                    st.rerun()                                             # Rerun to refresh the UI with new DEM data
                except Exception as ex:
                    st.session_state.dem_status = "error"                 # Mark fetch status as failed
                    st.session_state.dem_msg    = f"Fetch failed: {ex}"   # Store the error message for the status bar
                    st.session_state.dem_d      = None                    # Clear any stale distance array
                    st.session_state.dem_elev   = None                    # Clear any stale elevation array
                    st.session_state.dem_source = ""                      # Clear the stale source label
                    st.rerun()                                             # Rerun to display the error status
    else:
        # Terrain disabled — reset to default site coordinates and clear any stored DEM data
        siteLat, siteLon, azDeg = 30.0028686, 31.0719953, 80.0
        st.session_state.dem_d      = None
        st.session_state.dem_elev   = None
        st.session_state.dem_status = None
        st.session_state.dem_msg    = ""
        st.session_state.dem_source = ""

    st.divider()                                                           # Separator before the reset button
    if st.button("↺ Reset", use_container_width=True):                    # Full session state reset button
        for k in ['dem_d', 'dem_elev', 'dem_status', 'dem_msg', 'dem_source']:
            st.session_state[k] = "" if k in ("dem_msg", "dem_source") else None  # Clear each key appropriately
        st.rerun()                                                         # Rerun to apply the cleared state


# ─────────────────────────────────────────────────────
# COMPUTE GEOMETRY
# ─────────────────────────────────────────────────────
demD    = st.session_state.dem_d                                          # Retrieve distance array from session state
demElev = st.session_state.dem_elev                                       # Retrieve elevation array from session state
hasDem  = demD is not None and demElev is not None and len(demD) > 1     # True when a valid DEM profile is loaded

if hasDem and terrainOn:                                                   # Terrain mode: intersect lobe rays with DEM
    siteElev       = float(demElev[0])                                     # Site ground elevation (first DEM sample point)
    mainD          = rayIntersect(hM, siteElev, dtDeg,                demD, demElev)  # Main lobe terrain hit distance
    nearD          = rayIntersect(hM, siteElev, dtDeg + vbw / 2,     demD, demElev)  # Near lobe terrain hit distance
    farD           = rayIntersect(hM, siteElev, max(0.05, dtDeg - vbw/2), demD, demElev)  # Far lobe terrain hit distance
    flatGeomResult = flatGeomFull(hM, dtDeg, vbw)                         # Compute flat-earth fallback distances
    if not mainD: mainD = flatGeomResult[0]                                # Use flat-earth main if no terrain intersection found
    if not nearD: nearD = flatGeomResult[1]                                # Use flat-earth near if no terrain intersection found
    if not farD:  farD  = flatGeomResult[2]                                # Use flat-earth far if no terrain intersection found
    geoNote = "Terrain-adjusted"                                          # Label displayed in metric cards and status bar
else:                                                                      # Flat-earth mode: use pure geometric model
    mainD, nearD, farD = flatGeomFull(hM, dtDeg, vbw)
    geoNote = "Flat-earth model"                                          # Label displayed in metric cards


# ─────────────────────────────────────────────────────
# LIVE TERRAIN STATS
# ─────────────────────────────────────────────────────
liveStats = None                                                          # Will hold a stats dict when DEM is active
if hasDem and terrainOn:
    seStats    = float(demElev[0])                                        # Site elevation for the ray calculation
    rayZ       = seStats + hM - demD * np.tan(np.radians(dtDeg))         # Main lobe ray absolute height at each DEM sample
    aboveArr   = rayZ >= demElev                                          # Boolean array: True where ray clears the terrain
    nAbove     = int(np.sum(aboveArr))                                    # Count of samples where the ray is above terrain
    avgSig     = round(nAbove / len(demD) * 100)                         # Signal coverage as a percentage (0–100)
    shadowPct  = 100 - avgSig                                             # Shadowed/blocked percentage
    blockedArr = ~aboveArr                                                # Boolean array: True where terrain blocks the ray
    firstObs   = float(demD[np.argmax(blockedArr)]) if np.any(blockedArr) else None  # Distance (m) to the first obstruction
    liveStats  = dict(
        avg       = avgSig,
        shadow    = shadowPct,
        first_obs = firstObs,
        n         = len(demD),
    )                                                                     # Pack computed stats into a dict for the status bar


# ─────────────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────────────
demStatus = st.session_state.dem_status                                   # Fetch status: 'ok', 'error', or None
demSrc    = st.session_state.dem_source                                   # Human-readable elevation source label

# Build an HTML badge string reflecting the active elevation source
if demSrc == "Copernicus DEM GLO-30 (local)":
    srcHtml = '<span class="src-badge src-local">Copernicus DEM GLO-30</span>'   # Green badge for local Copernicus tiles
elif demSrc == "SRTM1 30m (local)":
    srcHtml = '<span class="src-badge src-local">SRTM1 30m local</span>'         # Green badge for local SRTM tiles
elif demSrc:
    srcHtml = '<span class="src-badge src-cloud">cloud API</span>'               # Yellow badge for cloud fallback
else:
    srcHtml = ''                                                          # No badge when no source is active

if demStatus == "ok" and liveStats:
    obsStr  = fmtD(liveStats['first_obs'], units) if liveStats['first_obs'] else "None"  # Format first obstruction distance
    liveMsg = (f"Elevation loaded ({liveStats['n']} samples) · "
               f"Avg signal: {liveStats['avg']}% · "
               f"Shadowed: {liveStats['shadow']}% · "
               f"1st obstruction: {obsStr}")
    st.markdown(f'<div class="status-ok">✓ {liveMsg} {srcHtml}</div>', unsafe_allow_html=True)
elif demStatus == "error":
    st.markdown(f'<div class="status-err">✕ {st.session_state.dem_msg}</div>', unsafe_allow_html=True)  # Error status bar
else:
    st.markdown(
        f'<div class="status-idle">Flat model · h={fmtD(hM, units)} AGL · '
        f'Main lobe={fmtD(mainD, units)}</div>',
        unsafe_allow_html=True)                                           # Idle status: flat-earth summary


# ─────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)                                                # Two equal-width columns side by side
with col1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Main Lobe Impact Point</div>
      <div class="metric-value">{fmtD(mainD, units)}</div>
      <div class="metric-sub">Ground intersection of central lobe ({geoNote}).</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Footprint Coverage</div>
      <div class="metric-value" style="font-size:1.05rem;">{fmtD(nearD, units)} to {fmtD(farD, units)}</div>
      <div class="metric-sub">{geoNote} · VBW {vbw:.1f}° · Width: {fmtD(farD - nearD, units)}</div>
    </div>""", unsafe_allow_html=True)

unitLabel = "Metric (m/km)" if units == "Metric (m, km)" else "USA (ft/mi)"  # Short unit system label for the assumption box
st.markdown(f"""<div class="assume">
  <strong>Assumption:</strong> Vertical beamwidth defaults to <code>{vbw:.1f} deg</code>,
  but you can modify it. Unit system: {unitLabel}.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TERRAIN STATS ROW
# ─────────────────────────────────────────────────────
if liveStats:
    obsStr = fmtD(liveStats['first_obs'], units) if liveStats['first_obs'] else "None"  # Formatted first obstruction distance
    st.markdown(f"""<div class="tstat-row">
      <div class="tstat">
        <div class="tstat-val">{obsStr}</div>
        <div class="tstat-lbl">1st Obstruction</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# VISUAL PROFILE CHART + SLIDER
# ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Visual Profile</div>', unsafe_allow_html=True)  # Section heading

sliderD = st.slider(
    "Footprint vs selected distance",
    min_value=0.0, max_value=float(distM),
    value=float(distM * 0.25),                                            # Default to 25% of the total profile distance
    step=max(1.0, distM / 500), format="%.0f m"                          # Step size: at most 500 discrete increments
)

sliderElev  = float(np.interp(sliderD, demD, demElev)) if hasDem else 0.0  # Terrain elevation (MSL) at the slider position
inFootprint = nearD <= sliderD <= farD                                    # True if the selected distance falls inside the footprint
colInfo1, colInfo2, colInfo3 = st.columns(3)                              # Three info columns beneath the slider
colInfo1.metric("📍 Position",     fmtD(sliderD, units))
colInfo2.metric("⛰ Terrain Elev", f"{sliderElev:.1f} m MSL" if hasDem else "—")
colInfo3.metric("📶 Footprint",    "✓ Inside" if inFootprint else "✗ Outside")

# Render the main terrain profile + lobe projection chart
terrainFig = buildChart(hM, dtDeg, vbw, distM, mainD, nearD, farD,
                        demD, demElev, sliderD, units, azDeg=azDeg)
st.plotly_chart(terrainFig, use_container_width=True,
                config={'displayModeBar': True, 'displaylogo': False})


# ─────────────────────────────────────────────────────
# SECTOR METRICS  (placed above the map/lobe grid)
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
    <div class="sm"><div class="sm-lbl">Main Lobe</div><div class="sm-val">{fmtD(mainD, units)}</div></div>
    <div class="sm"><div class="sm-lbl">Footprint</div>
      <div class="sm-val" style="font-size:0.8rem;">{fmtD(nearD, units)} to {fmtD(farD, units)}</div></div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# GRID ROW: Sector Map + Lobe Projection side by side
# ─────────────────────────────────────────────────────
mapCol, lobeCol = st.columns(2, gap="medium")                             # Two equal columns with a medium gap between them

with mapCol:
    st.markdown('<div class="sec-hdr">Sector Map</div>', unsafe_allow_html=True)
    st.caption("Sector built from site coordinates, azimuth, horizontal beamwidth.")
    mapObj  = buildMap(siteLat, siteLon, azDeg, hbw, mainD, nearD, farD, demD, distM)  # Build the Folium sector map
    mapData = st_folium(mapObj, width="100%", height=380,
                        returned_objects=["last_clicked"],                 # Return the coordinates of the last map click
                        key="sector_map")

with lobeCol:
    st.markdown('<div class="sec-hdr">Lobe Projection</div>', unsafe_allow_html=True)
    if hasDem:
        st.caption("Terrain-adjusted")                                    # Caption changes based on DEM availability
    else:
        st.caption("Flat-earth model.")
    lobeFig = buildLobeChart(
        hM, dtDeg, vbw,
        demD=demD, demElev=demElev,
        units=units, azDeg=azDeg,
    )
    st.plotly_chart(lobeFig, use_container_width=True,
                    config={'displayModeBar': True, 'displaylogo': False})


# ─────────────────────────────────────────────────────
# MAP CLICK HANDLER
# ─────────────────────────────────────────────────────
_clicked = (mapData or {}).get("last_clicked")                            # Extract the last-clicked coordinates from map callback data
if _clicked and _clicked.get("lat") is not None:
    _clickedLat, _clickedLng = _clicked["lat"], _clicked["lng"]          # Unpack clicked latitude and longitude
    _R          = 6_371_000.0                                             # Earth mean radius in metres for the Haversine formula
    _dLat       = math.radians(_clickedLat - siteLat)                    # Latitude difference in radians
    _dLon       = math.radians(_clickedLng - siteLon)                    # Longitude difference in radians
    _haversineA = (math.sin(_dLat / 2) ** 2                              # Haversine formula intermediate value 'a'
                   + math.cos(math.radians(siteLat))
                   * math.cos(math.radians(_clickedLat))
                   * math.sin(_dLon / 2) ** 2)
    _clickDist  = _R * 2 * math.atan2(math.sqrt(_haversineA), math.sqrt(1 - _haversineA))  # Great-circle distance in metres
    st.markdown(f"""
<div class="click-card">
  <div class="click-icon">📍</div>
  <div class="click-body">
    <div class="click-title">Selected Point — Map Click</div>
    <div class="click-fields">
      <div class="click-field"><span>Latitude</span>{_clickedLat:.6f}°</div>
      <div class="click-field"><span>Longitude</span>{_clickedLng:.6f}°</div>
      <div class="click-field"><span>Distance from Site</span>{fmtD(_clickDist, units)}</div>
    </div>
    <div class="click-hint">Click anywhere on the map to update</div>
  </div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="click-idle">🖱 Click anywhere on the map to see its coordinates and distance from the antenna site</div>',
        unsafe_allow_html=True)                                           # Placeholder shown before any map click is made


# ─────────────────────────────────────────────────────
# MAP FOOTER + KMZ EXPORT
# ─────────────────────────────────────────────────────
colMeta, colKmz = st.columns([3, 1])                                      # 3:1 column split: meta info left, download button right
with colMeta:
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#94a3b8;margin-top:8px;'>"
        f"Site {siteLat:.6f}, {siteLon:.6f} · Az {azDeg:.0f}° · "
        f"H-BW {hbw:.0f}° · Radius {fmtD(farD, units)}.</div>",
        unsafe_allow_html=True)                                           # One-line site summary in monospace style
with colKmz:
    kmzData = buildKmz(siteLat, siteLon, azDeg, hbw, mainD, nearD, farD)  # Generate the KMZ file bytes
    st.download_button("⬇ Export Sector KMZ", data=kmzData,
                       file_name=f"antenna_az{azDeg:.0f}.kmz",
                       mime="application/vnd.google-earth.kmz",
                       use_container_width=True)                          # Full-width download button for the KMZ export
