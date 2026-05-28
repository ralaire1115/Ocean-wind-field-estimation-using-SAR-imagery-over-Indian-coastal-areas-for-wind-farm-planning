import os
import io
import logging
import zipfile
from typing import Optional

import numpy as np
import requests
import ee

logger = logging.getLogger(__name__)

# Constants

# Sentinel-1 GRD collection identifier on GEE
S1_COLLECTION = "COPERNICUS/S1_GRD"

# Patch size (pixels) fetched from GEE.  Kept small for fast API responses.
PATCH_SIZE = 256          # ~640 m resolution at 20 m/px
SCALE_METRES = 20         # Sentinel-1 IW native GRD resolution ≈ 20 m

REGION_BBOX = {
    "tamil_nadu": [78.0, 8.0,  80.5, 13.5],
    "gujarat":    [68.0, 20.0, 74.5, 24.5],
}

# GEE bands we care about
POLARISATION = "VV"

# Initialisation

def _initialise_gee() -> None:
    key_path: Optional[str] = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

    if key_path and os.path.isfile(key_path):
        # Service-account flow (recommended for server deployments)
        credentials = ee.ServiceAccountCredentials(
            email=None,   # email is read from the JSON key file
            key_file=key_path,
        )
        ee.Initialize(credentials)
        logger.info("GEE initialised via service-account key: %s", key_path)
    else:
        # Interactive / default credential flow (local development)
        ee.Initialize(project='cec-open-ocean-wind-field-est')
        logger.info("GEE initialised via default credentials.")


# Public API

def fetch_sar_image(region_name: str, target_date: str) -> np.ndarray:
    """
    Fetch a Sentinel-1 GRD (VV, IW mode) image patch from GEE and return it
    as a normalised NumPy float32 array of shape ``(1, PATCH_SIZE, PATCH_SIZE)``.

    Parameters
    ----------
    region_name : str
        One of ``'tamil_nadu'`` or ``'gujarat'`` (case-insensitive).
    target_date : str
        ISO-8601 date string  (``'YYYY-MM-DD'``).  The collection is filtered
        over a ±1-day window centred on this date to improve availability.

    Returns
    -------
    np.ndarray
        Shape ``(1, 256, 256)``, dtype ``float32``.  Values are the raw
        Sentinel-1 σ₀ backscatter in dB, linearly scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``region_name`` is not recognised.
    RuntimeError
        If no Sentinel-1 image is found for the requested region/date, or if
        the GEE download fails.
    """
    # 1. Validate region
    key = region_name.lower().replace(" ", "_")
    if key not in REGION_BBOX:
        raise ValueError(
            f"Unknown region '{region_name}'. "
            f"Valid options: {list(REGION_BBOX.keys())}"
        )
    bbox = REGION_BBOX[key]                           

    # 2. Initialise GEE (idempotent – MUST happen before ee objects)
    _initialise_gee()
    
    # Now that GEE is initialized, we can create the Geometry object
    roi = ee.Geometry.Rectangle(bbox)

    # 3. Build date window (±1 day for better coverage)
    date_obj   = ee.Date(target_date)
    start_date = date_obj.advance(-5, "day")
    end_date   = date_obj.advance(5, "day")

    logger.info(
        "Querying S1-GRD | region=%s | date=%s | bbox=%s",
        region_name, target_date, bbox,
    )
    # 4. Filter Sentinel-1 GRD collection
    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        # IW (Interferometric Wide) is the standard ocean / land mode
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        # VV polarisation – sensitive to ocean surface roughness
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", POLARISATION))
        .select(POLARISATION)
    )

    size: int = collection.size().getInfo()
    if size == 0:
        raise RuntimeError(
            f"No Sentinel-1 IW/VV image found for region='{region_name}' "
            f"around date='{target_date}'.  Try a different date."
        )

    logger.info("Found %d S1 scene(s). Using the most recent one.", size)

    # Use the most recent scene in the window
    image: ee.Image = collection.sort("system:time_start", False).first()

    # 5. Download the image patch as a GeoTIFF
    sar_array = _download_image_as_array(image, roi)

    logger.info("SAR patch downloaded successfully. Shape: %s", sar_array.shape)
    return sar_array


# Private helpers

def _download_image_as_array(image: ee.Image, roi: ee.Geometry) -> np.ndarray:
    """
    Download a GEE image within ``roi`` as a raw NumPy array.

    Uses ``ee.Image.getDownloadURL`` (synchronous small-patch download) which
    returns a ZIP archive containing a GeoTIFF.

    Returns
    -------
    np.ndarray
        Shape ``(1, PATCH_SIZE, PATCH_SIZE)``, dtype ``float32``.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        _use_rasterio = True
    except ImportError:
        _use_rasterio = False
        logger.warning(
            "rasterio not installed – falling back to simulated SAR array."
        )

    if not _use_rasterio:
        return _simulate_sar_array()

    # Build the download URL for a small patch
    url: str = image.getDownloadURL(
        {
            "bands": [POLARISATION],
            "region": roi,
            "dimensions": f"{PATCH_SIZE}x{PATCH_SIZE}",
            "format": "GEO_TIFF",
        }
    )

    logger.debug("Downloading SAR patch from GEE URL: %s", url[:80] + "…")

    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"GEE download failed (HTTP {response.status_code}): "
            f"{response.text[:200]}"
        )

    # GEE wraps GeoTIFFs in a ZIP when multiple bands are requested;
    # for a single band it may return the GeoTIFF directly.
    raw_bytes = response.content

    if raw_bytes[:2] == b"PK":           # ZIP magic bytes
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            tif_name = [n for n in zf.namelist() if n.endswith(".tif")][0]
            raw_bytes = zf.read(tif_name)

    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        arr: np.ndarray = src.read(1).astype(np.float32)  # (H, W)

    arr = _preprocess(arr)                                 # normalise
    arr = arr[np.newaxis, ...]                             # → (1, H, W)
    return arr


def _preprocess(arr: np.ndarray) -> np.ndarray:
    """
    Normalise raw Sentinel-1 σ₀ dB values to the ``[0, 1]`` range.

    Typical ocean σ₀ VV ranges from roughly -25 dB (calm) to -5 dB (rough).
    Values outside this range are clipped before scaling.
    """
    DB_MIN, DB_MAX = -30.0, 0.0
    arr = np.clip(arr, DB_MIN, DB_MAX)
    arr = (arr - DB_MIN) / (DB_MAX - DB_MIN)
    return arr.astype(np.float32)


def _simulate_sar_array() -> np.ndarray:
    """
    Return a deterministic synthetic SAR patch for offline testing /
    when rasterio is unavailable.

    Shape: ``(1, PATCH_SIZE, PATCH_SIZE)``, dtype ``float32``.
    """
    logger.warning("Using SIMULATED SAR array (no real GEE data).")
    rng = np.random.default_rng(seed=42)
    # Simulate ocean-like speckle noise centred around -15 dB normalised
    arr = rng.normal(loc=0.5, scale=0.1, size=(PATCH_SIZE, PATCH_SIZE))
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr[np.newaxis, ...]           # → (1, 256, 256)
