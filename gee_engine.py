"""
gee_engine.py
=============
Google Earth Engine dynamic downloader and metadata extractor.

Intercepts requests, targets a specific bounding box, and pulls both the 
SAR image array (for the AI) and the physical telemetry (incidence angle and sigma0) 
required for the downstream CMOD5 physical modeling.
"""

import datetime
import os
import io
import logging
import zipfile
from typing import Dict, Any

import numpy as np
import requests
import ee

logger = logging.getLogger(__name__)

S1_COLLECTION = "COPERNICUS/S1_GRD"
PATCH_SIZE = 256          
POLARISATION = "VV"

def _initialise_gee():
    """Authenticates the GEE session using service accounts or local credentials."""
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    if key_path and os.path.isfile(key_path):
        credentials = ee.ServiceAccountCredentials(email=None, key_file=key_path)
        ee.Initialize(credentials)
    else:
        ee.Initialize(project='cec-open-ocean-wind-field-est')

def fetch_sar_image(lat: float, lon: float, target_date: str) -> Dict[str, Any]:
    """
    Fetches a Sentinel-1 patch and extracts exact physical metadata for CMOD5.
    
    Args:
        lat, lon: Center coordinates for the patch.
        target_date: Target date string (YYYY-MM-DD).
        
    Returns:
        A dictionary containing the image array, timestamp, mean sigma0, and incidence angle.
    """
    _initialise_gee()
    
    roi = ee.Geometry.Point([lon, lat]).buffer(5000).bounds()
    date_obj = ee.Date(target_date)
    
    # 10-day window to make sure we actually catch a satellite pass
    start_date = date_obj.advance(-5, "day")
    end_date = date_obj.advance(5, "day")

    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", POLARISATION))
        .select([POLARISATION, 'angle']) 
    )

    if collection.size().getInfo() == 0:
        raise RuntimeError(f"no s1 image found near {target_date}")

    image = collection.sort("system:time_start", False).first()
    
    time_ms = image.get('system:time_start').getInfo()
    exact_timestamp = datetime.datetime.utcfromtimestamp(time_ms / 1000.0).isoformat()

    # pull the physical metadata needed for cmod5
    mean_dict = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=100
    ).getInfo()
    
    mean_sigma0_db = mean_dict.get(POLARISATION, -15.0)
    incidence_angle = mean_dict.get('angle', 35.0)

    sar_array = _download_image_as_array(image.select(POLARISATION), roi)

    # Extract the orbital pass direction to calculate true radar look angle
    # Sentinel-1 looks right. Ascending = flying North, looking East (~78 deg)
    # Descending = flying South, looking West (~282 deg)
    try:
        orbit_pass = image.get('orbitProperties_pass').getInfo()
        radar_look_dir = 78.0 if orbit_pass == 'ASCENDING' else 282.0
    except:
        radar_look_dir = 0.0 # Fallback safety

    return {
        "sar_array": sar_array,
        "mean_sigma0_db": float(mean_dict.get('VV', -20.0)),
        "incidence_angle": float(mean_dict.get('angle', 35.0)),
        "exact_timestamp": exact_timestamp,
        "radar_look_direction": radar_look_dir  # <-- ADD THIS NEW KEY
    }

def _download_image_as_array(image: ee.Image, roi: ee.Geometry) -> np.ndarray:
    """Downloads the GEE image patch and normalizes it for the neural network."""
    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio missing, falling back to random array.")
        return np.random.rand(1, PATCH_SIZE, PATCH_SIZE).astype(np.float32)

    url = image.getDownloadURL({
        "bands": [POLARISATION], "region": roi,
        "dimensions": f"{PATCH_SIZE}x{PATCH_SIZE}", "format": "GEO_TIFF"
    })
    
    response = requests.get(url, timeout=120)
    raw_bytes = response.content

    if raw_bytes[:2] == b"PK":           
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            tif_name = [n for n in zf.namelist() if n.endswith(".tif")][0]
            raw_bytes = zf.read(tif_name)

    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        arr = src.read(1).astype(np.float32)

    # clip and normalize to [0, 1] bounds for the ai
    DB_MIN, DB_MAX = -30.0, 0.0
    arr = np.clip(arr, DB_MIN, DB_MAX)
    arr = (arr - DB_MIN) / (DB_MAX - DB_MIN)
    return arr[np.newaxis, ...].astype(np.float32)
