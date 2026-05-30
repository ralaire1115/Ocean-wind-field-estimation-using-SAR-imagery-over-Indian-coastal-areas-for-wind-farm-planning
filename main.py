import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

import logging
from datetime import date, datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gee_engine import fetch_sar_image, REGION_BBOX
from wind_model import calculate_wind_vectors, WindVector

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="SAR Ocean Wind Field Estimation API",
    description=(
        "Estimates ocean surface wind speed (m/s) and direction (°) over "
        "coastal Tamil Nadu and Gujarat using Sentinel-1 SAR imagery "
        "retrieved from Google Earth Engine and a ResNet-based deep-learning "
        "regression model."
    ),
    version="1.0.0",
    contact={
        "name": "Ocean Remote Sensing Team",
        "email": "sar-wind@example.in",
    },
    license_info={"name": "MIT"},
)


# Response schema

class WindFieldResponse(BaseModel):
    """Structured JSON response returned by the wind-field endpoint."""

    latitude:         float  # Exact coordinate queried
    longitude:        float
    date:             str    # Date queried  (YYYY-MM-DD)
    exact_timestamp:  str    # Exact ISO-8601 time of the satellite pass
    wind_speed_ms:    float  # Estimated wind speed  (m/s)
    wind_dir_deg:     float  # Estimated meteorological wind direction (0-360°)
    model_version:    str    # Model identifier for traceability
    sar_patch_pixels: int    # Side-length of the SAR patch used (pixels)
    note:             str    # Human-readable disclaimer


class ErrorResponse(BaseModel):
    detail: str


# Helper: validate date string

def _parse_date(date_str: str) -> str:
    """
    Validate that ``date_str`` is a parseable ISO-8601 date (YYYY-MM-DD)
    and falls within the operational lifespan of the Sentinel-1 mission.

    Returns the sanitised string, or raises ``HTTPException 400``.
    """
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid date format: '{date_str}'. "
                "Expected YYYY-MM-DD (e.g. 2024-01-15)."
            ),
        )

    # Sentinel-1 operational data availability cutoff
    min_date = date(2014, 10, 1)
    
    if parsed < min_date:
        raise HTTPException(
            status_code=400, 
            detail="Date is too early. Sentinel-1 SAR data is only available from 2014-10-01 onwards."
        )

    if parsed > date.today():
        raise HTTPException(
            status_code=400,
            detail=f"Date '{date_str}' is in the future. SAR data is not yet available.",
        )

    return date_str


# Validate region

def _validate_region(region: str) -> str:
    """
    Normalise and validate the ``region`` query parameter.

    Returns the normalised key used in ``REGION_BBOX``, or raises
    ``HTTPException 400``.
    """
    key = region.strip().lower().replace(" ", "_")
    if key not in REGION_BBOX:
        valid = ", ".join(REGION_BBOX.keys())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown region: '{region}'. "
                f"Supported regions: {valid}."
            ),
        )
    return key


# Routes

@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    """Health-check / redirect hint."""
    return {
        "message": "SAR Wind Field API is running.",
        "docs": "/docs",
        "endpoint": "/api/v1/wind-field",
    }


@app.get(
    "/api/v1/wind-field",
    response_model=WindFieldResponse,
    responses={
        200: {"description": "Wind field estimate successfully computed."},
        400: {"model": ErrorResponse, "description": "Invalid query parameters."},
        404: {"model": ErrorResponse, "description": "No SAR data found for the given date/region."},
        502: {"model": ErrorResponse, "description": "Google Earth Engine connection failure."},
        500: {"model": ErrorResponse, "description": "Unexpected internal server error."},
    },
    summary="Get ocean wind field estimate",
    tags=["Wind Field"],
)
async def get_wind_field(
    lat: float = Query(..., description="Latitude of the target location.", examples=[22.25]),
    lon: float = Query(..., description="Longitude of the target location.", examples=[71.25]),
    date: str = Query(
        ...,
        description="Target date for SAR acquisition in YYYY-MM-DD format.",
        examples=["2024-01-15"],
    ),
) -> WindFieldResponse:
    """
    **Estimate ocean surface wind speed and direction from Sentinel-1 SAR.**

    ### Pipeline
    1. Validate ``region`` and ``date`` parameters.
    2. Query the `COPERNICUS/S1_GRD` collection on **Google Earth Engine**
       for the closest Sentinel-1 IW/VV scene around the requested date.
    3. Download a 256×256 pixel SAR patch (σ₀ dB, normalised to [0, 1]).
    4. Pass the patch through a **ResNet-18 regression model** to predict
       wind speed (m/s) and meteorological wind direction (°).
    5. Return a structured JSON response.

    ### Notes
    - The deep-learning model is a **skeleton** in this version.  Replace
      `sar_wind_weights.pth` with real trained weights for production use.
    - GEE queries use a ±5-day window to improve scene availability.
    """

    # 1. Parameter validation
    validated_date    = _parse_date(date)

    logger.info(
        "Wind-field request | lat=%.2f | lon=%.2f | date=%s",
        lat, lon, validated_date,
    )

    # 2. Fetch SAR image from Google Earth Engine
    try:
        sar_array, exact_time = fetch_sar_image(
            lat=lat,
            lon=lon,
            target_date=validated_date,
        )
    except ValueError as exc:
        # Region or parameter validation failed inside gee_engine
        logger.warning("Region validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        error_msg = str(exc)
        if "No Sentinel-1" in error_msg:
            # No scene available for the requested date/region
            logger.warning("No SAR scene found: %s", error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            # GEE connectivity / download failure
            logger.error("GEE runtime error: %s", error_msg)
            raise HTTPException(
                status_code=502,
                detail=f"Google Earth Engine error: {error_msg}",
            )

    except Exception as exc:
        # Catch-all for unexpected GEE / network issues
        logger.exception("Unexpected error during GEE fetch.")
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected GEE error: {type(exc).__name__}: {exc}",
        )

    # 3. Run deep-learning wind inference
    try:
        wind: WindVector = calculate_wind_vectors(sar_array)
    except ValueError as exc:
        logger.error("Model input validation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model input error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during model inference.")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {type(exc).__name__}: {exc}",
        )

    # 4. Build and return the response
    patch_size: int = sar_array.shape[-1]   # last dim = width = height

    response = WindFieldResponse(
        latitude=lat,
        longitude=lon,
        date=validated_date,
        exact_timestamp=exact_time,
        wind_speed_ms=wind.wind_speed_ms,
        wind_dir_deg=wind.wind_dir_deg,
        model_version="SARWindNet-ResNet18-v1.0-skeleton",
        sar_patch_pixels=patch_size,
        note=(
            "Model weights are untrained placeholders. "
            "Fine-tune on ERA5/ECMWF-labeled SAR pairs for production use."
        ),
    )

    logger.info(
        "Response | speed=%.2f m/s | dir=%.1f° | lat=%.2f | lon=%.2f | date=%s",
        wind.wind_speed_ms, wind.wind_dir_deg,
        lat, lon, validated_date,
    )

    return response


# 5. Plotting the Quiver Plot Endpoint
@app.get(
    "/api/v1/wind-map", 
    responses={200: {"content": {"image/png": {}}}},
    response_class=StreamingResponse,
    summary="Get Wind Field Map Image",
    tags=["Visualization"]
)
def get_wind_map_image(
    region: str = Query("gujarat", description="Coastal region of India. Options: 'tamil_nadu', 'gujarat'.", examples=["gujarat"]),
    date: str = Query("2024-01-15", description="Target date (YYYY-MM-DD). Must be between 2014-10-01 and today.", examples=["2024-01-15"])
):
    """
    Generates a Quiver plot of the wind field and returns it as a PNG image.
    """
    # Validate inputs before processing the map
    normalised_region = _validate_region(region)
    validated_date = _parse_date(date)

    # 1. Generate the grid data (Simulated for visualization testing)
    lons = np.linspace(68.0, 74.0, 30)
    lats = np.linspace(20.0, 24.0, 20)
    X, Y = np.meshgrid(lons, lats)

    wind_speed = np.random.uniform(5.0, 12.0, X.shape)
    wind_dir = np.random.uniform(200, 260, X.shape)

    dir_rad = np.radians(wind_dir)
    U = -wind_speed * np.sin(dir_rad)
    V = -wind_speed * np.cos(dir_rad)

    # NORMALIZE VECTORS for arrows of uniform length
    magnitude = np.sqrt(U**2 + V**2) + 1e-10
    U_dir = U / magnitude
    V_dir = V / magnitude

    # 2. Initialize the Plot
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add landmass features
        ax.add_feature(cfeature.LAND, facecolor='#d3d3d3', zorder=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
        
        # EXPLICITLY FORCE TICKS FOR CARTOPY
        ax.set_xticks(np.arange(68.0, 75.0, 1.0), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(20.0, 25.0, 1.0), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        
    except ImportError:
        # Fallback if Cartopy isn't available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#d3d3d3') 
        
        # EXPLICITLY FORCE TICKS FOR MATPLOTLIB
        ax.set_xticks(np.arange(68.0, 75.0, 1.0))
        ax.set_yticks(np.arange(20.0, 25.0, 1.0))
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}°E"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}°N"))

    # 3. Background Gradient (Wind Speed)
    cf = ax.contourf(X, Y, wind_speed, levels=25, cmap='jet', alpha=0.85, zorder=0)
    
    # 4. Arrows (Direction Only)
    ax.quiver(X, Y, U_dir, V_dir, color='black', scale=35, headwidth=3, headlength=4, zorder=3)
    
    # 5. Axis Formatting
    ax.set_title(f"Ocean Wind Field - {normalised_region.upper().replace('_', ' ')} ({validated_date})", pad=15, fontsize=14)
    ax.set_xlabel("Longitude", fontweight='bold')
    ax.set_ylabel("Latitude", fontweight='bold')

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    # 6. Colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Wind Speed (m/s)', fontweight='bold')

    plt.tight_layout()

    # 7. Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig) 
    
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
 

# Application entry point

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
