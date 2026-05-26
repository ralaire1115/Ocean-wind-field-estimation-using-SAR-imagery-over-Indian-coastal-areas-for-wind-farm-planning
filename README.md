# SAR Ocean Wind Field Estimation API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Earth_Engine-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)

> A deep-learning-powered REST API that estimates ocean surface wind speed (m/s) and meteorological wind direction (°) over the coastal regions of India (Tamil Nadu and Gujarat).

This application automatically queries **Google Earth Engine (GEE)** for the closest Sentinel-1 Synthetic Aperture Radar (SAR) imagery, downloads the relevant patches, and processes them through a customized **ResNet-18 regression model**.

---

## ✨ Key Features

* **🛰️ Automated Data Pipeline:** Direct integration with the `COPERNICUS/S1_GRD` database to fetch Sentinel-1 IW/VV backscatter data dynamically.
* **🧠 Deep Learning Inference:** A modified PyTorch ResNet-18 backbone optimized for 1-channel SAR input, predicting continuous wind velocity vectors.
* **📊 Dual Output Modes:** * `JSON`: Clean, structured data for programmatic integration.
  * `PNG Map`: Real-time Matplotlib/Cartopy generated Quiver plots of the wind field.
* **🛡️ Robust Error Handling:** Strict date validations (post-2014) and explicit HTTP status codes for GEE connection issues and missing satellite passes.

---

## 📂 Project Structure

| File Name | Role | Description |
| :--- | :---: | :--- |
| **`requirements.txt`** | 📦 | Lists all third-party libraries needed to manage data and web routing. |
| **`gee_engine.py`** | 📡 | Authenticates with Google Cloud, queries satellite passes, and extracts imagery arrays. |
| **`wind_model.py`** | 🤖 | Houses the single-channel ResNet-18 neural network that estimates velocity vectors. |
| **`main.py`** | 🚀 | Runs the FastAPI app, manages routing, validates bounds, and generates the quiver maps. |

---

## 🛠️ Local Installation (Windows)

Follow these steps to set up the environment and bypass common Windows C++ compiler conflicts.

**1. Create & Activate Virtual Environment**
```powershell
python -m venv venv
venv\Scripts\activate

*(If scripts are disabled, switch to `cmd` and run `venv\Scripts\activate.bat`)*

```

**2. Install Core Dependencies**

```powershell
pip install -r requirements.txt

```

**3. Install PyTorch (Windows CPU)**
*Crucial: Install pre-compiled CPU binaries directly to prevent version conflicts.*

```powershell
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

```

**4. Install Cartopy (Map Visualizations)**

```powershell
pip install cartopy

```

**5. Authenticate Google Earth Engine**

```powershell
earthengine authenticate

```

*Note: The email you log in with must have the `Earth Engine Resource Viewer` role assigned within your Google Cloud Project.(use G-Suite ID)*

**6. Ignite the Server**

```powershell
python main.py

```

📍 **Live Dashboard:** Once running, access the interactive Swagger UI at `http://localhost:8000/docs`

---

## 🧭 API Endpoints

### 1. Get Wind Field Data (JSON)

Retrieves the numerical wind speed and direction estimates based on the AI inference.

* **URL:** `GET /api/v1/wind-field`
* **Parameters:**
* `region` *(string)*: `'tamil_nadu'` or `'gujarat'`
* `date` *(string)*: `YYYY-MM-DD` *(Must be between 2014-10-01 and today)*



**Response Example:**

```json
{
  "region": "tamil_nadu",
  "date": "2024-01-15",
  "wind_speed_ms": 12.435,
  "wind_dir_deg": 210.5,
  "model_version": "SARWindNet-ResNet18-v1.0-skeleton",
  "sar_patch_pixels": 256,
  "note": "Untrained placeholders. Fine-tune on ERA5/ECMWF for production."
}

```

### 2. Get Wind Field Map (Visualization)

Generates a Quiver plot showing wind direction arrows layered over a wind-speed gradient background and physical land boundaries.

* **URL:** `GET /api/v1/wind-map`
* **Parameters:** Same as above.
* **Returns:** `image/png`

---

## 🛑 HTTP Status Codes

The API uses standard HTTP response codes to indicate the success or failure of an API request.

| Code | Status | Description |
| --- | --- | --- |
| **`200`** | ✅ `OK` | Request succeeded; data or map generated successfully. |
| **`400`** | ⚠️ `Bad Request` | Unsupported region requested, or date is outside the 2014-present window. |
| **`404`** | 🚫 `Not Found` | GEE confirmed no Sentinel-1 image exists for that region within the $\pm$ 5-day sweep. |
| **`422`** | 🛑 `Validation Error` | Missing parameter or incorrect data type (e.g., malformed date string). |
| **`500`** | 💥 `Internal Error` | The PyTorch model crashed or received a malformed input array from GEE. |
| **`502`** | ⚡ `Bad Gateway` | Google Earth Engine servers timed out or refused the connection. |

---
