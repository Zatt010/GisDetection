# tests/conftest.py
import sys
import io
import os
import uuid
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ── Inject fake modules BEFORE any import of main.py ──────────────────────────
# This runs at collection time, so main.py never sees the real tensorflow/ee/xgb

def _make_mock_tf():
    tf = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros((1, 64, 64, 7), dtype=np.float32)
    tf.keras.models.load_model.return_value = mock_model
    return tf

# Stub every heavy module so Python never tries to import the real ones
sys.modules.setdefault("tensorflow",                    _make_mock_tf())
sys.modules.setdefault("tensorflow.keras",              MagicMock())
sys.modules.setdefault("tensorflow.keras.models",       MagicMock())
sys.modules.setdefault("tensorflow.keras.optimizers",   MagicMock())
sys.modules.setdefault("ee",                            MagicMock())
sys.modules.setdefault("xgboost",                       MagicMock())
sys.modules.setdefault("patchify",                      MagicMock())
sys.modules.setdefault("georaster",                     MagicMock())

# Now it's safe to import rasterio and the app
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from fastapi.testclient import TestClient

# ── Shared TIF factories ───────────────────────────────────────────────────────

def make_tif_bytes(
    bands=7, width=128, height=128,
    dtype="float32", pixel_value=5000.0, nodata=None,
) -> bytes:
    buf = io.BytesIO()
    transform = from_bounds(-66.35, -17.50, -65.90, -17.20, width, height)
    data = np.full((bands, height, width), pixel_value, dtype=dtype)
    profile = dict(
        driver="GTiff", dtype=dtype,
        width=width, height=height, count=bands,
        crs=CRS.from_epsg(32719), transform=transform,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.read()


def make_label_tif_bytes(width=128, height=128) -> bytes:
    buf = io.BytesIO()
    transform = from_bounds(-66.35, -17.50, -65.90, -17.20, width, height)
    class_values = [10, 20, 30, 40, 50, 60, 80]
    data = np.zeros((1, height, width), dtype=np.uint8)
    for i, val in enumerate(class_values):
        col_start = i * (width // len(class_values))
        col_end   = col_start + (width // len(class_values))
        data[0, :, col_start:col_end] = val
    with rasterio.open(buf, "w",
        driver="GTiff", dtype="uint8",
        width=width, height=height, count=1,
        crs=CRS.from_epsg(32719), transform=transform,
    ) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.read()


# ── Fixtures ───────────────────────────────────────────────────────────────────

# Mock httpx at session level to prevent real external calls
@pytest.fixture(scope="session", autouse=True)
def mock_httpx():
    import httpx
    import uuid
    
    def mock_post(url, **kwargs):
        # AI service responses
        if "/predict_area/" in str(url):
            return httpx.Response(200, json={"status": "success", "analisis_hectareas": {
                "Bosque": 10.5, "Matorrales": 8.2, "Pastizales": 15.1, "T_Agricolas": 12.3,
                "Infraestructura": 5.7, "Suelo_Desnudo": 3.9, "Agua": 2.4
            }, "processed_file_url": "http://test.com/output.tif"})
        
        # Tiling service responses
        elif "/upload_orthomosaic/" in str(url):
            job_id = uuid.uuid4().hex
            return httpx.Response(200, json={"status": "processing", "job_id": job_id})
        
        # Process orthomosaic responses
        elif "/process_orthomosaic/" in str(url):
            return httpx.Response(200, json={
                "status": "success", 
                "bounds": {"left": -66.35, "bottom": -17.50, "right": -65.90, "top": -17.20},
                "crs": "EPSG:32719",
                "processed_file_url": "http://test.com/processed_orthomosaic.tif"
            })
        
        # Export service responses
        elif "/export_vector/" in str(url):
            return httpx.Response(200, json={"status": "success", "download_url": "http://test.com/export.geojson"})
        
        # GEE service responses
        elif "/search_recent_image/" in str(url) or "/confirm_export/" in str(url):
            return httpx.Response(200, json={"status": "success"})
        
        # Default response
        return httpx.Response(200, json={"status": "success"})
    
    def mock_get(url, **kwargs):
        # Tiling status responses
        if "/tiling_status/" in str(url):
            import main
            # Extract job_id from URL
            job_id = str(url).split("/")[-1]
            if job_id in main.job_status:
                return httpx.Response(200, json=main.job_status[job_id])
            else:
                return httpx.Response(200, json={"status": "not_found"})
        
        # Download file responses - return 404 for non-existent files
        elif "/download/" in str(url):
            filename = str(url).split("/")[-1]
            if "does_not_exist" in filename or "nonexistent" in filename:
                return httpx.Response(404, content=b"File not found")
            else:
                # Return mock file content for existing files
                return httpx.Response(200, content=b"mock file content", headers={"content-type": "image/tiff"})
        
        # Default response
        return httpx.Response(200, json={"status": "success"})
    
    with patch('httpx.AsyncClient.post', side_effect=mock_post), \
         patch('httpx.AsyncClient.get', side_effect=mock_get):
        yield

@pytest.fixture(scope="session")
def client():
    from main import app
    return TestClient(app)

@pytest.fixture
def valid_sentinel_tif():
    return make_tif_bytes(bands=7, pixel_value=5000.0)

@pytest.fixture
def valid_sentinel_tif_13ch():
    return make_tif_bytes(bands=13, pixel_value=5000.0)

@pytest.fixture
def valid_label_tif():
    return make_label_tif_bytes()

@pytest.fixture
def small_rgb_tif():
    return make_tif_bytes(bands=3, width=64, height=64,
                          dtype="uint8", pixel_value=128.0)

@pytest.fixture
def large_orthomosaic_tif():
    return make_tif_bytes(bands=3, width=512, height=512,
                          dtype="uint8", pixel_value=100.0)

@pytest.fixture
def mock_model():
    m = MagicMock()
    probs = np.zeros((1, 64, 64, 7), dtype=np.float32)
    probs[..., 0] = 1.0
    m.predict.return_value = probs
    return m

@pytest.fixture
def temp_prediction_file(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    fname = f"mask_{uuid.uuid4().hex}.tif"
    fpath = tmp_path / fname
    data  = np.zeros((1, 64, 64), dtype=np.uint8)
    data[0, :32, :32] = 0
    data[0, :32, 32:] = 1
    data[0, 32:, :32] = 4
    data[0, 32:, 32:] = 5
    transform = from_bounds(-66.35, -17.50, -65.90, -17.20, 64, 64)
    with rasterio.open(str(fpath), "w",
        driver="GTiff", dtype="uint8",
        width=64, height=64, count=1,
        crs=CRS.from_epsg(32719),
        transform=transform, nodata=99,
    ) as dst:
        dst.write(data)
    return str(fpath), fname