import pytest
import os
import io
import numpy as np
import rasterio
from rasterio.transform import from_origin
from unittest.mock import patch, MagicMock

# ==============================================================================
# 1. MOCKS GLOBALES 
# ==============================================================================
import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['ee'] = MagicMock()

from fastapi.testclient import TestClient
from api_backend import app, TEMP_DIR

client = TestClient(app)

# ==============================================================================
# 2. FUNCIONES AUXILIARES PARA TESTING
# ==============================================================================
def create_dummy_tif(channels=7, width=128, height=128):
    """Crea un archivo GeoTIFF valido en memoria para enviar por el endpoint"""
    transform = from_origin(0, 0, 10, 10)  # Resolución 10x10
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': channels,
        'dtype': rasterio.uint16,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    dummy_data = np.random.randint(0, 10000, (channels, height, width), dtype=np.uint16)
    
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(dummy_data)
        
        tif_bytes = memfile.read()
    
    return io.BytesIO(tif_bytes)

# ==============================================================================
# 3. PRUEBAS DE INTEGRACIÓN Y UNITARIAS
# ==============================================================================

@patch('api_backend.model')
def test_predict_area_success(mock_model):
    """
    Prueba de Integracion: Endpoint /predict_area/
    Simula la subida de una imagen Sentinel-2 de 7 bandas y verifica la respuesta.
    """
    mock_model.predict.return_value = np.random.rand(4, 64, 64, 7).astype(np.float32)
    
    tif_file = create_dummy_tif(channels=7, width=128, height=128)
    
    response = client.post(
        "/predict_area/",
        files={"file": ("test_image.tif", tif_file, "image/tiff")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "analisis_hectareas" in data
    assert "processed_file_url" in data
    
    clases_esperadas = ['Bosque', 'Matorrales', 'Pastizales', 'T_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
    for clase in clases_esperadas:
        assert clase in data["analisis_hectareas"]
        
    url = data["processed_file_url"]
    filename = url.split("/")[-1]
    assert os.path.exists(os.path.join(TEMP_DIR, filename))

def test_predict_area_invalid_channels():
    """
    Prueba Unitaria: Validar que el endpoint rechaza imagenes con canales no soportados.
    """
    tif_file = create_dummy_tif(channels=5) # 5 canales no soportado para EJEM
    
    response = client.post(
        "/predict_area/",
        files={"file": ("test_image.tif", tif_file, "image/tiff")}
    )
    
    assert response.status_code == 200 
    assert response.json() == {"status": "error", "message": "Canales no soportados: 5"}

def test_download_file():
    """
    Prueba de Integración: Endpoint /download/{filename}
    """
    test_filename = "test_mask.tif"
    test_filepath = os.path.join(TEMP_DIR, test_filename)
    with open(test_filepath, "w") as f:
        f.write("datos de prueba")
        
    response = client.get(f"/download/{test_filename}")
    
    assert response.status_code == 200
    assert response.content == b"datos de prueba"
    
    os.remove(test_filepath)

@patch('api_backend.ee')
def test_search_recent_image(mock_ee):
    """
    Prueba Unitaria: Endpoint /search_recent_image/
    Simula la respuesta de Google Earth Engine sin conectarse a internet.
    """
    mock_collection = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_collection.filterBounds.return_value = mock_collection
    mock_collection.filterDate.return_value = mock_collection
    
    mock_ideal = MagicMock()
    mock_ideal.getInfo.return_value = {
        'id': 'COPERNICUS/S2_SR/IDEAL_ID',
        'properties': {
            'system:time_start': 1672531200000, 
            'CLOUDY_PIXEL_PERCENTAGE': 5.5
        }
    }
    mock_collection.filter.return_value.sort.return_value.first.return_value = mock_ideal
    
    mock_collection.sort.return_value.limit.return_value.getInfo.return_value = {
        'features': [
            {
                'id': 'COPERNICUS/S2_SR/RECENT_ID_1',
                'properties': {
                    'system:time_start': 1672617600000,
                    'CLOUDY_PIXEL_PERCENTAGE': 15.2
                }
            }
        ]
    }

    payload = {
        "coords": [[[-66.1, -17.3], [-66.0, -17.3], [-66.0, -17.4], [-66.1, -17.4], [-66.1, -17.3]]]
    }
    
    response = client.post("/search_recent_image/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["options"]) == 2 
    assert data["options"][0]["is_ideal"] is True
    assert data["options"][1]["is_ideal"] is False

@patch('api_backend.ee')
def test_confirm_export(mock_ee):
    """
    Prueba Unitaria: Endpoint /confirm_export/
    Verifica que se inicie la tarea de exportación a Drive correctamente.
    """
    mock_image = MagicMock()
    mock_ee.Image.return_value = mock_image
    mock_image.getInfo.return_value = {
        'properties': {'system:time_start': 1704067200000} # Año 2024
    }
    
    mock_task = MagicMock()
    mock_ee.batch.Export.image.toDrive.return_value = mock_task
    
    payload = {
        "coords": [[[-66.1, -17.3], [-66.0, -17.3], [-66.0, -17.4], [-66.1, -17.4], [-66.1, -17.3]]],
        "image_id": "COPERNICUS/S2_SR/TEST_ID"
    }
    
    response = client.post("/confirm_export/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "monitoringUrl" in data
    
    mock_task.start.assert_called_once()