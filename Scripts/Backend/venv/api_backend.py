import os
import numpy as np
import rasterio
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from patchify import patchify
from datetime import datetime
from fastapi import Body
import io
import uuid
import ee

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
FOLDER_NAME = 'Tesis_PNT_Sentinel'
# --- CONFIGURACIÓN DE RUTAS ---
BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"

# Modelo IA
MODEL_PATH = os.path.join(BASE_PATH, 'modelo_unet_final_tesis.keras')
model = tf.keras.models.load_model(MODEL_PATH) 

TEMP_DIR = os.path.join(BASE_PATH, "temp_outputs")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

PROJECT_ID = 'aifinal-480001' 

def initialize_gee():
    try:
        ee.Initialize(project=PROJECT_ID) 
        print("Earth Engine inicializado correctamente")
    except Exception as e:
        print(f"Error de inicialización: {e}")
        print("Intentando autenticación manual...")
      

def get_best_image_mosaic(year, roi):
    BANDS_TO_SELECT = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate(f'{year}-05-01', f'{year}-09-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') 
    
    image = collection.first()
    if not image:
        return None
    return image.select(BANDS_TO_SELECT).clip(roi).toFloat() 

def export_to_drive_task(image, year, roi):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'S2_PNT_{year}_Export',
        folder=FOLDER_NAME,
        fileNamePrefix=f'S2_PNT_{year}', 
        region=roi.bounds().getInfo()['coordinates'], 
        scale=10,
        fileFormat='GeoTIFF',
        maxPixels=1e9 
    )
    task.start()
    return task


initialize_gee()

@app.post("/predict_area/")
async def predict_area(file: UploadFile = File(...)):
    contents = await file.read()
    
    with rasterio.open(io.BytesIO(contents)) as src:
        profile = src.profile
        img_raw = src.read().transpose(1, 2, 0)
        res_x, res_y = src.res
        
    h, w, c = img_raw.shape
    
    valid_mask = np.max(img_raw, axis=-1) > 0 

    if c == 3:
        padding_channels = np.zeros((h, w, 4), dtype=img_raw.dtype)
        img_raw = np.concatenate([img_raw, padding_channels], axis=-1)
        img = np.nan_to_num(img_raw).astype(np.float32) * (10000.0 / 255.0)
    elif c == 7:
        img = np.nan_to_num(img_raw).astype(np.float32)
    else:
        return {"status": "error", "message": f"Canales no soportados: {c}"}

    img_normalized = img / 10000.0

    h_pad = (64 - h % 64) % 64
    w_pad = (64 - w % 64) % 64
    img_padded = np.pad(img_normalized, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
    
    patches = patchify(img_padded, (64, 64, 7), step=32) 
    output_probs = np.zeros((img_padded.shape[0], img_padded.shape[1], 7), dtype=np.float32)
    counts = np.zeros((img_padded.shape[0], img_padded.shape[1], 1), dtype=np.float32)

    for i in range(patches.shape[0]):
        preds = model.predict(patches[i, :, 0], verbose=0)
        for j in range(patches.shape[1]):
            y, x = i * 32, j * 32
            output_probs[y:y+64, x:x+64, :] += preds[j]
            counts[y:y+64, x:x+64] += 1.0

    final_map = np.argmax(output_probs / np.maximum(counts, 1.0), axis=-1).astype(np.uint8)
    
    final_map = final_map[:h, :w]
    
    final_map[~valid_mask] = 99 

    
    area_px = abs(res_x * res_y)
    classes = ['Bosque', 'Matorrales', 'Pastizales', 'T_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
    
    
    results = {
        name: round(float((np.sum((final_map == i) & valid_mask).item() * area_px) / 10000.0), 2) 
        for i, name in enumerate(classes)
    }
    
    area_px = abs(res_x * res_y)
    classes = ['Bosque', 'Matorrales', 'Pastizales', 'T_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
    
    
    results = {
        name: round(float((np.sum((final_map == i) & valid_mask).item() * area_px) / 10000.0), 2) 
        for i, name in enumerate(classes)
    }

    result_id = f"mask_{uuid.uuid4().hex}.tif"
    result_path = os.path.join(TEMP_DIR, result_id)
    new_profile = profile.copy()

    new_profile.update(count=1, dtype='uint8', nodata=99)
    
    with rasterio.open(result_path, 'w', **new_profile) as dst:
        dst.write(final_map, 1)

    return {
        "analisis_hectareas": results, 
        "processed_file_url": f"http://127.0.0.1:8000/download/{result_id}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(os.path.join(TEMP_DIR, filename))

@app.post("/search_recent_image/")
async def search_recent_image(data: dict = Body(...)):
    try:
        coords = data.get("coords")
        roi = ee.Geometry.Polygon(coords)
        
        # Fecha actual del sistema
        now = datetime.now().strftime('%Y-%m-%d')
        
        # Buscar en Sentinel-2 los últimos 3 meses desde hoy para asegurar encontrar algo sin nubes
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate('2024-10-01', now) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .sort('system:time_start', False) # El más reciente primero
        
        image = collection.first()
        
        if not image:
            return {"status": "error", "message": "No se encontró imagen reciente sin nubes."}
        
        # Obtener metadatos para mostrar al usuario
        info = image.getInfo()
        date_str = datetime.fromtimestamp(info['properties']['system:time_start']/1000).strftime('%Y-%m-%d %H:%M')
        clouds = info['properties']['CLOUDY_PIXEL_PERCENTAGE']
        
        return {
            "status": "success",
            "date": date_str,
            "clouds": f"{clouds:.2f}%",
            "id": info['id']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/confirm_export/")
async def confirm_export(data: dict = Body(...)):
    try:
        coords = data.get("coords")
        roi = ee.Geometry.Polygon(coords)
        
        now = datetime.now().strftime('%Y-%m-%d')
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate('2024-01-01', now) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)) \
            .sort('system:time_start', False)
        
        recent_img = collection.first()
        
        if not recent_img:
            return {"status": "error", "message": "No se encontró una imagen válida para exportar."}

        info = recent_img.getInfo()
        timestamp = info['properties']['system:time_start']
        anio_detectado = datetime.fromtimestamp(timestamp/1000).year
        
        BANDS_TO_SELECT = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        final_img = recent_img.select(BANDS_TO_SELECT).clip(roi).toFloat()

        task = ee.batch.Export.image.toDrive(
            image=final_img,
            description=f'S2_PNT_Reciente_{anio_detectado}_Export',
            folder=FOLDER_NAME,
            fileNamePrefix=f'S2_PNT_Reciente_{anio_detectado}', 
            region=roi.bounds().getInfo()['coordinates'], 
            scale=10,
            fileFormat='GeoTIFF',
            maxPixels=1e9 
        )
        task.start()

        monitoring_url = "https://code.earthengine.google.com/tasks"
        
        return {
            "status": "success", 
            "message": f"Exportación iniciada para el año {anio_detectado}",
            "monitoringUrl": monitoring_url
        }
    except Exception as e:
        print(f"Error en confirm_export: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)