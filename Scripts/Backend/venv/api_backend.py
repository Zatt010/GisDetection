from fastapi.staticfiles import StaticFiles
import os
import numpy as np
import rasterio
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from patchify import patchify
from datetime import datetime, timedelta
from fastapi import Body
from fastapi import BackgroundTasks
import json
import io
import uuid
import ee
import xgboost as xgb
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import subprocess
from rasterio.warp import transform_bounds
import sys
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import math
from PIL import Image
import io

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
BASE_PATH = r"C:\Users\afuhe\Desktop\materias\PG\Scripts\IA"

# Modelo IA
MODEL_PATH = os.path.join(BASE_PATH, 'modelo_unet_pro_final.keras')
model = tf.keras.models.load_model(MODEL_PATH) 

TEMP_DIR = os.path.join(BASE_PATH, "temp_outputs")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

PROJECT_ID = 'aifinal-480001' 



CLASES_ENTRENADAS = [10, 50, 60]

app.mount("/tiles", StaticFiles(directory="tiles_dron"), name="tiles")
TILES_DIR = os.path.join(BASE_PATH, "tiles_outputs")
os.makedirs(TILES_DIR, exist_ok=True)
app.mount("/tiles_outputs", StaticFiles(directory=TILES_DIR), name="tiles_outputs")
job_status = {}

def initialize_gee():
    try:
        ee.Initialize(project=PROJECT_ID) 
        print("Earth Engine inicializado correctamente")
    except Exception as e:
        print(f"Error de inicialización: {e}")
        print("Intentando autenticación manual...")
      
def tile_to_bbox(x, y, z):
    """Convert XYZ tile to lat/lng bounds"""
    n = 2 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max

def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 
             1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def run_tiling_job(job_id: str, tif_path: str, file_id: str, bounds_wgs84: tuple):
    try:
        job_status[job_id] = {"status": "reprojecting", "progress": 10}

        tiles_output_path = os.path.join(TILES_DIR, file_id)
        os.makedirs(tiles_output_path, exist_ok=True)

        # Reproject to EPSG:4326 in memory using rasterio
        with rasterio.open(tif_path) as src:
            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "driver": "GTiff"
            })

            reprojected_path = os.path.join(TEMP_DIR, f"ortho_{file_id}_4326.tif")
            with rasterio.open(reprojected_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.lanczos
                    )

        job_status[job_id] = {"status": "tiling", "progress": 40}

        # Generate XYZ tiles zoom 12-18
        MIN_ZOOM = 12
        MAX_ZOOM = 18

        west, south, east, north = bounds_wgs84

        with rasterio.open(reprojected_path) as src:
            band_count = src.count

            for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
                x_min, y_max = latlon_to_tile(south, west, zoom)
                x_max, y_min = latlon_to_tile(north, east, zoom)

                zoom_dir = os.path.join(tiles_output_path, str(zoom))

                for x in range(x_min, x_max + 1):
                    x_dir = os.path.join(zoom_dir, str(x))
                    os.makedirs(x_dir, exist_ok=True)

                    for y in range(y_min, y_max + 1):
                        tile_west, tile_south, tile_east, tile_north = tile_to_bbox(x, y, zoom)

                        # Read the raster data for this tile extent
                        window = rasterio.windows.from_bounds(
                            tile_west, tile_south, tile_east, tile_north,
                            transform=src.transform
                        )

                        try:
                            # Read at 256x256 resolution
                            data = src.read(
                                out_shape=(band_count, 256, 256),
                                window=window,
                                resampling=Resampling.lanczos
                            )

                            # Skip empty tiles
                            if data.max() == 0:
                                continue

                            # Convert to RGB image
                            if band_count >= 3:
                                r = data[0]
                                g = data[1]
                                b = data[2]
                            else:
                                r = g = b = data[0]

                            # Normalize if needed (16-bit to 8-bit)
                            if r.max() > 255:
                                r = (r / r.max() * 255).astype(np.uint8)
                                g = (g / g.max() * 255).astype(np.uint8)
                                b = (b / b.max() * 255).astype(np.uint8)
                            else:
                                r = r.astype(np.uint8)
                                g = g.astype(np.uint8)
                                b = b.astype(np.uint8)

                            img_array = np.stack([r, g, b], axis=-1)
                            img = Image.fromarray(img_array, mode="RGB")

                            tile_path = os.path.join(x_dir, f"{y}.png")
                            img.save(tile_path, "PNG")

                        except Exception:
                            continue  # skip tiles outside bounds

                progress = 40 + int((zoom - MIN_ZOOM + 1) / (MAX_ZOOM - MIN_ZOOM + 1) * 55)
                job_status[job_id] = {"status": "tiling", "progress": progress}

        # Cleanup
        if os.path.exists(tif_path): os.remove(tif_path)
        if os.path.exists(reprojected_path): os.remove(reprojected_path)

        job_status[job_id] = {
            "status": "done",
            "progress": 100,
            "tile_url": f"http://127.0.0.1:8000/tiles_outputs/{file_id}/{{z}}/{{x}}/{{y}}.png",
            "bounds": {
                "south": south, "west": west,
                "north": north, "east": east,
            }
        }

    except Exception as e:
        print(f"TILING JOB ERROR: {e}")
        import traceback; traceback.print_exc()
        job_status[job_id] = {"status": "error", "message": str(e)}
        if os.path.exists(tif_path): os.remove(tif_path)

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
@app.post("/upload_orthomosaic/")
async def upload_orthomosaic(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        file_id = uuid.uuid4().hex
        job_id  = uuid.uuid4().hex
        tif_path = os.path.join(TEMP_DIR, f"ortho_{file_id}.tif")

        contents = await file.read()
        with open(tif_path, "wb") as f:
            f.write(contents)

        from rasterio.warp import transform_bounds
        with rasterio.open(tif_path) as src:
            bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        job_status[job_id] = {"status": "queued", "progress": 0}
        background_tasks.add_task(run_tiling_job, job_id, tif_path, file_id, bounds_wgs84)

        # ← Returns IMMEDIATELY, doesn't wait for tiling
        return {"status": "processing", "job_id": job_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/tiling_status/{job_id}")
async def tiling_status(job_id: str):
    return job_status.get(job_id, {"status": "not_found"})

@app.post("/process_orthomosaic/")
async def process_orthomosaic(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Create a temporary file
        temp_filename = f"ortho_{uuid.uuid4().hex}.tif"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(contents)
        
        # Open with rasterio to get info and process
        with rasterio.open(temp_path) as src:
            # Get bounds and CRS
            bounds = src.bounds
            crs = src.crs
            
            # Read a lower resolution version for web display
            # Calculate decimation factor based on image size
            width, height = src.width, src.height
            max_dimension = 2048  # Max dimension for web display
            
            if width > max_dimension or height > max_dimension:
                scale_factor = max(width, height) / max_dimension
                new_width = int(width / scale_factor)
                new_height = int(height / scale_factor)
            else:
                new_width, new_height = width, height
            
            # Read the data at lower resolution
            data = src.read(out_shape=(src.count, new_height, new_width))
            
            # Create transform for the new resolution
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
            
            # Create output filename
            output_filename = f"ortho_processed_{uuid.uuid4().hex}.tif"
            output_path = os.path.join(TEMP_DIR, output_filename)
            
            # Write the processed orthomosaic
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[-2],
                width=data.shape[-1],
                count=src.count,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                compress='lzw'  # Compression to reduce file size
            ) as dst:
                dst.write(data)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Return the processed file info
        return {
            "status": "success",
            "message": "Ortomosaico procesado correctamente",
            "processed_file_url": f"http://127.0.0.1:8000/temp_outputs/{output_filename}",
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top
            },
            "crs": str(crs)
        }
        
    except Exception as e:
        print(f"Error processing orthomosaic: {e}")
        return {"status": "error", "message": f"Error al procesar ortomosaico: {str(e)}"}

@app.get("/temp_outputs/{filename}")
async def get_temp_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"status": "error", "message": "File not found"}

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

@app.post("/export_vector/{filename}")
async def export_vector(filename: str, formato: str = "geojson"):
    try:
        # Path to the processed prediction file
        result_path = os.path.join(TEMP_DIR, filename)
        
        if not os.path.exists(result_path):
            return {"status": "error", "message": "Archivo de predicción no encontrado"}
        
        # Read the prediction raster
        with rasterio.open(result_path) as src:
            prediction_map = src.read(1)
            profile = src.profile
            transform = src.transform
            crs = src.crs
            
        # Convert raster to vector shapes
        results = [
            {'properties': {'class_index': int(value), 'class_name': ''}, 'geometry': geometry}
            for geometry, value in shapes(prediction_map, mask=(prediction_map != 99), transform=transform)
            if value != 99
        ]
        
        if not results:
            return {"status": "error", "message": "No se encontraron áreas clasificadas"}
        
        # Create GeoDataFrame
        geometries = [shape(result['geometry']) for result in results]
        class_data = [result['properties'] for result in results]
        
        gdf = gpd.GeoDataFrame(class_data, geometry=geometries, crs=crs)
        
        # Class names mapping
        classes = ['Bosque', 'Matorrales', 'Pastizales', 'T_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
        gdf['class_name'] = gdf['class_index'].map(lambda x: classes[x] if 0 <= x < len(classes) else 'Desconocido')
        gdf['area_ha'] = gdf.geometry.area / 10000  # Convert to hectares
        
        print("GeoDataFrame columns:", gdf.columns)
        print("GeoDataFrame head:\n", gdf.head())
        print("Unique classes found:", gdf['class_name'].unique())
        
        # Export based on requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato.lower() == "shapefile":
            import zipfile
            import shutil
            
            # Create directory for shapefile
            shp_dir = os.path.join(TEMP_DIR, f"shapefile_{timestamp}")
            os.makedirs(shp_dir, exist_ok=True)
            
            # Export shapefile components
            shp_path = os.path.join(shp_dir, f"prediction_{timestamp}")
            gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            
            # Create zip file
            zip_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in os.listdir(shp_dir):
                    file_path = os.path.join(shp_dir, file)
                    zipf.write(file_path, file)
            
            # Clean up directory
            shutil.rmtree(shp_dir)
            
            return {
                "status": "success",
                "download_url": f"http://127.0.0.1:8000/download/prediction_{timestamp}.zip",
                "format": "shapefile"
            }
            
        elif formato.lower() == "gpkg":
            gpkg_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.gpkg")
            
            # Add color attributes to GeoDataFrame for manual styling
            CLASS_COLORS = {
                0: '#006400',  # Bosque - Dark Green
                1: '#228B22',  # Matorrales - Forest Green  
                2: '#ADFF2F',  # Pastizales - Yellow Green
                3: '#FFFF00',  # T_Agricolas - Yellow
                4: '#FF0000',  # Infraestructura - Red
                5: '#8B4513',  # Suelo_Desnudo - Brown
                6: '#0000FF',  # Agua - Blue
            }
            
            # Add color attributes properly
            for idx, row in gdf.iterrows():
                class_idx = int(row['class_index'])
                if class_idx in CLASS_COLORS:
                    gdf.at[idx, 'fill_color'] = CLASS_COLORS[class_idx]
                    gdf.at[idx, 'stroke_color'] = CLASS_COLORS[class_idx]
                else:
                    gdf.at[idx, 'fill_color'] = '#CCCCCC'  # Default gray
                    gdf.at[idx, 'stroke_color'] = '#CCCCCC'
            
            gdf['stroke_width'] = 0.5
            gdf['fill_opacity'] = 0.7
            
            print("GeoDataFrame with colors:")
            print(gdf[['class_index', 'class_name', 'fill_color']].head(10))
            print("Unique fill colors:", gdf['fill_color'].unique())
            
            # Save GeoPackage with all attributes
            gdf.to_file(gpkg_path, driver='GPKG', encoding='utf-8')
            
            return {
                "status": "success", 
                "download_url": f"http://127.0.0.1:8000/download/prediction_{timestamp}.gpkg",
                "format": "geopackage",
                "message": "GeoPackage exportado con atributos de color. Para visualizar correctamente: 1) Abre en QGIS 2) Ve a Propiedades de Capa > Simbología 3) Cambia 'Símbolo único' a 'Graduado' 4) Selecciona 'fill_color' como columna de color"
            }
            
        elif formato.lower() == "kml":
            kml_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kml")
            gdf.to_crs('EPSG:4326').to_file(kml_path, driver='KML', encoding='utf-8')
            
            return {
                "status": "success",
                "download_url": f"http://127.0.0.1:8000/download/prediction_{timestamp}.kml", 
                "format": "kml"
            }
            
        elif formato.lower() == "kmz":
            import zipfile
            
            # First create KML
            kml_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kml")
            gdf.to_crs('EPSG:4326').to_file(kml_path, driver='KML', encoding='utf-8')
            
            # Create KMZ (zipped KML)
            kmz_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kmz")
            with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(kml_path, 'doc.kml')
            
            # Remove temporary KML
            os.remove(kml_path)
            
            return {
                "status": "success",
                "download_url": f"http://127.0.0.1:8000/download/prediction_{timestamp}.kmz",
                "format": "kmz"
            }
            
        else:
            return {"status": "error", "message": f"Formato no soportado: {formato}"}
            
    except Exception as e:
        print(f"Error en export_vector: {e}")
        return {"status": "error", "message": f"Error al exportar: {str(e)}"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(os.path.join(TEMP_DIR, filename))

@app.get("/legend/{filename}")
async def download_legend(filename: str):
    legend_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(legend_path):
        return FileResponse(legend_path, media_type='image/png')
    else:
        return {"status": "error", "message": "Legend file not found"}

@app.post("/search_recent_image/")
async def search_recent_image(data: dict = Body(...)):
    try:
        coords = data.get("coords")
        roi = ee.Geometry.Polygon(coords)
        
        
        ahora = datetime.now()
        hace_4_meses = (ahora - timedelta(days=120)).strftime('%Y-%m-%d')
        hoy_str = ahora.strftime('%Y-%m-%d')
        
        
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(hace_4_meses, hoy_str)

        # --- OPCIÓN A: LA "IDEAL" 
        ideal_img = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                              .sort('system:time_start', False) \
                              .first()

        
        recent_images = collection.sort('system:time_start', False).limit(3).getInfo()['features']

        options = []

        if ideal_img:
            info_ideal = ideal_img.getInfo()
            options.append({
                "label": "Óptima (Pocas nubes)",
                "date": datetime.fromtimestamp(info_ideal['properties']['system:time_start']/1000).strftime('%Y-%m-%d %H:%M'),
                "clouds": f"{info_ideal['properties']['CLOUDY_PIXEL_PERCENTAGE']:.2f}%",
                "id": info_ideal['id'],
                "is_ideal": True
            })

        for img in recent_images:
            img_id = img['id']
            # Evitar repetir la imagen
            if ideal_img and img_id == info_ideal['id']:
                continue
                
            date_val = datetime.fromtimestamp(img['properties']['system:time_start']/1000).strftime('%Y-%m-%d %H:%M')
            options.append({
                "label": "Reciente",
                "date": date_val,
                "clouds": f"{img['properties']['CLOUDY_PIXEL_PERCENTAGE']:.2f}%",
                "id": img_id,
                "is_ideal": False
            })

        if not options:
            return {"status": "error", "message": "No se encontraron imágenes en los últimos 4 meses."}
        
        return {
            "status": "success",
            "options": options[:4] 
        }
        
    except Exception as e:
        print(f"Error en search_recent_image: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/confirm_export/")
async def confirm_export(data: dict = Body(...)):
    try:
        coords = data.get("coords")
        image_id = data.get("image_id") 
        roi = ee.Geometry.Polygon(coords)
        
        if not image_id:
            return {"status": "error", "message": "No se seleccionó ninguna imagen."}

        selected_img = ee.Image(image_id)
        
        info = selected_img.getInfo()
        timestamp = info['properties']['system:time_start']
        anio_detectado = datetime.fromtimestamp(timestamp/1000).year
        
        BANDS_TO_SELECT = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        final_img = selected_img.select(BANDS_TO_SELECT).clip(roi).toFloat()

        task = ee.batch.Export.image.toDrive(
            image=final_img,
            description=f'S2_PNT_{anio_detectado}_Manual',
            folder=FOLDER_NAME,
            fileNamePrefix=f'S2_PNT_{anio_detectado}_sel', 
            region=roi.bounds().getInfo()['coordinates'], 
            scale=10,
            fileFormat='GeoTIFF',
            maxPixels=1e9 
        )
        task.start()

        return {
            "status": "success", 
            "message": f"Exportación iniciada para la fecha seleccionada",
            "monitoringUrl": "https://code.earthengine.google.com/tasks"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)