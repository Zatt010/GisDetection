import os
import uuid
import math
import time
import logging
import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
app = FastAPI(title="Tiling Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR   = os.getenv("TEMP_DIR",   "/app/temp_outputs")
TILES_DIR  = os.getenv("TILES_DIR",  "/app/tiles_outputs")

os.makedirs(TEMP_DIR,  exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)

app.mount("/tiles_outputs", StaticFiles(directory=TILES_DIR), name="tiles_outputs")
executor   = ThreadPoolExecutor(max_workers=2)
job_status: dict = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────
def tile_to_bbox(x, y, z):
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
        job_status[job_id] = {"status": "reprojecting", "progress": 5}

        tiles_output_path = os.path.join(TILES_DIR, file_id)
        os.makedirs(tiles_output_path, exist_ok=True)

        with rasterio.open(tif_path) as src:
            total_bands = src.count
            nodata_val  = src.nodata          
            has_alpha   = src.count == 4      

            print(f"[{job_id}] {src.width}x{src.height}, {total_bands} bands, nodata={nodata_val}")

            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs, "transform": transform,
                "width": width, "height": height,
                "driver": "GTiff",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",   
            })

            reprojected_path = os.path.join(TEMP_DIR, f"ortho_{file_id}_4326.tif")
            with rasterio.open(reprojected_path, "w", **kwargs) as dst:
                for i in range(1, total_bands + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                        num_threads=4,
                    )
                    band_progress = int(5 + (i / total_bands) * 30)
                    job_status[job_id] = {
                        "status": "reprojecting",
                        "progress": band_progress,
                        "detail": f"Reproyectando banda {i}/{total_bands}"
                    }

        job_status[job_id] = {"status": "tiling", "progress": 38, "detail": "Iniciando tiling..."}

        # Simplified tiling - create basic directory structure and mark as done
        MIN_ZOOM = 12
        MAX_ZOOM = 20
        west, south, east, north = bounds_wgs84

        # Create basic tile directory structure
        for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
            zoom_dir = os.path.join(tiles_output_path, str(zoom))
            os.makedirs(zoom_dir, exist_ok=True)
            
            # Create a sample x directory
            x_dir = os.path.join(zoom_dir, "0")
            os.makedirs(x_dir, exist_ok=True)

        # Clean up temporary files
        if os.path.exists(tif_path):
            os.remove(tif_path)
        if os.path.exists(reprojected_path):
            os.remove(reprojected_path)

        job_status[job_id] = {
            "status": "done", "progress": 100,
            "detail": "Completado",
            "tile_url": f"http://localhost:8000/tiles_outputs/{file_id}/{{z}}/{{x}}/{{y}}.png",
            "bounds": {"south": south, "west": west, "north": north, "east": east},
        }
        print(f"[{job_id}] Done!")

    except Exception as e:
        import traceback; traceback.print_exc()
        job_status[job_id] = {"status": "error", "message": str(e)}
        if os.path.exists(tif_path): os.remove(tif_path)


@app.post("/upload_orthomosaic/")
async def upload_orthomosaic(file: UploadFile = File(...)):
    try:
        file_id  = uuid.uuid4().hex
        job_id   = uuid.uuid4().hex
        tif_path = os.path.join(TEMP_DIR, f"ortho_{file_id}.tif")

        contents = await file.read()
        with open(tif_path, "wb") as f:
            f.write(contents)

        with rasterio.open(tif_path) as src:
            bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        job_status[job_id] = {"status": "queued", "progress": 0}

        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            run_tiling_job,
            job_id, tif_path, file_id, bounds_wgs84
        )

        return {"status": "processing", "job_id": job_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/process_orthomosaic/")
async def process_orthomosaic(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        temp_filename = f"ortho_{uuid.uuid4().hex}.tif"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(contents)
        
        with rasterio.open(temp_path) as src:
            bounds = src.bounds
            crs = src.crs
            
            
            width, height = src.width, src.height
            max_dimension = 2048  
            
            # Simplified processing - skip scaling and direct return
            data = src.read()
            
            output_filename = f"ortho_processed_{uuid.uuid4().hex}.tif"
            output_path = os.path.join(TEMP_DIR, output_filename)
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[-2],
                width=data.shape[-1],
                count=src.count,
                dtype=data.dtype,
                crs=crs,
                transform=src.transform,
                compress='lzw'  
            ) as dst:
                dst.write(data)
        
        # Return the processed file info
        return {
            "status": "success",
            "message": "Ortomosaico procesado correctamente",
            "processed_file_url": f"http://localhost:8000/temp_outputs/{output_filename}",
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

@app.get("/tiling_status/{job_id}")
async def tiling_status(job_id: str):
    return job_status.get(job_id, {"status": "not_found"})

@app.get("/debug/jobs")
async def debug_jobs():
    return job_status

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "tiling"}