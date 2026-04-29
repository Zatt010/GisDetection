import os
import ee
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

app = FastAPI(title="GEE Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID  = os.getenv("GEE_PROJECT_ID", "aifinal-480001")
FOLDER_NAME = "Tesis_PNT_Sentinel"


@app.on_event("startup")
def startup_event():
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"GEE initialized with project: {PROJECT_ID}")
    except Exception as e:
        print(f"GEE init error: {e}")

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "gee"}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/search_recent_image/")
async def search_recent_image(data: dict = Body(...)):
    try:
        coords = data.get("coords")
        roi    = ee.Geometry.Polygon(coords)

        ahora        = datetime.now()
        hace_4_meses = (ahora - timedelta(days=120)).strftime("%Y-%m-%d")
        hoy_str      = ahora.strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(hace_4_meses, hoy_str)
        )

        # Best (low cloud) image
        ideal_img = (
            collection
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .sort("system:time_start", False)
            .first()
        )

        # 3 most recent regardless of clouds
        recent_images = (
            collection.sort("system:time_start", False).limit(3).getInfo()["features"]
        )

        options    = []
        info_ideal = None

        if ideal_img:
            info_ideal = ideal_img.getInfo()
            options.append({
                "label":    "Óptima (Pocas nubes)",
                "date":     datetime.fromtimestamp(
                                info_ideal["properties"]["system:time_start"] / 1000
                            ).strftime("%Y-%m-%d %H:%M"),
                "clouds":   f"{info_ideal['properties']['CLOUDY_PIXEL_PERCENTAGE']:.2f}%",
                "id":       info_ideal["id"],
                "is_ideal": True,
            })

        for img in recent_images:
            img_id = img["id"]
            if info_ideal and img_id == info_ideal["id"]:
                continue
            options.append({
                "label":    "Reciente",
                "date":     datetime.fromtimestamp(
                                img["properties"]["system:time_start"] / 1000
                            ).strftime("%Y-%m-%d %H:%M"),
                "clouds":   f"{img['properties']['CLOUDY_PIXEL_PERCENTAGE']:.2f}%",
                "id":       img_id,
                "is_ideal": False,
            })

        if not options:
            return {"status": "error", "message": "No se encontraron imágenes en los últimos 4 meses."}

        return {"status": "success", "options": options[:4]}

    except Exception as e:
        print(f"Error en search_recent_image: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/confirm_export/")
async def confirm_export(data: dict = Body(...)):
    try:
        coords   = data.get("coords")
        image_id = data.get("image_id")
        roi      = ee.Geometry.Polygon(coords)

        if not image_id:
            return {"status": "error", "message": "No se seleccionó ninguna imagen."}

        selected_img = ee.Image(image_id)
        info         = selected_img.getInfo()
        timestamp    = info["properties"]["system:time_start"]
        anio         = datetime.fromtimestamp(timestamp / 1000).year

        BANDS     = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]
        final_img = selected_img.select(BANDS).clip(roi).toFloat()

        task = ee.batch.Export.image.toDrive(
            image=final_img,
            description=f"S2_PNT_{anio}_Manual",
            folder=FOLDER_NAME,
            fileNamePrefix=f"S2_PNT_{anio}_sel",
            region=roi.bounds().getInfo()["coordinates"],
            scale=10,
            fileFormat="GeoTIFF",
            maxPixels=1e9,
        )
        task.start()

        return {
            "status":        "success",
            "message":       f"Exportación iniciada para {anio}",
            "monitoringUrl": "https://code.earthengine.google.com/tasks",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}