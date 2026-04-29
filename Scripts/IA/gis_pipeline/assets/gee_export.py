"""
Stage 1 — Export Sentinel-2 imagery and WorldCover labels from Google Earth Engine.

This asset kicks off two GEE export tasks to Google Drive:
  • S2_Data_v4.tif    — 13-channel Sentinel-2 composite (9 bands + 4 indices)
  • Labels_Data_v4.tif — ESA WorldCover 2021 labels

The asset does NOT wait for GEE tasks to finish (they run server-side).
It returns the task IDs so you can monitor them at:
  https://code.earthengine.google.com/tasks
"""
import time
from dagster import asset, AssetExecutionContext, Output, MetadataValue
import ee

from gis_pipeline.resources import PipelineConfig


# ── Sentinel-2 helpers ────────────────────────────────────────────────────────

def _mask_s2_clouds(image):
    """Pixel-level cloud masking using the Scene Classification Layer (SCL)."""
    scl = image.select("SCL")
    cloud_mask = (
        scl.neq(8).And(scl.neq(9))   # medium/high-prob clouds
           .And(scl.neq(10))          # thin cirrus
           .And(scl.neq(7))           # unclassified
           .And(scl.neq(3))           # cloud shadows
    )
    return image.updateMask(cloud_mask)


def _add_spectral_indices(image):
    """
    Add 4 spectral indices critical for separating difficult classes:
      NDVI  → vegetation density   (Bosque / Matorrales / Pastizales)
      NDWI  → water detection      (Agua)
      NDBI  → built-up index       (Infraestructura vs Suelo_Desnudo)
      BSI   → bare soil index      (Suelo_Desnudo vs Tierras_Agricolas)
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    bsi = image.expression(
        "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
        {
            "B11": image.select("B11"),
            "B4":  image.select("B4"),
            "B8":  image.select("B8"),
            "B2":  image.select("B2"),
        },
    ).rename("BSI")
    return image.addBands([ndvi, ndwi, ndbi, bsi])


def _build_sentinel_composite(aoi, date_start, date_end, cloud_pct):
    """Return a 13-band median composite clipped to AOI."""
    raw_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
    all_bands  = raw_bands + ["NDVI", "NDWI", "NDBI", "BSI"]

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(date_start, date_end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_s2_clouds)
        .map(_add_spectral_indices)
    )
    return collection.median().select(all_bands).clip(aoi).toFloat()


# ── Dagster asset ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_ingestion",
    description=(
        "Exports Sentinel-2 (13 bands) and WorldCover labels from GEE to "
        "Google Drive. Returns GEE task IDs for monitoring."
    ),
)
def gee_export(context: AssetExecutionContext, config: PipelineConfig) -> Output[dict]:
    """
    Kick off GEE export tasks.  The asset completes immediately;
    actual export runs asynchronously on Google's servers (~5-20 min).
    """
    context.log.info("Initializing Google Earth Engine...")
    ee.Initialize(project=config.gee_project_id)

    west, south, east, north = config.aoi_coords
    aoi = ee.Geometry.Rectangle([west, south, east, north])

    # ── Sentinel-2 composite ──────────────────────────────────────────────
    context.log.info(
        f"Building S2 composite: {config.date_start} → {config.date_end}, "
        f"cloud < {config.cloud_pct}%"
    )
    s2_image = _build_sentinel_composite(
        aoi, config.date_start, config.date_end, config.cloud_pct
    )

    task_s2 = ee.batch.Export.image.toDrive(
        image          = s2_image,
        description    = "S2_v4_WithIndices",
        folder         = config.gee_folder,
        fileNamePrefix = "S2_Data_v4",
        region         = aoi,
        scale          = 10,
        fileFormat     = "GeoTIFF",
        maxPixels      = 1e13,
    )
    task_s2.start()
    s2_task_id = task_s2.id
    context.log.info(f"✓ S2 export started  →  task ID: {s2_task_id}")

    # ── WorldCover labels ─────────────────────────────────────────────────
    worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(aoi)

    task_labels = ee.batch.Export.image.toDrive(
        image          = worldcover,
        description    = "WorldCover_Labels_v4",
        folder         = config.gee_folder,
        fileNamePrefix = "Labels_Data_v4",
        region         = aoi,
        scale          = 10,
        fileFormat     = "GeoTIFF",
        maxPixels      = 1e13,
    )
    task_labels.start()
    labels_task_id = task_labels.id
    context.log.info(f"✓ Labels export started  →  task ID: {labels_task_id}")

    # ── QA pixel-count map ────────────────────────────────────────────────
    s2_raw = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(config.date_start, config.date_end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.cloud_pct))
        .map(_mask_s2_clouds)
    )
    pixel_count = s2_raw.select("B2").count().clip(aoi)
    task_qa = ee.batch.Export.image.toDrive(
        image          = pixel_count,
        description    = "QA_PixelCount_v4",
        folder         = config.gee_folder,
        fileNamePrefix = "QA_PixelCount_v4",
        region         = aoi,
        scale          = 10,
        fileFormat     = "GeoTIFF",
        maxPixels      = 1e13,
    )
    task_qa.start()
    context.log.info(f"✓ QA map export started  →  task ID: {task_qa.id}")

    result = {
        "s2_task_id":     s2_task_id,
        "labels_task_id": labels_task_id,
        "qa_task_id":     task_qa.id,
        "monitor_url":    "https://code.earthengine.google.com/tasks",
        "output_folder":  config.gee_folder,
        "s2_filename":    "S2_Data_v4.tif",
        "labels_filename": "Labels_Data_v4.tif",
    }

    return Output(
        value=result,
        metadata={
            "s2_task_id":     MetadataValue.text(s2_task_id),
            "labels_task_id": MetadataValue.text(labels_task_id),
            "monitor_url":    MetadataValue.url(result["monitor_url"]),
            "gee_folder":     MetadataValue.text(config.gee_folder),
            "date_range":     MetadataValue.text(f"{config.date_start} → {config.date_end}"),
            "aoi":            MetadataValue.text(str(config.aoi_coords)),
            "bands":          MetadataValue.int(13),
        },
    )