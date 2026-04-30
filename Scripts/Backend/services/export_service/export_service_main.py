import os
import zipfile
import shutil
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Export Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.getenv("TEMP_DIR", "/app/temp_outputs")
os.makedirs(TEMP_DIR, exist_ok=True)

CLASSES = ["Bosque", "Matorrales", "Pastizales", "T_Agricolas", "Infraestructura", "Suelo_Desnudo", "Agua"]

CLASS_COLORS = {
    0: "#006400",  # Bosque         – Dark Green
    1: "#228B22",  # Matorrales     – Forest Green
    2: "#ADFF2F",  # Pastizales     – Yellow Green
    3: "#FFFF00",  # T_Agricolas    – Yellow
    4: "#FF0000",  # Infraestructura– Red
    5: "#8B4513",  # Suelo_Desnudo  – Brown
    6: "#0000FF",  # Agua           – Blue
}


def _load_gdf(filename: str):
    result_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(result_path):
        return None, "Archivo de predicción no encontrado"

    with rasterio.open(result_path) as src:
        prediction_map = src.read(1)
        transform      = src.transform
        crs            = src.crs

    results = [
        {"properties": {"class_index": int(value)}, "geometry": geometry}
        for geometry, value in shapes(prediction_map, mask=(prediction_map != 99), transform=transform)
        if value != 99
    ]

    if not results:
        return None, "No se encontraron áreas clasificadas"

    gdf = gpd.GeoDataFrame(
        [r["properties"] for r in results],
        geometry=[shape(r["geometry"]) for r in results],
        crs=crs,
    )
    gdf["class_name"] = gdf["class_index"].map(
        lambda x: CLASSES[x] if 0 <= x < len(CLASSES) else "Desconocido"
    )
    gdf["area_ha"] = gdf.geometry.area / 10000

    for idx, row in gdf.iterrows():
        ci = int(row["class_index"])
        color = CLASS_COLORS.get(ci, "#CCCCCC")
        gdf.at[idx, "fill_color"]   = color
        gdf.at[idx, "stroke_color"] = color

    gdf["stroke_width"]  = 0.5
    gdf["fill_opacity"]  = 0.7

    return gdf, None


def _export_kml_custom(gdf, output_path):
    """Export GeoDataFrame to KML without using Fiona drivers"""
    # Convert to WGS84 if needed
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    # Create KML document
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    
    # Add name
    ET.SubElement(document, "name").text = "Prediction Results"
    
    # Add styles for each class
    for class_idx, class_name in enumerate(CLASSES):
        style = ET.SubElement(document, "Style", id=f"style_{class_idx}")
        poly_style = ET.SubElement(style, "PolyStyle")
        color = CLASS_COLORS.get(class_idx, "#CCCCCC")
        # Convert hex color to KML format (AABBGGRR)
        kml_color = "ff" + color[5:7] + color[3:5] + color[1:3]
        ET.SubElement(poly_style, "color").text = kml_color
        ET.SubElement(poly_style, "fill").text = "1"
        ET.SubElement(poly_style, "outline").text = "1"
    
    # Add features
    for idx, row in gdf_wgs84.iterrows():
        placemark = ET.SubElement(document, "Placemark")
        ET.SubElement(placemark, "name").text = f"{row['class_name']} - {row['area_ha']:.2f} ha"
        ET.SubElement(placemark, "styleUrl").text = f"#style_{int(row['class_index'])}"
        
        # Add geometry
        geom = row.geometry
        if geom.geom_type == "Polygon":
            coords = geom.exterior.coords
            coordinates = " ".join([f"{lon},{lat},0" for lon, lat in coords])
            
            polygon = ET.SubElement(placemark, "Polygon")
            outer_boundary = ET.SubElement(polygon, "outerBoundaryIs")
            linear_ring = ET.SubElement(outer_boundary, "LinearRing")
            ET.SubElement(linear_ring, "coordinates").text = coordinates
    
    # Write to file
    tree = ET.ElementTree(kml)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/export_vector/{filename}")
async def export_vector(filename: str, formato: str = "geojson"):
    try:
        gdf, err = _load_gdf(filename)
        if err:
            return {"status": "error", "message": err}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fmt = formato.lower()

        # ── GeoJSON ──────────────────────────────────────────────────────────
        if fmt == "geojson":
            out_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.geojson")
            gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON", encoding="utf-8")
            return {
                "status":       "success",
                "download_url": f"http://localhost:8004/download/prediction_{timestamp}.geojson",
                "format":       "geojson",
            }

        # ── Shapefile ─────────────────────────────────────────────────────────
        elif fmt == "shapefile":
            shp_dir  = os.path.join(TEMP_DIR, f"shapefile_{timestamp}")
            os.makedirs(shp_dir, exist_ok=True)
            
            shp_path = os.path.join(shp_dir, f"prediction_{timestamp}.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

            zip_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for f in os.listdir(shp_dir):
                    full_path = os.path.join(shp_dir, f)
                    if os.path.isfile(full_path):  
                        zipf.write(full_path, f)   

            shutil.rmtree(shp_dir)

            return {
                "status":       "success",
                "download_url": f"http://localhost:8004/download/prediction_{timestamp}.zip",
                "format":       "shapefile",
            }

        # ── GeoPackage ────────────────────────────────────────────────────────
        elif fmt == "gpkg":
            gpkg_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.gpkg")
            gdf.to_file(gpkg_path, driver="GPKG", encoding="utf-8")
            return {
                "status":       "success",
                "download_url": f"http://localhost:8004/download/prediction_{timestamp}.gpkg",
                "format":       "geopackage",
                "message":      (
                    "GeoPackage exportado. Para visualizar en QGIS: "
                    "Propiedades de Capa > Simbología > Categorizado > Columna: fill_color"
                ),
            }

        # ── KML ───────────────────────────────────────────────────────────────
        elif fmt == "kml":
            kml_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kml")
            _export_kml_custom(gdf, kml_path)
            return {
                "status":       "success",
                "download_url": f"http://localhost:8004/download/prediction_{timestamp}.kml",
                "format":       "kml",
            }

        # ── KMZ ───────────────────────────────────────────────────────────────
        elif fmt == "kmz":
            kml_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kml")
            _export_kml_custom(gdf, kml_path)

            kmz_path = os.path.join(TEMP_DIR, f"prediction_{timestamp}.kmz")
            with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(kml_path, "doc.kml")
            os.remove(kml_path)

            return {
                "status":       "success",
                "download_url": f"http://localhost:8004/download/prediction_{timestamp}.kmz",
                "format":       "kmz",
            }

        else:
            return {"status": "error", "message": f"Formato no soportado: {formato}"}

    except Exception as e:
        print(f"Error en export_vector: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    return {"status": "error", "message": "File not found"}


@app.get("/legend/{filename}")
async def download_legend(filename: str):
    legend_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(legend_path):
        return FileResponse(legend_path, media_type="image/png")
    return {"status": "error", "message": "Legend not found"}


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "export"}