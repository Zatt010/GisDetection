import ee
import geemap
import os

try:
    ee.Initialize()
    print("GEE Initialized.")
except Exception as e:
    print(f"Error GEE: {e}")

AOI = ee.Geometry.Rectangle([-66.35, -17.50, -65.90, -17.20])


CLASES_A_EXTRAER = {
    'Bosque': 10,
    'Matorrales': 20,
    'Pastizales': 30,
    'Tierras_Agricolas': 40,
    'Infraestructura': 50,
    'Suelo_Desnudo': 60,  
    'Agua': 80 
}

PATCH_SIZE = 128  # Parche en pixeles (128x128)
BANDAS_S2 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']

def get_sentinel_image(aoi):
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2024-05-01', '2024-09-30') \
        .filterBounds(aoi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) 
    
    image_composite = s2_collection.median()
    return image_composite.select(BANDAS_S2).clip(aoi).toFloat()

def export_cnn_data(s2_image, aoi):
    
    print("Iniciando exportación para CNN...")

    task_img = ee.batch.Export.image.toDrive(
        image=s2_image,
        description='Sentinel2_Image_For_CNN',
        folder='CNN_Training_Data',
        fileNamePrefix='S2_Data',
        region=aoi,
        scale=10,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    task_img.start()
    print("- Tarea de imagen satelital enviada a Drive.")

    worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(aoi)
    
    task_label = ee.batch.Export.image.toDrive(
        image=worldcover,
        description='WorldCover_Labels_For_CNN',
        folder='CNN_Training_Data',
        fileNamePrefix='Labels_Data',
        region=aoi,
        scale=10,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    task_label.start()
    print("- Tarea de etiquetas (labels) enviada a Drive.")

if __name__ == "__main__":
    if 'ee' in globals():
        try:
            s2_img = get_sentinel_image(AOI)
            export_cnn_data(s2_img, AOI)
            
            print("\n--- PROCESO INICIADO ---")
            print("Revisa: https://code.earthengine.google.com/tasks")
            
        except Exception as e:
            print(f"Error general: {e}")