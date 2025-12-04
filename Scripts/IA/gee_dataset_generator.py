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
    'Cobertura_1_Bosque': 10, 
    'Cobertura_2_SueloDesnudo': 80, 
    'Cobertura_3_Infraestructura': 50 
}

NUM_MUESTRAS_POR_CLASE = 5000 


BANDAS_S2 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']


# Carga la imagen de Sentinel-2 para las bandas espectrales
def get_sentinel_image(aoi):

    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2024-05-01', '2024-09-30') \
        .filterBounds(aoi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) 
    
    
    image_composite = s2_collection.median()
    
    print(f"Generating Median composite (May-Sept) for the area.")
    
    # Seleccion de bandas y recorte
    return image_composite.select(BANDAS_S2).clip(aoi).toFloat()

worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(AOI)


def export_training_samples(s2_image, class_map, aoi, num_samples):
    export_tasks = [] 
    for class_name, wc_value in class_map.items():
        print(f"Sampling: {class_name}...")
        
        class_mask = worldcover.eq(wc_value)
        training_image = s2_image.updateMask(class_mask).addBands(ee.Image.constant(CLASES_A_EXTRAER[class_name]).rename('class'))
        
        samples = training_image.sample(
            region=aoi,
            scale=10, 
            numPixels=num_samples, 
            seed=42,
            tileScale=8,
            geometries=False 
        )
        
        task = ee.batch.Export.table.toDrive(
            collection=samples,
            description=f'Dataset_{class_name}', 
            folder='GEE_Training_Data_FIXED', 
            fileNamePrefix=class_name,
            fileFormat='CSV',
            selectors=BANDAS_S2 + ['class'] 
        )
        task.start()
        export_tasks.append(task)
    return export_tasks


if __name__ == "__main__":
    if 'ee' in globals():
        try:
            s2_img = get_sentinel_image(AOI)
            
            tasks = export_training_samples(s2_img, CLASES_A_EXTRAER, AOI, NUM_MUESTRAS_POR_CLASE)
            
            print("\nExport started. CHECK GEE to monitor progress:")
         
            print("   - URL: https://code.earthengine.google.com/tasks")
            print(f"   - Se crearán {len(tasks)} archivos CSV en la carpeta 'GEE_Training_Data' de tu Drive.")
            
        except ValueError as ve:
            print(f"Error de datos: {ve}")
        except Exception as e:
            print(f"Error general: {e}")