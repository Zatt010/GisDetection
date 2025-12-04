import ee

PROJECT_ID = 'aifinal-480001' 
FOLDER_NAME = 'Tesis_PNT_Sentinel_Dataset' # Folder Google Drive
SCALE = 10 
YEARS = [2016, 2020, 2024, 2025] 

try:
    ee.Initialize(project=PROJECT_ID) 
    print("¡Earth Engine initialized successfully!")

except Exception as e:
    print(f"Error: {e}")

aoi_coords = [
    [
        [-66.22589320019163, -17.327984858056993],
        [-66.2177308689185, -17.31408528970765],
        [-66.15662368614169, -17.328511336185215],
        [-66.10180370448111, -17.36862452688139],
        [-66.12584949120544, -17.365992696779358],
        [-66.22589320019163, -17.327984858056993]
    ]
]
roi = ee.Geometry.Polygon(aoi_coords)


def get_best_image_mosaic(year, roi):
    
    BANDS_TO_SELECT = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'] # 7 Bands
    
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate(f'{year}-05-01', f'{year}-09-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') 
    
    image = collection.first()
    
    if not image:
        raise ValueError(f"No clean image found for {year} in the dry season.")

    
    selected_image = image.select(BANDS_TO_SELECT).clip(roi)
    
    return selected_image.toFloat() 


def export_to_drive_task(image, year, roi):
    
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'S2_PNT_{year}_Export_7BANDS',
        folder=FOLDER_NAME,
        fileNamePrefix=f'S2_PNT_{year}_7B', 
        region=roi.bounds().getInfo()['coordinates'], 
        scale=SCALE,
        fileFormat='GeoTIFF',
        maxPixels=1e9 
    )
    task.start()
    return task


print("Starting export tasks to Google Drive...")
print(f"Images will be saved in the folder: {FOLDER_NAME}")
tasks = []

for year in YEARS:
    try:
        image = get_best_image_mosaic(year, roi)
        
        task = export_to_drive_task(image, year, roi)
        tasks.append(task)
        
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        print(f"Export task for {year} started. Acquisition Date: {date} (Task ID: {task.id}).")

    except Exception as e:
        print(f"Error initiating export for {year}: {e}")

print("\nMonitor the 'Tasks' tab in the GEE Code Editor to track progress.")