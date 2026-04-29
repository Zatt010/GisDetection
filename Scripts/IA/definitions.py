import os
import subprocess
from dagster import asset, Definitions, AssetExecutionContext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asset(
    description="Paso 1: Descarga y prepara los datos de Sentinel-2 usando Google Earth Engine."
)
def dataset_sentinel(context: AssetExecutionContext):
    context.log.info("Iniciando la descarga de datos desde GEE...")
    
    script_path = os.path.join(BASE_DIR, "gee_dataset_generator.py")
    
    result = subprocess.run(["uv", "run", "python", script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        context.log.error(f"Error en gee_dataset_generator.py:\n{result.stderr}")
        raise Exception("Falló la generación del dataset desde GEE.")
        
    context.log.info(f"Salida:\n{result.stdout}")
    return "Datos descargados y listos."

@asset(
    deps=[dataset_sentinel], 
    description="Paso 2: Carga el modelo U-Net (.keras) y ejecuta la inferencia sobre el TIF."
)
def mapa_clasificacion(context: AssetExecutionContext):
    context.log.info("Iniciando la inferencia del modelo U-Net...")
    
    script_path = os.path.join(BASE_DIR, "inferencia_final_tesis.py")
    
    result = subprocess.run(["uv", "run", "python", script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        context.log.error(f"Error en inferencia_final_tesis.py:\n{result.stderr}")
        raise Exception("Falló la inferencia del modelo.")
        
    context.log.info(f"Salida:\n{result.stdout}")
    return "Mapa de clasificación generado exitosamente."

defs = Definitions(
    assets=[dataset_sentinel, mapa_clasificacion],
)