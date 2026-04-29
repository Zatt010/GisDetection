# IA/ML - GIS Detection

Componentes de machine learning e inteligencia artificial para el proyecto GIS Detection.

## Descripción General

La carpeta IA contiene todos los componentes de inteligencia artificial y machine learning utilizados para el análisis de detecciones geoespaciales. Incluye modelos de deep learning, scripts de entrenamiento, inferencia y procesamiento de datos geoespaciales con técnicas avanzadas de IA.

## Estructura de Carpetas

```
IA/
├── Configuration/          # Configuraciones y dependencias
│   ├── Conda/              # Archivos de entorno conda
│   └── requirements.txt    # Lista de dependencias
├── Entrenamiento/          # Scripts y notebooks de entrenamiento
├── Classification_Results/ # Resultados de clasificaciones
├── Tif/                   # Archivos TIFF geoespaciales
├── gis_pipeline/          # Pipeline Dagster moderno
│   ├── __init__.py        # Definiciones del pipeline
│   ├── definitions.py     # Definiciones de Dagster
│   ├── resources.py       # Configuración y recursos
│   └── assets/            # Assets del pipeline
│       ├── __init__.py
│       ├── gee_export.py  # Exportación desde GEE
│       ├── training.py    # Entrenamiento del modelo
│       └── evaluation.py  # Evaluación del modelo
├── inferencia_final_tesis.py # Script de inferencia
├── modelo_unet_pro_final.keras # Modelo pre-entrenado
├── pyproject.toml         # Gestión de dependencias con UV
└── README.md              # Este archivo
```

## Instalación y Ejecución

### Prerrequisitos

- **Python**: Exactamente `3.9.25` (versión utilizada en el entorno conda)
- **UV**: Gestor de paquetes Python (instalar con `pip install uv`)
- **CUDA**: Para aceleración GPU (opcional pero recomendado)
- **Memoria RAM**: Mínimo 8GB para entrenamiento

### Instalación con UV

1. **Navegar a la carpeta IA:**
   ```bash
   cd IA
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   uv sync
   ```

3. **Activar entorno virtual:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

## Uso Básico

### Entrenamiento de Modelos

#### Entrenamiento de CNN para Land Cover Classification
```bash
uv run python train_cnn_landcover.py
```

Ejemplo de salida esperada:
```
Epoch 1/100
1000/1000 [==============================] - 45s 45ms/step - loss: 0.8234 - accuracy: 0.7234
...
Modelo guardado en: models/modelo_unet_pro_final.keras
```

### Inferencia y Predicción

#### Ejecutar Inferencia sobre Datos Nuevos
```bash
uv run python inferencia_final_tesis.py --input data/test_images/ --output results/
```

#### Inferencia Programática
```python
from tensorflow.keras.models import load_model
import numpy as np

# Cargar modelo pre-entrenado
model = load_model('modelo_unet_pro_final.keras')

# Realizar predicción
predictions = model.predict(image_data)
```

### Jupyter Notebooks

#### Iniciar Jupyter para Desarrollo Interactivo
```bash
uv run jupyter notebook
```

Acceso a notebooks en: http://localhost:8888

## Requisitos Técnicos

### Dependencias Principales

| Componente | Versión | Descripción |
|-------------|---------|-------------|
| **Python** | `3.9.25` | Versión exacta del intérprete |
| **TensorFlow** | `2.20.0+` | Framework de deep learning |
| **Keras** | `3.10.0+` | API de alto nivel para redes neuronales |
| **OpenCV** | `4.13.0+` | Procesamiento de imágenes y visión por computadora |
| **Scikit-learn** | `1.6.1+` | Machine learning tradicional |

### Librerías Clave

- **Deep Learning**: `tensorflow`, `keras`, `ml-dtypes`
- **Procesamiento de Imágenes**: `opencv-python`, `pillow`, `scikit-image`
- **Geoespaciales**: `geopandas`, `rasterio`, `shapely`, `pyproj`
- **Visualización**: `matplotlib`, `seaborn`, `plotly`
- **Datos**: `numpy`, `pandas`, `xarray`, `pyarrow`

### Configuración Requerida

1. **Variables de Entorno**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   export TF_ENABLE_ONEDNN_OPTS=1  # Optimización Intel
   export CUDA_VISIBLE_DEVICES=0  # GPU selection
   ```

2. **Configuración de TensorFlow**:
   ```python
   import tensorflow as tf
   print(f"TensorFlow Version: {tf.__version__}")
   print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
   ```

## Modelos y Arquitecturas

### Modelo Principal: UNet++

- **Arquitectura**: UNet++ mejorada para segmentación semántica
- **Clases**: 7 clases de land cover (agua, vegetación, urbano, etc.)
- **Input**: 256x256 píxeles, 3 canales RGB
- **Output**: Máscaras de segmentación 256x256
- **Accuracy**: ~92% en conjunto de prueba

### Métricas de Evaluación
- **Precision**: 0.91
- **Recall**: 0.89
- **F1-Score**: 0.90
- **IoU**: 0.85


## 🤖 Pipeline GIS con Dagster

El proyecto incluye un pipeline completo de orquestación usando Dagster para automatizar todo el proceso de ML geoespacial:

### Arquitectura del Pipeline

```
Dagster GIS Pipeline
├── 1. gee_export              # Exporta datos desde GEE a Google Drive
├── 2. train_unet             # Entrena modelo U-Net con datos preparados
├── 3. evaluate_model         # Evalúa el modelo entrenado
└── 4. pipeline_report        # Genera reporte final del pipeline
```

### Instalación y Configuración

```bash
# Ya incluido en pyproject.toml
uv sync  # Instalará dagster>=1.5.0
```

### Configuración de Google Earth Engine

1. **Autenticación**:
   ```bash
   uv run python -c "import ee; ee.Authenticate()"
   ```

2. **Configurar project ID**:
   El project ID está configurado en `gis_pipeline/resources.py`:
   ```python
   gee_project_id: str = Field(default="aifinal-480001")
   ```

### Ejecución del Pipeline

#### Modo Desarrollo (UI Web)
```bash
cd IA
uv run dagster dev -f gis_pipeline/definitions.py
```

Acceso a la UI de Dagster: http://localhost:3000

#### Ejecución Programática
```python
from dagster import build_assets_job
from gis_pipeline.definitions import all_assets

# Crear y ejecutar job
job = build_assets_job("gis_job", all_assets)
result = job.execute_in_process()
```

### Assets del Pipeline

#### 1. gee_export
- **Descripción**: Exporta imágenes Sentinel-2 (13 bandas) y etiquetas WorldCover desde GEE
- **Input**: Configuración de AOI y fechas
- **Output**: Task IDs de GEE para monitoreo
- **Archivo**: `gis_pipeline/assets/gee_export.py`

#### 2. train_unet  
- **Descripción**: Entrena modelo U-Net Pro con datos exportados
- **Input**: Imágenes y etiquetas TIFF
- **Output**: Modelo entrenado (.keras) e historial
- **Archivo**: `gis_pipeline/assets/training.py`

#### 3. evaluate_model
- **Descripción**: Evalúa modelo con métricas completas
- **Input**: Modelo entrenado y datos de prueba
- **Output**: Métricas, matriz de confusión y visualizaciones
- **Archivo**: `gis_pipeline/assets/evaluation.py`

### Configuración del Pipeline

Toda la configuración está centralizada en `gis_pipeline/resources.py`:

```python
class PipelineConfig(ConfigurableResource):
    # Rutas
    base_path: str = "C:/Users/afuhe/Desktop/materias/PG/Scripts/IA"
    gee_folder: str = "CNN_Training_Data"
    
    # Dataset
    aoi_coords: list = [-66.35, -17.50, -65.90, -17.20]  # [west, south, east, north]
    date_start: str = "2023-05-01"
    date_end: str = "2023-09-30"
    
    # Hiperparámetros
    patch_size: int = 64
    patch_step: int = 32
    channels: int = 13
    num_classes: int = 7
    batch_size: int = 16
    epochs: int = 100
    
    # Clases
    class_names: list = ["Bosque", "Matorrales", "Pastizales", 
                        "Tierras_Agricolas", "Infraestructura", 
                        "Suelo_Desnudo", "Agua"]
```

### Flujo de Trabajo

1. **Exportación de Datos**: GEE exporta imágenes Sentinel-2 y etiquetas WorldCover
2. **Preparación Automática**: Formateo y normalización de datos para entrenamiento
3. **Entrenamiento**: U-Net Pro con 13 canales (9 bandas + 4 índices)
4. **Evaluación**: Métricas completas y visualizaciones
5. **Reporte**: Generación automática de resultados

### Monitoreo y Logs

Dagster proporciona:
- **UI Web**: Monitoreo en tiempo real en http://localhost:3000
- **Logs Detallados**: Cada asset con logging estructurado
- **Historial**: Registro de todas las ejecuciones
- **Métricas**: Tiempos de ejecución y rendimiento
- **Dependencies**: Gestión automática de dependencias entre assets

### Salidas del Pipeline

- **Modelo**: `modelo_unet_pro_v4.keras`
- **Historial**: `training_history_v4.json`
- **Métricas**: Guardadas como metadata de Dagster
- **Visualizaciones**: Matriz de confusión y gráficos
- **Reporte**: Metadata completa en la UI de Dagster

##  Ejemplos de Uso

### 1. Ejecución del Pipeline Completo
```bash
# Iniciar Dagster UI
cd IA
uv run dagster dev -f gis_pipeline/definitions.py

# Acceder a http://localhost:3000
# Seleccionar "Materialize all" para ejecutar todo el pipeline
```

### 2. Ejecución Individual de Assets
```python
from gis_pipeline.assets.gee_export import gee_export
from gis_pipeline.assets.training import train_unet
from gis_pipeline.resources import PipelineConfig
from dagster import build_asset_context

# Configurar
config = PipelineConfig()
context = build_asset_context()

# Exportar datos desde GEE
gee_result = gee_export(context, config)

# Entrenar modelo
training_result = train_unet(context, config)
```

### 2. Batch Processing
```python
import glob
from inferencia_final_tesis import process_batch

# Procesar múltiples imágenes
image_files = glob.glob('data/*.tif')
results = process_batch(image_files, model_path='modelo_unet_pro_final.keras')
```

### 3. Integración con Google Earth Engine
```python
import ee
from gis_pipeline.resources import PipelineConfig

# Configuración
config = PipelineConfig()

# Inicializar GEE
ee.Initialize(project=config.gee_project_id)

# Definir región de interés
west, south, east, north = config.aoi_coords
aoi = ee.Geometry.Rectangle([west, south, east, north])

# Consultar imágenes Sentinel-2
s2_collection = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(aoi) \
    .filterDate(config.date_start, config.date_end) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', config.cloud_pct))
```

## Optimización y Rendimiento

### GPU Acceleration
```bash
# Verificar disponibilidad de GPU
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Habilitar crecimiento de memoria GPU
uv run python -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
"
```

### Optimización de Memoria
```python
# Configurar TensorFlow para uso eficiente de memoria
import tensorflow as tf
tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices('GPU')[0], True
)
```

## Notas Importantes

1. **Versión Python**: Es crucial usar Python 3.9.25 para compatibilidad con TensorFlow
2. **Memoria**: El entrenamiento requiere mínimo 16GB RAM para datasets grandes
3. **GPU**: Se recomienda GPU con mínimo 8GB VRAM para entrenamiento eficiente
4. **Dataset**: Los datos de entrenamiento deben estar en formato TIFF georreferenciado

##  Troubleshooting

### Problemas Comunes

1. **Error de memoria GPU**:
   ```bash
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

2. **Error de dependencias**:
   ```bash
   uv sync --refresh
   ```

3. **Error de CUDA**:
   ```bash
   # Verificar compatibilidad CUDA
   uv run python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
   ```

4. **Error de carga de modelo**:
   ```bash
   # Verificar ruta y formato del modelo
   uv run python -c "
   from tensorflow.keras.models import load_model
   model = load_model('modelo_unet_pro_final.keras')
   print(model.summary())
   "
   ```

## Documentación Adicional

- [Documentación TensorFlow](https://www.tensorflow.org/)
- [Documentación Keras](https://keras.io/)
- [Documentación OpenCV](https://opencv.org/)
- [Guía UV](https://docs.astral.sh/uv/)
