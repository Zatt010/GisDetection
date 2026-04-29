# Backend - GIS Detection

Backend de servicios para el proyecto GIS Detection con integración de Google Earth Engine.

## Descripción General

El backend es responsable de procesar datos geoespaciales, gestionar solicitudes de la API, y proporcionar servicios de inteligencia artificial para el análisis de detecciones geográficas. Utiliza FastAPI como framework web principal y se integra con Google Earth Engine para el procesamiento de datos satelitales.

## Estructura de Carpetas

```
Backend/
├── gateway/                 # Gateway principal y punto de entrada
│   ├── gateway_main.py      # Archivo principal de la aplicación FastAPI
│   ├── Dockerfile          # Configuración Docker para el gateway
│   └── requirements.txt    # Dependencias específicas del gateway
├── services/               # Microservicios especializados
│   ├── ai_service/         # Servicio de IA y ML
│   ├── export_service/     # Servicio de exportación de datos
│   ├── gee_service/        # Servicio de Google Earth Engine
│   └── tiling_service/     # Servicio de procesamiento de mosaicos
├── models/                 # Modelos de datos y esquemas
│   └── modelo_unet_pro_final.keras  # Modelo de deep learning pre-entrenado
├── gee_credentials/        # Credenciales para Google Earth Engine
│   └── aifinal-480001-f32e66f5f6ec.json
├── tests/                  # Pruebas unitarias y de integración
├── docker-compose.yml      # Configuración Docker Compose
├── pyproject.toml         # Gestión de dependencias con UV
└── README.md              # Este archivo
```

## Instalación y Ejecución

### Prerrequisitos

- **Python**: Exactamente `3.9.25` (versión utilizada en el entorno conda)
- **Docker**: Versión 20.10+ (para contenerización)
- **UV**: Gestor de paquetes Python (instalar con `pip install uv`)
- **FastAPI**: Framework web para APIs

### Instalación con UV

1. **Clonar el repositorio y navegar al backend:**
   ```bash
   cd Backend
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

### Ejecución

#### Modo Desarrollo
```bash
uv run uvicorn gateway_main:app --reload --host 0.0.0.0 --port 8000
```

#### Modo Producción con Docker
```bash
docker-compose up --build
```

## 📖 Uso Básico

### Ejemplo de Uso - API Endpoint

Una vez iniciado el servidor, puedes acceder a:

1. **Documentación API**: http://localhost:8000/docs
2. **Health Check**: http://localhost:8000/health

#### Ejemplo de llamada a la API:
```python
import requests

# Verificar estado del servicio
response = requests.get("http://localhost:8000/health")
print(response.json())

# Ejemplo de procesamiento geoespacial
data = {
    "coordinates": [-74.0060, 40.7128],
    "date_range": ["2023-01-01", "2023-12-31"]
}
response = requests.post("http://localhost:8000/process", json=data)
```

### Uso de Servicios Específicos

#### Servicio de IA
```bash
# Ejecutar análisis de detección
uv run python services/ai_service/detection_processor.py
```

#### Servicio de Google Earth Engine
```bash
# Procesar datos satelitales
uv run python services/gee_service/satellite_processor.py
```

## Requisitos Técnicos

### Dependencias Principales

| Componente | Versión | Descripción |
|-------------|---------|-------------|
| **Python** | `3.9.25` | Versión exacta del intérprete |
| **FastAPI** | `0.123.5+` | Framework web principal |
| **UVicorn** | `0.38.0+` | Servidor ASGI |
| **Docker** | `20.10+` | Contenerización |
| **Google Earth Engine API** | `1.6.15` | Procesamiento satelital |

### Librerías Clave

- **Geoespaciales**: `geopandas`, `rasterio`, `shapely`, `pyproj`
- **Procesamiento**: `numpy`, `pandas`, `scikit-learn`
- **Cloud**: `google-cloud-storage`, `google-auth`
- **Web**: `requests`, `httpx`, `pydantic`

### Configuración Requerida

1. **Credenciales de Google Earth Engine**:
   - Colocar el archivo JSON en `gee_credentials/`
   - Configurar variable de entorno `GOOGLE_APPLICATION_CREDENTIALS`

2. **Variables de Entorno**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   export GEE_PROJECT_ID="your-project-id"
   export API_HOST="0.0.0.0"
   export API_PORT="8000"
   ```

## 🐳 Docker

### Construcción y Ejecución
```bash
# Construir imagen
docker build -t gis-backend .

# Ejecutar contenedor
docker run -p 8000:8000 gis-backend

# O con docker-compose
docker-compose up
```

### Docker Compose Services
- **gateway**: Servicio API principal
- **redis**: Caché y sesiones
- **postgres**: Base de datos (opcional)

## 🧪 Testing

```bash
# Ejecutar todas las pruebas
uv run pytest

# Ejecutar con cobertura
uv run pytest --cov=services

# Ejecutar pruebas específicas
uv run pytest tests/test_gateway.py
```

## Notas Importantes

1. **Versión Python**: Es crucial usar Python 3.9.25 para compatibilidad con TensorFlow y otras dependencias
2. **Memoria**: El servicio requiere mínimo 4GB RAM para procesamiento geoespacial
3. **Credenciales**: Las credenciales de GEE deben ser válidas y activas
4. **Red**: Asegurar conexión a internet para acceso a APIs de Google

## Troubleshooting

### Problemas Comunes

1. **Error de dependencias**:
   ```bash
   uv sync --refresh
   ```

2. **Error de credenciales GEE**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="Backend/gee_credentials/aifinal-480001-f32e66f5f6ec.json"
   ```

3. **Error de puerto en uso**:
   ```bash
   uv run uvicorn gateway_main:app --port 8001
   ```

## Documentación Adicional

- [Documentación FastAPI](https://fastapi.tiangolo.com/)
- [Documentación Google Earth Engine](https://developers.google.com/earth-engine)
- [Guía UV](https://docs.astral.sh/uv/)
