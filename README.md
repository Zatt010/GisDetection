# GIS Detection - Proyecto Completo

Sistema integral de detección geoespacial utilizando inteligencia artificial y procesamiento de datos satelitales con Google Earth Engine.

## Descripción General

GIS Detection es un proyecto completo que combina backend de servicios, frontend web, y modelos de machine learning para el análisis y detección de patrones en datos geoespaciales. El sistema procesa imágenes satelitales, aplica modelos de deep learning para clasificación de land cover, y proporciona una interfaz interactiva para visualización y análisis.

## Estructura del Proyecto

```
Scripts/
├── Backend/                # Servicios backend y API
│   ├── gateway/            # Gateway principal FastAPI
│   ├── services/           # Microservicios especializados
│   ├── models/             # Modelos de datos y ML
│   ├── gee_credentials/    # Credenciales Google Earth Engine
│   ├── tests/              # Pruebas unitarias
│   ├── docker-compose.yml  # Configuración Docker
│   ├── pyproject.toml      # Dependencias UV
│   └── README.md           # Documentación backend
├── Frontend/               # Aplicación web React
│   ├── public/             # Assets estáticos
│   ├── src/                # Código fuente TypeScript/React
│   ├── package.json        # Dependencias npm
│   ├── tsconfig.json       # Configuración TypeScript
│   └── README.md           # Documentación frontend
├── IA/                     # Componentes de IA/ML
│   ├── Configuration/      # Configuraciones y dependencias
│   ├── Entrenamiento/      # Scripts de entrenamiento
│   ├── Classification_Results/ # Resultados
│   ├── Tif/               # Archivos TIFF geoespaciales
│   ├── definitions.py     # Utilidades comunes
│   ├── train_cnn_landcover.py # Entrenamiento CNN
│   ├── inferencia_final_tesis.py # Inferencia
│   ├── modelo_unet_pro_final.keras # Modelo pre-entrenado
│   ├── pyproject.toml     # Dependencias UV
│   └── README.md          # Documentación IA
├── UV_SETUP.md            # Guía de configuración UV
└── README.md              # Este archivo
```

## Instalación General

### Prerrequisitos Globales

- **Python**: `3.9.25` (exactamente esta versión)
- **Node.js**: `18.0.0+` para el frontend
- **UV**: Gestor de paquetes Python
- **Docker**: `20.10+` para contenerización
- **Git**: Para control de versiones

### Instalación de UV (si no está instalado)
```bash
pip install uv
```

## 🔧 Configuración por Componentes

### 1. Backend - Servicios API

```bash
# Navegar al backend
cd Backend

# Instalar dependencias
uv sync

# Activar entorno
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Iniciar servidor
uv run uvicorn gateway_main:app --reload --host 0.0.0.0 --port 8000
```

**Acceso**: http://localhost:8000/docs

### 2. IA/ML - Modelos de Inteligencia Artificial

```bash
# Navegar a IA
cd IA

# Instalar dependencias
uv sync

# Activar entorno
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Entrenar modelo
uv run python train_cnn_landcover.py

# Ejecutar inferencia
uv run python inferencia_final_tesis.py

# Iniciar Jupyter
uv run jupyter notebook
```

**Acceso Jupyter**: http://localhost:8888

### 3. Frontend - Interfaz Web

```bash
# Navegar al frontend
cd Frontend

# Instalar dependencias
npm install

# Iniciar desarrollo
npm run dev
```

**Acceso**: http://localhost:5173

## Flujo de Trabajo Completo

### 1. Inicialización del Sistema

```bash
# Terminal 1 - Backend
cd Backend
uv run uvicorn gateway_main:app --reload

# Terminal 2 - Frontend  
cd Frontend
npm run dev

# Terminal 3 - IA (opcional, para procesamiento)
cd IA
uv run jupyter notebook
```

### 2. Ejemplo de Uso Completo

#### Paso 1: Procesamiento de Datos con IA
```bash
# En la carpeta IA
uv run python inferencia_final_tesis.py \
  --input data/satellite_images/ \
  --output results/detections/ \
  --model modelo_unet_pro_final.keras
```

#### Paso 2: Consumo de API del Backend
```python
import requests

# Subir imagen para procesamiento
with open('satellite_image.tif', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/process',
        files={'file': f}
    )

# Obtener resultados
results = response.json()
print(f"Detecciones encontradas: {len(results['detections'])}")
```

#### Paso 3: Visualización en Frontend
```tsx
// En el componente React
import { useEffect, useState } from 'react';

function DetectionMap() {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/detections')
      .then(res => res.json())
      .then(data => setDetections(data));
  }, []);

  return (
    <MapComponent detections={detections} />
  );
}
```

## Requisitos Técnicos Detallados

### Stack Tecnológico

| Componente | Tecnología | Versión | Propósito |
|-------------|------------|---------|----------|
| **Backend** | Python | `3.9.25` | Servicios API y procesamiento |
| **Framework** | FastAPI | `0.123.5+` | API REST de alto rendimiento |
| **ML/AI** | TensorFlow | `2.20.0+` | Deep learning y modelos |
| **Frontend** | React | `18.2.0+` | Interfaz de usuario |
| **Tipado** | TypeScript | `5.0.0+` | Tipado estático frontend |
| **Build** | Vite | `5.0.0+` | Build tool frontend |
| **Geoespacial** | Google Earth Engine | `1.6.15` | Procesamiento satelital |
| **Container** | Docker | `20.10+` | Contenerización |

### Dependencias Críticas

#### Backend
- `fastapi`, `uvicorn` - Framework web
- `geopandas`, `rasterio` - Procesamiento geoespacial
- `google-cloud-storage` - Almacenamiento cloud
- `pydantic` - Validación de datos

#### IA/ML
- `tensorflow`, `keras` - Deep learning
- `opencv-python` - Visión por computadora
- `scikit-learn` - Machine learning tradicional
- `numpy`, `pandas` - Procesamiento de datos

#### Frontend
- `react`, `react-dom` - Framework UI
- `typescript` - Tipado estático
- `vite` - Build y desarrollo

### Configuración de Entorno

#### Variables de Entorno Globales
```bash
# Google Earth Engine
export GOOGLE_APPLICATION_CREDENTIALS="Backend/gee_credentials/aifinal-480001-f32e66f5f6ec.json"
export GEE_PROJECT_ID="your-gcp-project-id"

# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_DEBUG="true"

# Frontend
export VITE_API_URL="http://localhost:8000"
export VITE_MAP_API_KEY="your-map-api-key"
```

## 🐳 Docker - Despliegue Completo

### Docker Compose
```bash
# Construir y levantar todos los servicios
docker-compose up --build

# Solo backend
docker-compose up backend

# Ver logs
docker-compose logs -f
```

### Servicios Docker
- **backend**: API FastAPI con Python 3.9
- **frontend**: Nginx sirviendo React build
- **redis**: Caché y sesiones
- **postgres**: Base de datos persistente

## 🧪 Testing y Validación

### Tests del Backend
```bash
cd Backend
uv run pytest --cov=services
```

### Tests del Frontend
```bash
cd Frontend
npm run test
npm run test:coverage
```

### Validación de Modelos IA
```bash
cd IA
uv run python validate_model.py --model modelo_unet_pro_final.keras

```

## Monitoreo y Rendimiento

### Métricas del Backend
- **Health Check**: http://localhost:8000/health
- **Métricas**: http://localhost:8000/metrics
- **Docs API**: http://localhost:8000/docs

### Monitoreo de Modelos
- **Accuracy**: ~92% en land cover classification
- **Inference Time**: ~500ms por imagen 256x256
- **Memory Usage**: ~2GB RAM para inferencia

### Performance Frontend
- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <2s
- **Bundle Size**: ~500KB gzipped

## Troubleshooting General

### Problemas Comunes del Sistema

1. **Error de Conexión Backend-Frontend**:
   ```bash
   # Verificar CORS en backend
   # Revisar variables de entorno VITE_API_URL
   ```

2. **Error de Credenciales GEE**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="Backend/gee_credentials/aifinal-480001-f32e66f5f6ec.json"
   ```

3. **Error de Dependencias Python**:
   ```bash
   cd Backend  # o IA
   uv sync --refresh
   ```

4. **Error de Dependencias Node**:
   ```bash
   cd Frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

### Verificación del Sistema

```bash
# Script de verificación completo
#!/bin/bash
echo "Verificando sistema GIS Detection..."

# Backend
cd Backend && uv run python -c "import fastapi; print('✓ Backend OK')" && cd ..

# IA
cd IA && uv run python -c "import tensorflow; print('✓ TensorFlow OK')" && cd ..

# Frontend
cd Frontend && npm list react >/dev/null && echo "✓ Frontend OK" && cd ..

echo "Verificación completada"
```

##  Recursos y Documentación

### Documentación por Componente
- **Backend**: [Backend README](Backend/README.md)
- **IA/ML**: [IA README](IA/README.md)  
- **Frontend**: [Frontend README](Frontend/README.md)
- **UV Setup**: [UV Setup Guide](UV_SETUP.md)

### Enlaces Externos
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [React Documentation](https://react.dev/)
- [Google Earth Engine](https://developers.google.com/earth-engine)
- [UV Package Manager](https://docs.astral.sh/uv/)

## Contribución

### Estándares de Código
- **Python**: Seguir PEP 8, usar type hints
- **TypeScript**: Configuración ESLint estricta


Este proyecto se trata de un proyecto de grado de la Universidad Catolica Boliviana sin fines de lucro.

---

## Resumen Rápido

| Comando | Propósito | Puerto |
|---------|-----------|---------|
| `cd Backend && uv run uvicorn gateway_main:app --reload` | Iniciar backend | 8000 |
| `cd Frontend && npm run dev` | Iniciar frontend | 5173 |
| `cd IA && uv run jupyter notebook` | Iniciar Jupyter | 8888 |
| `docker-compose up` | Iniciar todo con Docker | Varios |

**URLs Importantes**:
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:5173
- Jupyter: http://localhost:8888
