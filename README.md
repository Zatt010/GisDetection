# Sistema de Alerta Temprana para Detección de Cambios de Cobertura de Suelo - PNT

Este proyecto implementa un sistema automatizado para el monitoreo y detección de cambios en el uso y cobertura de suelo en el Parque Nacional Tunari (PNT), Cochabamba. El sistema integra la descarga de imágenes satelitales de Sentinel-2 (Google Earth Engine), el procesamiento mediante Redes Neuronales Convolucionales (Arquitectura U-Net y modelos comparativos como XGBoost/CatBoost) y una plataforma web interactiva estructurada en un entorno Frontend (React) y Backend (FastAPI).

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instaladas las siguientes herramientas en tu sistema:

* **Python:** Versión 3.9 (Recomendado gestionarlo mediante Miniconda/Anaconda).
* **Node.js:** Versión 16 o superior (Requerido para el entorno de React).
* **uv:** Gestor ultrarrápido de dependencias y proyectos en Python. [Instrucciones de instalación](https://docs.astral.sh/uv/).
* **Git:** Para clonar el repositorio.
* **Cuenta de Google Earth Engine (GEE):** Necesaria para autenticar la API y descargar imágenes satelitales.

---

## Instalación

Sigue estos pasos rigurosamente para levantar todo el entorno de desarrollo.

### 1. Clonar el repositorio
Abre tu terminal y ejecuta:
```bash
git clone https://github.com/Zatt010/GisDetection
```
### 2. Configuración del Módulo de Inteligencia Artificial (IA)
Este módulo se encarga del procesamiento geoespacial y la inferencia del modelo.

Navegar a la carpeta de IA
cd Scripts/IA

Inicializar y sincronizar dependencias usando uv
uv sync

Autenticar Google Earth Engine (Solo es necesario la primera vez)
uv run earthengine authenticate

### 3. Configuración del Servidor Backend (API)
El backend sirve los datos procesados y expone los endpoints necesarios para el frontend.
Volver a la carpeta raíz y entrar al Backend
cd ../Backend

Inicializar y sincronizar dependencias de la API
uv sync

### 4. Configuración del Cliente Frontend (React)
La interfaz de usuario web para visualizar los mapas y las alertas.

Volver a la carpeta raíz y entrar al Frontend
cd ../Frontend

Instalar los paquetes de Node
npm install 
# (o usar 'yarn install' / 'pnpm install' dependiendo de tu gestor)

#  Uso (Flujo de Trabajo Paso a Paso)

El flujo de trabajo principal de este proyecto se ejecuta en etapas secuenciales. Dado el volumen de los datos geoespaciales, la extracción inicial se procesa en la nube y requiere una transferencia manual antes de la clasificación local.

### 1. Generación y Extracción de Datos (Google Earth Engine)
Este script se conecta a GEE, procesa el área de interés del Parque Nacional Tunari (PNT) y envía las imágenes satelitales de Sentinel-2 a tu cuenta de Google Drive.

```bash
cd Scripts/IA
uv run python gee_dataset_generator.py
```
Paso Manual Importante: Ingresa a tu Google Drive, descarga las imágenes .tif que el script acaba de generar y guárdalas en tu carpeta local de trabajo (por ejemplo, dentro de Scripts/IA/Tif/).

### 2. Entrenamiento del Modelo (Opcional)
Si necesitas reentrenar la Red Neuronal desde cero con los datos recién descargados, ejecuta el script de entrenamiento:
uv run python train_cnn_landcover.py

(Nota: El repositorio ya incluye el modelo final entrenado modelo_unet_final_tesis.keras, por lo que puedes saltar este paso si solo deseas hacer predicciones).

### 3. Generación de Mapas de Clasificación (Inferencia)
Una vez que las imágenes .tif están en tu disco local, ejecuta este script. Tomará las imágenes y el modelo pre-entrenado para generar los mapas finales de cobertura de suelo.

uv run python inferencia_final_tesis.py

Los mapas resultantes se guardarán en la carpeta de resultados (ej. Classification_Results/).

### 4. Iniciar el Servidor Backend (API)
Levanta la API que se encarga de leer los mapas generados y servirlos a la plataforma web.

cd ../Backend
uv run uvicorn api_backend:app --reload

La API estará disponible en: http://127.0.0.1:8000

Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

### 5. Iniciar la Interfaz Web (Frontend React)
Levanta la plataforma de usuario para visualizar los mapas y las alertas tempranas de forma interactiva.

cd ../Frontend
npm start
# (Usa 'npm run dev' si el proyecto fue creado con Vite)

La aplicación web estará disponible en: http://localhost:3000

# FORMA DAGSTER AI

Ejecutar el Pipeline de Datos e IA (Dagster)
El flujo de trabajo geoespacial está orquestado con Dagster. Esto permite visualizar el proceso de descarga y clasificación paso a paso.

Bash

cd Scripts/IA
uv run dagster dev
Acceso: Abre tu navegador en http://127.0.0.1:3000.

Acción: Ve a la pestaña "Assets" y haz clic en Materialize All para iniciar la descarga de Sentinel-2 y la posterior inferencia con U-Net.


# Requisitos y Tecnologías Principales
Módulo IA (Python 3.9):

tensorflow: Entorno base para la Red Neuronal U-Net (modelo_unet_final_tesis.keras).

earthengine-api & geemap: Conexión y extracción de datasets satelitales.

geopandas, rasterio: Lectura y procesamiento de archivos .tif y geometrías espaciales.

dagster, dagster-webserver: Orquestación del pipeline de datos.

Módulo Backend (Python 3.9):

fastapi: Framework principal de la API.

uvicorn: Servidor web ASGI.

pytest: Framework para pruebas unitarias y de integración.

Módulo Frontend (Node.js):

react: Librería principal para la construcción de interfaces.

(Otras dependencias específicas estarán listadas en tu package.json, como librerías de mapas tipo Leaflet o Mapbox).


