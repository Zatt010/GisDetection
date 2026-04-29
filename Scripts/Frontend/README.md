# Frontend - GIS Detection

Interfaz de usuario web para el proyecto GIS Detection construida con React, TypeScript y Vite.

## Descripción General

El frontend es una aplicación web moderna que proporciona una interfaz interactiva para visualizar y analizar datos geoespaciales procesados por el backend. Incluye mapas interactivos, visualizaciones de detecciones, y herramientas para el análisis de datos satelitales.

## Estructura de Carpetas

```
Frontend/
├── public/                 # Archivos estáticos públicos
│   ├── vite.svg           # Logo de Vite
│   └── ...                # Otros assets públicos
├── src/                   # Código fuente principal
│   ├── assets/            # Imágenes y recursos estáticos
│   ├── components/        # Componentes React reutilizables
│   │   ├── MapComponent/  # Componentes de mapas
│   │   ├── DetectionPanel/ # Panel de detecciones
│   │   └── ...            # Otros componentes
│   ├── hooks/             # Hooks personalizados de React
│   ├── App.tsx            # Componente principal de la aplicación
│   ├── App.css            # Estilos principales
│   └── main.tsx           # Punto de entrada de la aplicación
├── .gitignore             # Archivos ignorados por Git
├── README.md              # Este archivo
├── eslint.config.js       # Configuración ESLint
├── index.html             # Plantilla HTML principal
├── package.json           # Dependencias y scripts del proyecto
├── tsconfig.app.json      # Configuración TypeScript para la app
├── tsconfig.node.json     # Configuración TypeScript para Node.js
└── tsconfig.json          # Configuración TypeScript general
```

## Instalación y Ejecución

### Prerrequisitos

- **Node.js**: Versión `18.0.0` o superior
- **npm**: Versión `8.0.0` o superior (o yarn)
- **Navegador moderno**: Chrome, Firefox, Safari, Edge

### Instalación

1. **Navegar a la carpeta Frontend:**
   ```bash
   cd Frontend
   ```

2. **Instalar dependencias:**
   ```bash
   npm install
   # o con yarn
   yarn install
   ```

### Ejecución

#### Modo Desarrollo
```bash
npm run dev
# o con yarn
yarn dev
```

La aplicación estará disponible en: http://localhost:5173

#### Modo Producción
```bash
# Construir para producción
npm run build
# o con yarn
yarn build

# Previsualizar producción
npm run preview
# o con yarn
yarn preview
```

## Uso Básico

### Navegación y Funcionalidades

Una vez iniciada la aplicación, tendrás acceso a:

1. **Mapa Interactivo Principal**: Visualización de datos geoespaciales
2. **Panel de Detecciones**: Herramientas para análisis de detecciones
3. **Control de Capas**: Gestión de capas de datos
4. **Filtros Temporales**: Selección de rangos de fechas

### Ejemplo de Uso - Componente de Mapa

```tsx
import React from 'react';
import { MapComponent } from './components/MapComponent';

function App() {
  return (
    <div className="app">
      <header>
        <h1>GIS Detection</h1>
      </header>
      <main>
        <MapComponent 
          center={[40.7128, -74.0060]}
          zoom={10}
          layers={['satellite', 'detections']}
        />
      </main>
    </div>
  );
}
```

### API Integration

```tsx
// Ejemplo de consumo de API del backend
import { useState, useEffect } from 'react';

interface DetectionData {
  id: string;
  coordinates: [number, number];
  confidence: number;
  timestamp: string;
}

export function DetectionPanel() {
  const [detections, setDetections] = useState<DetectionData[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/detections')
      .then(response => response.json())
      .then(data => setDetections(data))
      .catch(error => console.error('Error:', error));
  }, []);

  return (
    <div className="detection-panel">
      {detections.map(detection => (
        <div key={detection.id} className="detection-item">
          <span>Confianza: {detection.confidence}</span>
          <span>Coordenadas: {detection.coordinates.join(', ')}</span>
        </div>
      ))}
    </div>
  );
}
```

## Requisitos Técnicos

### Dependencias Principales

| Componente | Versión | Descripción |
|-------------|---------|-------------|
| **Node.js** | `18.0.0+` | Entorno de ejecución JavaScript |
| **React** | `18.2.0+` | Framework de frontend |
| **TypeScript** | `5.0.0+` | Superset tipado de JavaScript |
| **Vite** | `5.0.0+` | Herramienta de build y desarrollo |

### Librerías Clave

- **Framework**: `react`, `react-dom`
- **Tipado**: `typescript`, `@types/react`, `@types/node`
- **Build**: `vite`, `@vitejs/plugin-react`
- **Estilos**: `css`, módulos CSS
- **Mapas**: `leaflet`, `react-leaflet` (si aplica)
- **HTTP**: `axios` o `fetch` para API calls

### Configuración de Desarrollo

1. **Variables de Entorno** (crear `.env.local`):
   ```bash
   VITE_API_URL=http://localhost:8000
   VITE_MAP_API_KEY=your_map_api_key
   VITE_APP_TITLE=GIS Detection
   ```

2. **Configuración Vite** (`vite.config.ts`):
   ```typescript
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'

   export default defineConfig({
     plugins: [react()],
     server: {
       port: 5173,
       proxy: {
         '/api': {
           target: 'http://localhost:8000',
           changeOrigin: true
         }
       }
     }
   })
   ```

## Componentes Principales

### MapComponent
- Visualización de mapas interactivos
- Soporte para múltiples capas
- Controles de zoom y navegación
- Integración con APIs de mapas

### DetectionPanel
- Panel lateral de detecciones
- Filtros y búsqueda
- Visualización de resultados
- Exportación de datos

### DataVisualization
- Gráficos y estadísticas
- Visualizaciones temporales
- Análisis de tendencias
- Exportación de reportes

## Testing

### Ejecutar Tests
```bash
# Tests unitarios
npm run test
# o con yarn
yarn test

# Tests con cobertura
npm run test:coverage
# o con yarn
yarn test:coverage

# Tests E2E (si aplica)
npm run test:e2e
```

### Linting y Formato
```bash
# Linting con ESLint
npm run lint
# o con yarn
yarn lint

# Formato con Prettier (si configura)
npm run format
# o con yarn
yarn format
```

## Responsive Design

La aplicación está diseñada para funcionar en:

- **Desktop**: 1024px y superior
- **Tablet**: 768px - 1023px
- **Mobile**: 320px - 767px

SOLO se adapta a estos tamaños, no a otros y solo se tiene la certeza de que funciona al 100% en Desktop.
### Media Queries Example
```css
/* Mobile */
@media (max-width: 767px) {
  .map-container {
    height: 60vh;
  }
}

/* Tablet */
@media (min-width: 768px) and (max-width: 1023px) {
  .map-container {
    height: 70vh;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .map-container {
    height: 80vh;
  }
}
```

## Despliegue

### Build de Producción
```bash
npm run build
```
```bash
npm run dev
```

El resultado estará en la carpeta `dist/`.

#### Docker
```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Notas Importantes

1. **CORS**: Configurar CORS en el backend para permitir peticiones del frontend
2. **Environment Variables**: Usar variables de entorno para configuración sensible
3. **Performance**: Optimizar imágenes y assets para mejor rendimiento
4. **Accessibility**: Seguir WCAG 2.1 para accesibilidad

## Troubleshooting

### Problemas Comunes

1. **Error de dependencias**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Error de puerto en uso**:
   ```bash
   npm run dev -- --port 3000
   ```

3. **Error de TypeScript**:
   ```bash
   npm run type-check
   ```

4. **Error de API Connection**:
   - Verificar que el backend esté corriendo
   - Revisar configuración de CORS
   - Validar variables de entorno

## 📚 Documentación Adicional

- [Documentación React](https://react.dev/)
- [Documentación TypeScript](https://www.typescriptlang.org/)
- [Documentación Vite](https://vite.dev/)
- [Documentación ESLint](https://eslint.org/)
