import React, { useState, useEffect, useCallback } from 'react';
// @ts-ignore
import { MapContainer, TileLayer, Marker, Popup, useMap, FeatureGroup } from 'react-leaflet';
// @ts-ignore
import { EditControl } from "react-leaflet-draw";
import * as L from 'leaflet';

import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css'; 
// @ts-ignore
import parseGeoraster from 'georaster';
// @ts-ignore
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import axios from 'axios';
import './App.css'; 

// Fix para iconos de Leaflet
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png",
});

const droneIcon = new L.Icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/683/683214.png',
    iconSize: [35, 35],
});

const MapController = ({ setMap }: { setMap: (map: L.Map) => void }) => {
    const map = useMap();
    useEffect(() => { if (map) setMap(map); }, [map, setMap]);
    return null;
};

const App: React.FC = () => {
    // Estados de UI
    const [isDragging, setIsDragging] = useState(false);
    const [loading, setLoading] = useState(false);
    
    // Estados de Datos
    const [mapInstance, setMapInstance] = useState<L.Map | null>(null);
    const [hectareas, setHectareas] = useState<any>(null);
    const [droneMarkers] = useState<any[]>([]);
    const [tempCoords, setTempCoords] = useState<any>(null);
    const [imageInfo, setImageInfo] = useState<any>(null);

    // Helpers de Estilo
    const getBadgeClass = (clase: string) => {
        const c = clase.toLowerCase();
        if (c.includes('bosque')) return 'bg-bosque';
        if (c.includes('infraestructura')) return 'bg-infraestructura';
        if (c.includes('pastizal')) return 'bg-pastizales';
        if (c.includes('suelo')) return 'bg-suelo';
        if (c.includes('agricola')) return 'bg-agricola';
        return 'bg-secondary';
    };

    
    const processFile = useCallback(async (file: File) => {
        if (!file) return;
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            // Inferencia de IA
            const response = await axios.post('http://127.0.0.1:8000/predict_area/', formData);
            setHectareas(response.data.analisis_hectareas);

            // Renderizado de Raster si es TIF
            if (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')) {
                const processedResp = await axios.get(response.data.processed_file_url, { responseType: 'arraybuffer' });
                const georaster = await parseGeoraster(processedResp.data);
                const layer = new GeoRasterLayer({
                    georaster: georaster,
                    opacity: 0.7,
                    resolution: 256,
                    pixelValuesToColorFn: (values: any) => {
                        const pixel = values[0]; 
                        
                        
                        if (pixel === 99) return null; 
                        
                        
                        if (pixel === 0) return '#006400'; // Bosque
                        if (pixel === 1) return '#228B22'; // Matorrales
                        if (pixel === 2) return '#ADFF2F'; // Pastizales
                        if (pixel === 3) return '#FFFF00'; // T_Agricolas
                        if (pixel === 4) return '#FF0000'; // Infraestructura
                        if (pixel === 5) return '#8B4513'; // Suelo_Desnudo
                        if (pixel === 6) return '#0000FF'; // Agua
                        return null; 
                    }
                });
                
                if (mapInstance) {
                    mapInstance.eachLayer((l: any) => { if (l instanceof GeoRasterLayer) mapInstance.removeLayer(l); });
                    layer.addTo(mapInstance);
                    mapInstance.fitBounds(layer.getBounds());
                }
            }
        } catch (error) { 
            console.error("Error al procesar archivo:", error); 
            alert("Error al analizar la imagen.");
        } finally { 
            setLoading(false); 
        }
    }, [mapInstance]);

    // Handlers de Input y Drag
    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) processFile(file);
    };

    // --- Lógica de Google Earth Engine ---
    const _onCreated = async (e: any) => {
        const { layerType, layer } = e;
        if (layerType === 'polygon') {
            const leafletCoords = layer.getLatLngs()[0];
            const geeCoords = leafletCoords.map((latlng: any) => [latlng.lng, latlng.lat]);
            geeCoords.push(geeCoords[0]); 

            setTempCoords([geeCoords]);

            try {
                setLoading(true);
                const res = await axios.post('http://127.0.0.1:8000/search_recent_image/', { coords: [geeCoords] });
                if(res.data.status === "success") setImageInfo(res.data);
            } catch (err) {
                console.error("Error GEE:", err);
            } finally {
                setLoading(false);
            }
        }
    };

    const handleConfirmDownload = async () => {
        if (!tempCoords) return;
        setLoading(true);
        try {
            const res = await axios.post('http://127.0.0.1:8000/confirm_export/', { coords: tempCoords });
            if (res.data.status === "success") {
                window.open("https://code.earthengine.google.com/tasks", '_blank');
                setImageInfo(null);
                setTempCoords(null);
            }
        } catch (error) {
            alert("Error en la comunicación con el servidor.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>Sincronizando con satélites...</p>
                </div>
            )}

            <aside className="sidebar">
                <header>
                    <h2>🛰️ Tunari Guard</h2>
                    <p className="subtitle">Monitoreo Ambiental Pro</p>
                </header>

                {/* Zona de Carga con Drag & Drop */}
                <section 
                    className={`upload-zone ${isDragging ? 'dragging' : ''} ${loading ? 'disabled' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                >
                    <div className="upload-icon">🌍</div>
                    <div className="upload-text">
                        <strong>Analizar área local</strong>
                        <span>Arrastra TIF o haz clic aquí</span>
                    </div>
                    <input 
                        type="file" 
                        className="file-input-hidden"
                        accept=".tif, .tiff, .jpg, .jpeg" 
                        onChange={handleFileInput} 
                        disabled={loading} 
                    />
                </section>

                {/* Panel Sentinel GEE */}
                {imageInfo && (
                    <div className="confirm-box">
                        <div className="confirm-header">
                            <h4>Sentinel-2 Detectada</h4>
                            <button className="close-btn" onClick={() => setImageInfo(null)}>×</button>
                        </div>
                        <div className="info-grid">
                            <span>📅 {imageInfo.date}</span>
                            <span>☁️ {imageInfo.clouds}% nubes</span>
                        </div>
                        <button className="btn-primary" onClick={handleConfirmDownload}>
                            Exportar a Drive
                        </button>
                    </div>
                )}

                {/* Resultados de Cobertura */}
                {hectareas && (
                    <div className="results-container">
                        <h3>Análisis de Cobertura</h3>
                        <div className="results-list">
                            {Object.entries(hectareas).map(([clase, ha]: any) => (
                                <div key={clase} className="result-item">
                                    <span className={`clase-badge ${getBadgeClass(clase)}`}>
                                        {clase}
                                    </span>
                                    <span className="value">{ha} ha</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </aside>

            <main className="map-viewport">
                <MapContainer center={[-17.33, -66.22]} zoom={13}>
                    <MapController setMap={setMapInstance} />
                    <TileLayer 
                        url="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                        subdomains={['mt0','mt1','mt2','mt3']}
                        attribution="&copy; Google Satellite"
                    />

                    <FeatureGroup>
                        <EditControl
                            position='topleft'
                            onCreated={_onCreated}
                            draw={{
                                polygon: {
                                    allowIntersection: false,
                                    shapeOptions: { color: '#1a73e8', weight: 3 }
                                },
                                rectangle: true,
                                circle: false,
                                marker: true,
                                polyline: false,
                                circlemarker: false
                            }}
                        />
                    </FeatureGroup>
                    
                    {droneMarkers.map(m => (
                        <Marker key={m.id} position={m.pos as any} icon={droneIcon}>
                            <Popup><strong>Dron: {m.name}</strong></Popup>
                        </Marker>
                    ))}
                </MapContainer>
            </main>
        </div>
    );
};

export default App;