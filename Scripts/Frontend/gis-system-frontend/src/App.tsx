import React, { useState, useEffect } from 'react';
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
// @ts-ignore
import EXIF from 'exif-js';
import './App.css'; 

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
    const [mapInstance, setMapInstance] = useState<L.Map | null>(null);
    const [hectareas, setHectareas] = useState<any>(null);
    const [droneMarkers, setDroneMarkers] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [tempCoords, setTempCoords] = useState<any>(null);
    const [imageInfo, setImageInfo] = useState<any>(null);

    const getBadgeClass = (clase: string) => {
        const c = clase.toLowerCase();
        if (c.includes('bosque')) return 'bg-bosque';
        if (c.includes('infraestructura')) return 'bg-infraestructura';
        if (c.includes('pastizal')) return 'bg-pastizales';
        if (c.includes('suelo')) return 'bg-suelo';
        if (c.includes('agricola')) return 'bg-agricola';
        return 'bg-secondary';
    };

    
    const _onCreated = async (e: any) => {
        const { layerType, layer } = e;
        
        if (layerType === 'polygon') {
            
            const leafletCoords = layer.getLatLngs()[0];
            
            
            console.log("Polígono detectado");
            console.log("Coordenadas originales (Leaflet):", leafletCoords);

            // Convertir a formato GEE [lng, lat]
            const geeCoords = leafletCoords.map((latlng: any) => [latlng.lng, latlng.lat]);
            
            
            geeCoords.push(geeCoords[0]); 

            
            console.log("Coordenadas formateadas para GEE [lng, lat]:", geeCoords);

            setTempCoords([geeCoords]); // Guardar

            try {
                setLoading(true);
                
                
                const res = await axios.post('http://127.0.0.1:8000/search_recent_image/', { 
                    coords: [geeCoords] 
                });

                if(res.data.status === "success") {
                    console.log("Respuesta GEE recibida:", res.data);
                    setImageInfo(res.data);
                } else {
                    console.error("El backend respondió con error:", res.data.message);
                }
            } catch (err) {
                console.error("Error de conexión con el backend:", err);
            } finally {
                setLoading(false);
            }
        }
    };

    // 2. CONFIRMACION PARA GUARDAR EN DRIVE
    const handleConfirmDownload = async () => {
        if (!tempCoords) return;
        setLoading(true);
        try {
            const res = await axios.post('http://127.0.0.1:8000/confirm_export/', { coords: tempCoords });
            
            if (res.data.status === "success") {
                
                const monitoringUrl = "https://code.earthengine.google.com/tasks";
                
                alert(`¡Tareas de exportación enviadas!\n\nAl darle a OK, se abrirá el Gestor de Tareas de GEE para que veas el progreso del procesamiento.`);
                
                
                window.open(monitoringUrl, '_blank');
                
                setImageInfo(null);
                setTempCoords(null);
            } else {
                alert("Error: " + res.data.message);
            }
        } catch (error) {
            console.error("Error al exportar:", error);
            alert("Fallo en la comunicación con el servidor.");
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await axios.post('http://127.0.0.1:8000/predict_area/', formData);
            setHectareas(response.data.analisis_hectareas);

            if (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')) {
                const processedResp = await axios.get(response.data.processed_file_url, { responseType: 'arraybuffer' });
                const georaster = await parseGeoraster(processedResp.data);
                const layer = new GeoRasterLayer({
                    georaster: georaster,
                    opacity: 0.7,
                    resolution: 256,
                    pixelValuesToColorFn: (values: any) => {
                        const pixel = values[0]; 
                        if (pixel === 0) return '#006400'; 
                        if (pixel === 1) return '#228B22'; 
                        if (pixel === 2) return '#ADFF2F'; 
                        if (pixel === 3) return '#FFFF00'; 
                        if (pixel === 4) return '#FF0000'; 
                        if (pixel === 5) return '#8B4513'; 
                        if (pixel === 6) return '#0000FF'; 
                        return null; 
                    }
                });
                if (mapInstance) {
                    mapInstance.eachLayer((l: any) => { if (l instanceof GeoRasterLayer) mapInstance.removeLayer(l); });
                    layer.addTo(mapInstance);
                    mapInstance.fitBounds(layer.getBounds());
                }
            }
        } catch (error) { console.error("Error:", error); }
        finally { setLoading(false); }
    };

    return (
        <div className="app-container">
            {/* Pantalla de carga para procesos pesados de IA o GEE */}
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <h3 style={{marginTop: '15px', color: '#1a73e8'}}>Procesando en Google Earth Engine...</h3>
                </div>
            )}

            <div className="sidebar">
                <h2>🛰️ Tunari Guard</h2>
                <p style={{color: '#666', fontSize: '0.85rem', marginBottom: '20px'}}>
                    Interfaz Geoespacial para Monitoreo Ambiental
                </p>
                <hr style={{opacity: 0.1, marginBottom: '20px'}} />
                
                <div className="upload-section">
                    <label style={{fontSize: '0.9rem', fontWeight: 600, color: '#4a5568'}}>Subir Datos (TIF/JPG)</label>
                    <input 
                        type="file" 
                        className="file-input"
                        accept=".tif, .tiff, .jpg, .jpeg" 
                        onChange={handleFileUpload} 
                        disabled={loading} 
                    />

                    {/* PANEL DE CONFIRMACIÓN GEE*/}
                    {imageInfo && (
                        <div className="confirm-box" style={{
                            padding: '15px', 
                            background: '#e3f2fd', 
                            border: '1px solid #1e88e5', 
                            borderRadius: '8px', 
                            marginTop: '20px',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                        }}>
                            <h4 style={{margin: '0 0 10px 0', color: '#0d47a1', fontSize: '1rem'}}>🛰️ Imagen Sentinel-2 Detectada</h4>
                            <p style={{fontSize: '0.85rem', margin: '5px 0', color: '#333'}}>
                                <strong>Fecha más cercana:</strong> {imageInfo.date}
                            </p>
                            <p style={{fontSize: '0.85rem', margin: '5px 0', color: '#333'}}>
                                <strong>Nubosidad:</strong> {imageInfo.clouds}
                            </p>
                            
                            <button 
                                onClick={handleConfirmDownload}
                                style={{
                                    background: '#28a745', 
                                    color: 'white', 
                                    border: 'none', 
                                    padding: '10px', 
                                    borderRadius: '5px', 
                                    cursor: 'pointer', 
                                    width: '100%', 
                                    fontWeight: 'bold', 
                                    marginTop: '10px'
                                }}
                            >
                                Confirmar Descarga TIF
                            </button>
                            <button 
                                onClick={() => setImageInfo(null)}
                                style={{
                                    background: 'transparent', 
                                    color: '#666', 
                                    border: 'none', 
                                    fontSize: '0.75rem', 
                                    marginTop: '8px', 
                                    width: '100%', 
                                    cursor: 'pointer',
                                    textDecoration: 'underline'
                                }}
                            >
                                Cancelar y borrar selección
                            </button>
                        </div>
                    )}
                </div>

                {/* Resultados de la Inferencia U-Net */}
                {hectareas && (
                    <div className="results-container" style={{marginTop: '25px'}}>
                        <h3 style={{fontSize: '1.1rem', marginBottom: '15px', color: '#2d3748'}}>Cobertura Detectada</h3>
                        {Object.entries(hectareas).map(([clase, ha]: any) => (
                            <div key={clase} className="result-item">
                                <span className={`clase-badge ${getBadgeClass(clase)}`}>
                                    {clase}
                                </span>
                                <span style={{fontWeight: 700, color: '#2d3748'}}>{ha} ha</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div style={{ flex: 1, position: 'relative' }}>
                {/* Contenedor del Mapa Leaflet */}
                <MapContainer center={[-17.33, -66.22]} zoom={13} style={{ height: '100%', width: '100%' }}>
                    <MapController setMap={setMapInstance} />
                    <TileLayer 
                        url="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                        subdomains={['mt0','mt1','mt2','mt3']}
                        attribution="&copy; Google Satellite"
                    />

                    {/* Herramientas de Dibujo configuradas para puntos ilimitados */}
                    <FeatureGroup>
                        <EditControl
                            position='topleft'
                            onCreated={_onCreated}
                            draw={{
                                polygon: {
                                    allowIntersection: true, // Solución para evitar cortes
                                    showArea: true,
                                    metric: true,
                                    shapeOptions: {
                                        color: '#333333',
                                        fill: true,
                                        fillColor: '#666666',
                                        fillOpacity: 0.5,
                                        weight: 2
                                    },
                                    maxPoints: 0 // Puntos infinitos
                                },
                                rectangle: {
                                    shapeOptions: { fill: true, fillColor: '#666666', fillOpacity: 0.5 }
                                },
                                polyline: {
                                    shapeOptions: { color: '#333333', weight: 3 }
                                },
                                circle: false,
                                circlemarker: false,
                                marker: true,
                            }}
                        />
                    </FeatureGroup>
                    
                    {/* Renderizado de fotos de drones con GPS */}
                    {droneMarkers.map(m => (
                        <Marker key={m.id} position={m.pos as any} icon={droneIcon}>
                            <Popup>
                                <div style={{textAlign: 'center'}}>
                                    <strong>Dron DJI: {m.name}</strong><br />
                                    <img src={m.preview} width="180" style={{ marginTop: '10px', borderRadius: '4px' }} alt="preview" />
                                </div>
                            </Popup>
                        </Marker>
                    ))}
                </MapContainer>
            </div>
        </div>
    );
};

export default App;