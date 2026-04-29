import React, { useState, useCallback } from 'react';
import axios from 'axios';
import exifr from 'exifr';
// @ts-ignore
import parseGeoraster from 'georaster';
// @ts-ignore
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import * as L from 'leaflet';

import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css'; 
import './App.css'; 

// Import extracted components
import { MapView } from './components/map';
import { UploadZone } from './components/upload';
import { ResultsDisplay, ExportButtons } from './components/results';
import { OrthomosaicControls } from './components/orthomosaic';
import { ImageSelector } from './components/gee';
import { LoadingOverlay } from './components/common';

// Fix para iconos de Leaflet
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png",
});

const App: React.FC = () => {
    // Estados de UI
    const [isDragging, setIsDragging] = useState(false);
    const [loading, setLoading] = useState(false);
    
    // Estados de Datos
    const [mapInstance, setMapInstance] = useState<any>(null);
    const [hectareas, setHectareas] = useState<any>(null);
    const [droneMarkers, setDroneMarkers] = useState<any[]>([]);
    const [tempCoords, setTempCoords] = useState<any>(null);
    const [imageInfo, setImageInfo] = useState<any>(null);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [processedFileId, setProcessedFileId] = useState<string | null>(null);
    const [geoJsonData, setGeoJsonData] = useState<any>(null);
    const [orthomosaicLayer, setOrthomosaicLayer] = useState<any>(null);
    const [orthomosaicOpacity, setOrthomosaicOpacity] = useState(0.8);
    const [showOrthomosaic, setShowOrthomosaic] = useState(true);
    const [loadingMsg, setLoadingMsg] = useState("Procesando...");
    const [tileLayerRef, setTileLayerRef] = useState<any>(null);

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
            const isJpg = file.name.toLowerCase().endsWith('.jpg') || file.name.toLowerCase().endsWith('.jpeg');
            const isTif = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
            const isGeoJson = file.name.toLowerCase().endsWith('.geojson');

            if (isGeoJson) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    if (event.target && event.target.result) {
                        const parsedData = JSON.parse(event.target.result as string);
                        setGeoJsonData(parsedData);
                        setLoading(false);
                    }
                };
                reader.readAsText(file);
                return;
            }

            if (isJpg) {
                const gpsData = await exifr.gps(file);
                if (gpsData && mapInstance) {
                    const { latitude, longitude } = gpsData;
                    mapInstance.flyTo([latitude, longitude], 19, {
                        animate: true,
                        duration: 1.5 
                    });
                    setDroneMarkers(prev => [...prev, { 
                        id: Date.now(), 
                        pos: [latitude, longitude], 
                        name: file.name 
                    }]);
                    setLoading(false);
                    return; 
                } else {
                    alert("Esta imagen no contiene metadatos GPS válidos.");
                    setLoading(false);
                    return;
                }
            }

            if (isTif) {
                const formData = new FormData();
                formData.append('file', file);
                
                const endpoint = 'http://localhost:8000/predict_area/';
                const response = await axios.post(endpoint, formData);
                
                setHectareas(response.data.analisis_hectareas);

                const fileUrl = response.data.processed_file_url;
                const fileName = fileUrl.split('/').pop(); 
                setProcessedFileId(fileName);

                const processedResp = await axios.get(fileUrl, { responseType: 'arraybuffer' });
                const georaster = await parseGeoraster(processedResp.data);
                
                const layer = new GeoRasterLayer({
                    georaster: georaster,
                    opacity: 0.7,
                    resolution: 256,
                    pixelValuesToColorFn: (values: any) => {
                        const pixel = values[0]; 
                        if (pixel === 99) return null; 
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
                    mapInstance.eachLayer((l: any) => { 
                        if (l instanceof GeoRasterLayer) mapInstance.removeLayer(l); 
                    });
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

    const processOrthomosaic = useCallback(async (file: File) => {
        if (!file) return;
        setLoading(true);
        setLoadingMsg("Subiendo archivo...");

        try {
            const isTif = file.name.toLowerCase().endsWith('.tif') || 
                        file.name.toLowerCase().endsWith('.tiff');
            if (!isTif) throw new Error("El archivo debe ser un TIF/TIFF georreferenciado");

            const formData = new FormData();
            formData.append('file', file);

            const uploadResp = await axios.post(
                'http://localhost:8000/upload_orthomosaic/', 
                formData
            );

            if (uploadResp.data.status === "error") {
                throw new Error(uploadResp.data.message);
            }

            const { job_id } = uploadResp.data;

            const result = await new Promise<any>((resolve, reject) => {
                const interval = setInterval(async () => {
                    try {
                        const statusResp = await axios.get(
                            `http://localhost:8000/tiling_status/${job_id}`
                        );
                        const data = statusResp.data;
                        console.log("Tiling status:", data.status);

                        if (data.status === "queued") {
                            setLoadingMsg("En cola, iniciando proceso...");
                        } else if (data.status === "reprojecting") {
                            setLoadingMsg("Reproyectando imagen...");
                        } else if (data.status === "tiling") {
                            setLoadingMsg("Generando tiles... puede tomar varios minutos");
                        } else if (data.status === "done") {
                            clearInterval(interval);
                            resolve(data);
                        } else if (data.status === "error") {
                            clearInterval(interval);
                            reject(new Error(data.message || "Error desconocido en el servidor"));
                        }

                    } catch (e) {
                        clearInterval(interval);
                        reject(e);
                    }
                }, 5000);
            });

            if (tileLayerRef && mapInstance) {
                mapInstance.removeLayer(tileLayerRef);
            }

            if (mapInstance) {
                const newTileLayer = L.tileLayer(result.tile_url, {
                    opacity: orthomosaicOpacity,
                    maxZoom: 19,
                    minZoom: 10,
                });

                newTileLayer.addTo(mapInstance);
                setTileLayerRef(newTileLayer);

                const leafletBounds = L.latLngBounds(
                    [result.bounds.south, result.bounds.west],
                    [result.bounds.north, result.bounds.east]
                );
                mapInstance.fitBounds(leafletBounds, { padding: [20, 20] });
            }

            setOrthomosaicLayer(true as any);
            setShowOrthomosaic(true);
            setLoadingMsg("Procesando...");

        } catch (error) {
            console.error("Error completo:", error);
            const msg = error instanceof Error ? error.message : JSON.stringify(error);
            alert(`Error: ${msg}`);
        } finally {
            setLoading(false);
        }
    }, [mapInstance, orthomosaicOpacity, tileLayerRef]);

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
    };

    const handleOrthomosaicInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processOrthomosaic(file);
    };

    const handleOrthomosaicDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (file && (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff'))) {
            processOrthomosaic(file);
        }
    };

    const toggleOrthomosaicVisibility = () => {
        if (tileLayerRef && mapInstance) {
            if (showOrthomosaic) {
                mapInstance.removeLayer(tileLayerRef);
            } else {
                mapInstance.addLayer(tileLayerRef);
            }
            setShowOrthomosaic(!showOrthomosaic);
        }
    };

    const updateOrthomosaicOpacity = (newOpacity: number) => {
        setOrthomosaicOpacity(newOpacity);
        if (tileLayerRef) {
            tileLayerRef.setOpacity(newOpacity);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) processFile(file);
    };

    const _onCreated = async (e: any) => {
        const { layerType, layer } = e;
        if (layerType === 'polygon') {
            const leafletCoords = layer.getLatLngs()[0];
            const geeCoords = leafletCoords.map((latlng: any) => [latlng.lng, latlng.lat]);
            geeCoords.push(geeCoords[0]); 

            setTempCoords([geeCoords]);

            try {
                setLoading(true);
                const res = await axios.post('http://localhost:8000/search_recent_image/', { coords: [geeCoords] });
                if(res.data.status === "success") setImageInfo(res.data);
            } catch (err) {
                console.error("Error GEE:", err);
            } finally {
                setLoading(false);
            }
        }
    };

    const handleConfirmDownload = async (imgId: string | null) => {
        if (!tempCoords || !imgId) return;
        setLoading(true);
        try {
            const res = await axios.post('http://localhost:8000/confirm_export/', { 
                coords: tempCoords,
                image_id: imgId 
            });
            
            if (res.data.status === "success") {
                window.open("https://code.earthengine.google.com/tasks", '_blank');
                setImageInfo(null);
                setTempCoords(null);
                setSelectedId(null);
            }
        } catch (error) {
            alert("Error al iniciar exportación.");
        } finally {
            setLoading(false);
        }
    };

    const handleExportVector = async (format: string) => {
        if (!processedFileId) {
            alert("Primero procesa una imagen con el modelo de predicción");
            return;
        }
        
        try {
            setLoading(true);
            const response = await axios.post(`http://localhost:8000/export_vector/${processedFileId}?formato=${format}`);
            
            if (response.data.status === "success") {
                window.open(response.data.download_url, '_blank');
            } else {
                alert(`Error al exportar: ${response.data.message}`);
            }
        } catch (error) {
            console.error("Export error:", error);
            alert("Error al conectar con el servidor de exportación");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <LoadingOverlay loading={loading} message={loadingMsg} />

            <aside className="sidebar">
                <header>
                    <h2>🛰️ GIS Detection</h2>
                    <p className="subtitle">Monitoreo Ambiental Pro</p>
                </header>

                <UploadZone
                    isDragging={isDragging}
                    loading={loading}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onFileInput={handleFileInput}
                    title="Analizar área local"
                    subtitle="Arrastra TIF o haz clic aquí"
                    icon="🌍"
                    accept=".tif, .tiff, .jpg, .jpeg"
                />

                <UploadZone
                    isDragging={false}
                    loading={loading}
                    onDragOver={(e) => { e.preventDefault(); }}
                    onDragLeave={() => {}}
                    onDrop={handleOrthomosaicDrop}
                    onFileInput={handleOrthomosaicInput}
                    title="Cargar Ortomosaico"
                    subtitle="Arrastra TIF georreferenciado"
                    icon="🗺️"
                    accept=".tif, .tiff"
                />

                {orthomosaicLayer && (
                    <OrthomosaicControls
                        orthomosaicOpacity={orthomosaicOpacity}
                        showOrthomosaic={showOrthomosaic}
                        onToggleVisibility={toggleOrthomosaicVisibility}
                        onOpacityChange={updateOrthomosaicOpacity}
                    />
                )}

                <ResultsDisplay
                    hectareas={hectareas}
                    getBadgeClass={getBadgeClass}
                />

                <ExportButtons
                    processedFileId={processedFileId}
                    loading={loading}
                    onExportVector={handleExportVector}
                />

                <ImageSelector
                    imageInfo={imageInfo}
                    selectedId={selectedId}
                    onImageSelect={setSelectedId}
                    onClose={() => setImageInfo(null)}
                    onConfirm={handleConfirmDownload}
                />
            </aside>

            <main className="map-viewport">
                <MapView
                    setMapInstance={setMapInstance}
                    droneMarkers={droneMarkers}
                    geoJsonData={geoJsonData}
                    onCreated={_onCreated}
                />
            </main>
        </div>
    );
};

export default App;
