import React, { useState } from 'react';
import * as L from 'leaflet';

import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css'; 
import './App.css'; 

// Import custom hooks
import { useFileProcessor } from './hooks/useFileProcessor';
import { useOrthomosaicProcessor } from './hooks/useOrthomosaicProcessor';
import { useGoogleEarthEngine } from './hooks/useGoogleEarthEngine';
import { useExportService } from './hooks/useExportService';

// Import extracted components
import { MapView } from './components/map';
import { UploadZone } from './components/upload';
import { ResultsDisplay, ExportButtons } from './components/results';
import { OrthomosaicControls } from './components/orthomosaic';
import { ImageSelector } from './components/gee';
import { LoadingOverlay } from './components/common';
import ErrorAlert from './components/common/ErrorAlert';

// Fix para iconos de Leaflet
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png",
});

const App: React.FC = () => {
    // UI State
    const [isDragging, setIsDragging] = useState(false);
    const [mapInstance, setMapInstance] = useState<any>(null);

    // Custom hooks for different functionalities
    const fileProcessor = useFileProcessor();
    const orthomosaicProcessor = useOrthomosaicProcessor();
    const geeProcessor = useGoogleEarthEngine();
    const exportService = useExportService();

    // Helper function for styling badges
    const getBadgeClass = (clase: string) => {
        const c = clase.toLowerCase();
        if (c.includes('bosque')) return 'bg-bosque';
        if (c.includes('infraestructura')) return 'bg-infraestructura';
        if (c.includes('pastizal')) return 'bg-pastizales';
        if (c.includes('suelo')) return 'bg-suelo';
        if (c.includes('agricola')) return 'bg-agricola';
        return 'bg-secondary';
    };

    // File handling functions
    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) fileProcessor.processFile(file, mapInstance);
    };

    const handleOrthomosaicInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) orthomosaicProcessor.processOrthomosaic(file, mapInstance);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) fileProcessor.processFile(file, mapInstance);
    };

    const handleOrthomosaicDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (file && (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff'))) {
            orthomosaicProcessor.processOrthomosaic(file, mapInstance);
        }
    };

    // Orthomosaic controls
    const toggleOrthomosaicVisibility = () => {
        if (orthomosaicProcessor.tileLayerRef && mapInstance) {
            if (orthomosaicProcessor.showOrthomosaic) {
                mapInstance.removeLayer(orthomosaicProcessor.tileLayerRef);
            } else {
                mapInstance.addLayer(orthomosaicProcessor.tileLayerRef);
            }
            orthomosaicProcessor.toggleOrthomosaicVisibility();
        }
    };

    // Export functionality - now handled by exportService hook

    // Determine overall loading state
    const isLoading = fileProcessor.loading || orthomosaicProcessor.loading || geeProcessor.loading;
    const loadingMessage = fileProcessor.loadingMsg || orthomosaicProcessor.loadingMsg || "Procesando...";
    const tilingProgress = orthomosaicProcessor.tilingProgress;

    return (
        <div className="app-container">
            <LoadingOverlay 
                loading={isLoading} 
                message={loadingMessage} 
                progress={tilingProgress}
            />
            <ErrorAlert message={exportService.error} onClose={exportService.clearError} />

            <aside className="sidebar">
                <header>
                    <h2>🛰️ GIS Detection</h2>
                    <p className="subtitle">Monitoreo Ambiental Pro</p>
                </header>

                <UploadZone
                    isDragging={isDragging}
                    loading={fileProcessor.loading}
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
                    loading={orthomosaicProcessor.loading}
                    onDragOver={(e) => { e.preventDefault(); }}
                    onDragLeave={() => {}}
                    onDrop={handleOrthomosaicDrop}
                    onFileInput={handleOrthomosaicInput}
                    title="Cargar Ortomosaico"
                    subtitle="Arrastra TIF georreferenciado"
                    icon="🗺️"
                    accept=".tif, .tiff"
                />

                {orthomosaicProcessor.orthomosaicLayer && (
                    <OrthomosaicControls
                        orthomosaicOpacity={orthomosaicProcessor.orthomosaicOpacity}
                        showOrthomosaic={orthomosaicProcessor.showOrthomosaic}
                        onToggleVisibility={toggleOrthomosaicVisibility}
                        onOpacityChange={orthomosaicProcessor.updateOrthomosaicOpacity}
                    />
                )}

                <ResultsDisplay
                    hectareas={fileProcessor.hectareas}
                    getBadgeClass={getBadgeClass}
                />

                <ExportButtons
                    processedFileId={fileProcessor.processedFileId}
                    loading={exportService.loading}
                    onExportVector={(format) => exportService.exportVector(fileProcessor.processedFileId!, format)}
                />

                <ImageSelector
                    imageInfo={geeProcessor.imageInfo}
                    selectedId={geeProcessor.selectedId}
                    onImageSelect={geeProcessor.setSelectedId}
                    onClose={() => geeProcessor.resetState()}
                    onConfirm={geeProcessor.handleConfirmDownload}
                />
            </aside>

            <main className="map-viewport">
                <MapView
                    setMapInstance={setMapInstance}
                    droneMarkers={fileProcessor.droneMarkers}
                    geoJsonData={fileProcessor.geoJsonData}
                    onCreated={geeProcessor.handlePolygonCreated}
                />
            </main>
        </div>
    );
};

export default App;
