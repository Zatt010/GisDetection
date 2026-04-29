import React from 'react';
import type { ExportButtonsProps } from './types.ts';

const ExportButtons: React.FC<ExportButtonsProps> = ({
    processedFileId,
    loading,
    onExportVector
}) => {
    if (!processedFileId) return null;

    return (
        <div className="export-section" style={{ marginTop: '20px' }}>
            <h4>📥 Descargar Mapa de Predicciones</h4>
            <div className="export-grid">
                <button 
                    className="btn-primary"
                    onClick={() => onExportVector('shapefile')}
                    disabled={loading}
                >
                    📁 Shapefile (ZIP)
                </button>
                <button 
                    className="btn-secondary"
                    onClick={() => onExportVector('gpkg')}
                    disabled={loading}
                >
                    📦 GeoPackage
                </button>
                <button 
                    className="btn-secondary"
                    onClick={() => onExportVector('kml')}
                    disabled={loading}
                >
                    🌐 KML
                </button>
                <button 
                    className="btn-secondary"
                    onClick={() => onExportVector('kmz')}
                    disabled={loading}
                >
                    🗜️ KMZ
                </button>
            </div>
        </div>
    );
};

export default ExportButtons;
