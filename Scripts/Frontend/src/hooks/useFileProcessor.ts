import { useState, useCallback } from 'react';
import { fileService } from '../services/fileService.ts';
import { mapService } from '../services/mapService.ts';

export interface FileProcessorState {
    hectareas: any;
    droneMarkers: any[];
    geoJsonData: any;
    processedFileId: string | null;
    loading: boolean;
    loadingMsg: string;
}

export interface FileProcessorActions {
    processFile: (file: File, mapInstance: any) => Promise<void>;
    resetState: () => void;
}

export const useFileProcessor = (): FileProcessorState & FileProcessorActions => {
    const [state, setState] = useState<FileProcessorState>({
        hectareas: null,
        droneMarkers: [],
        geoJsonData: null,
        processedFileId: null,
        loading: false,
        loadingMsg: "Procesando..."
    });

    const updateState = useCallback((updates: Partial<FileProcessorState> | ((prev: FileProcessorState) => Partial<FileProcessorState>)) => {
        setState(prev => ({ 
            ...prev, 
            ...(typeof updates === 'function' ? updates(prev) : updates)
        }));
    }, []);

    const processFile = useCallback(async (file: File, mapInstance: any) => {
        if (!file) return;
        updateState({ loading: true });
        
        try {
            const fileType = fileService.getFileType(file.name);

            switch (fileType) {
                case 'geojson':
                    await fileService.processGeoJson(file, (geoJsonData: any) => {
                        updateState({ geoJsonData, loading: false });
                    });
                    break;

                case 'jpg':
                    await fileService.processJpg(file, mapInstance, (marker: any) => {
                        updateState((prev: any) => ({
                            droneMarkers: [...prev.droneMarkers, marker],
                            loading: false
                        }));
                    });
                    break;

                case 'tif':
                    await fileService.processTif(file, async (response: any) => {
                        const { hectareas, processedFileId } = await mapService.createPredictionLayer(
                            response,
                            mapInstance
                        );
                        
                        updateState({
                            hectareas,
                            processedFileId,
                            loading: false
                        });
                    });
                    break;

                default:
                    throw new Error("Tipo de archivo no soportado");
            }
            
        } catch (error) { 
            console.error("Error al procesar archivo:", error); 
            alert("Error al analizar la imagen.");
            updateState({ loading: false });
        }
    }, [updateState]);

    const resetState = useCallback(() => {
        setState({
            hectareas: null,
            droneMarkers: [],
            geoJsonData: null,
            processedFileId: null,
            loading: false,
            loadingMsg: "Procesando..."
        });
    }, []);

    return {
        ...state,
        processFile,
        resetState
    };
};
