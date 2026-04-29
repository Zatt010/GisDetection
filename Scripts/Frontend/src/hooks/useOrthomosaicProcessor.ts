import { useState, useCallback } from 'react';
import { mapService } from '../services/mapService';
import { orthomosaicService, TilingStatus } from '../services/orthomosaicService.ts';

export interface OrthomosaicProcessorState {
    orthomosaicLayer: boolean;
    tileLayerRef: any;
    orthomosaicOpacity: number;
    showOrthomosaic: boolean;
    loading: boolean;
    loadingMsg: string;
    tilingProgress: TilingStatus | null;
}

export interface OrthomosaicProcessorActions {
    processOrthomosaic: (file: File, mapInstance: any) => Promise<void>;
    toggleOrthomosaicVisibility: () => void;
    updateOrthomosaicOpacity: (newOpacity: number) => void;
    resetState: () => void;
}

export const useOrthomosaicProcessor = (): OrthomosaicProcessorState & OrthomosaicProcessorActions => {
    const [state, setState] = useState<OrthomosaicProcessorState>({
        orthomosaicLayer: false,
        tileLayerRef: null,
        orthomosaicOpacity: 0.8,
        showOrthomosaic: true,
        loading: false,
        loadingMsg: "Procesando...",
        tilingProgress: null
    });

    const updateState = useCallback((updates: Partial<OrthomosaicProcessorState>) => {
        setState(prev => ({ ...prev, ...updates }));
    }, []);

    const processOrthomosaic = useCallback(async (file: File, mapInstance: any) => {
        if (!file) return;
        
        console.log('🚀 Starting orthomosaic processing for file:', file.name);
        updateState({ loading: true, loadingMsg: "Subiendo archivo...", tilingProgress: null });

        try {
            console.log('📤 Calling orthomosaic service...');
            const result = await orthomosaicService.uploadAndProcess(
                file, 
                (status: TilingStatus) => {
                    console.log('📊 Progress callback received:', status);
                    
                    // Update progress with detailed information
                    const progressMsg = orthomosaicService.getProgressMessage(status);
                    const timeMsg = orthomosaicService.getTimeEstimate(status);
                    const detailedMsg = `${progressMsg}\n${timeMsg}`;
                    
                    console.log('💬 Progress message:', detailedMsg);
                    
                    if (status.tiles_per_second) {
                        console.log(`⚡ Tiling Progress: ${status.tiles_processed}/${status.total_tiles} tiles, ${status.tiles_per_second} tiles/sec`);
                    }
                    
                    console.log('🔄 Updating state with progress...');
                    updateState({ 
                        loadingMsg: detailedMsg,
                        tilingProgress: status
                    });
                }
            );
            
            console.log('✅ Orthomosaic processing completed, result:', result);
            
            // Remove existing tile layer
            if (state.tileLayerRef && mapInstance) {
                mapInstance.removeLayer(state.tileLayerRef);
            }

            // Add new tile layer
            if (mapInstance) {
                const newTileLayer = mapService.createTileLayer(result.tile_url, state.orthomosaicOpacity);
                newTileLayer.addTo(mapInstance);
                
                updateState({
                    tileLayerRef: newTileLayer,
                    orthomosaicLayer: true,
                    showOrthomosaic: true,
                    loadingMsg: "¡Ortomosaico cargado exitosamente!",
                    tilingProgress: null
                });

                mapService.fitMapToBounds(mapInstance, result.bounds);
            }

            // Reset loading message after a delay
            setTimeout(() => {
                updateState({ loadingMsg: "Procesando..." });
            }, 3000);

        } catch (error) {
            console.error('❌ Error in orthomosaic processing:', error);
            const msg = error instanceof Error ? error.message : JSON.stringify(error);
            updateState({ loadingMsg: `Error: ${msg}`, tilingProgress: null });
            alert(`Error: ${msg}`);
        } finally {
            console.log('🏁 Processing finished, setting loading to false');
            updateState({ loading: false });
        }
    }, [state.tileLayerRef, state.orthomosaicOpacity, updateState]);

    const toggleOrthomosaicVisibility = useCallback(() => {
        const { showOrthomosaic } = state;
        
        // This would need access to mapInstance - we'll handle this in the component
        updateState({ showOrthomosaic: !showOrthomosaic });
    }, [state.showOrthomosaic, updateState]);

    const updateOrthomosaicOpacity = useCallback((newOpacity: number) => {
        updateState({ orthomosaicOpacity: newOpacity });
        
        if (state.tileLayerRef) {
            state.tileLayerRef.setOpacity(newOpacity);
        }
    }, [state.tileLayerRef, updateState]);

    const resetState = useCallback(() => {
        setState({
            orthomosaicLayer: false,
            tileLayerRef: null,
            orthomosaicOpacity: 0.8,
            showOrthomosaic: true,
            loading: false,
            loadingMsg: "Procesando...",
            tilingProgress: null
        });
    }, []);

    return {
        ...state,
        processOrthomosaic,
        toggleOrthomosaicVisibility,
        updateOrthomosaicOpacity,
        resetState
    };
};
