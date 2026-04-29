import { useState, useCallback } from 'react';
import axios from 'axios';

export interface GeeState {
    tempCoords: any;
    imageInfo: any;
    selectedId: string | null;
    loading: boolean;
}

export interface GeeActions {
    handlePolygonCreated: (e: any) => Promise<void>;
    handleConfirmDownload: (imgId: string | null) => Promise<void>;
    setSelectedId: (id: string | null) => void;
    resetState: () => void;
}

export const useGoogleEarthEngine = (): GeeState & GeeActions => {
    const [state, setState] = useState<GeeState>({
        tempCoords: null,
        imageInfo: null,
        selectedId: null,
        loading: false
    });

    const updateState = useCallback((updates: Partial<GeeState>) => {
        setState(prev => ({ ...prev, ...updates }));
    }, []);

    const handlePolygonCreated = useCallback(async (e: any) => {
        const { layerType, layer } = e;
        if (layerType === 'polygon') {
            const leafletCoords = layer.getLatLngs()[0];
            const geeCoords = leafletCoords.map((latlng: any) => [latlng.lng, latlng.lat]);
            geeCoords.push(geeCoords[0]);

            updateState({ tempCoords: [geeCoords] });

            try {
                updateState({ loading: true });
                const res = await axios.post('http://localhost:8000/search_recent_image/', { coords: [geeCoords] });
                if(res.data.status === "success") {
                    updateState({ imageInfo: res.data });
                }
            } catch (err) {
                console.error("Error GEE:", err);
            } finally {
                updateState({ loading: false });
            }
        }
    }, [updateState]);

    const handleConfirmDownload = useCallback(async (imgId: string | null) => {
        if (!state.tempCoords || !imgId) return;
        
        updateState({ loading: true });
        try {
            const res = await axios.post('http://localhost:8000/confirm_export/', { 
                coords: state.tempCoords,
                image_id: imgId 
            });
            
            if (res.data.status === "success") {
                window.open("https://code.earthengine.google.com/tasks", '_blank');
                updateState({
                    imageInfo: null,
                    tempCoords: null,
                    selectedId: null
                });
            }
        } catch (error) {
            alert("Error al iniciar exportación.");
        } finally {
            updateState({ loading: false });
        }
    }, [state.tempCoords, updateState]);

    const setSelectedId = useCallback((id: string | null) => {
        updateState({ selectedId: id });
    }, [updateState]);

    const resetState = useCallback(() => {
        setState({
            tempCoords: null,
            imageInfo: null,
            selectedId: null,
            loading: false
        });
    }, []);

    return {
        ...state,
        handlePolygonCreated,
        handleConfirmDownload,
        setSelectedId,
        resetState
    };
};
