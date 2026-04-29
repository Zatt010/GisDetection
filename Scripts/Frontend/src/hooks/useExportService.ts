import { useState, useCallback } from 'react';
import { exportService } from '../services/exportService';

export interface ExportState {
    loading: boolean;
    error: string | null;
}

export interface ExportActions {
    exportVector: (fileId: string, format: string) => Promise<void>;
    clearError: () => void;
}

export const useExportService = (): ExportState & ExportActions => {
    const [state, setState] = useState<ExportState>({
        loading: false,
        error: null
    });

    const exportVector = useCallback(async (fileId: string, format: string) => {
        if (!fileId) {
            setState({ error: "Primero procesa una imagen con el modelo de predicción", loading: false });
            return;
        }

        setState({ loading: true, error: null });

        try {
            const result = await exportService.exportVector({ format: format as any, fileId });
            
            if (result.status === 'success' && result.download_url) {
                window.open(result.download_url, '_blank');
            } else {
                setState({ error: result.message || 'Error al exportar', loading: false });
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Error desconocido';
            setState({ error: `Error al exportar: ${errorMessage}`, loading: false });
        } finally {
            setState(prev => ({ ...prev, loading: false }));
        }
    }, []);

    const clearError = useCallback(() => {
        setState(prev => ({ ...prev, error: null }));
    }, []);

    return {
        ...state,
        exportVector,
        clearError
    };
};
