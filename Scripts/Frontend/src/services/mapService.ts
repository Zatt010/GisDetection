import axios from 'axios';
// @ts-ignore
import parseGeoraster from 'georaster';
// @ts-ignore
import GeoRasterLayer from 'georaster-layer-for-leaflet';

export interface PredictionResult {
    analisis_hectareas: any;
    processed_file_url: string;
}

export interface LayerResult {
    hectareas: any;
    processedFileId: string;
    layer: any;
}

export class MapService {
    static async createPredictionLayer(
        responseData: PredictionResult,
        mapInstance: any
    ): Promise<LayerResult> {
        const { analisis_hectareas, processed_file_url } = responseData;
        
        const fileName = processed_file_url.split('/').pop() || '';
        const processedResp = await axios.get(processed_file_url, { responseType: 'arraybuffer' });
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

        return {
            hectareas: analisis_hectareas,
            processedFileId: fileName,
            layer
        };
    }

    static createTileLayer(tileUrl: string, opacity: number): any {
        // @ts-ignore
        return L.tileLayer(tileUrl, {
            opacity,
            maxZoom: 19,
            minZoom: 10,
        });
    }

    static fitMapToBounds(mapInstance: any, bounds: { south: number; west: number; north: number; east: number }) {
        if (!mapInstance) return;
        
        // @ts-ignore
        const leafletBounds = L.latLngBounds(
            [bounds.south, bounds.west],
            [bounds.north, bounds.east]
        );
        mapInstance.fitBounds(leafletBounds, { padding: [20, 20] });
    }
}

export const mapService = MapService;
