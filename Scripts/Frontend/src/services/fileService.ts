import axios from 'axios';
import exifr from 'exifr';

export type FileType = 'jpg' | 'tif' | 'geojson' | 'unknown';

export class FileService {
    static getFileType(filename: string): FileType {
        const name = filename.toLowerCase();
        if (name.endsWith('.jpg') || name.endsWith('.jpeg')) return 'jpg';
        if (name.endsWith('.tif') || name.endsWith('.tiff')) return 'tif';
        if (name.endsWith('.geojson')) return 'geojson';
        return 'unknown';
    }

    static async processGeoJson(file: File, onSuccess: (geoJsonData: any) => void): Promise<void> {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (event.target && event.target.result) {
                const parsedData = JSON.parse(event.target.result as string);
                onSuccess(parsedData);
            }
        };
        reader.readAsText(file);
    }

    static async processJpg(
        file: File, 
        mapInstance: any, 
        onSuccess: (marker: { id: number; pos: number[]; name: string }) => void
    ): Promise<void> {
        const gpsData = await exifr.gps(file);
        if (gpsData && mapInstance) {
            const { latitude, longitude } = gpsData;
            mapInstance.flyTo([latitude, longitude], 19, {
                animate: true,
                duration: 1.5 
            });
            
            onSuccess({ 
                id: Date.now(), 
                pos: [latitude, longitude], 
                name: file.name 
            });
        } else {
            throw new Error("Esta imagen no contiene metadatos GPS válidos.");
        }
    }

    static async processTif(file: File, onSuccess: (response: any) => Promise<void>): Promise<void> {
        const formData = new FormData();
        formData.append('file', file);
        
        const endpoint = 'http://localhost:8000/predict_area/';
        const response = await axios.post(endpoint, formData);
        await onSuccess(response.data);
    }
}

export const fileService = FileService;
