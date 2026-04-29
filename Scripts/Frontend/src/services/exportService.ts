import axios from 'axios';

export interface ExportOptions {
    format: 'shapefile' | 'gpkg' | 'kml' | 'kmz';
    fileId: string;
}

export interface ExportResult {
    status: 'success' | 'error';
    download_url?: string;
    message?: string;
}

export class ExportService {
    private static readonly BASE_URL = 'http://localhost:8000';

    static async exportVector(options: ExportOptions): Promise<ExportResult> {
        try {
            const { format, fileId } = options;
            
            // Validate format
            const supportedFormats = ['shapefile', 'gpkg', 'kml', 'kmz'];
            if (!supportedFormats.includes(format)) {
                throw new Error(`Formato no soportado: ${format}`);
            }

            const response = await axios.post(
                `${this.BASE_URL}/export_vector/${fileId}?formato=${format}`
            );

            return response.data;
        } catch (error) {
            console.error('Export service error:', error);
            
            if (axios.isAxiosError(error)) {
                const message = error.response?.data?.message || error.message;
                return {
                    status: 'error',
                    message: `Error del servidor: ${message}`
                };
            }
            
            return {
                status: 'error',
                message: 'Error desconocido al exportar'
            };
        }
    }

    static getFormatDisplayName(format: string): string {
        const formatNames: Record<string, string> = {
            'shapefile': 'Shapefile (ZIP)',
            'gpkg': 'GeoPackage',
            'kml': 'KML',
            'kmz': 'KMZ'
        };
        return formatNames[format] || format;
    }

    static getFormatIcon(format: string): string {
        const formatIcons: Record<string, string> = {
            'shapefile': '📁',
            'gpkg': '📦',
            'kml': '🌐',
            'kmz': '🗜️'
        };
        return formatIcons[format] || '📄';
    }
}

export const exportService = ExportService;
