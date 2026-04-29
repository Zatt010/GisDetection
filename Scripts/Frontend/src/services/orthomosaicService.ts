import axios from 'axios';

export interface TilingStatus {
    status: 'queued' | 'reprojecting' | 'tiling' | 'done' | 'error' | 'not_found';
    progress: number;
    current_step?: string;
    start_time?: number;
    estimated_total_time?: number;
    estimated_remaining_time?: number;
    elapsed_time?: number;
    tiles_processed?: number;
    total_tiles?: number;
    tiles_per_second?: number;
    total_time?: number;
    message?: string;
    tile_url?: string;
    bounds?: {
        south: number;
        west: number;
        north: number;
        east: number;
    };
}

export interface TilingResult {
    tile_url: string;
    bounds: {
        south: number;
        west: number;
        north: number;
        east: number;
    };
}

export class OrthomosaicService {
    static async uploadAndProcess(
        file: File, 
        onProgress?: (status: TilingStatus) => void
    ): Promise<TilingResult> {
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

        const result = await new Promise<TilingResult>((resolve, reject) => {
            const interval = setInterval(async () => {
                try {
                    const statusResp = await axios.get(
                        `http://localhost:8000/tiling_status/${job_id}`
                    );
                    const data: TilingStatus = statusResp.data;

                    // Call progress callback if provided
                    if (onProgress) {
                        onProgress(data);
                    }

                    if (data.status === "done") {
                        clearInterval(interval);
                        resolve({
                            tile_url: data.tile_url!,
                            bounds: data.bounds!
                        });
                    } else if (data.status === "error") {
                        clearInterval(interval);
                        reject(new Error(data.message || "Error desconocido en el servidor"));
                    }

                } catch (e) {
                    clearInterval(interval);
                    reject(e);
                }
            }, 2000); // Update every 2 seconds for better UX
        });

        return result;
    }

    static formatTime(seconds: number): string {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    static getProgressMessage(status: TilingStatus): string {
        if (status.status === 'queued') {
            return 'En cola, iniciando proceso...';
        } else if (status.status === 'reprojecting') {
            return 'Reproyectando imagen...';
        } else if (status.status === 'tiling') {
            if (status.tiles_processed && status.total_tiles) {
                const percentage = Math.round((status.tiles_processed / status.total_tiles) * 100);
                return `Generando tiles: ${status.tiles_processed}/${status.total_tiles} (${percentage}%)`;
            }
            return 'Generando tiles...';
        } else if (status.status === 'done') {
            return '¡Completado!';
        } else if (status.status === 'error') {
            return `Error: ${status.message || 'Error desconocido'}`;
        }
        return status.current_step || 'Procesando...';
    }

    static getTimeEstimate(status: TilingStatus): string {
        if (status.status === 'done' && status.total_time) {
            return `Completado en ${this.formatTime(status.total_time)}`;
        }
        
        if (status.estimated_remaining_time !== undefined && status.estimated_remaining_time > 0) {
            return `Tiempo restante: ${this.formatTime(status.estimated_remaining_time)}`;
        }
        
        if (status.elapsed_time && status.elapsed_time > 0) {
            return `Tiempo transcurrido: ${this.formatTime(status.elapsed_time)}`;
        }
        
        return 'Estimando tiempo...';
    }
}

export const orthomosaicService = OrthomosaicService;
