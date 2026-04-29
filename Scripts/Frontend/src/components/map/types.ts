export interface MapViewProps {
    setMapInstance: (map: any) => void;
    droneMarkers: Array<{
        id: number;
        pos: [number, number];
        name: string;
    }>;
    geoJsonData: any;
    onCreated: (e: any) => void;
}
