import { useEffect } from 'react';
import { useMap } from 'react-leaflet';

interface MapControllerProps {
    setMap: (map: any) => void;
}

const MapController = ({ setMap }: MapControllerProps) => {
    const map = useMap();
    useEffect(() => { 
        if (map) setMap(map); 
    }, [map, setMap]);
    
    return null;
};

export default MapController;
