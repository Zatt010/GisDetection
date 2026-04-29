import React from 'react';
// @ts-ignore
import { MapContainer, TileLayer, Marker, Popup, FeatureGroup, GeoJSON } from 'react-leaflet';
// @ts-ignore
import { EditControl } from "react-leaflet-draw";
import * as L from 'leaflet';

import MapController from './MapController';
import type { MapViewProps } from './types.ts';

const droneIcon = new L.Icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/683/683214.png',
    iconSize: [35, 35],
});

const MapView: React.FC<MapViewProps> = ({
    setMapInstance,
    droneMarkers,
    geoJsonData,
    onCreated
}) => {
    return (
        <MapContainer center={[-17.33, -66.22]} zoom={13}>
            <MapController setMap={setMapInstance} />
            <TileLayer 
                url="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                subdomains={['mt0','mt1','mt2','mt3']}
                attribution="&copy; Google Satellite"
            />

            <FeatureGroup>
                <EditControl
                    position='topleft'
                    onCreated={onCreated}
                    draw={{
                        polygon: {
                            allowIntersection: false,
                            shapeOptions: { color: '#1a73e8', weight: 3 }
                        },
                        rectangle: true,
                        circle: false,
                        marker: true,
                        polyline: false,
                        circlemarker: false
                    }}
                />
            </FeatureGroup>
            
            {droneMarkers.map(m => (
                <Marker key={m.id} position={m.pos as any} icon={droneIcon}>
                    <Popup><strong>Dron: {m.name}</strong></Popup>
                </Marker>
            ))}
            
            {geoJsonData && (
                <GeoJSON data={geoJsonData} />
            )}
        </MapContainer>
    );
};

export default MapView;
