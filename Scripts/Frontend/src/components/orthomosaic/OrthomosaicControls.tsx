import React from 'react';
import type { OrthomosaicControlsProps } from './types.ts';

const OrthomosaicControls: React.FC<OrthomosaicControlsProps> = ({
    orthomosaicOpacity,
    showOrthomosaic,
    onToggleVisibility,
    onOpacityChange
}) => {
    return (
        <div className="orthomosaic-controls">
            <h4>🗺️ Controles del Ortomosaico</h4>
            
            <div className="control-row">
                <button 
                    className={`btn-toggle ${showOrthomosaic ? 'active' : ''}`}
                    onClick={onToggleVisibility}
                >
                    {showOrthomosaic ? 'Visible' : 'Oculto'}
                </button>
            </div>

            <div className="control-row">
                <label>Transparencia: {Math.round(orthomosaicOpacity * 100)}%</label>
                <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={orthomosaicOpacity * 100}
                    onChange={(e) => onOpacityChange(Number(e.target.value) / 100)}
                    className="opacity-slider"
                />
            </div>
        </div>
    );
};

export default OrthomosaicControls;
