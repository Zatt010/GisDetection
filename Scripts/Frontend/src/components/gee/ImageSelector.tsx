import React from 'react';
import { ImageSelectorProps } from './types.js';

const ImageSelector: React.FC<ImageSelectorProps> = ({
    imageInfo,
    selectedId,
    onImageSelect,
    onClose,
    onConfirm
}) => {
    if (!imageInfo || !imageInfo.options) return null;

    return (
        <div className="confirm-box">
            <div className="confirm-header">
                <h4>Seleccionar Imagen Sentinel-2</h4>
                <button className="close-btn" onClick={onClose}>×</button>
            </div>
            
            <div className="options-selector-container">
                {imageInfo.options.map((opt: any) => (
                    <div 
                        key={opt.id}
                        className={`option-item ${selectedId === opt.id ? 'selected' : ''} ${opt.is_ideal ? 'ideal' : ''}`}
                        onClick={() => onImageSelect(opt.id)}
                    >
                        <div className="option-radio">
                            <div className="radio-circle"></div>
                        </div>
                        <div className="option-details">
                            <span className="opt-date">📅 {opt.date}</span>
                            <span className={`opt-clouds ${parseFloat(opt.clouds) > 15 ? 'cloudy' : 'clear'}`}>
                                ☁️ {opt.clouds} nubes
                            </span>
                            {opt.is_ideal && <span className="best-tag">RECOMENDADA</span>}
                        </div>
                    </div>
                ))}
            </div>

            <button 
                className="btn-primary" 
                onClick={() => onConfirm(selectedId)}
                disabled={!selectedId}
            >
                Confirmar y Exportar
            </button>
        </div>
    );
};

export default ImageSelector;
