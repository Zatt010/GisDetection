import React from 'react';
import { UploadZoneProps } from './types.ts';

const UploadZone: React.FC<UploadZoneProps> = ({
    isDragging,
    loading,
    onDragOver,
    onDragLeave,
    onDrop,
    onFileInput,
    title,
    subtitle,
    icon,
    accept
}) => {
    return (
        <section 
            className={`upload-zone ${isDragging ? 'dragging' : ''} ${loading ? 'disabled' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
        >
            <div className="upload-icon">{icon}</div>
            <div className="upload-text">
                <strong>{title}</strong>
                <span>{subtitle}</span>
            </div>
            <input 
                type="file" 
                className="file-input-hidden"
                accept={accept} 
                onChange={onFileInput} 
                disabled={loading} 
            />
        </section>
    );
};

export default UploadZone;
