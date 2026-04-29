import React from 'react';
import { TilingStatus } from '../../services/orthomosaicService';

interface LoadingOverlayProps {
    loading: boolean;
    message: string;
    progress?: TilingStatus | null;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ loading, message, progress }) => {
    if (!loading) return null;

    const progressPercentage = progress?.progress || 0;
    const messageLines = message.split('\n').filter(line => line.trim());

    return (
        <div className="loading-overlay">
            <div className="loading-content">
                <div className="spinner"></div>
                
                {progressPercentage > 0 && (
                    <div className="progress-container">
                        <div className="progress-bar">
                            <div 
                                className="progress-fill" 
                                style={{ width: `${progressPercentage}%` }}
                            ></div>
                        </div>
                        <span className="progress-text">{progressPercentage}%</span>
                    </div>
                )}
                
                <div className="loading-message">
                    {messageLines.map((line, index) => (
                        <p key={index}>{line}</p>
                    ))}
                </div>
                
                {progress && (
                    <div className="detailed-progress">
                        {progress.tiles_processed && progress.total_tiles && (
                            <p className="tiles-info">
                                Tiles: {progress.tiles_processed}/{progress.total_tiles}
                                {progress.tiles_per_second && (
                                    <span className="tiles-per-sec">
                                        ({progress.tiles_per_second} tiles/sec)
                                    </span>
                                )}
                            </p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default LoadingOverlay;
