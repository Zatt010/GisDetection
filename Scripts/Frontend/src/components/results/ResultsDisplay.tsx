import React from 'react';
import { ResultsDisplayProps } from './types.ts';

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
    hectareas,
    getBadgeClass
}) => {
    if (!hectareas) return null;

    return (
        <div className="results-container">
            <h3>Análisis de Cobertura</h3>
            <div className="results-list">
                {Object.entries(hectareas).map(([clase, ha]: any) => (
                    <div key={clase} className="result-item">
                        <span className={`clase-badge ${getBadgeClass(clase)}`}>
                            {clase}
                        </span>
                        <span className="value">{ha} ha</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ResultsDisplay;
