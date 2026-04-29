export interface ResultsDisplayProps {
    hectareas: any;
    getBadgeClass: (clase: string) => string;
}

export interface ExportButtonsProps {
    processedFileId: string | null;
    loading: boolean;
    onExportVector: (format: string) => void;
}
