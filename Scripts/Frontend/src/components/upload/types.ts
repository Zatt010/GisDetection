export interface UploadZoneProps {
    isDragging: boolean;
    loading: boolean;
    onDragOver: (e: React.DragEvent) => void;
    onDragLeave: () => void;
    onDrop: (e: React.DragEvent) => void;
    onFileInput: (e: React.ChangeEvent<HTMLInputElement>) => void;
    title: string;
    subtitle: string;
    icon: string;
    accept: string;
}
