export interface ImageSelectorProps {
    imageInfo: any;
    selectedId: string | null;
    onImageSelect: (id: string) => void;
    onClose: () => void;
    onConfirm: (selectedId: string | null) => void;
}
