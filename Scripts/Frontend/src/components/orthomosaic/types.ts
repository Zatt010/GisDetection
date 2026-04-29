export interface OrthomosaicControlsProps {
    orthomosaicOpacity: number;
    showOrthomosaic: boolean;
    onToggleVisibility: () => void;
    onOpacityChange: (newOpacity: number) => void;
}
