import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"
MAPA_PATH = os.path.join(BASE_PATH, 'Classification_Results', 'Mapa_Final_UNET_Blanco.tif')

legend_dict = {
    'Bosque': [0, 100, 0],
    'Matorrales': [128, 128, 0],
    'Pastizales': [173, 255, 47],
    'T. Agrícolas': [255, 255, 0],
    'Infraestructura (ROJO)': [255, 0, 0],
    'Suelo Desnudo': [139, 69, 19],
    'Agua / Fondo': [255, 255, 255] 
}

def plot_final_result():
    if not os.path.exists(MAPA_PATH):
        print("Error: No se encuentra el archivo .tif")
        return

    with rasterio.open(MAPA_PATH) as src:
        img = src.read()
        img_display = np.transpose(img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(14, 10))
        
        ax.imshow(img_display)
        
        patches = [mpatches.Patch(color=np.array(c)/255, label=l) for l, c in legend_dict.items()]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  title="Leyenda de Cobertura", fontsize=10, shadow=True)

        ax.set_title("Mapa Final de Clasificación de Cobertura de Suelo (U-Net)\nParque Nacional Tunari - Gestión 2025", 
                     fontsize=15, fontweight='bold', pad=20)
        
        ax.set_xlabel("Longitud (Píxeles)")
        ax.set_ylabel("Latitud (Píxeles)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_final_result()