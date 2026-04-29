import os
import numpy as np
import rasterio
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import colors

BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"
MODEL_PATH = os.path.join(BASE_PATH, 'modelo_xgboost_comparativo.json')
IMG_PATH = os.path.join(BASE_PATH, 'Tif', 'S2_Data.tif')

COLOR_MAP = ['#006400', '#228B22', '#ADFF2F', '#FFFF00', '#FF0000', '#8B4513', '#0000FF']
CLASS_NAMES = ['Bosque', 'Matorrales', 'Pastizales', 'Agricola', 'Infraestructura', 'Suelo', 'Agua']

def predict_xgboost_image():
    print(f"Cargando modelo desde: {MODEL_PATH}")
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    print(f"Leyendo imagen: {IMG_PATH}")
    with rasterio.open(IMG_PATH) as src:
        profile = src.profile
        img = src.read().transpose(1, 2, 0)
        h, w, c = img.shape
        
        X_flat = img.reshape(-1, c)
        
        X_flat = np.nan_to_num(X_flat).astype('float32') / 10000.0

    print(f"Realizando inferencia en {X_flat.shape[0]} pixeles...")
    
    preds_flat = model.predict(X_flat)
    
    final_map = preds_flat.reshape(h, w)
    
    print("Prediccion finalizada. Generando visualizacion...")

    plt.figure(figsize=(12, 10))
    
    cmap = colors.ListedColormap(COLOR_MAP)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(final_map, cmap=cmap, norm=norm)
    
    patches = [plt.Rectangle((0,0),1,1, color=COLOR_MAP[i]) for i in range(len(CLASS_NAMES))]
    plt.legend(patches, CLASS_NAMES, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title("Resultado Clasificacion XGBoost (Pixel-Based)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(BASE_PATH, 'resultado_xgboost.tif')
    profile.update(count=1, dtype=rasterio.uint8, nodata=99)
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(final_map.astype(rasterio.uint8), 1)
    print(f"Resultado guardado en: {save_path}")

if __name__ == "__main__":
    predict_xgboost_image()