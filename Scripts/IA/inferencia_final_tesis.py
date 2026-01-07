import os
import numpy as np
import rasterio
import tensorflow as tf
from patchify import patchify
from collections import Counter

# --- CONFIGURACION DE RUTAS ---
BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"
MODEL_PATH = os.path.join(BASE_PATH, 'modelo_unet_final_tesis.keras')
INPUT_TIF = os.path.join(BASE_PATH, 'Tif', 'S2_PNT_2025_7B.tif')
OUTPUT_DIR = os.path.join(BASE_PATH, 'Classification_Results')
OUTPUT_TIF = os.path.join(OUTPUT_DIR, 'Mapa_Final_UNET_Blanco.tif')

PATCH_SIZE = 64
CHANNELS = 7
STEP = 32 

CLASS_INFO = {
    0: {'nombre': 'Bosque', 'color': [0, 100, 0]},
    1: {'nombre': 'Matorrales', 'color': [128, 128, 0]},
    2: {'nombre': 'Pastizales', 'color': [173, 255, 47]},
    3: {'nombre': 'Tierras_Agricolas', 'color': [255, 255, 0]},
    4: {'nombre': 'Infraestructura', 'color': [255, 0, 0]},
    5: {'nombre': 'Suelo_Desnudo', 'color': [139, 69, 19]},
    6: {'nombre': 'Agua', 'color': [0, 0, 255]}
}

def run_final_inference():
    print("Cargando modelo y ejecutando limpieza de bordes...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encontró el modelo en {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    
    with rasterio.open(INPUT_TIF) as src:
        profile = src.profile
        res_x, res_y = src.res
        img_raw = src.read().transpose(1, 2, 0)
        
        # MÁSCARA BINARIA: 1 donde hay datos, 0 donde es negro absoluto
        valid_mask = np.max(img_raw, axis=-1) > 0 
        
        img = np.nan_to_num(img_raw) / 10000.0 
        
    h, w, _ = img.shape
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    patches = patchify(img_padded, (PATCH_SIZE, PATCH_SIZE, CHANNELS), step=STEP)
    p_h, p_w, _, _, _, _ = patches.shape
    
    output_probs = np.zeros((img_padded.shape[0], img_padded.shape[1], len(CLASS_INFO)), dtype=np.float32)
    counts = np.zeros((img_padded.shape[0], img_padded.shape[1], 1), dtype=np.float32)

    for i in range(p_h):
        row_patches = patches[i, :, 0] 
        preds = model.predict(row_patches, verbose=0, batch_size=16) 
        for j in range(p_w):
            y, x = i * STEP, j * STEP
            output_probs[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] += preds[j]
            counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    final_map = np.argmax(output_probs / np.maximum(counts, 1.0), axis=-1).astype(np.uint8)
    final_map = final_map[:h, :w]

    # --- LIMPIEZA TOTAL ---
    final_map[~valid_mask] = 99

    rgb_output = np.full((3, h, w), 255, dtype=np.uint8)
    
    pixel_counts = Counter(final_map[valid_mask].flatten())
    
    print("\nREPORTE DE COBERTURA REAL")
    area_pixel_m2 = abs(res_x * res_y)
    for idx, info in CLASS_INFO.items():
        mask = (final_map == idx)
        for b in range(3):
            rgb_output[b, mask] = info['color'][b]
        
        hectareas = (pixel_counts.get(idx, 0) * area_pixel_m2) / 10000.0
        print(f"- {info['nombre']}: {hectareas:.2f} ha")

    profile.update(dtype=rasterio.uint8, count=3, nodata=255)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with rasterio.open(OUTPUT_TIF, 'w', **profile) as dst:
        dst.write(rgb_output)
        
    print(f"\n¡ÉXITO!: {OUTPUT_TIF}")

if __name__ == "__main__":
    run_final_inference()