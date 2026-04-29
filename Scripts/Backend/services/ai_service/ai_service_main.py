import os
import io
import uuid
import numpy as np
import rasterio
import tensorflow as tf
from patchify import patchify
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="AI Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/modelo_unet_pro_final.keras")
TEMP_DIR   = os.getenv("TEMP_DIR",   "/app/temp_outputs")
os.makedirs(TEMP_DIR, exist_ok=True)

model = None
print(f"Loading model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"SUCCESS: Model loaded. Input: {model.input_shape}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
else:
    print(f"ERROR: Model not found at {MODEL_PATH}")


@app.post("/predict_area/")
async def predict_area(file: UploadFile = File(...)):
    if model is None:
        return {"status": "error", "message": "Model not loaded"}

    contents = await file.read()

    with rasterio.open(io.BytesIO(contents)) as src:
        profile      = src.profile
        img_raw      = src.read().transpose(1, 2, 0)
        res_x, res_y = src.res

    h, w, c    = img_raw.shape
    valid_mask = np.max(img_raw, axis=-1) > 0
    print(f"Image shape: {h}x{w}x{c}")

    if c == 3:
        padding = np.zeros((h, w, 4), dtype=img_raw.dtype)
        img_raw = np.concatenate([img_raw, padding], axis=-1)
        img     = np.nan_to_num(img_raw).astype(np.float32) * (10000.0 / 255.0)
    elif c == 7:
        img = np.nan_to_num(img_raw).astype(np.float32)
    else:
        return {"status": "error", "message": f"Canales no soportados: {c}"}

    img_normalized = img / 10000.0

    # Pad to multiple of 64, minimum 64x64
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    img_padded = np.pad(img_normalized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    print(f"Padded shape: {img_padded.shape}")

    patches      = patchify(img_padded, (64, 64, 7), step=32)
    output_probs = np.zeros((img_padded.shape[0], img_padded.shape[1], 7), dtype=np.float32)
    counts       = np.zeros((img_padded.shape[0], img_padded.shape[1], 1), dtype=np.float32)

    for i in range(patches.shape[0]):
        batch = patches[i, :, 0]
        if batch.shape[0] == 0:
            continue
        preds = model.predict(batch, verbose=0)
        for j in range(patches.shape[1]):
            y, x = i * 32, j * 32
            output_probs[y:y+64, x:x+64, :] += preds[j]
            counts[y:y+64, x:x+64]           += 1.0

    final_map              = np.argmax(output_probs / np.maximum(counts, 1.0), axis=-1).astype(np.uint8)
    final_map              = final_map[:h, :w]
    final_map[~valid_mask] = 99

    area_px = abs(res_x * res_y)
    classes = ["Bosque", "Matorrales", "Pastizales", "T_Agricolas", "Infraestructura", "Suelo_Desnudo", "Agua"]
    results = {
        name: round(float(np.sum((final_map == i) & valid_mask).item() * area_px / 10000.0), 2)
        for i, name in enumerate(classes)
    }

    result_id   = f"mask_{uuid.uuid4().hex}.tif"
    result_path = os.path.join(TEMP_DIR, result_id)
    new_profile = profile.copy()
    new_profile.update(count=1, dtype="uint8", nodata=99)

    with rasterio.open(result_path, "w", **new_profile) as dst:
        dst.write(final_map, 1)

    print(f"Prediction done. Results: {results}")
    return {
        "analisis_hectareas":   results,
        "processed_file_url":   f"http://localhost:8004/download/{result_id}",
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"status": "error", "message": "File not found"}

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "ai"}