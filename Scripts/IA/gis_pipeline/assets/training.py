"""
Stage 2 — Prepare patches and train the U-Net Pro model.

Inputs  : S2_Data_v4.tif (13 channels) + Labels_Data_v4.tif
Outputs : modelo_unet_pro_v4.keras  +  training history JSON
"""
import os
import json
import numpy as np
import rasterio
from patchify import patchify
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from dagster import asset, AssetExecutionContext, Output, MetadataValue

from gis_pipeline.resources import PipelineConfig


# ── Data preparation ──────────────────────────────────────────────────────────

def _load_and_patch(config: PipelineConfig):
    """
    Load TIF files, normalize correctly (raw bands ÷ 10000, indices as-is),
    and slice into (64×64×13) patches with 50% overlap.
    """
    # ── Image ────────────────────────────────────────────────────────────
    with rasterio.open(config.img_path) as src:
        img_raw = src.read().transpose(1, 2, 0).astype(np.float32)

    img_normalized = np.zeros_like(img_raw)
    # Bands 0-8: raw Sentinel-2 reflectance (divide by 10 000)
    img_normalized[:, :, :9]  = np.nan_to_num(img_raw[:, :, :9])  / 10000.0
    # Bands 9-12: NDVI, NDWI, NDBI, BSI — already in [-1, 1]
    img_normalized[:, :, 9:]  = np.nan_to_num(img_raw[:, :, 9:])

    # ── Labels ───────────────────────────────────────────────────────────
    with rasterio.open(config.label_path) as src:
        label_raw = src.read(1)

    label = np.zeros(label_raw.shape, dtype=np.uint8)
    for worldcover_val, class_idx in config.class_mapping.items():
        label[label_raw == int(worldcover_val)] = class_idx

    # ── Align spatial dims ───────────────────────────────────────────────
    h = min(img_normalized.shape[0], label.shape[0])
    w = min(img_normalized.shape[1], label.shape[1])
    img_normalized = img_normalized[:h, :w, :]
    label          = label[:h, :w]

    # ── Patchify ─────────────────────────────────────────────────────────
    ps, step, ch = config.patch_size, config.patch_step, config.channels
    img_patches   = patchify(img_normalized, (ps, ps, ch), step=step)
    label_patches = patchify(label,           (ps, ps),     step=step)

    X, Y = [], []
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            X.append(img_patches[i, j, 0])
            Y.append(label_patches[i, j])

    X = np.array(X, dtype=np.float32)
    Y = np.expand_dims(np.array(Y, dtype=np.uint8), axis=-1)

    return X, Y


# ── Model architecture ────────────────────────────────────────────────────────

def _conv_block(x, filters: int):
    """Double conv + BN + ReLU block."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet_pro(input_shape: tuple, num_classes: int, lr: float) -> tf.keras.Model:
    """
    U-Net with 3 encoder levels + bottleneck + 3 decoder levels.
    Dropout at every pooling and bottleneck layer to prevent overfitting.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = _conv_block(inputs, 64);  p1 = layers.Dropout(0.2)(layers.MaxPooling2D()(c1))
    c2 = _conv_block(p1, 128);     p2 = layers.Dropout(0.2)(layers.MaxPooling2D()(c2))
    c3 = _conv_block(p2, 256);     p3 = layers.Dropout(0.2)(layers.MaxPooling2D()(c3))

    # Bottleneck
    c4 = layers.Dropout(0.3)(_conv_block(p3, 512))

    # Decoder
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    c5 = _conv_block(layers.concatenate([u5, c3]), 256)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    c6 = _conv_block(layers.concatenate([u6, c2]), 128)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    c7 = _conv_block(layers.concatenate([u7, c1]), 64)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c7)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Dagster asset ─────────────────────────────────────────────────────────────

@asset(
    group_name="model_training",
    description=(
        "Trains the U-Net Pro model on 13-channel Sentinel-2 patches. "
        "Saves the model and training history."
    ),
)
def train_unet(context: AssetExecutionContext, config: PipelineConfig) -> Output[dict]:

    # ── Validate inputs ───────────────────────────────────────────────────
    for path, name in [(config.img_path, "Image TIF"), (config.label_path, "Labels TIF")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at: {path}\n"
                "Run the 'gee_export' asset first and download the files from Google Drive."
            )

    # ── Load data ─────────────────────────────────────────────────────────
    context.log.info("Loading and patching raster data...")
    X, Y = _load_and_patch(config)
    context.log.info(f"Total patches: {len(X)}  |  shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=config.test_size, random_state=42
    )
    context.log.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Build model ───────────────────────────────────────────────────────
    input_shape = (config.patch_size, config.patch_size, config.channels)
    context.log.info(f"Building U-Net Pro  →  input {input_shape}, classes {config.num_classes}")
    model = build_unet_pro(input_shape, config.num_classes, config.learning_rate)
    context.log.info(f"Parameters: {model.count_params():,}")

    # ── Callbacks ─────────────────────────────────────────────────────────
    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=config.model_output_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    context.log.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test, y_test),
        callbacks=cb_list,
        verbose=1,
    )

    # ── Persist history ───────────────────────────────────────────────────
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(config.base_path, "training_history_v4.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    best_val_acc  = float(max(history.history["val_accuracy"]))
    best_val_loss = float(min(history.history["val_loss"]))
    epochs_run    = len(history.history["accuracy"])

    context.log.info(f"Training done — best val_accuracy: {best_val_acc:.4f}")

    result = {
        "model_path":    config.model_output_path,
        "history_path":  history_path,
        "best_val_acc":  best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_run":    epochs_run,
        "total_patches": len(X),
        "train_patches": len(X_train),
        "test_patches":  len(X_test),
    }

    return Output(
        value=result,
        metadata={
            "best_val_accuracy":  MetadataValue.float(best_val_acc),
            "best_val_loss":      MetadataValue.float(best_val_loss),
            "epochs_run":         MetadataValue.int(epochs_run),
            "total_patches":      MetadataValue.int(len(X)),
            "model_saved_to":     MetadataValue.path(config.model_output_path),
            "channels":           MetadataValue.int(config.channels),
            "patch_size":         MetadataValue.int(config.patch_size),
        },
    )