import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from patchify import patchify
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURACIÓN DE RUTAS ---
BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"
IMG_PATH = os.path.join(BASE_PATH, 'Tif', 'S2_Data.tif') 
LABEL_PATH = os.path.join(BASE_PATH, 'Entrenamiento', 'Labels_Data.tif')

PATCH_SIZE = 64
CHANNELS = 7 
CLASS_NAMES = ['Bosque', 'Matorrales', 'Pastizales', 'Tierras_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
CLASS_MAPPING = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6}
NUM_CLASSES = len(CLASS_MAPPING)

def prepare_data_unet():
    print("Cargando imágenes para segmentación U-Net...")
    with rasterio.open(IMG_PATH) as src:
        img = src.read().transpose(1, 2, 0)
        img = np.nan_to_num(img) / 10000.0 
        
    with rasterio.open(LABEL_PATH) as src:
        label = src.read(1)
        new_label = np.zeros(label.shape, dtype=np.uint8)
        for val, idx in CLASS_MAPPING.items():
            new_label[label == val] = idx

    min_h, min_w = min(img.shape[0], new_label.shape[0]), min(img.shape[1], new_label.shape[1])
    img, new_label = img[:min_h, :min_w, :], new_label[:min_h, :min_w]

    STEP = 32 
    img_patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, CHANNELS), step=STEP)
    label_patches = patchify(new_label, (PATCH_SIZE, PATCH_SIZE), step=STEP)

    X, Y = [], []
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            X.append(img_patches[i, j, 0])
            Y.append(label_patches[i, j])

    X, Y = np.array(X), np.array(Y)
    Y = np.expand_dims(Y, axis=-1)
    
    print(f"Total de pares de entrenamiento: {X.shape[0]}")
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def build_unet(input_shape=(64, 64, 7), num_classes=7):
    inputs = layers.Input(input_shape)
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    # Bridge
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    # Decoder
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo (Accuracy)')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

def plot_semantic_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Normalizar para ver porcentajes
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusión Normalizada (Píxel a Píxel)')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data_unet()
    model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, CHANNELS), num_classes=NUM_CLASSES)
    
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

    print("Iniciando entrenamiento de U-Net...")
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[lr_reducer, early_stop]
    )
    
    model.save('modelo_unet_final_tesis.keras')
    
    # 1. Gráficos de Entrenamiento
    plot_history(history)

    # 2. Evaluación y Matriz
    y_pred = model.predict(X_test)
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
    y_test_flat = y_test.flatten()
    
    print("\nReporte de Clasificación Píxel a Píxel:")
    print(classification_report(y_test_flat, y_pred_flat, target_names=CLASS_NAMES))
    
    plot_semantic_confusion_matrix(y_test_flat, y_pred_flat)