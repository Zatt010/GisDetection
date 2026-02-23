import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA"
IMG_PATH = os.path.join(BASE_PATH, 'Tif', 'S2_Data.tif')
LABEL_PATH = os.path.join(BASE_PATH, 'Entrenamiento', 'Labels_Data.tif')

CLASS_NAMES = ['Bosque', 'Matorrales', 'Pastizales', 'Tierras_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']
CLASS_MAPPING = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6}

def prepare_data_tabular():
    print("Cargando datos...")
    
    with rasterio.open(IMG_PATH) as src:
        img = src.read().transpose(1, 2, 0)
        h, w, c = img.shape
        X = img.reshape(-1, c)
        X = np.nan_to_num(X).astype('float32') / 10000.0

    with rasterio.open(LABEL_PATH) as src:
        lbl = src.read(1)
        y = lbl.reshape(-1)

    valid_mask = np.isin(y, list(CLASS_MAPPING.keys()))
    
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    mp = np.vectorize(CLASS_MAPPING.get)
    y_mapped = mp(y_valid)

    return X_valid, y_mapped

def train_random_forest():
    X, y = prepare_data_tabular()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    print("Iniciando entrenamiento...")
    model.fit(X_train, y_train)
    print("Entrenamiento finalizado")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusion Random Forest')
    plt.ylabel('Realidad')
    plt.xlabel('Prediccion')
    plt.show()

    importances = model.feature_importances_
    band_names = [f"Banda {i+1}" for i in range(len(importances))]
    
    plt.figure(figsize=(8, 5))
    plt.barh(band_names, importances, color='teal')
    plt.xlabel('Importancia')
    plt.title('Importancia de las Bandas (Random Forest)')
    plt.show()

    save_path = os.path.join(BASE_PATH, 'modelo_rf_comparativo.joblib')
    joblib.dump(model, save_path)
    print(f"Modelo guardado en: {save_path}")

if __name__ == "__main__":
    train_random_forest()