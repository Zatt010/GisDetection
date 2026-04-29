import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

def train_catboost():
    X, y = prepare_data_tabular()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        custom_metric=['Accuracy'],
        task_type='CPU',
        verbose=10
    )
    
    print("Iniciando entrenamiento...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    print("Entrenamiento finalizado")

    y_pred = model.predict(X_test).flatten()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusion CatBoost')
    plt.ylabel('Realidad')
    plt.xlabel('Prediccion')
    plt.show()

    feature_importances = model.get_feature_importance()
    band_names = [f"Banda {i+1}" for i in range(len(feature_importances))]
    
    plt.figure(figsize=(8, 5))
    plt.barh(band_names, feature_importances, color='coral')
    plt.xlabel('Importancia')
    plt.title('Importancia de las Bandas (CatBoost)')
    plt.show()

    save_path = os.path.join(BASE_PATH, 'modelo_catboost_comparativo.cbm')
    model.save_model(save_path)
    print(f"Modelo guardado en: {save_path}")

if __name__ == "__main__":
    train_catboost()