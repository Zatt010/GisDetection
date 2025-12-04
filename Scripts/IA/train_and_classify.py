import os
import numpy as np
import rasterio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import colors
from rasterio.plot import show 



BASE_PATH = r"C:\Users\afuhe\OneDrive\Escritorio\materias\PG\Scripts\IA" 

# INPUT PATHS
TRAINING_SAMPLES_DIR = os.path.join(BASE_PATH, 'Entrenamiento') 
INPUT_IMAGE_FILE = os.path.join(BASE_PATH, 'Tif', 'S2_PNT_2025_7B.tif') 

# OUTPUT PATHS
OUTPUT_DIR = os.path.join(BASE_PATH, 'Classification_Results')
OUTPUT_CLASSIFIED_IMAGE = os.path.join(OUTPUT_DIR, 'RF_Classification_2025.tif')

CLASES_MAP_GEE = {
    'Cobertura_1_Bosque': 10,  # Arboles/vegetacion (WorldCover ID 10)
    'Cobertura_2_SueloDesnudo': 80, # Suelos desnudo (WorldCover ID 80)
    'Cobertura_3_Infraestructura': 50 # Infraestructura (WorldCover ID 50)
}

BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'] #7 Bandas

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_training_data(training_dir, class_map):
    print("\nStarting GEE CSV data loading...")
    
    all_pixels = []
    all_labels = []
    
    for folder_name, class_id in class_map.items():
        csv_filename = folder_name + '.csv'
        csv_path = os.path.join(training_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"WARNING: CSV file not found: {csv_path}. Skipping class {class_id}.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            pixels = df[BAND_NAMES].values
            
            labels = df['class'].values 
            
            all_pixels.extend(pixels)
            all_labels.extend(labels)
            
            print(f"   - Loaded {pixels.shape[0]} samples for class: {folder_name}")

        except Exception as e:
            print(f"   - Error reading {csv_filename}: {e}")

    if not all_pixels:
        raise ValueError("Training dataset is empty. Please verify the CSV files and paths.")

    pixels = np.array(all_pixels, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.uint8)

    print(f"Data loaded successfully. Total training pixels: {pixels.shape[0]}")
    return pixels, labels


def train_and_classify(pixels, labels, image_path, output_path, band_names):
    
    print("\nStarting Preprocessing and Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining Random Forest (RF) Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred)) 
    
    print(f"\n5. Classifying full image: {os.path.basename(image_path)}...")
    
    with rasterio.open(image_path) as src:
        metadata_original = src.profile
        
        imagen_masked = src.read(masked=True) 
        
        bands, height, width = imagen_masked.shape
        
        imagen_data = imagen_masked.filled(0).astype(np.float32)
        
        valid_mask = np.isfinite(imagen_data).all(axis=0) & np.any(imagen_data != 0, axis=0)
        
        print(f"   - Total Pixels: {height * width}")
        print(f"   - Valid Pixels: {np.count_nonzero(valid_mask)}")

        classified_image = np.zeros((height, width), dtype=np.int32)
        
        
        pixels_to_predict = imagen_data[:, valid_mask].transpose()
        
        if pixels_to_predict.shape[0] > 0:
            pixels_to_predict = scaler.transform(pixels_to_predict)
            
            pixels_to_predict = np.nan_to_num(pixels_to_predict)
            
            print("   - Prediciendo...")
            predictions = rf_model.predict(pixels_to_predict)
            
            classified_image[valid_mask] = predictions
        else:
            print("CRITICAL ERROR: No valid pixels found in the image.")

        metadata_original.update(
            dtype=rasterio.int32,
            count=1, 
            nodata=0 
        )

        with rasterio.open(output_path, 'w', **metadata_original) as dst:
            dst.write(classified_image.astype(rasterio.int32), 1)
            
    print(f" Classification saved to: {output_path}")


def visualize_classification(classified_image_path, classes_map):
    print("\n6. Starting Visualization...")
    
    color_map_raw = {
        10: '#006400',  # Bosque
        80: '#A0522D',  # Suelo
        50: '#FF0000',  # Infraestructura
    }
    
    class_ids = sorted(color_map_raw.keys())
    class_colors = [color_map_raw[id] for id in class_ids]
    
    inv_map = {v: k for k, v in classes_map.items()}
    class_labels = [inv_map[id] for id in class_ids]
    
    cmap = colors.ListedColormap(class_colors)
    # Fondo blanco para valores nulos ---
    cmap.set_bad(color='white') 
    
    bounds = class_ids + [class_ids[-1] + 1] 
    norm = colors.BoundaryNorm(bounds, cmap.N)

    try:
        with rasterio.open(classified_image_path) as src:
            classified_data = src.read(1)
            
            masked_data = np.ma.masked_where(classified_data == 0, classified_data)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_facecolor('white') 
            
            # Plot
            img = ax.imshow(masked_data, cmap=cmap, norm=norm, interpolation='nearest')
            
            cbar = plt.colorbar(img, ax=ax, ticks=np.array(class_ids) + 0.5, fraction=0.04, pad=0.04)
            cbar.ax.set_yticklabels(class_labels, va="center")
            
            plt.title(f"Final classication: {os.path.basename(classified_image_path)}")
            plt.show()
            
    except Exception as e:
        print(f"Error visualization: {e}")

if __name__ == "__main__":
    print("--- STARTING GEE-TRAINED CLASSIFICATION PIPELINE ---")
    
    try:
        X, Y = load_training_data(TRAINING_SAMPLES_DIR, CLASES_MAP_GEE)
        
        train_and_classify(X, Y, INPUT_IMAGE_FILE, OUTPUT_CLASSIFIED_IMAGE, BAND_NAMES)
        
        visualize_classification(OUTPUT_CLASSIFIED_IMAGE, CLASES_MAP_GEE)
        
        print("\n CLASSIFICATION COMPLETE.")
        
    except ValueError as ve:
        print(f"\n DATA ERROR: {ve}")
    except Exception as e:
        print(f"\n FATAL EXECUTION ERROR: {e}")