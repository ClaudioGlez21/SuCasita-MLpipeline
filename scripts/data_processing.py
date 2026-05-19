import pandas as pd
import numpy as np
import logging
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from BorutaShap import BorutaShap

## Configuración del logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data-processing")

def column_ratio(X):
    """Realiza el cociente entre la primera y segunda columna de un arreglo."""
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    """Regresa el nombre de las variables de entrada con el sufijo 'ratio'."""
    return ["ratio"]

def load_data(file_path):
    """Carga los datos desde un archivo csv y asegura tipos de datos correctos."""
    logger.info(f"Cargando datos limpios desde {file_path}")
    df = pd.read_csv(file_path)
    if 'price' in df.columns:
        logger.info("Aplicando Data Type Casting a la variable objetivo (price)...")
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Eliminar posibles nulos que se generaron al forzar el numérico
        filas_antes = len(df)
        df = df.dropna(subset=['price']).reset_index(drop=True)
        filas_borradas = filas_antes - len(df)
        
        if filas_borradas > 0:
            logger.warning(f"Se eliminaron {filas_borradas} registros con precios inválidos o nulos.")    
    return df

def split_data(df, target_col="price"):
    """Realiza el split estratificado basado en cuantiles del precio."""
    logger.info("Realizando Train-Test Split estratificado")
    
    price_cuts = np.quantile(df[target_col], [0.2, 0.4, 0.6, 0.8])
    price_discretized = pd.cut(df[target_col],
                               bins=[-np.inf, price_cuts[0], price_cuts[1],
                                     price_cuts[2], price_cuts[3], np.inf],
                               labels=[1, 2, 3, 4, 5])
    
    train, test = train_test_split(df, test_size=0.3, stratify=price_discretized, random_state=42)
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col].copy()
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col].copy()
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline():
    """Construye y retorna el ColumnTransformer completo."""
    logger.info("Construyendo el pipeline de preprocesamiento")
    
    # Pipelines individuales
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )
    
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, np.exp, feature_names_out="one-to-one"),
        StandardScaler()
    )
    
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    
    # Orquestación final
    preprocessing = ColumnTransformer([
        ("default", default_num_pipeline, ["baths", "beds"]),
        ("baths", ratio_pipeline, ["baths", "sqft"]),
        ("beds", ratio_pipeline, ["beds", "sqft"]),
        ("log", log_pipeline, ["sqft"]),
        ("cat", cat_pipeline, make_column_selector(dtype_exclude=np.number))
    ], remainder=default_num_pipeline)
    
    return preprocessing

def process_data(input_file, output_dir, artifacts_dir):
    """
    Función orquestadora: Carga, divide, procesa, selecciona variables y guarda artefactos.
    """
    logger.info("Iniciando procesamiento de datos y feature engineering")
    
    # 1. Asegurar directorios
    out_path = Path(output_dir)
    art_path = Path(artifacts_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    art_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Carga y Split
    df = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Construir y ajustar el Pipeline de Preprocesamiento
    preprocessing = build_preprocessing_pipeline()
    logger.info("Ajustando el pipeline con datos de entrenamiento")
    X_train_prepared = preprocessing.fit_transform(X_train)
    X_test_prepared = preprocessing.transform(X_test)
    
    # Convertir de vuelta a DataFrames para BorutaShap
    feature_names = preprocessing.get_feature_names_out()
    X_train_prepared_df = pd.DataFrame(X_train_prepared, columns=feature_names, index=X_train.index)
    X_test_prepared_df = pd.DataFrame(X_test_prepared, columns=feature_names, index=X_test.index)
    
    # 4. Selección de características con BorutaSHAP
    logger.info("Ejecutando BorutaSHAP para selección de características (100 trials)...")
    feature_selector = BorutaShap(importance_measure="shap", classification=False)
    feature_selector.fit(X=X_train_prepared_df, y=y_train, n_trials=100)
    
    # Guardar gráfica de BorutaSHAP (Trazabilidad MLOps)
    logger.info("Generando y guardando gráfica de importancia de variables")
    feature_selector.plot(which_features="all")
    plt.savefig(art_path / "boruta_shap_importance.png", bbox_inches="tight")
    plt.close()
    
    # 5. Filtrar features seleccionados
    selected_features = feature_selector.Subset().columns.tolist()
    logger.info(f"Variables seleccionadas ({len(selected_features)}): {selected_features}")
    
    X_train_final = X_train_prepared_df[selected_features].copy()
    X_test_final = X_test_prepared_df[selected_features].copy()
    
    # Reintegrar el target para guardarlo
    df_train_final = X_train_final.copy()
    df_train_final["price"] = y_train
    
    df_test_final = X_test_final.copy()
    df_test_final["price"] = y_test
    
    # 6. Guardado de Datasets (CSV)
    train_out = out_path / "sacramento_train_processed.csv"
    test_out = out_path / "sacramento_test_processed.csv"
    df_train_final.to_csv(train_out, index=False)
    df_test_final.to_csv(test_out, index=False)
    logger.info(f"Datasets de entrenamiento y prueba guardados en {output_dir}")
    
    # 7. Persistencia de Artefactos (PKL para Inferencia en Tiempo Real)
    logger.info("Guardando artefactos de transformación (Pipelines y Features) para MLOps...")
    joblib.dump(preprocessing, art_path / "preprocessing_pipeline.pkl")
    joblib.dump(selected_features, art_path / "selected_features.pkl")
    logger.info(f"Artefactos guardados exitosamente en {artifacts_dir}")


if __name__ == "__main__":
    # Base path of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "sacramento_cleaned.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "models") # Carpeta para guardar los .pkl y gráficas
    
    process_data(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        artifacts_dir=ARTIFACTS_DIR
    )