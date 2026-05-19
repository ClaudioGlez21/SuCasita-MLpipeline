import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train-model")
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenamiento y registro de modelos en MLflow")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--training", type=str, required=True, help="Ruta al dataset de entrenamiento procesado (CSV)")
    parser.add_argument("--test", type=str, required=True, help="Ruta al dataset de prueba procesado (CSV)")
    parser.add_argument("--models-dir", type=str, required=True, help="Directorio para guardar el artefacto local (.pkl)")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="URI del servidor de MLflow (opcional)")
    return parser.parse_args()

def load_config(config_path: Path) -> Dict[str, Any]:
    """Carga y valida el archivo de configuración YAML."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error al cargar el archivo de configuración {config_path}: {e}")
        sys.exit(1)

def get_model_instance(name: str, params: Dict[str, Any]) -> Any:
    """Retorna una instancia del modelo scikit-learn configurada."""
    model_map = {
        "LinearRegression": LinearRegression,
        "KNeighbors": KNeighborsRegressor,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor
    }

    if name not in model_map:
        raise ValueError(f"Modelo no soportado por el pipeline: '{name}'")

    return model_map[name](**params)

def prepare_data(train_path: Path, test_path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga los datasets y separa las features del target."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        logger.error(f"Error al encontrar los archivos de datos: {e}")
        sys.exit(1)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        logger.error(f"La variable objetivo '{target_col}' no se encuentra en los datasets.")
        sys.exit(1)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test

def main(
    config_path: Path, 
    train_path: Path, 
    test_path: Path, 
    models_dir: Path, 
    tracking_uri: str | None
) -> None:
    """Lógica principal de entrenamiento y registro orquestada para MLOps."""
    
    ## 1. Cargar Configuración
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "default_model")
    target_variable = model_cfg.get("target_variable", "price")
    best_model_name = model_cfg.get("best_model", "RandomForest")
    model_parameters = model_cfg.get("parameters", {})

    ## 2. Inicializar tracking de MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(model_name)

    ## 3. Cargar y preparar datos
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path, target_variable)

    ## 4. Obtener la instancia del modelo a entrenar
    model = get_model_instance(best_model_name, model_parameters)

    ## 5. Ejecución rastreada por MLflow
    with mlflow.start_run(run_name="final-training") as run:
        logger.info(f"Iniciando entrenamiento del modelo: {best_model_name}")

        ## Entrenar y Predecir
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ## Calcular Métricas
        metrics = {
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred)
        }

        ## Registro de Parámetros y Métricas
        mlflow.log_params(model_parameters)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, y_pred)
        
        model_reqs = ["scikit-learn==1.8.0", "skops==0.13.0", "pandas"]

        logger.info("Guardando modelo y registrándolo en MLflow...")
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            name="tuned_model", 
            signature=signature,
            registered_model_name=model_name, 
            serialization_format="skops",
            pip_requirements=model_reqs # <--- FIX: Le damos las dependencias en la mano
        )
        
        ## Asignar el Alias (Etiqueta de entorno)
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name, 
            alias="staging", 
            version=model_info.registered_model_version
        )

        ## Guardado del modelo local
        models_dir.mkdir(parents=True, exist_ok=True)
        local_save_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, local_save_path)
        
        logger.info(f"Pipeline finalizado. Artefacto local guardado en: {local_save_path}")

if __name__ == "__main__":
    ## Dependency Injection
    parsed_args = parse_args()
    
    main(
        config_path=Path(parsed_args.config),
        train_path=Path(parsed_args.training),
        test_path=Path(parsed_args.test),
        models_dir=Path(parsed_args.models_dir),
        tracking_uri=parsed_args.mlflow_tracking_uri
    )