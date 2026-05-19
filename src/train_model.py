import logging
import argparse
import joblib
from mlflow.tracking import MlflowClient
import platform
import sklearn
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import yaml

## Configurar el registro
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

## Definir el parser de los argumentos
def parse_args():
    parser = argparse.ArgumentParser(description = "Entrenar y registrar el modelo")
    parser.add_argument("--config", type = str, required = True, help = "Ruta al archivo de configuración del modelo")
    parser.add_argument("--training", type = str, required = True, help = "Ruta al conjunto de entrenamiento procesado")
    parser.add_argument("--test", type = str, required = True, help = "Ruta al conjunto de prueba procesado")
    parser.add_argument("--models-dir", type = str, required = True, help = "Directorio para guardar el modelo entrenado")
    parser.add_argument("--mlflow-tracking-uri", type = str, default = None, help = "URI de seguimiento de MLflow")
    return parser.parse_args()

## Definir una función para cargar
## el modelo desde la configuración
def get_model_instance(name, params):
    model_map = {"LinearRegression": LinearRegression,
                 "KNeighbors": KNeighborsRegressor,
                 "RandomForest": RandomForestRegressor,
                 "GradientBoosting": GradientBoostingRegressor
                 #"XGBoost": xgb.XGBRegressor
                 }

    if name not in model_map:
        raise ValueError(f"Modelo no soportado: {name}")

    return model_map[name](**params)

## Función principal
def main(args):
    ## Cargar la configuración
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ## Extraer el modelo
    model_cfg = config["model"]

    ## Inicializar el tracking de MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg["name"])

    ## Cargar los datos de entrenamiento y de prueba
    train = pd.read_csv(args.training)
    test = pd.read_csv(args.test)

    ## Definir la(s) variable(s) objetivo
    target = model_cfg["target_variable"]

    ## Generar el conjunto de features y targets de entrenamiento
    X_train = train.drop(columns = [target])
    y_train = train[target].copy()

    ## Generar el conjunto de features y targets de prueba
    X_test = test.drop(columns = [target])
    y_test = test[target].copy()

    ## Obtener la instancia del model
    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])

    ## Iniciar la ejecución de MLflow
    with mlflow.start_run(run_name = "final-training"):
        ## Registro inicial
        logger.info(f"Entrenando modelo: {model_cfg["best_model"]}")

        ## Entrenar modelo
        model.fit(X_train, y_train)

        ## Obtener predicciones en el test set
        y_pred = model.predict(X_test)

        ## Obtener métricas de evaluación
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        ## Registrar parámetros y métricas
        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics({"mape": mape,
                            "mae": mae,
                            "rmse": rmse})

        ## Crear un log para el modelo
        ## junto con una URI distintiva
        mlflow.sklearn.log_model(model, "tuned_model")
        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        ## Registrar modelo en MLflow
        logger.info("Registrando modelo al Model Registry de MLflow")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  ## Ya existe

        ## Crear una versión del modelo
        model_version = client.create_model_version(name = model_name,
                                                    source = model_uri,
                                                    run_id = mlflow.active_run().info.run_id)

        ## Promover el modelo a etapa de pruebas
        client.transition_model_version_stage(name = model_name,
                                              version = model_version.version,
                                              stage = "Staging")

        ## Guardar el modelo localmente
        save_path = f"{args.models_dir}/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Model guardado en: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
