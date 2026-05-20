import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import make_column_selector, ColumnTransformer
import logging
import joblib
import sys
import os

## Cargar funciones para pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "api"))
from utils import column_ratio, ratio_name

## Configuración del logging
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data-processing")

## Función para realizar el procesamiento de datos
def create_preprocessor():
    """
    Crea una pipeline de preprocesamiento de datos.
    """

    ## Iniciando el procesamiento
    logger.info("Creando una pipeline de procesamiento")

    ## Pipeline para ratios
    logger.info("Pipeline de ratios creada")
    ratio_pipeline = make_pipeline(SimpleImputer(strategy = "median"),
                                   FunctionTransformer(column_ratio,
                                                       feature_names_out = ratio_name),
                                   StandardScaler())

    ## Pipeline para transformaciones logarítmicas
    logger.info("Pipeline logarítmica creada")
    log_pipeline = make_pipeline(SimpleImputer(strategy = "median"),
                                 FunctionTransformer(np.log, np.exp,
                                                     feature_names_out = "one-to-one"),
                                 StandardScaler())

    ## Pipeline numérico por defecto
    logger.info("Pipeline numérica por defecto creada")
    default_num_pipeline = make_pipeline(SimpleImputer(strategy = "median"),
                                         StandardScaler())

    ## Pipeline para variables categóricas
    logger.info("Pipeline categórica creada")
    cat_pipeline = make_pipeline(SimpleImputer(strategy = "most_frequent"),
                                 OneHotEncoder(handle_unknown = "ignore"))

    ## Pipeline final
    logger.info("Pipeline completa creada")
    preprocessing = ColumnTransformer([("default", default_num_pipeline, ["baths", "beds"]),
                                       ("baths", ratio_pipeline, ["baths", "sqft"]),
                                       ("beds", ratio_pipeline, ["beds", "sqft"]),
                                       ("log", log_pipeline, ["sqft"]),
                                       ("cat", cat_pipeline,
                                        make_column_selector(dtype_exclude=np.number))],
                                      remainder = default_num_pipeline)

    return preprocessing

## Ejecutar pipeline
def run_preprocessor(input_file, training_output_file, test_output_file, preprocessor_file):
    """
    Ejecuta la pipeline completa.
    """

    ## Cargar los datos
    logger.info(f"Cargando datos desde {input_file}")
    df = pd.read_csv(input_file)

    ## Crear la discretización del precio de la vivienda
    price_cuts = np.quantile(df["price"], [0.2, 0.4, 0.6, 0.8])
    price_discretized = pd.cut(df["price"],
                               bins = [-np.inf, price_cuts[0], price_cuts[1],
                                       price_cuts[2], price_cuts[3], np.inf],
                               labels = [1, 2, 3, 4, 5])

    ## Generar el train-test split
    train, test = train_test_split(df,
                                   test_size = 0.3,
                                   stratify = price_discretized,
                                   random_state = 42)

    ## Crear y ejecutar el procesamiento
    preprocessor = create_preprocessor()
    X_train = train.drop(columns = ["price"], errors = "ignore")
    y_train = train["price"] if "price" in train.columns else None
    X_train_prepared = preprocessor.fit_transform(X_train)
    logger.info("Se ejecutó el procesamiento de datos de entrenamiento")

    ## Ejecutar la selección de BorutaSHAP
    X_train_prepared = pd.DataFrame(X_train_prepared,
                                    columns = preprocessor.get_feature_names_out(),
                                    index = X_train.index)
    X_train_prepared = X_train_prepared[["remainder__latitude", "baths__ratio",
                                         "remainder__longitude", "beds__ratio",
                                         "log__sqft"]]

    ## Guardar los datos de entrenamiento procesados
    df_train = X_train_prepared.copy()
    if y_train is not None:
        df_train["price"] = y_train.values
    df_train.to_csv(training_output_file, index = False)
    logger.info(f"Datos de entrenamiento procesados guardados en {training_output_file}")

    ## Procesamiento sobre el test set
    X_test = test.drop(columns = ["price"], errors = "ignore")
    y_test = test["price"] if "price" in test.columns else None
    X_test_prepared = preprocessor.transform(X_test)
    logger.info("Se ejecutó el procesamiento de datos de prueba")

    ## Guardar los datos de test procesados
    df_test = pd.DataFrame(X_test_prepared,
                           columns = preprocessor.get_feature_names_out(),
                           index = X_test.index)
    df_test = df_test[X_train_prepared.columns]
    if y_test is not None:
        df_test["price"] = y_test.values
    df_test.to_csv(test_output_file, index = False)
    logger.info(f"Datos de prueba procesados guardados en {test_output_file}")

    ## Guardar el procesador
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Procesador guardado en {preprocessor_file}")

    return df_train, df_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Procesamiento de datos")
    parser.add_argument("--input", required = True, help = "Ruta del archivo csv limpio")
    parser.add_argument("--train_output", required = True, help = "Ruta para guardar el training set procesado")
    parser.add_argument("--test_output", required = True, help = "Ruta para guardar el test set procesado")
    parser.add_argument("--preprocessor", required = True, help = "Ruta para guardar el procesador")
    args = parser.parse_args()
    run_preprocessor(args.input, args.train_output, args.test_output, args.preprocessor)
