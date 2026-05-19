import pandas as pd
from pathlib import Path
import logging
import os

## Configuración del logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data-static-cleaning")


def load_data(file_path):
    """
    Carga los datos desde un archivo csv.
    """

    ## Cargar el archivo de datos
    logger.info(f"Cargando datos desde {file_path}")

    return pd.read_csv(file_path)


def clean_data(input_file, output_file):
    """
    Realiza una limpieza estática de datos.
    """

    ## Iniciando limpieza
    logger.info("Limpieza del dataset")

    ## Asegurar que la carpeta de destino exista
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    ## Carga de datos
    df = load_data(input_file)
    logger.info(f"Datos cargados con dimensión: {df.shape}")

    ## Hacer una copia para evitar modificar el data frame original
    df_cleaned = df.copy()

    ## Este proceso incluye tareas como:
    ## 1. Normalización de tipos de datos (data type casting).
    ## 2. Estandarización de texto (sanitización).
    ##     - Case folding: Pasar todo a minúsculas o mayúsculas.
    ##     - Stripping: Eliminar espacios en blanco al inicio o al final.
    ##     - Limpieza de caracteres especiales: Eliminar acentos, emojis o símbolos extraños.
    ##     - Traducción de etiquetas: Homogenizar las categorías.
    ## 3. Eliminación de duplicados exactos.
    ## 4. Filtrado de columnas irrelevantes (dropping).
    ## 5. Lógica de negocio y restricciones físicas.

    ## Sanitización de la variable de tipo
    df_cleaned["type"] = df_cleaned["type"].str.lower().str.strip()

    ## Eliminar registros totalmente duplicados
    df_cleaned.drop_duplicates(inplace=True)

    ## Filtrar columnas irrelevantes
    df_cleaned.drop(columns=["city", "zip"], inplace=True)

    ## Guardar datos limpios
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Datos limpios guardados en {output_file}")

    return df_cleaned


if __name__ == "__main__":
    # Base path of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ## Ejemplo de uso
    clean_data(
        input_file=os.path.join(BASE_DIR, "data", "raw", "Sacramento_Data.csv"),
        output_file=os.path.join(BASE_DIR, "data", "processed", "sacramento_cleaned.csv"),
    )
