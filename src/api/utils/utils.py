import numpy as np

## Función para el cociente de variables
def column_ratio(X):
    """
    Realiza el cociente entre la primera y segunda columna de un arreglo.

    Parámetros
    ----------
    X: np.array
        Matriz de datos.

    Output
    ------
    np.array
        Arreglo con el cociente.
    """

    ## Considerar la división entre cero

    return X[:, [0]]/X[:, [1]]

## Función de utilidad para la compatibilidad
def ratio_name(function_transformer, feature_names_in):
    """
    Función de utilidad para regresar los
    nombres de las variables transformadas
    con un sufijo específico.

    Parámetros
    ----------
    function_transformer: column_ratio transformer.
    feature_names_in: Nombres de las variables de entrada.

    Output
    ------
    Regresa el nombre de las variables de
    entrada con el sufijo "ratio".
    """

    return ["ratio"]
