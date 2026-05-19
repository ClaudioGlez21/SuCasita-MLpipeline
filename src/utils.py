import numpy as np

def column_ratio(X):
    """Realiza el cociente entre la primera y segunda columna de un arreglo."""
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    """Regresa el nombre de las variables de entrada con el sufijo 'ratio'."""
    return ["ratio"]
