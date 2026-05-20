from fastapi import APIRouter
import joblib
import pandas as pd
from schemas.prediction import HousePredictionRequest

router = APIRouter(tags = ["Inference"])


MODEL_PATH = "../../models/su_casita_model.pkl"
PROCESSOR_PATH = "../../models/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    processor = joblib.load(PROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo o procesamiento de datos: {str(e)}")

@router.get("/model-health")
def model_load():
    """
    Endpoint para verificar la salud del modelo.
    """
    response = {"response": model.get_params()}

    return response