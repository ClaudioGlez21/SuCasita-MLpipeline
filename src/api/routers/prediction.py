from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os
from src.api.schemas.schemas_prediction import PropertyData, PredictionResponse
import sys

# Add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import column_ratio, ratio_name

# Trick joblib into finding the functions if they were saved as part of __main__ or scripts.data_processing
import scripts.data_processing
sys.modules['scripts.data_processing'] = scripts.data_processing
# Also try to put them in __main__ just in case
import __main__
__main__.column_ratio = column_ratio
__main__.ratio_name = ratio_name

router = APIRouter(prefix="/predict", tags=["prediction"])

# Base path of the project
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load artifacts
try:
    PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    MODEL_PATH = os.path.join(MODELS_DIR, "su_casita_model.pkl")
    SELECTED_FEATURES_PATH = os.path.join(MODELS_DIR, "selected_features.pkl")

    preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    preprocessing_pipeline = None
    model = None
    selected_features = None

@router.post("/", response_model=PredictionResponse)
def predict(data: PropertyData):
    if model is None or preprocessing_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    try:
        # Preprocess the data
        # Note: The pipeline expects the original features before selection
        X_processed = preprocessing_pipeline.transform(input_df)
        
        # Convert back to DataFrame to handle feature selection
        # We need the feature names from the preprocessing pipeline
        feature_names = preprocessing_pipeline.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Select features
        X_final = X_processed_df[selected_features]
        
        # Predict
        prediction = model.predict(X_final)
        
        return PredictionResponse(predicted_price=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
