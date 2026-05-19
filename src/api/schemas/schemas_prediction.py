from pydantic import BaseModel

class PropertyData(BaseModel):
    beds: int
    baths: float
    sqft: int
    type: str
    latitude: float
    longitude: float

class PredictionResponse(BaseModel):
    predicted_price: float
