from pydantic import BaseModel, Field
from typing import Literal

class HousePredictionRequest(__________):
    baths: float = Field(..., gt = 0, description = "Número de baños")
    beds: int = Field(..., ge = 1, description = "Número de recámaras")
    sqft: float = Field(..., gt = 0, description = "Área de la vivienda en pies cuadrados")
    latitude: float = Field(..., ge = -90, le = 90, description = "Latitud")
    longitude: float = Field(..., ge = -180, le = 180, description = "Longitud")
    type: Literal ["residential", "condo", "multi_family"] = Field(..., description = "Tipo de propiedad")

class PredictionResponse(BaseModel):
    price: float = Field(..., gt = 0, description = "Precio predicho de la vivienda")