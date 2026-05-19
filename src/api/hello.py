from fastapi import FastAPI
from src.api.routers import health
from src.api.routers import shopping
from src.api.routers import prediction


app = FastAPI(title = "SuCasita API 🏠",
              description = "API para predicción de precios de bienes raíces en Sacramento",
              version = "1.0")

@app.get("/")

def hello():
    '''
    Devuelve un mensaje de bienvenida
    '''

    response = {"message": "Bienvenido a la API de SuCasita"}

    return response

app.include_router(health.router)
app.include_router(shopping.router)
app.include_router(prediction.router)