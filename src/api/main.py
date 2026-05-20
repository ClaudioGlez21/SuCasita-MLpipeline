from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import health, inference


app = FastAPI(title = "API predicción Su Casita",
              description = "API para realizar predicciones de precios de vivienda",
              version = "0.1")


app.add_middleware(CORSMiddleware,
                   allow_origins = ["*"],
                   allow_credentials = True,
                   allow_methods = ["*"],
                   allow_headers = ["*"])


app.include_router(health.router)
app.include_router(inference.router)