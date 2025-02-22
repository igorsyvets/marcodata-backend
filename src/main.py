from fastapi import FastAPI
from src.api.endpoints import router

app = FastAPI(title="Test API")

app.include_router(router, prefix="/api/v1")