from fastapi import FastAPI
from app.routes.summarize import router as summarize_router

app = FastAPI()

app.include_router(summarize_router)
