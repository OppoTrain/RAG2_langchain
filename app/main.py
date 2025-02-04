from fastapi import FastAPI
from app.routes.summarize import router as summarize_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.include_router(summarize_router)

origins = [
    "http://localhost:3000",  # React app running locally
    "http://127.0.0.1:3000",  # In case you're accessing from localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)