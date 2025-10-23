# uvicorn app.main:app --reload   .\venv\Scripts\activate.bat
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import pdf_routes, chat_routes, recommendation_routes

app = FastAPI(title="PDF RAG Pipeline")

# --- CORS ---
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(pdf_routes.router)
app.include_router(chat_routes.router)
app.include_router(recommendation_routes.router)
