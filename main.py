from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import pdf_routes, chat_routes, recommendation_routes
import os

app = FastAPI(title="PDF RAG Pipeline")

# --- CORS ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nextjs-fyp-tau.vercel.app",
]
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

@app.get("/")
def root():
    return {"message": "🚀 FastAPI PDF RAG Pipeline is running successfully!"}

# --- Run (Render + Local) ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
