import traceback, logging, os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.chat_models import AskRequest
from db.supabase_client import supabase
from services.llm_engine import RAG
from core.retriever import Retriever

logger = logging.getLogger(__name__)

# Setup retriever + RAG 
retriever = Retriever(supabase)
rag = RAG(retriever=retriever, llm_name="gemini-2.5-flash-preview-09-2025", api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

@router.post("/ask_stream/")
async def ask_question_stream(request: AskRequest):
    try:
    
        return StreamingResponse(
            rag.stream_query(request.question, pdf_id=request.pdf_id, top_k=10),
            media_type="text/plain"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    