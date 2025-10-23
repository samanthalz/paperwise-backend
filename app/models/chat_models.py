# app/models/ask_models.py
from pydantic import BaseModel

class AskRequest(BaseModel):
    pdf_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
