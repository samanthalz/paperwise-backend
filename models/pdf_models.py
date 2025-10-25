from pydantic import BaseModel
from typing import List

class PDFRequest(BaseModel):
    arxiv_id: str
    arxiv_url: str
    title: str
    summary: str
    authors: List[str]
    published: str

class PDFResponse(BaseModel):
    status: str
    pdf_url: str
    pdf_id: str
    checkpoint: str
    message: str | None = None
