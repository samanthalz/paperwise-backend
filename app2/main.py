# app/main.py uvicorn app.main:app --reload
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from uuid import UUID
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import requests
import fitz  # PyMuPDF
from .utils import process_pdf, download_pdf, replace_base64_images, generate_pdf_thumbnail
from .chunk_embed import process_markdown
from PIL import Image
from .supabase_retriever import Retriever
from .supabase_client import supabase  
from .llm_engine import RAG
import logging, traceback
logger = logging.getLogger(__name__)
import os
import time
import re
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI(title="PDF RAG Pipeline")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Explicit origins only
    allow_credentials=True,         # Needed for cookies / auth headers
    allow_methods=["*"],            # Allow all standard methods
    allow_headers=["*"],            # Allow all headers
)

embedding_model = SentenceTransformer("google/embeddinggemma-300m")
retriever = Retriever(supabase)
rag = RAG(retriever=retriever, llm_name="gemini-2.5-flash-lite", api_key=GEMINI_API_KEY)

# --- Request body ---
class PDFRequest(BaseModel):
    arxiv_id: str
    arxiv_url: str
    title: str
    summary: str
    authors: List[str]
    published: str

# --- Output folder ---
TMP_DIR = Path("tmp_pdfs")
TMP_DIR.mkdir(exist_ok=True)

@app.post("/process_pdf/")
async def process_pdf_endpoint(data: PDFRequest):
    print("Received request data:", data.dict())
    try:
        pdf_url = data.arxiv_url
        if not pdf_url:
            raise HTTPException(status_code=400, detail="Missing arxiv_url")
        
        arxiv_id = data.arxiv_id
        title = data.title
        summary = data.summary
        authors = data.authors
        published = data.published

        logger.info(f"[process_pdf] Processing paper {arxiv_id}: {title}")

        # Check if paper already exists
        try:
            resp = supabase.table("papers").select("*").eq("arxiv_id", arxiv_id).execute()

            if resp.data:
                paper = resp.data[0]
                pdf_id = paper["pdf_id"]
                processed = paper.get("processed", False)

                logger.info(f"[process_pdf] Found existing paper with pdf_id={pdf_id}, processed={processed}")

                if processed:
                    return {
                        "status": "skipped",
                        "pdf_url": pdf_url,
                        "pdf_id": pdf_id,
                        "checkpoint": "Already processed",
                        "message": "Paper already processed"
                    }

            else:
                insert_resp = supabase.table("papers").insert({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "published": published,
                    "filename": pdf_url.split("/")[-1],
                    "processed": False,
                    "checkpoint": "Inserted metadata"
                }).execute()

                if not insert_resp.data or len(insert_resp.data) == 0:
                    raise HTTPException(status_code=500, detail="Failed to insert paper metadata")

                pdf_id = insert_resp.data[0]["pdf_id"]
                logger.info(f"[process_pdf] Inserted new paper with pdf_id={pdf_id}")

                # Return early so frontend can start polling
                return {
                    "status": "processing",
                    "pdf_url": pdf_url,
                    "pdf_id": pdf_id,
                    "processing": True,
                    "checkpoint": "Metadata inserted"
                }

        except Exception as e:
            logger.warning(f"[process_pdf] Exception checking/inserting paper metadata: {e}")
            raise

        # Download PDF
        checkpoint = "Downloading PDF"
        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        pdf_path = download_pdf(pdf_url)
        logger.info(f"[process_pdf] PDF downloaded to: {pdf_path}")

        # Generate + upload thumbnail
        checkpoint = "Generating thumbnail"
        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        thumbnail_path = generate_pdf_thumbnail(pdf_path)
        with open(thumbnail_path, "rb") as f:
            supabase.storage.from_("thumbnails").upload(f"papers/{pdf_id}.png", f.read(), {"content-type": "image/png"})

        # Convert PDF to Markdown
        checkpoint = "Extracting text from PDF"
        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        markdown_text, summaries = process_pdf(pdf_path, api_key=GEMINI_API_KEY)

        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        markdown_text = replace_base64_images(markdown_text, summaries.copy())

        if not markdown_text.strip():
            raise HTTPException(status_code=500, detail="No text extracted from PDF")

        checkpoint = "Saving embeddings and chunks"
        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        md_path = f"{Path(pdf_path).stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        process_markdown(md_path, pdf_id=pdf_id)

        # Extract and persist keypoints
        checkpoint = "Extracting keypoints"
        supabase.table("papers").update({"checkpoint": checkpoint}).eq("pdf_id", pdf_id).execute()
        keypoints = rag.extract_keypoints(pdf_id=pdf_id)
        supabase.table("papers").update({
            "problem_statement": keypoints.get("Problem Statement"),
            "objectives": keypoints.get("Objectives"),
            "methodology": keypoints.get("Methodology"),
            "results": keypoints.get("Results"),
            "discussions": keypoints.get("Discussions"),
            "limitations": keypoints.get("Limitations"),
            "processed": True,
            "checkpoint": "Processing complete"
        }).eq("pdf_id", pdf_id).execute()

        return {
            "status": "success",
            "pdf_url": pdf_url,
            "markdown_length": len(markdown_text),
            "summaries_count": len(summaries),
            "pdf_id": pdf_id,
            "checkpoint": "Processing complete"
        }

    except Exception as e:
        logger.error("[process_pdf] Unhandled exception: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/process_pdf/")
# async def process_pdf_endpoint(data: PDFRequest):
#     print("Received request data:", data.dict())
#     try:
#         pdf_url = data.arxiv_url
#         if not pdf_url:
#             raise HTTPException(status_code=400, detail="Missing arxiv_url")
        
#         arxiv_id = data.arxiv_id
#         title = data.title
#         summary = data.summary
#         authors = data.authors
#         published = data.published

#         logger.info(f"[process_pdf] Processing paper {arxiv_id}: {title}")

#         # Check if paper already exists
#         try:
#             resp = supabase.table("papers").select("*").eq("arxiv_id", arxiv_id).execute()

#             if resp.data:
#                 # Paper exists → resp.data is a list of dicts
#                 paper = resp.data[0]   # ✅ take the first row
#                 pdf_id = paper["pdf_id"]
#                 processed = paper.get("processed", False)

#                 logger.info(f"[process_pdf] Found existing paper with pdf_id={pdf_id}, processed={processed}")

#                 # Skip processing if already processed
#                 if processed:
#                     logger.info(f"[process_pdf] Paper {arxiv_id} already processed. Skipping.")
#                     return {
#                         "status": "skipped",
#                         "pdf_url": pdf_url,
#                         "pdf_id": pdf_id,
#                         "message": "Paper already processed"
#                     }
                    
#             else:
#                 # Paper does not exist, insert new
#                 insert_resp = supabase.table("papers").insert({
#                     "arxiv_id": arxiv_id,
#                     "title": title,
#                     "abstract": summary,
#                     "authors": authors,
#                     "published": published,
#                     "filename": pdf_url.split("/")[-1],
#                     "processed": False
#                 }).execute()

#                 if not insert_resp.data or len(insert_resp.data) == 0:
#                     raise HTTPException(status_code=500, detail="Failed to insert paper metadata")

#                 pdf_id = insert_resp.data[0]["pdf_id"]
#                 logger.info(f"[process_pdf] Inserted new paper with pdf_id={pdf_id}")

#                 return {
#                     "status": "processing",
#                     "pdf_url": pdf_url,
#                     "pdf_id": pdf_id,
#                     "processing": True,
#                 }

#         except Exception as e:
#             logger.warning(f"[process_pdf] Exception checking/inserting paper metadata: {e}")
#             raise

#         # Download PDF
#         t0 = time.time()
#         pdf_path = download_pdf(pdf_url)
#         logger.info(f"[process_pdf] PDF downloaded to: {pdf_path} (took {time.time()-t0:.2f}s)")
        
#         logger.info(f"[process_pdf] Starting thumbnail generation for: {pdf_path}")
#         # Generate + upload thumbnail
#         try:
#             thumbnail_path = generate_pdf_thumbnail(pdf_path)
#             logger.info(f"[process_pdf] Thumbnail generated at: {thumbnail_path}")

#             # Upload to Supabase storage
#             with open(thumbnail_path, "rb") as f:
#                 supabase.storage.from_("thumbnails").upload(
#                     f"papers/{pdf_id}.png", f.read(), {"content-type": "image/png"}
#                 )

#             # Generate a signed URL (7 days)
#             signed_url_resp = supabase.storage.from_("thumbnails").create_signed_url(
#                 f"papers/{pdf_id}.png", 60 * 60 * 24 * 7
#             )
#             logger.info(f"[process_pdf] Signed URL response: {signed_url_resp}")

#             signed_url = signed_url_resp.get("signedURL") or signed_url_resp.get("data", {}).get("signed_url")

#             if signed_url:
#                 supabase.table("papers").update({
#                     "thumbnail_url": signed_url
#                 }).eq("pdf_id", pdf_id).execute()
#                 logger.info(f"[process_pdf] Thumbnail URL saved for pdf_id={pdf_id}")

#         except Exception as e:
#             logger.error(f"[process_pdf] Thumbnail upload failed: {e}", exc_info=True)


#         # Convert PDF to Markdown + summaries
#         t0 = time.time()
#         markdown_text, summaries = process_pdf(pdf_path, api_key=GEMINI_API_KEY)
#         logger.info(f"[process_pdf] Markdown length={len(markdown_text)}, summaries={len(summaries)} (took {time.time()-t0:.2f}s)")

#         # Replace base64 images
#         t0 = time.time()
#         markdown_text = replace_base64_images(markdown_text, summaries.copy())
#         logger.info(f"[process_pdf] Markdown after image replacement length={len(markdown_text)} (took {time.time()-t0:.2f}s)")

#         if not markdown_text.strip():
#             raise HTTPException(status_code=500, detail="No text extracted from PDF")

#         # Save embeddings + chunks
#         t0 = time.time()
#         md_path = f"{Path(pdf_path).stem}.md"
#         with open(md_path, "w", encoding="utf-8") as f:
#             f.write(markdown_text)
#         process_markdown(md_path, pdf_id=pdf_id)  # Pass pdf_id to link chunks
#         # Path(md_path).unlink(missing_ok=True)
#         logger.info(f"[process_pdf] Embeddings + chunks saved (took {time.time()-t0:.2f}s)")
        
#         # Extract and persist keypoints
#         try:
#             keypoints = rag.extract_keypoints(pdf_id=pdf_id)

#             supabase.table("papers").update({
#                 "problem_statement": keypoints.get("Problem Statement"),
#                 "objectives": keypoints.get("Objectives"),
#                 "methodology": keypoints.get("Methodology"),
#                 "results": keypoints.get("Results"),
#                 "discussions": keypoints.get("Discussions"),
#                 "limitations": keypoints.get("Limitations"),
#                 "processed": True
#             }).eq("pdf_id", pdf_id).execute()

#             logger.info(f"[process_pdf] Keypoints extracted and saved for pdf_id={pdf_id}")
#         except Exception as e:
#             logger.warning(f"[process_pdf] Failed to extract keypoints: {e}")

#         # Mark as processed
#         logger.info(f"[process_pdf] Completed successfully for {pdf_url}")

#         return {
#             "status": "success",
#             "pdf_url": pdf_url,
#             "markdown_length": len(markdown_text),
#             "summaries_count": len(summaries),
#             "pdf_id": pdf_id
#         }

#     except Exception as e:
#         logger.error("[process_pdf] Unhandled exception: %s\n%s", str(e), traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    arxiv_id: str
    question: str
    
@app.post("/ask_stream/")
async def ask_question_stream(request: AskRequest):
    try:
        # Get paper
        paper_resp = supabase.table("papers").select("*").eq("arxiv_id", request.arxiv_id).single().execute()
        if not paper_resp.data:
            raise HTTPException(status_code=404, detail="Paper not found")
        pdf_id = paper_resp.data["pdf_id"]

        return StreamingResponse(
            rag.stream_query(request.question, pdf_id=pdf_id, top_k=10),
            media_type="text/plain"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))