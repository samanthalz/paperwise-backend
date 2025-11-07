from pathlib import Path
import traceback, logging, os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.pdf_models import PDFRequest
from db.supabase_client import supabase
from utils.pdf_utils import download_pdf, generate_pdf_thumbnail, replace_base64_images, rename_paper, move_paper, delete_paper, delete_folder
from core.pdf_parser import parse_pdf
from core.chunk_embed import process_markdown
from services.llm_engine import RAG
from services.img_summarizer import summarize_with_gemini_pil
from core.retriever import Retriever
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- Setup retriever + RAG ---
embedding_model = SentenceTransformer("google/embeddinggemma-300m")
retriever = Retriever(supabase)
rag = RAG(retriever=retriever, llm_name="gemini-2.5-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

# --- Background pipeline ---
def run_full_pipeline(pdf_url, pdf_id, extract_summary: bool = True):
    try:
        # Download PDF
        supabase.table("papers").update({"checkpoint": "Downloading PDF"}).eq("pdf_id", pdf_id).execute()
        pdf_path, file_hash = download_pdf(pdf_url)
        
        existing = supabase.table("papers").select("pdf_id, title").eq("file_hash", file_hash).execute()

        if extract_summary and existing.data and len(existing.data) > 0:
            # Duplicate found
            existing_pdf_id = existing.data[0]["pdf_id"]
            supabase.table("papers").update({
                "checkpoint": f"Duplicate found: {existing_pdf_id}"
            }).eq("pdf_id", pdf_id).execute()
            print(f"Duplicate detected for pdf_id={pdf_id}. Already exists as {existing_pdf_id}")
            return {"status": "duplicate"}
        else:
            supabase.table("papers").update({
                "checkpoint": f"Background processing started"
            }).eq("pdf_id", pdf_id).execute()
        
        # Save hash to DB
        supabase.table("papers").update({
            "file_hash": file_hash,
        }).eq("pdf_id", pdf_id).execute()
        
        # Generate thumbnail
        supabase.table("papers").update({"checkpoint": "Generating thumbnail"}).eq("pdf_id", pdf_id).execute()
        thumbnail_path = generate_pdf_thumbnail(pdf_path)

        with open(thumbnail_path, "rb") as f:
            image_bytes = f.read()

        storage_path = f"papers/{pdf_id}.png"
        try:
            supabase.storage.from_("thumbnails").remove([storage_path])
        except Exception as e:
            print(f": Failed to remove old thumbnail (may not exist): {e}")

        try:
            supabase.storage.from_("thumbnails").upload(
                storage_path,
                image_bytes,
                {"content-type": "image/png"},
            )
        except Exception as e:
            print(f"Upload warning: {e}")

        thumbnail_url = supabase.storage.from_("thumbnails").get_public_url(storage_path)
        supabase.table("papers").update({"thumbnail_url": thumbnail_url}).eq("pdf_id", pdf_id).execute()

        # Convert PDF to Markdown
        supabase.table("papers").update({"checkpoint": "Extracting text from PDF"}).eq("pdf_id", pdf_id).execute()
        markdown_text, image_elements, conv_res = parse_pdf(pdf_path)
        summaries = summarize_with_gemini_pil(
            image_elements=image_elements,
            conv_res=conv_res,
            api_key=os.getenv("GEMINI_API_KEY"),
            doc_filename=Path(pdf_path).name
        )
        markdown_text = replace_base64_images(markdown_text, summaries.copy())

        if not markdown_text.strip():
            supabase.table("papers").update({"checkpoint": "Error: No text extracted"}).eq("pdf_id", pdf_id).execute()
            return

        # Save embeddings and chunks
        supabase.table("papers").update({"checkpoint": "Saving embeddings and chunks"}).eq("pdf_id", pdf_id).execute()
        md_path = f"{Path(pdf_path).stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        process_markdown(md_path, pdf_id=pdf_id)

        # Extract summary/authors ONLY if flag is True
        if extract_summary:
            supabase.table("papers").update({"checkpoint": "Extracting summary/authors"}).eq("pdf_id", pdf_id).execute()
            try:
                extracted_info = rag.extract_summary_and_authors(pdf_id, pdf_path=pdf_path)
                title = extracted_info.get("title")
                authors = extracted_info.get("authors")
                summary = extracted_info.get("summary")
                supabase.table("papers").update({
                    "title": title,
                    "authors": authors,
                    "abstract": summary
                }).eq("pdf_id", pdf_id).execute()
            except Exception as e:
                print(f"Failed to extract summary/authors: {e}")

        # Extract keypoints
        supabase.table("papers").update({"checkpoint": "Extracting keypoints"}).eq("pdf_id", pdf_id).execute()
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

    except Exception as e:
        logger.error(f"[Background pipeline] Error processing PDF {pdf_id}: {e}\n{traceback.format_exc()}")
        supabase.table("papers").update({"checkpoint": f"Error: {str(e)}"}).eq("pdf_id", pdf_id).execute()


# --- Endpoint ---
@router.post("/process_pdf/")
async def process_pdf_endpoint(data: PDFRequest, background_tasks: BackgroundTasks):
    logger.info(f"[process_pdf] Received request: {data.dict()}")
    pdf_url = data.arxiv_url
    if not pdf_url:
        raise HTTPException(status_code=400, detail="Missing arxiv_url")

    arxiv_id, title, summary, authors, published = (
        data.arxiv_id, data.title, data.summary, data.authors, data.published
    )

    # --- Check if paper exists ---
    resp = supabase.table("papers").select("*").eq("arxiv_id", arxiv_id).execute()
    if resp.data:
        paper = resp.data[0]
        pdf_id, processed = paper["pdf_id"], paper.get("processed", False)
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
        if not insert_resp.data:
            raise HTTPException(status_code=500, detail="Failed to insert paper metadata")
        pdf_id = insert_resp.data[0]["pdf_id"]

    # --- Return early to start frontend polling ---
    background_tasks.add_task(run_full_pipeline, pdf_url, pdf_id, extract_summary=False)
    return {
        "status": "processing",
        "pdf_url": pdf_url,
        "pdf_id": pdf_id,
        "checkpoint": "Metadata inserted; background processing started"
    }

class PDFIdRequest(BaseModel):
    pdf_id: str

@router.post("/process_existing_pdf/")
async def process_existing_pdf_endpoint(data: PDFIdRequest, background_tasks: BackgroundTasks):
    pdf_id = data.pdf_id

    # --- Check if PDF exists and has supabase_url ---
    resp = supabase.table("papers").select("*").eq("pdf_id", pdf_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail=f"PDF with id {pdf_id} not found")

    paper = resp.data[0]
    pdf_url = paper.get("supabase_url")
    if not pdf_url:
        raise HTTPException(status_code=400, detail=f"PDF with id {pdf_id} does not have a valid supabase_url")

    # --- Check if already processed ---
    if paper.get("processed", False):
        return {
            "status": "skipped",
            "pdf_id": pdf_id,
            "checkpoint": "Already processed",
            "message": "PDF has already been processed"
        }

    # --- Schedule background processing ---
    background_tasks.add_task(run_full_pipeline, pdf_url, pdf_id, extract_summary=True)

class RenameRequest(BaseModel):
    pdf_id: str
    new_title: str

class MoveRequest(BaseModel):
    pdf_id: str
    folder_id: Optional[str] = None

class DeleteRequest(BaseModel):
    pdf_id: str
    
class DeleteFolderRequest(BaseModel):
    folder_id: str
    
@router.post("/rename_pdf/")
async def rename_pdf(req: RenameRequest):
    success = rename_paper(req.pdf_id, req.new_title)
    if not success:
        raise HTTPException(status_code=500, detail="Rename failed.")
    return {"success": True}

@router.post("/move_pdf/")
async def move_pdf(req: MoveRequest):
    success = move_paper(req.pdf_id, req.folder_id)
    if not success:
        raise HTTPException(status_code=400, detail="Move failed")
    return {"success": True, "message": f"Moved {req.pdf_id} to folder {req.folder_id}"}


@router.post("/delete_pdf/")
async def delete_pdf(req: DeleteRequest):
    success = delete_paper(req.pdf_id)
    if not success:
        raise HTTPException(status_code=400, detail="Delete failed")
    return {"success": True, "message": f"Deleted {req.pdf_id}"}

@router.post("/delete_folder/")
async def delete_folder_endpoint(req: DeleteFolderRequest):
    success = delete_folder(req.folder_id)
    if not success:
        raise HTTPException(status_code=400, detail="Folder deletion failed")
    return {"success": True, "message": f"Deleted folder {req.folder_id}"}