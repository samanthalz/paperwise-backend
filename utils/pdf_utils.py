from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, AcceleratorDevice, AcceleratorOptions
import re
from collections import OrderedDict
import requests
from pathlib import Path
import fitz  # PyMuPDF
from typing import Union
from urllib.parse import urlparse
import hashlib
from db.supabase_client import supabase

def download_pdf(url: str, tmp_dir="tmp_pdfs"):
    """
    Downloads a PDF from a given URL, saves it locally,
    and returns both the file path and its SHA-256 hash.
    """
    Path(tmp_dir).mkdir(exist_ok=True)

    # Extract filename from URL safely
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name or "downloaded.pdf"
    file_path = Path(tmp_dir) / filename

    # Download PDF content
    r = requests.get(url)
    r.raise_for_status()
    pdf_bytes = r.content

    # Save file locally
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    # Compute SHA-256 hash of the file
    try:
        print("Computing file hash...")
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        print("file hash is", file_hash)
    except Exception as e:
        print("❌ Error computing hash:", e)

    return file_path, file_hash

#Replace each base64 image with its corresponding summary
def replace_base64_images(md_text, summary_dict):
    pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

    def replacement(match):
        # Get next unused key from the summaries dict
        if summary_dict:
            key, value = summary_dict.popitem(last=False)  # pop the first item
            return f"\n\n{value}\n\n"
        else:
            return "\n\n[Image removed - no summary available]\n\n"

    return re.sub(pattern, replacement, md_text)

def generate_pdf_thumbnail(pdf_path: Union[str, Path], output_dir: Union[str, Path] = "tmp_thumbnails") -> Path:
    """
    Generate a thumbnail (PNG) from the first page of a PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the thumbnail

    Returns:
        Path to the generated PNG thumbnail
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # first page

    # Render page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # scale 2x for better resolution
    thumbnail_path = output_dir / f"{pdf_path.stem}_thumbnail.png"
    pix.save(thumbnail_path)
    doc.close()

    return thumbnail_path

def rename_paper(pdf_id: str, new_title: str) -> bool:
    """Rename a paper's title in Supabase."""
    try:
        res = supabase.table("papers").update({"title": new_title}).eq("pdf_id", pdf_id).execute()
        print("Rename result:", res.data)
        return True
    except Exception as e:
        print("Rename failed:", e)
        return False


def move_paper(pdf_id: str, folder_id: str) -> bool:
    """Move a paper to a different folder (update user_papers.folder_id)."""
    try:
        res = supabase.table("user_papers").update({"folder_id": folder_id}).eq("pdf_id", pdf_id).execute()
        if res.data:
            print(f"Paper {pdf_id} moved to folder {folder_id}")
            return True
        else:
            print("Move returned empty response")
            return False
    except Exception as e:
        print("Move failed:", e)
        return False


def delete_paper(pdf_id: str) -> bool:
    """Delete a paper and its link records if no user references remain.
    Only delete from papers table if it's a user-uploaded paper (user_id is not null).
    """
    try:
        # Delete from user_papers first
        supabase.table("user_papers").delete().eq("pdf_id", pdf_id).execute()

        # Check if any remaining links exist
        count_resp = supabase.table("user_papers").select("*", count="exact", head=True).eq("pdf_id", pdf_id).execute()
        count = count_resp.count or 0

        # Only delete paper if no more references
        if count == 0:
            # Fetch the paper to check user_id
            paper_resp = supabase.table("papers").select("user_id").eq("pdf_id", pdf_id).single().execute()
            paper = paper_resp.data
            if paper and paper.get("user_id"):
                # Only delete user-uploaded papers
                supabase.table("papers").delete().eq("pdf_id", pdf_id).execute()
                print(f"User-uploaded paper {pdf_id} deleted from database")
            else:
                print(f"Public paper {pdf_id} not deleted from papers table, only user_papers removed")
        else:
            print(f"Paper {pdf_id} still linked to other users, skipped paper deletion")

        return True
    except Exception as e:
        print("Delete failed:", e)
        return False

def delete_folder(folder_id: str) -> bool:
    """Delete a folder and all papers inside if no user references remain."""
    try:
        # 1️⃣ Get all papers in this folder
        papers_resp = supabase.table("user_papers").select("pdf_id").eq("folder_id", folder_id).execute()
        papers = [p["pdf_id"] for p in papers_resp.data] if papers_resp.data else []

        # 2️⃣ Delete all user_papers links for these papers
        for pdf_id in papers:
            supabase.table("user_papers").delete().eq("pdf_id", pdf_id).execute()

            # Check if paper has remaining links
            count_resp = supabase.table("user_papers").select("*", count="exact", head=True).eq("pdf_id", pdf_id).execute()
            count = count_resp.count or 0
            if count == 0:
                supabase.table("papers").delete().eq("pdf_id", pdf_id).execute()
                print(f"Paper {pdf_id} deleted from database")
            else:
                print(f"Paper {pdf_id} still linked to other users, skipped paper deletion")

        # 3️⃣ Delete the folder itself
        supabase.table("folders").delete().eq("id", folder_id).execute()
        print(f"Folder {folder_id} deleted successfully")

        return True
    except Exception as e:
        print("Delete folder failed:", e)
        return False

