from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter

# import the supabase client
from .supabase_client import supabase  

# ----------------------------
# Read Markdown
# ----------------------------
def read_markdown(md_path: str) -> str:
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------
# Chunk text
# ----------------------------
def chunk_text(text: str):
    """
    Split Markdown into chunks based on headings, keeping heading metadata.
    Returns list of dicts: {"text": chunk_text, "heading": heading}
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("###", "h3")],
        return_each_line=False,
        strip_headers=False,
    )
    
    # Split text into Document objects
    docs = splitter.split_text(text)

    chunk_dicts = []
    current_heading = "Untitled"  # Default heading if none is found

    for doc in docs:
        content = doc.page_content.strip()
        lines = content.splitlines()

        for line in lines:
            if line.startswith("##") or line.startswith("###"):
                current_heading = line.lstrip("#").strip()
                break

        # Ensure the chunk has some heading
        heading_to_use = current_heading if current_heading else "Untitled"

        chunk_dicts.append({
            "text": content,
            "heading": heading_to_use
        })

    return chunk_dicts

# ----------------------------
# Embedding Model
# ----------------------------
def load_gemma_model():
    model = SentenceTransformer("google/embeddinggemma-300m")
    return model

def embed_chunks(model, chunks):
    texts = [c["text"] for c in chunks]  # extract all texts
    embeddings = model.encode(texts)      # batch encode
    return [emb.astype(np.float32).tolist() for emb in embeddings]

# ----------------------------
# Save to Supabase
# ----------------------------
def save_md_and_chunks(chunks: list, embeddings: list, pdf_id: str, batch_size: int = 50):
    """
    Save Markdown chunks with embeddings and heading metadata to Supabase.
    """
    print(f"Saving chunks for paper pdf_id={pdf_id}")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        records = []
        for idx, (chunk_dict, emb) in enumerate(zip(batch_chunks, batch_embeddings), start=i):
            records.append({
                "pdf_id": pdf_id,
                "chunk_index": idx,
                "chunk_text": chunk_dict["text"],
                "heading": chunk_dict.get("heading", ""),
                "embedding": emb,
            })
        chunk_resp = supabase.table("chunks").insert(records).execute()
        if not chunk_resp.data:
            raise Exception(f"Error inserting chunk batch starting at {i}: {chunk_resp}")
        
        print(f"Inserted chunk batch {i} -> {i + len(records) - 1}")
              
# ----------------------------
# Full pipeline
# ----------------------------
def process_markdown(md_path: str, pdf_id: str):
    # Read Markdown
    text = read_markdown(md_path)
    if not text.strip():
        raise ValueError("Markdown is empty!")

    # Chunk
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    # Embed
    model = load_gemma_model()
    embeddings = embed_chunks(model, chunks)

    save_md_and_chunks(chunks, embeddings, pdf_id)
    print("Markdown embedding complete.")
