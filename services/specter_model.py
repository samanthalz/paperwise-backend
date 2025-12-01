from sentence_transformers import SentenceTransformer, util
import torch

# Load the SPECTER2-base model once
model = SentenceTransformer("allenai/specter2_base")

async def rank_by_similarity(title: str, abstract: str, papers: list, top_k: int = 10):
    """Rank related papers by similarity to the main paper using embeddings."""
    main_text = f"{title}. {abstract or ''}"
    main_emb = model.encode(main_text, convert_to_tensor=True, normalize_embeddings=True)

    paper_texts = [
        f"{p.get('title', '')}. {p.get('abstract', '')}" for p in papers if p.get("title")
    ]
    embeddings = model.encode(paper_texts, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(main_emb, embeddings)[0]
    ranked_idx = torch.argsort(sims, descending=True)[:top_k]

    ranked_papers = []
    for idx in ranked_idx:
        p = papers[idx]
        ranked_papers.append({
            **p,  # preserve all existing data 
            "score": float(sims[idx]),  # add similarity score
        })
    return ranked_papers
