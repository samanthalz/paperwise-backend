from fastapi import APIRouter, HTTPException
from app.services.semantic_api import (
    resolve_paper,
    fetch_citations,
    fetch_references,
    fetch_recommendations,
    clean_papers
)
from app.services.specter_model import rank_by_similarity
import httpx

router = APIRouter()

@router.post("/recommendations")
async def get_recommendations(request: dict):
    query = request.get("query")
    arxiv_id = request.get("arxiv_id")

    async with httpx.AsyncClient() as client:
        # --- Resolve main paper ---
        paper_id, title, abstract = await resolve_paper(client, arxiv_id, query)
        if not paper_id:
            raise HTTPException(status_code=404, detail="Paper not found")

        # --- Fetch citations and references ---
        citations_raw = await fetch_citations(client, paper_id)
        references_raw = await fetch_references(client, paper_id)
        # print(f"Fetched: {len(citations_raw)} citations, {len(references_raw)} references")
        # print("Sample before cleaning:")
        # for p in citations_raw[:3]:
        #     print(p.get("title"), p.get("externalIds"))

        
        citations = clean_papers(citations_raw)
        references = clean_papers(references_raw)
        # print(f"After clean: {len(citations)} citations, {len(references)} references")
        # print("Sample after cleaning:")
        # for p in citations[:3]:
        #     print(p.get("title"), p.get("externalIds"))


        # --- Fetch API recommendations ---
        recommendations_raw = await fetch_recommendations(client, paper_id)
        recommendations = clean_papers(recommendations_raw)
        # print(f"After clean: {len(recommendations)} recommendations")

        # --- Combine papers for ranking ---
        if recommendations:
            combined_papers = citations + references + recommendations
            source = "api + specter2-base"
        else:
            combined_papers = citations + references
            source = "specter2-base"

        combined_papers = [p for p in combined_papers if p.get("abstract")]
        # print(f"Total papers to rank: {len(combined_papers)}")

        if not combined_papers:
            raise HTTPException(status_code=404, detail="No papers with abstracts found for ranking.")

        # --- Rank using SPECTER2 embeddings ---
        ranked = await rank_by_similarity(title, abstract, combined_papers)
        # print(f"Ranked {len(ranked)} papers, showing top {min(10, len(ranked))}")

        return {
            "source": source,
            "recommendations": ranked,
        }

