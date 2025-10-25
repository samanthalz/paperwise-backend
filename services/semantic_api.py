import os
import asyncio
import httpx
from typing import List, Dict, Any

SEMANTIC_API_KEY = os.getenv("SEMANTIC_API_KEY")
SEMANTIC_API_URL = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATION_API_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
FIELDS = "paperId,title,abstract,url,authors,year,citationCount,externalIds,openAccessPdf"

HEADERS = {"x-api-key": SEMANTIC_API_KEY} if SEMANTIC_API_KEY else {}

async def safe_get(client: httpx.AsyncClient, url: str, retries: int = 3, delay: float = 2.0, timeout: float = 30.0) -> Any:
    for attempt in range(retries):
        try:
            response = await client.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited: exponential backoff
                await asyncio.sleep(delay * (attempt + 1))
            else:
                response.raise_for_status()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError) as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(delay * (attempt + 1))
    # Failed after retries
    print(f"Failed to fetch {url} after {retries} attempts")
    return None

# Resolve paperId, title, and abstract from query or arXiv ID
async def resolve_paper(client: httpx.AsyncClient, arxiv_id: str = None, query: str = None):
    if arxiv_id:
        url = f"{SEMANTIC_API_URL}/paper/ARXIV:{arxiv_id}?fields=paperId,title,abstract"
    elif query:
        url = f"{SEMANTIC_API_URL}/paper/search?query={query}&limit=1&fields=paperId,title,abstract"
    else:
        return None, None, None

    data = await safe_get(client, url)
    if not data:
        return None, None, None

    if "data" in data and len(data["data"]) > 0:
        item = data["data"][0]
        return item.get("paperId"), item.get("title"), item.get("abstract")
    else:
        return data.get("paperId"), data.get("title"), data.get("abstract")

def clean_papers(papers: list):
    cleaned = []
    for p in papers:
        # Some API responses wrap data under 'citedPaper' or 'citingPaper'
        paper = p.get("citedPaper") or p.get("citingPaper") or p

        # Skip invalid or unlinked papers
        if not paper.get("paperId"):
            continue

        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()

        # Skip junk or too short titles
        if not title or len(title.split()) < 3:
            continue

        authors = paper.get("authors", [])

        # Skip if authors is None or an empty list
        if not authors:
            continue

        cleaned.append({
            "paperId": paper["paperId"],
            "title": title,
            "abstract": abstract,
            "url": paper.get("url"),
            "authors": authors,
            "year": paper.get("year") or paper.get("published"),
            "citationCount": paper.get("citationCount", 0),
            "openAccessPdf": {"url": paper.get("pdf_url")} if paper.get("pdf_url") else None,
            "externalIds": paper.get("externalIds"),
        })
    return cleaned

# Fetch citations for a given paper
async def fetch_citations(client: httpx.AsyncClient, paper_id: str) -> List[Dict[str, Any]]:
    url = f"{SEMANTIC_API_URL}/paper/{paper_id}/citations?fields={FIELDS}"
    data = await safe_get(client, url)
    if not data:
        return []
    return [item["citingPaper"] for item in data.get("data", []) if "citingPaper" in item]


# Fetch references for a given paper
async def fetch_references(client: httpx.AsyncClient, paper_id: str) -> List[Dict[str, Any]]:
    url = f"{SEMANTIC_API_URL}/paper/{paper_id}/references?fields={FIELDS}"
    data = await safe_get(client, url)
    if not data:
        return []
    return [item["citedPaper"] for item in data.get("data", []) if "citedPaper" in item]


# Fetch API-based recommendations
async def fetch_recommendations(client: httpx.AsyncClient, paper_id: str) -> List[Dict[str, Any]]:
    #url = f"{RECOMMENDATION_API_URL}/{paper_id}?fields=title,url,venue,abstract,openAccessPdf,authors"
    url = f"{RECOMMENDATION_API_URL}/{paper_id}?fields={FIELDS}"
    data = await safe_get(client, url)
    if not data or "recommendedPapers" not in data:
        return []
    return data.get("recommendedPapers", [])
