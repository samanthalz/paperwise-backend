import time
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, supabase, model_name="multi-qa-mpnet-base-dot-v1"):
        self.supabase = supabase
        self.embed_model = SentenceTransformer(model_name)

    def _normalize(self, vec):
        vec = np.array(vec)
        return (vec / np.linalg.norm(vec)).tolist()

    def _expand_query(self, query):
        return [query, f"What is {query}?", f"Explain {query}", f"Details about {query}"]

    def search(self, query, top_k, pdf_id=None):
        # Expand query → encode → average embeddings
        alt_queries = self._expand_query(query)
        embeddings = self.embed_model.encode(alt_queries)
        query_embedding = self._normalize(embeddings.mean(axis=0))

        start_time = time.time()
        response = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "match_pdf_id": pdf_id
            }
        ).execute()
        end_time = time.time()

        print(f"Execution time for pgvector search: {end_time - start_time:.4f} seconds")
        return response.data
    
    def get_all_chunks(self, pdf_id: str):
        """
        Return all chunks for a given PDF ID, without any similarity ranking.
        """
        start_time = time.time()
        response = self.supabase.table("chunks").select("*").eq("pdf_id", pdf_id).execute()
        end_time = time.time()
        print(f"Execution time for fetching all chunks: {end_time - start_time:.4f} seconds")
        
        return response.data or []

