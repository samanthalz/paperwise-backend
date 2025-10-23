# llm_engine.py
import google.generativeai as genai
import re

class RAG:
    def __init__(self, retriever, llm_name="gemini-2.5-flash-lite", api_key=None):
        if not api_key:
            raise ValueError("Gemini API key required")
        genai.configure(api_key=api_key)

        self.llm_name = llm_name
        self.retriever = retriever
        self.qa_prompt_tmpl_str = """You are answering questions based on a research paper.

Context information is below (may be empty):
---------------------
{context}
---------------------

Instructions:
- If the provided context contains enough information, answer based only on the context.
- If the context does NOT contain enough information (or is empty):
    1. Start your answer with exactly this sentence: "No relevant information found in the paper."
    2. Then continue on the next line with: "Based on general knowledge: ..." and provide a clear, factual answer to the user’s question.
    3. Do NOT simply restate that the context has no information—always give a real answer from your general knowledge.
- If the user’s question expects a yes/no answer, respond clearly with "Yes" or "No" and then provide a short explanation.
- Always provide a concise and accurate answer.
- The context may contain Markdown headings or formatting (like *, **, _, etc.). Ignore all such formatting symbols and do not include them in your answer.
- Use headings only to identify titles or sections, but do not include Markdown symbols in your final answer.

Query: {query}
---------------------
Answer:"""

    def _normalize_text(self, t: str):
        return re.sub(r'\s+', ' ', (t or "")).strip()

    def _keyword_score(self, text: str, query: str):
        # simple heuristic: number of query words found (can be improved)
        qwords = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
        lower = text.lower()
        return sum(1 for w in qwords if w in lower)

    def rerank_and_select(self, chunks, query, final_k=8):
        """
        Takes overfetched chunks (each dict must contain 'chunk_text' and 'similarity'), 
        returns ordered, deduped list of top final_k chunks.
        - boosts chunks containing any term from must_include_terms
        - uses exact substring checks to boost very high
        """

        scored = []
        for c in chunks:
            text = c.get("chunk_text", "") or ""
            sim = float(c.get("similarity", 0.0))
            # base score = similarity (assumes similarity in 0..1; adjust if distance)
            score = sim

            # keyword overlap boost
            score += 0.01 * self._keyword_score(text, query)
            scored.append((score, c))

        # sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # dedupe by short fingerprint (avoid near-duplicates)
        selected = []
        seen_hashes = set()
        for score, c in scored:
            short = (self._normalize_text(c.get("chunk_text",""))[:200]).lower()
            h = hash(short)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            selected.append((score, c))
            if len(selected) >= final_k:
                break

        return [c for _, c in selected]

    def generate_context(self, query, pdf_id=None, top_k=20, final_k=8):
        """
        Generate context string from top relevant chunks for RAG.
        Includes Markdown headings if available.
        """
        # 1️⃣ Retrieve top_k chunks from Supabase
        raw = self.retriever.search(query, pdf_id=pdf_id, top_k=top_k) or []

        print(f"Retriever returned: {len(raw)} candidates")
        for i, entry in enumerate(raw[:6]):
            preview = (entry.get("chunk_text", "")[:200]).replace("\n", " ")
            print(f"[raw {i}] sim={entry.get('similarity', 0.0):.4f} preview={preview}...")

        # 2️⃣ Rerank & select top_final
        top_final = self.rerank_and_select(raw, query, final_k=final_k)

        # 3️⃣ Build combined context with heading metadata
        combined = []
        for idx, e in enumerate(top_final):
            text = e.get("chunk_text", "")
            heading = e.get("heading", "")
            metadata_parts = []
            if heading:
                metadata_parts.append(f"Heading: {heading}")

            meta_str = f"[{' | '.join(metadata_parts)}]" if metadata_parts else ""
            combined.append(f"[Chunk {idx+1}]{meta_str}\n{text}")

        print(f"Final context chunks used: {len(combined)}")
        for i, c in enumerate(combined):
            preview = c[:200].replace("\n", " ")
            print(f"[used {i}] len={len(c)} preview={preview}...")

        return "\n\n---\n\n".join(combined)

    def stream_query(self, query, pdf_id=None, top_k=20):
        context = self.generate_context(query, pdf_id=pdf_id, top_k=top_k)

        # just pass context or empty string
        context_for_prompt = context if context.strip() else ""

        prompt = self.qa_prompt_tmpl_str.format(
            context=context_for_prompt,
            query=query
        )

        model = genai.GenerativeModel(self.llm_name)
        response = model.generate_content(
            prompt,
            stream=True,
            generation_config={"max_output_tokens": 300}
        )

        try:
            for chunk in response:
                if hasattr(chunk, "candidates"):
                    for candidate in chunk.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                text = getattr(part, "text", None)
                                if text:
                                    yield text
                else:
                    yield "⚠️ Gemini returned no content."
        except Exception as e:
            yield f"⚠️ Gemini streaming error: {str(e)}"

    def extract_keypoints(self, pdf_id: str):
        if not pdf_id:
            raise ValueError("pdf_id must be provided for keypoint extraction.")

        sections = [
            "Problem Statement",
            "Objectives",
            "Methodology",
            "Results",
            "Discussions",
            "Limitations"
        ]

        # 1️⃣ Fetch all chunks
        all_chunks = self.retriever.get_all_chunks(pdf_id=pdf_id) or []
        if not all_chunks:
            print(f"⚠️ No chunks found for pdf_id={pdf_id}")
        
        # 2️⃣ Combine chunks into a single text context
        full_text = ""
        for idx, chunk in enumerate(all_chunks):
            heading = chunk.get("heading") or ""
            text = chunk.get("chunk_text") or ""
            meta_str = f"[Heading: {heading}]" if heading else ""
            full_text += f"[Chunk {idx+1}]{meta_str}\n{text}\n\n---\n\n"

        # 3️⃣ Initialize model
        model = genai.GenerativeModel(self.llm_name)

        # 4️⃣ Extract keypoints section by section
        keypoints = {}
        for section in sections:
            prompt = f"""
Context (entire paper):
-----------------------
{full_text}
-----------------------

Instruction: Based on the entire paper, extract a concise {section}.
- Understand from the entire paper's content.
- Do not just repeat sentences from the paper; infer the section's meaning.
- If the information is not explicitly present, respond with "Not mentioned".
- The context may contain formatting (like *, **, _, etc.). Remove all such formatting symbols and do not include them in your answer.
"""

            try:
                response = model.generate_content(
                    prompt,
                    stream=False,
                    generation_config={"max_output_tokens": 500}
                )
                keypoints[section] = getattr(response, "text", "").strip() or "Not mentioned"
            except Exception as e:
                print(f"❌ Failed to extract {section}: {e}")
                keypoints[section] = "⚠️ Failed"

        return keypoints

