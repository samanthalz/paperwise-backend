# llm_engine.py
import fitz  # PyMuPDF
import google.generativeai as genai
import re

class RAG:
    def __init__(self, retriever, llm_name="gemini-2.5-flash-lite", api_key=None):
        if not api_key:
            raise ValueError("Gemini API key required")
        genai.configure(api_key=api_key)

        self.llm_name = llm_name
        self.retriever = retriever
        self.qa_prompt_tmpl_str = """You are an expert research assistant answering questions based on a paper.

Context (may be empty):
---------------------
{context}
---------------------

Instructions:
- If the provided context contains enough information, answer strictly using it.
- If the context does NOT contain enough information (or is empty):
    1. Begin your answer with: "No relevant information found in the paper."
    2. Then add a second sentence starting with: "Based on general knowledge, ..." and give a clear, factual answer.
    3. Always provide a meaningful answer even if the paper lacks the information.
- If the question expects a yes/no response, answer clearly with "Yes" or "No," followed by a short explanation.
- Be concise, factual, and avoid Markdown or unnecessary formatting symbols.
- Ignore any Markdown (*, **, _, etc.) present in the context text.

Question: {query}
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

        # print(f"Retriever returned: {len(raw)} candidates")
        # for i, entry in enumerate(raw[:6]):
        #     preview = (entry.get("chunk_text", "")[:200]).replace("\n", " ")
        #     print(f"[raw {i}] sim={entry.get('similarity', 0.0):.4f} preview={preview}...")

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

        # print(f"Final context chunks used: {len(combined)}")
        # for i, c in enumerate(combined):
        #     preview = c[:200].replace("\n", " ")
        #     print(f"[used {i}] len={len(c)} preview={preview}...")
            
        metadata = self.retriever.get_metadata(pdf_id)
        title = metadata.get("title")

        context_str = f"Paper title: {title}" + "\n\n---\n\n".join(combined)
        return context_str


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
- The context may contain Markdown headings or formatting (like *, **, _, etc.). Ignore all such formatting symbols and do not include them in your answer.
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
    
    def extract_summary_and_authors(self, pdf_id: str, pdf_path: str = None):
        """
        Extracts the concise summary (abstract), title, and authors of a PDF given its pdf_id.
        - Summary: uses entire paper text from all chunks (or the abstract if found).
        - Title + Authors: uses direct text from page 1 of the original PDF for better accuracy.
        """
        if not pdf_id:
            raise ValueError("pdf_id must be provided for extraction.")

        model = genai.GenerativeModel(self.llm_name)
        extracted = {"title": None, "authors": [], "summary": None, "summary_source": None}

        # 1️⃣ Gather all chunks for full-text summary
        all_chunks = self.retriever.get_all_chunks(pdf_id=pdf_id) or []
        if not all_chunks:
            print(f"⚠️ No chunks found for pdf_id={pdf_id}")

        full_text = ""
        for idx, chunk in enumerate(all_chunks):
            heading = chunk.get("heading") or ""
            text = chunk.get("chunk_text") or ""
            meta_str = f"[Heading: {heading}]" if heading else ""
            full_text += f"[Chunk {idx+1}]{meta_str}\n{text}\n\n---\n\n"

        # 2️⃣ Try reading first page directly from PDF (for title/authors)
        first_page_text = ""
        if pdf_path:
            try:
                doc = fitz.open(pdf_path)
                if len(doc) > 0:
                    first_page_text = doc.load_page(0).get_text("text")
                doc.close()
            except Exception as e:
                print(f"⚠️ Failed to read PDF: {e}")
        else:
            first_page_chunk = next((c for c in all_chunks if c.get("page_number") == 1), all_chunks[0] if all_chunks else {})
            first_page_text = first_page_chunk.get("chunk_text", "")

        # 3️⃣ Extract TITLE
        try:
            title_prompt = f"""
    Context (first page of paper):
    -----------------------------
    {first_page_text}
    -----------------------------

    Instruction:
    Extract the paper title as it appears at the top.
    - Respond with only the title (no extra text).
    - If not found, respond exactly with "Not mentioned".
    """
            response = model.generate_content(title_prompt, stream=False, generation_config={"max_output_tokens": 100})
            title = getattr(response, "text", "").strip()
            extracted["title"] = None if not title or title.lower() == "not mentioned" else title
        except Exception as e:
            print(f"❌ Failed to extract title: {e}")

        # 4️⃣ Extract AUTHORS
        try:
            authors_prompt = f"""
    Context (first page of paper):
    -----------------------------
    {first_page_text}
    -----------------------------

    Instruction:
    List only the paper's authors (not affiliations or citations), separated by commas.
    - Example: "John Doe, Jane Smith, Alex Tan"
    - If not found, respond exactly with "Not mentioned".
    """
            response = model.generate_content(authors_prompt, stream=False, generation_config={"max_output_tokens": 200})
            authors_str = getattr(response, "text", "").strip()
            if authors_str and authors_str.lower() != "not mentioned":
                extracted["authors"] = [a.strip() for a in authors_str.split(",") if a.strip()]
        except Exception as e:
            print(f"❌ Failed to extract authors: {e}")

        # 5️⃣ Extract ABSTRACT or generate SUMMARY
        try:
            abstract_text = ""
            abstract_patterns = [
                # Case 1: Abstract followed by Keywords / Introduction
                r"(?is)\babstract\b[:\-]?\s*(.+?)(?=\n\s*(keywords|introduction|1[\s\.]|i[\s\.]|background)\b)",
                # Case 2: Abstract till double newline or end of section
                r"(?is)\babstract\b[:\-]?\s*([\s\S]{0,1500}?)(?=\n{2,}|$)"
            ]

            for pattern in abstract_patterns:
                match = re.search(pattern, full_text)
                if match:
                    abstract_text = match.group(1).strip()
                    # Stop at the first match that looks reasonable (not too long)
                    if 100 < len(abstract_text) < 2000:
                        break

            # Clean and post-process
            if abstract_text:
                abstract_text = re.sub(r"[\n\r]+", " ", abstract_text)  # merge lines
                abstract_text = re.sub(r"\s{2,}", " ", abstract_text)   # collapse spaces
                abstract_text = re.sub(r"[*_#`]+", "", abstract_text)   # remove markdown junk
                abstract_text = re.split(r"\b(∗|†|‡|---|listing order|equal contribution)", abstract_text, 1)[0].strip()
                extracted["summary"] = abstract_text
                extracted["summary_source"] = "abstract"
            else:
                summary_prompt = f"""
    Context (entire paper):
    -----------------------
    {full_text}
    -----------------------

    Instruction:
    Write a concise academic-style summary (like an abstract).
    - Use clear, objective tone.
    - Remove all formatting symbols (*, **, _, etc.).
    - If unclear, respond "Not mentioned".
    """
                response = model.generate_content(summary_prompt, stream=False, generation_config={"max_output_tokens": 500})
                summary = getattr(response, "text", "").strip() or "Not mentioned"
                extracted["summary"] = summary
                extracted["summary_source"] = "generated"

        except Exception as e:
            print(f"❌ Failed to extract summary: {e}")
            extracted["summary"] = "⚠️ Failed"
            extracted["summary_source"] = "error"

        return extracted

