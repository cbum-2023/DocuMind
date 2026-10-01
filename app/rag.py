"""Core RAG logic - document processing, vector search, and LLM answering."""
import time
import uuid
from typing import Optional, List
from openai import OpenAI
from . import config
from .store import get_collection
from .models import ChatMessage

def get_openai_client():
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")
    return OpenAI(api_key=config.OPENAI_API_KEY)

def chunk_pages_with_metadata(pages: List[str], chunk_size=1000, overlap=200):
    """Concatenates pages to preserve sentence context across boundaries, 
    mapping each chunk back to its page number range."""
    page_spans = []
    full_text = ""
    
    for page_num, p_text in enumerate(pages, start=1):
        p_clean = p_text.strip()
        if not p_clean:
            continue
        start_idx = len(full_text)
        full_text += p_clean + "\n\n"
        end_idx = len(full_text)
        page_spans.append((start_idx, end_idx, page_num))
        
    if not full_text.strip():
        return []

    chunks = []
    start = 0
    full_len = len(full_text)
    
    while start < full_len:
        end = min(start + chunk_size, full_len)
        chunk_str = full_text[start:end].strip()
        
        if chunk_str:
            # find pages overlapping with start..end
            pages_involved = []
            for p_start, p_end, p_num in page_spans:
                if max(start, p_start) < min(end, p_end):
                    pages_involved.append(p_num)
                    
            primary_page = pages_involved[0] if pages_involved else 1
            chunks.append({
                "text": chunk_str,
                "page": primary_page,
                "pages": pages_involved
            })
            
        if end >= full_len:
            break
        start = end - overlap

    return chunks

def add_document(filename: str, pages: List[str]):
    """Ingests PDF page texts, creates embeddings, and saves to ChromaDB."""
    col = get_collection()
    doc_id = str(uuid.uuid4())[:8]

    chunk_objs = chunk_pages_with_metadata(pages)
    if not chunk_objs:
        return doc_id, 0

    ids, docs, metas = [], [], []
    for i, c in enumerate(chunk_objs):
        cid = f"{doc_id}_{i}"
        ids.append(cid)
        docs.append(c["text"])
        metas.append({
            "doc_id": doc_id, 
            "filename": filename, 
            "page": c["page"]
        })

    embeddings = embed_texts(docs)
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return doc_id, len(ids)

def embed_texts(texts: List[str]):
    """Get embeddings from OpenAI."""
    client = get_openai_client()
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def search(question: str, doc_id: Optional[str] = None, n_results=4):
    col = get_collection()
    if col.count() == 0:
        return []
        
    q_emb = embed_texts([question])[0]
    
    where_clause = {"doc_id": doc_id} if doc_id else None
    
    try:
        res = col.query(
            query_embeddings=[q_emb], 
            n_results=min(n_results, col.count()),
            where=where_clause
        )
    except Exception as e:
        print(f"[search error] {e}")
        return []

    if not res or not res.get("documents") or not res["documents"][0]:
        return []

    out = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        out.append({
            "text": doc, 
            "page": meta.get("page", 1), 
            "filename": meta.get("filename", "Document")
        })
    return out

def is_safe(question: str):
    """Basic guardrail against prompt injection."""
    bad = ["ignore previous instructions", "system prompt", "jailbreak mode"]
    q = question.lower()
    return not any(b in q for b in bad)

def answer_question(question: str, doc_id: Optional[str] = None, history: Optional[List[ChatMessage]] = None):
    start = time.time()

    if not is_safe(question):
        return "Blocked: suspicious prompt detected.", []

    chunks = search(question, doc_id=doc_id)
    if not chunks:
        if doc_id:
            return "I couldn't find any relevant information in the selected document.", []
        return "No documents uploaded yet, or no relevant information was found.", []

    context = "\n\n".join([f"--- Source: {c['filename']} (Page {c['page']}) ---\n{c['text']}" for c in chunks])

    messages = [
        {
            "role": "system",
            "content": (
                "You are DocuMind, an intelligent document analysis assistant. "
                "Answer the user's question accurately using ONLY the provided context snippets below. "
                "If the information is not contained in the context, state clearly: 'I don't know based on the provided document(s).' "
                "Format your answer using clean Markdown (bullet points, bold text, headers where appropriate)."
            )
        }
    ]

    # Include recent conversation history if provided (up to last 6 messages)
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

    # Add context + question as final user prompt
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_prompt})

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content

    elapsed = int((time.time() - start) * 1000)
    print(f"[ask] q='{question[:50]}' chunks={len(chunks)} time={elapsed}ms")

    return answer, chunks

