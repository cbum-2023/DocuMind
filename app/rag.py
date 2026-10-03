"""Core RAG logic using LangChain for text splitting, embeddings, Chroma vector store, and Gemini LLM generation."""
import time
import uuid
from typing import Optional, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from . import config
from .store import get_collection, get_client
from .models import ChatMessage

def get_embeddings():
    """Initialize LangChain Google Generative AI Embeddings."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{config.EMBEDDING_MODEL}",
        google_api_key=config.GEMINI_API_KEY
    )

def get_vectorstore():
    """Get LangChain Chroma vector store instance."""
    return Chroma(
        client=get_client(),
        collection_name="documind",
        embedding_function=get_embeddings()
    )


def chunk_pages_with_langchain(pages: List[str]) -> List[Document]:
    """Split pages into semantic text chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = []
    for page_num, page_text in enumerate(pages, start=1):
        clean_text = page_text.strip()
        if not clean_text:
            continue
            
        page_chunks = text_splitter.split_text(clean_text)
        for chunk in page_chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page_num}
            ))
            
    return docs

def add_document(filename: str, pages: List[str]):
    """Ingest document pages using LangChain text splitters and Chroma vectorstore."""
    doc_id = str(uuid.uuid4())[:8]
    langchain_docs = chunk_pages_with_langchain(pages)
    
    if not langchain_docs:
        return doc_id, 0

    chunk_ids = []
    for i, doc in enumerate(langchain_docs):
        doc.metadata["doc_id"] = doc_id
        doc.metadata["filename"] = filename
        chunk_ids.append(f"{doc_id}_{i}")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=langchain_docs, ids=chunk_ids)
    return doc_id, len(langchain_docs)

def search(question: str, doc_id: Optional[str] = None, top_k=4):
    """Retrieve relevant chunks from Chroma using LangChain similarity search."""
    vectorstore = get_vectorstore()
    filter_kwargs = {"doc_id": doc_id} if doc_id else None
    
    results = vectorstore.similarity_search(
        query=question,
        k=top_k,
        filter=filter_kwargs
    )

    out = []
    for doc in results:
        out.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page", 1),
            "filename": doc.metadata.get("filename", "Document")
        })
    return out

def is_safe(question: str) -> bool:
    """Basic guardrail against prompt injection."""
    bad_keywords = ["ignore previous instructions", "system prompt", "jailbreak mode"]
    q_lower = question.lower()
    return not any(b in q_lower for b in bad_keywords)

def answer_question(question: str, doc_id: Optional[str] = None, history: Optional[List[ChatMessage]] = None):
    """Answer question using LangChain RAG pipeline (ChatPromptTemplate + ChatGoogleGenerativeAI + StrOutputParser)."""
    start_time = time.time()

    if not is_safe(question):
        return "Blocked: suspicious prompt detected.", []

    chunks = search(question, doc_id=doc_id)
    if not chunks:
        if doc_id:
            return "I couldn't find any relevant information in the selected document.", []
        return "No documents uploaded yet, or no relevant information was found.", []

    context_str = "\n\n".join([
        f"--- Source: {c['filename']} (Page {c['page']}) ---\n{c['text']}" 
        for c in chunks
    ])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are DocuMind, an intelligent document analysis assistant. "
         "Answer the user's question accurately using ONLY the provided context snippets below. "
         "If the information is not contained in the context, state clearly: 'I don't know based on the provided document(s).' "
         "Format your answer using clean Markdown (bullet points, bold text, headers where appropriate).\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2
    )

    # LangChain Expression Language (LCEL) chain
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    elapsed = int((time.time() - start_time) * 1000)
    print(f"[ask] q='{question[:50]}' chunks={len(chunks)} time={elapsed}ms")

    return answer, chunks




