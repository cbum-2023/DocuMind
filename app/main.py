import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from .models import AskRequest, AskResponse, Source, DocumentListResponse, DocumentInfo
from .rag import add_document, answer_question
from .store import list_stored_documents, delete_stored_document, clear_all_documents

app = FastAPI(title="DocuMind API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "DocuMind API"}

@app.get("/documents", response_model=DocumentListResponse)
def get_documents():
    """Lists all uploaded documents with chunk statistics."""
    try:
        docs = list_stored_documents()
        doc_models = [DocumentInfo(**d) for d in docs]
        return DocumentListResponse(documents=doc_models)
    except Exception as e:
        print(f"[get_documents error] {e}")
        raise HTTPException(500, f"Failed to retrieve document list: {str(e)}")

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a specific document by its ID."""
    try:
        delete_stored_document(doc_id)
        return {"ok": True, "message": f"Document {doc_id} deleted successfully."}
    except Exception as e:
        print(f"[delete_document error] {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

@app.delete("/documents")
def delete_all_documents():
    """Wipes all documents from the vector database."""
    try:
        clear_all_documents()
        return {"ok": True, "message": "All documents cleared."}
    except Exception as e:
        print(f"[delete_all_documents error] {e}")
        raise HTTPException(500, f"Failed to clear documents: {str(e)}")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Uploaded file is empty.")

        pdf_stream = io.BytesIO(contents)
        reader = PdfReader(pdf_stream)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        if not any(p.strip() for p in pages):
            raise HTTPException(400, "Could not extract readable text from PDF. It may be scanned or image-only.")

        doc_id, n_chunks = add_document(file.filename, pages)
        return {"doc_id": doc_id, "chunks": n_chunks, "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload error] {e}")
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if len(q) < 2:
        raise HTTPException(400, "Question is too short.")
    if len(q) > 2000:
        raise HTTPException(400, "Question is too long.")

    try:
        answer, chunks = answer_question(question=q, doc_id=req.doc_id, history=req.history)
        sources = [Source(text=c["text"][:400], page=c["page"], filename=c["filename"]) for c in chunks]
        return AskResponse(answer=answer, sources=sources)
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        print(f"[ask error] {e}")
        raise HTTPException(500, f"Something went wrong while processing your question: {str(e)}")

