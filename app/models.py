from typing import Optional, List
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

class Source(BaseModel):
    text: str
    page: int
    filename: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

