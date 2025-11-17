from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    topic_id: str
    message: str
    chunks_created: int

class ChatRequest(BaseModel):
    topic_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    image_id: Optional[str] = None
    image_filename: Optional[str] = None
    image_title: Optional[str] = None
    sources: List[str]

class ImageMetadata(BaseModel):
    id: str
    filename: str
    title: str
    keywords: List[str]
    description: str
    embeddings: Optional[List[float]] = None

class TextChunk(BaseModel):
    id: str
    text: str
    embeddings: List[float]
    page: int