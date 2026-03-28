from pydantic import BaseModel, Field


class DocumentContentResponse(BaseModel):
    document_id: str
    status: str
    content: str | None = None


class DocumentSummaryResponse(BaseModel):
    document_id: str
    status: str
    title: str | None = None
    author: str | None = None
    summary: str | None = None


class IndexResponse(BaseModel):
    document_id: str
    status: str
    chunks_indexed: int = 0


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str = "READY"
    filename: str | None = None


class ChatSessionRequest(BaseModel):
    document_id: str
    tenant_id: str


class ChatSessionResponse(BaseModel):
    session_id: str
    document_id: str
    user_id: str


class ChatMessageRequest(BaseModel):
    message: str
    top_k: int | None = None


class ChatMessageResponse(BaseModel):
    answer: str
    citations: list[dict]
