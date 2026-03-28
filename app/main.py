import os
import uvicorn
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.security import decode_bearer_token
from app.db.session import engine, get_db
from app.models import Base, Document, DocumentContent, DocumentChunk
from app.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionRequest,
    ChatSessionResponse,
    IndexResponse,
    DocumentUploadResponse,
    DocumentContentResponse,
    DocumentSummaryResponse,
)
from app.services import (
    answer_with_rag,
    create_chat_session,
    create_or_update_document,
    extract_document_text,
    run_indexing_job,
    save_document_file,
    generate_document_summary,
)
from app.models.entities import IndexStatus

app = FastAPI(title="Document Agent Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on app startup."""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize database on startup: {e}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post(f"{settings.api_prefix}/documents/{{document_id}}/upload", response_model=DocumentUploadResponse)
async def upload_document(
    document_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> DocumentUploadResponse:
    limiter.allow(f"upload:{user.get('sub', 'anon')}", limit=30, period_seconds=60)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    data = await file.read()
    max_size = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_size:
        raise HTTPException(status_code=400, detail="File too large")

    tenant_id = user.get("tenant_id", "default")
    title = f"Document {document_id}"
    saved_path = save_document_file(document_id, file.filename, data)

    try:
        text = extract_document_text(Path(saved_path))
    except ValueError as exc:
        Path(saved_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        Path(saved_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read document") from None

    if not text.strip():
        Path(saved_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Document contains no extractable text")

    doc = create_or_update_document(
        db,
        document_id=document_id,
        tenant_id=tenant_id,
        title=title,
        owner_id=user.get("sub"),
    )
    content = DocumentContent(document_id=document_id, source_type="upload", language=None, full_text=text)
    db.add(content)
    doc.index_status = IndexStatus.pending
    db.commit()

    return DocumentUploadResponse(document_id=document_id, status="READY", filename=file.filename)


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/content", response_model=DocumentContentResponse)
def get_content(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> DocumentContentResponse:
    limiter.allow(f"content:{user.get('sub', 'anon')}", limit=120, period_seconds=60)
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    content = db.query(DocumentContent).filter(DocumentContent.document_id == document_id).order_by(DocumentContent.id.desc()).first()
    if not content:
        return DocumentContentResponse(document_id=document_id, status="PENDING", content=None)
    return DocumentContentResponse(document_id=document_id, status="READY", content=content.full_text)


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/summary", response_model=DocumentSummaryResponse)
def get_summary(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> DocumentSummaryResponse:
    limiter.allow(f"summary:{user.get('sub', 'anon')}", limit=60, period_seconds=60)
    try:
        title, author, summary = generate_document_summary(db, document_id)
        return DocumentSummaryResponse(
            document_id=document_id,
            status="READY",
            title=title,
            author=author,
            summary=summary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc!s}") from exc


@app.post(f"{settings.api_prefix}/documents/{{document_id}}/index", response_model=IndexResponse)
def index_document(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> IndexResponse:
    limiter.allow(f"index:{user.get('sub', 'anon')}", limit=20, period_seconds=60)
    try:
        count = run_indexing_job(db, document_id)
    except ValueError as exc:
        detail = str(exc)
        code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=code, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc!s}") from exc
    return IndexResponse(document_id=document_id, status="READY", chunks_indexed=count)


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/index/status", response_model=IndexResponse)
def index_status(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> IndexResponse:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = (
        db.query(func.count(DocumentChunk.id)).filter(DocumentChunk.document_id == document_id).scalar() or 0
    )
    return IndexResponse(document_id=document_id, status=doc.index_status.value, chunks_indexed=int(chunks))


@app.post(f"{settings.api_prefix}/chat/sessions", response_model=ChatSessionResponse)
def create_session(
    payload: ChatSessionRequest,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> ChatSessionResponse:
    limiter.allow(f"session:{user.get('sub', 'anon')}", limit=60, period_seconds=60)
    session = create_chat_session(db, payload.document_id, payload.tenant_id, user.get("sub", "anonymous"))
    return ChatSessionResponse(session_id=session.id, document_id=session.document_id, user_id=session.user_id)


@app.post(f"{settings.api_prefix}/chat/sessions/{{session_id}}/messages", response_model=ChatMessageResponse)
def send_message(
    session_id: str,
    payload: ChatMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(decode_bearer_token),
) -> ChatMessageResponse:
    limiter.allow(f"chat:{user.get('sub', 'anon')}", limit=120, period_seconds=60)
    top_k = min(payload.top_k or settings.default_top_k, settings.max_top_k)
    try:
        answer, citations = answer_with_rag(db, session_id, payload.message, top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatMessageResponse(answer=answer, citations=citations)


@app.websocket(f"{settings.api_prefix}/chat/stream")
async def stream_chat(socket: WebSocket) -> None:
    await socket.accept()
    data = await socket.receive_json()
    answer = data.get("message", "")
    await socket.send_json({"type": "token", "content": "Streaming is not implemented for Ollama RAG yet. "})
    await socket.send_json({"type": "token", "content": answer})
    await socket.send_json({"type": "final", "answer": answer, "citations": []})
    await socket.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
