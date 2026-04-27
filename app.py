"""
FastAPI app — REST API for LexRAG.
Endpoints: ingest, query, chat (stateless), sessions (persistent chat).
"""
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from src.ingestion.ingestor import Ingestor
from src.generation.chain import RAGChain, ChatMessage
from src.sessions import init_db, create_session, list_sessions, get_session, append_messages, delete_session


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LexRAG API...")
    await init_db()
    app.state.ingestor = Ingestor()
    app.state.rag_chain = RAGChain()
    logger.success("LexRAG API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="LexRAG", version="0.2.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str

class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int

class ChatMessageModel(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessageModel] = []
    filters: dict | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    history: list[ChatMessageModel]

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"

class SessionChatRequest(BaseModel):
    question: str
    filters: dict | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_sources(chunks) -> list[dict]:
    return [
        {
            "filename": c.filename,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
            "preview": c.content[:200] + "...",
        }
        for c in chunks
    ]


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = "unknown",
    jurisdiction: str = "unknown",
):
    """Upload and ingest a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        metadata = {"doc_type": doc_type, "jurisdiction": jurisdiction}
        n_chunks = app.state.ingestor.ingest_file(tmp_path, metadata)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(filename=file.filename, chunks_ingested=n_chunks)


# ── Stateless query / chat ────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Single-turn query — no history."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = app.state.rag_chain.query(req.question, filters=req.filters)
    return QueryResponse(answer=result.answer, sources=_format_sources(result.sources), query=result.query)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Stateless multi-turn chat — caller manages history."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [ChatMessage(role=m.role, content=m.content) for m in req.history]
    result = app.state.rag_chain.chat(req.question, history=history, filters=req.filters)

    updated_history = list(req.history) + [
        ChatMessageModel(role="user", content=req.question),
        ChatMessageModel(role="assistant", content=result.answer),
    ]
    return ChatResponse(answer=result.answer, sources=_format_sources(result.sources), history=updated_history)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming query — returns tokens as they arrive."""
    def token_generator():
        for token in app.state.rag_chain.stream_query(req.question, req.filters):
            yield token
    return StreamingResponse(token_generator(), media_type="text/plain")


# ── Persistent sessions ───────────────────────────────────────────────────────

@app.post("/sessions")
async def new_session(req: CreateSessionRequest):
    """Create a new chat session."""
    return await create_session(req.title)


@app.get("/sessions")
async def get_sessions():
    """List all sessions, newest first."""
    return await list_sessions()


@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get a session with its full message history."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a session and all its messages."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    await delete_session(session_id)
    return {"deleted": session_id}


@app.post("/sessions/{session_id}/chat")
async def session_chat(session_id: str, req: SessionChatRequest):
    """
    Persistent chat — history is loaded from and saved to the DB automatically.
    Just send the new question; no need to manage history on the client.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Rebuild history from DB
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in session["messages"]
    ]

    result = app.state.rag_chain.chat(req.question, history=history, filters=req.filters)
    sources = _format_sources(result.sources)

    # Auto-title session from first user message
    is_first = len(session["messages"]) == 0
    new_title = req.question[:60] if is_first else None

    await append_messages(
        session_id,
        [
            {"role": "user", "content": req.question, "sources": []},
            {"role": "assistant", "content": result.answer, "sources": sources},
        ],
        new_title=new_title,
    )

    return {"answer": result.answer, "sources": sources, "session_id": session_id}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": "ollama"}
