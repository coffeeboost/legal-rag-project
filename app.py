"""
FastAPI app — REST API for LexRAG.
Endpoints: ingest, query, list documents.
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


# ── Lifespan — load models once at startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LexRAG API...")
    app.state.ingestor = Ingestor()
    app.state.rag_chain = RAGChain()
    logger.success("LexRAG API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="LexRAG", version="0.1.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None  # e.g. {"doc_type": "contract", "jurisdiction": "NY"}

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str

class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int

class ChatMessageRequest(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessageRequest] = []
    filters: dict | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    history: list[ChatMessageRequest]  # updated history to pass back next turn


# ── Routes ────────────────────────────────────────────────────────────────────

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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Query the RAG system."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = app.state.rag_chain.query(req.question, filters=req.filters)

    return QueryResponse(
        answer=result.answer,
        sources=[
            {
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
                "preview": c.content[:200] + "...",
            }
            for c in result.sources
        ],
        query=result.query,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Conversational RAG — maintains context across turns.
    Pass the returned history back in your next request to continue the conversation."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [ChatMessage(role=m.role, content=m.content) for m in req.history]
    result = app.state.rag_chain.chat(req.question, history=history, filters=req.filters)

    updated_history = list(req.history) + [
        ChatMessageRequest(role="user", content=req.question),
        ChatMessageRequest(role="assistant", content=result.answer),
    ]

    return ChatResponse(
        answer=result.answer,
        sources=[
            {
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
                "preview": c.content[:200] + "...",
            }
            for c in result.sources
        ],
        history=updated_history,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming query endpoint — returns tokens as they're generated."""
    def token_generator():
        for token in app.state.rag_chain.stream_query(req.question, req.filters):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "ollama"}
