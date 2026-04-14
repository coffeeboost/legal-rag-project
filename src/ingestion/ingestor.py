"""
Ingestion pipeline — parse, chunk, embed, and store legal documents.
Supports PDFs (contracts, court opinions).
"""
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

from src.config import settings


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE,
               overlap: int = settings.CHUNK_OVERLAP) -> list[str]:
    """
    Sliding-window chunker.
    For production, swap this with a semantic or section-aware splitter.
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def doc_id(path: str, chunk_index: int) -> str:
    """Stable, unique ID for each chunk."""
    return hashlib.md5(f"{path}::{chunk_index}".encode()).hexdigest()


# ── Main ingestor ─────────────────────────────────────────────────────────────

class Ingestor:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)

        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION}' ready.")

    def ingest_file(self, path: str, metadata: dict = None) -> int:
        """
        Parse, chunk, embed, and store a single PDF.

        metadata example:
            {"doc_type": "contract", "jurisdiction": "NY", "date": "2024-01-15"}
        """
        path = str(path)
        metadata = metadata or {}

        logger.info(f"Ingesting: {path}")
        text = load_pdf(path)
        chunks = chunk_text(text)
        logger.info(f"  → {len(chunks)} chunks")

        ids, embeddings, documents, metadatas = [], [], [], []

        for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
            chunk_meta = {
                "source": path,
                "chunk_index": i,
                "filename": Path(path).name,
                **metadata,
            }
            ids.append(doc_id(path, i))
            embeddings.append(self.embedder.encode(chunk).tolist())
            documents.append(chunk)
            metadatas.append(chunk_meta)

        # Upsert in batches of 100
        batch_size = 100
        for b in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[b: b + batch_size],
                embeddings=embeddings[b: b + batch_size],
                documents=documents[b: b + batch_size],
                metadatas=metadatas[b: b + batch_size],
            )

        logger.success(f"Ingested {len(chunks)} chunks from {Path(path).name}")
        return len(chunks)

    def ingest_directory(self, directory: str, metadata: dict = None) -> int:
        """Ingest all PDFs in a directory."""
        total = 0
        for pdf_path in Path(directory).glob("**/*.pdf"):
            total += self.ingest_file(str(pdf_path), metadata)
        return total
