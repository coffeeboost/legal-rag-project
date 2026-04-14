"""
Hybrid Retriever — combines dense (vector) + sparse (BM25) search,
then re-ranks with a cross-encoder.

This is the core differentiator vs. a basic RAG system.
"""
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from loguru import logger

from src.config import settings


@dataclass
class RetrievedChunk:
    content: str
    source: str
    filename: str
    chunk_index: int
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)

        logger.info("Loading re-ranker model...")
        self.reranker = CrossEncoder(settings.RERANK_MODEL)

        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index is built lazily from the collection
        self._bm25 = None
        self._bm25_docs = None

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int,
                      filters: dict = None) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode(query).tolist()
        # Strip empty/invalid filter values before passing to ChromaDB
        clean = {k: v for k, v in (filters or {}).items() if v not in (None, {}, [])}
        where = clean if clean else None

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(
                content=doc,
                source=meta.get("source", ""),
                filename=meta.get("filename", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=1 - dist,  # cosine distance → similarity
                metadata=meta,
            ))
        return chunks

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────

    def _build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB."""
        logger.info("Building BM25 index...")
        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs = list(zip(result["documents"], result["metadatas"]))
        tokenized = [doc.lower().split() for doc in result["documents"]]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(tokenized)} chunks.")

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            self._build_bm25_index()

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        chunks = []
        for idx in top_indices:
            doc, meta = self._bm25_docs[idx]
            chunks.append(RetrievedChunk(
                content=doc,
                source=meta.get("source", ""),
                filename=meta.get("filename", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=float(scores[idx]),
                metadata=meta,
            ))
        return chunks

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[RetrievedChunk],
                top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve(self, query: str, filters: dict = None) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
          1. Dense search (vector similarity)
          2. Sparse search (BM25 keyword)
          3. Merge & deduplicate
          4. Re-rank with cross-encoder
        """
        logger.debug(f"Retrieving for query: '{query[:80]}...'")

        dense_results = self._dense_search(query, settings.DENSE_TOP_K, filters)
        sparse_results = self._sparse_search(query, settings.SPARSE_TOP_K)

        # Deduplicate by (source, chunk_index)
        seen = set()
        merged = []
        for chunk in dense_results + sparse_results:
            key = (chunk.source, chunk.chunk_index)
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        logger.debug(f"Merged {len(merged)} unique candidates before re-ranking.")

        reranked = self._rerank(query, merged, settings.FINAL_TOP_K)
        logger.debug(f"Returning top {len(reranked)} chunks after re-ranking.")

        return reranked
