"""
RAG generation chain — assembles context from retrieved chunks
and calls Ollama (local LLM) to generate a grounded, cited answer.
"""
from dataclasses import dataclass

import ollama
from loguru import logger

from src.config import settings
from src.retrieval.retriever import HybridRetriever, RetrievedChunk

# Use configured base URL so Docker containers can reach host Ollama
_ollama = ollama.Client(host=settings.OLLAMA_BASE_URL)


SYSTEM_PROMPT = """You are LexRAG, an expert legal research assistant.

Your job is to answer legal questions based ONLY on the provided context passages.

Rules:
1. Answer using ONLY information from the context. Never use outside knowledge.
2. Always cite the source filename and chunk index for every claim, like: [Source: filename.pdf, chunk 3]
3. If the context does not contain enough information to answer, say: "The provided documents do not contain sufficient information to answer this question."
4. Be precise and use proper legal terminology.
5. Do not speculate or interpret beyond what the text states.
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    query: str


@dataclass
class ChatMessage:
    role: str   # "user" or "assistant"
    content: str


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Passage {i+1}] Source: {chunk.filename}, chunk {chunk.chunk_index}\n"
            f"{chunk.content}\n"
        )
    return "\n---\n".join(parts)


class RAGChain:
    def __init__(self):
        self.retriever = HybridRetriever()

    def query(self, question: str, filters: dict = None,
              stream: bool = False) -> RAGResponse:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build context
          3. Generate grounded answer via local LLM

        filters example:
            {"doc_type": "contract", "jurisdiction": "NY"}
        """
        logger.info(f"Query: {question}")

        # Step 1: Retrieve
        chunks = self.retriever.retrieve(question, filters=filters)

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found. Please ingest some documents first.",
                sources=[],
                query=question,
            )

        # Step 2: Build context
        context = build_context(chunks)

        # Step 3: Generate
        user_message = f"""Context passages:
{context}

Question: {question}

Answer (cite sources using [Source: filename, chunk N] format):"""

        logger.debug("Calling Ollama...")
        response = _ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer = response["message"]["content"]
        logger.success("Answer generated.")

        return RAGResponse(answer=answer, sources=chunks, query=question)

    def chat(self, question: str, history: list[ChatMessage],
             filters: dict = None) -> RAGResponse:
        """
        Conversational RAG — retrieves context for the latest question,
        then passes the full conversation history to the LLM so it can
        reference prior turns.
        """
        logger.info(f"Chat turn: {question}")

        chunks = self.retriever.retrieve(question, filters=filters)

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found. Please ingest some documents first.",
                sources=[],
                query=question,
            )

        context = build_context(chunks)

        # Build message list: system → prior turns → new user turn with context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({
            "role": "user",
            "content": f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer (cite sources using [Source: filename, chunk N] format):",
        })

        logger.debug("Calling Ollama (chat)...")
        response = _ollama.chat(model=settings.OLLAMA_MODEL, messages=messages)
        answer = response["message"]["content"]
        logger.success("Chat answer generated.")

        return RAGResponse(answer=answer, sources=chunks, query=question)

    def stream_query(self, question: str, filters: dict = None):
        """
        Generator version — yields text tokens for streaming UIs.
        Usage: for token in chain.stream_query("..."): print(token, end="")
        """
        chunks = self.retriever.retrieve(question, filters=filters)

        if not chunks:
            yield "No relevant documents found."
            return

        context = build_context(chunks)
        user_message = f"""Context passages:
{context}

Question: {question}

Answer (cite sources):"""

        stream = _ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )

        for chunk_response in stream:
            token = chunk_response["message"]["content"]
            if token:
                yield token

        # Yield sources at the end as metadata
        yield "\n\n__SOURCES__\n" + "\n".join(
            f"- {c.filename} (chunk {c.chunk_index}, score: {c.score:.3f})"
            for c in chunks
        )
