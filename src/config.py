from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-base-en-v1.5"

    # Re-ranker
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 5

    # Retrieval
    DENSE_TOP_K: int = 20       # candidates from vector search
    SPARSE_TOP_K: int = 20      # candidates from BM25
    FINAL_TOP_K: int = 5        # after re-ranking

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "lexrag_docs"

    # Sessions DB
    DATABASE_PATH: str = "./data/sessions.db"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"


settings = Settings()
