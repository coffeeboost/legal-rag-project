# ⚖️ LexRAG — Legal Document Intelligence

A production-grade RAG system for legal contract analysis and case law search.
Built with a fully local, free stack: Ollama + ChromaDB + sentence-transformers.

## Architecture

```
PDF Documents
     │
     ▼
Ingestion (PyMuPDF → chunks → embeddings → ChromaDB)
     │
     ▼
Hybrid Retrieval (Dense vector search + BM25 sparse search)
     │
     ▼
Re-ranking (Cross-encoder)
     │
     ▼
Generation (Ollama / Llama 3.1 8B) with citation grounding
     │
     ▼
FastAPI + Streamlit UI
```

## Stack (100% Free / Local)

| Component      | Tool                                  |
|----------------|---------------------------------------|
| LLM            | Ollama (Llama 3.1 8B)                 |
| Embeddings     | BAAI/bge-base-en-v1.5                 |
| Vector DB      | ChromaDB (local)                      |
| Sparse search  | BM25 (rank_bm25)                      |
| Re-ranker      | cross-encoder/ms-marco-MiniLM-L-6-v2  |
| API            | FastAPI                               |
| UI             | Streamlit                             |
| Evaluation     | RAGAS                                 |

## Setup

### 1. Install Ollama and pull the model
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b
```

### 2. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ingest documents
```bash
# Single file
python scripts/ingest.py --path data/raw/contract.pdf --doc_type contract --jurisdiction NY

# Entire directory
python scripts/ingest.py --path data/raw/ --doc_type court_opinion
```

### 4. Start the API
```bash
uvicorn src.api.app:app --reload
```

### 5. Start the UI
```bash
streamlit run ui/app.py
```

### 6. Run RAGAS evaluation
```bash
python evals/evaluate.py
```

## Free Data Sources

- [CUAD Dataset](https://huggingface.co/datasets/cuad) — 500 annotated contracts
- [CourtListener API](https://www.courtlistener.com/api/) — US court opinions
- [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) — Public company filings

## Project Structure

```
lexrag/
├── data/
│   ├── raw/          # PDFs go here
│   └── processed/    # ChromaDB stored here
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   └── ingestor.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── chain.py
│   └── api/
│       └── app.py
├── ui/
│   └── app.py
├── scripts/
│   └── ingest.py
├── evals/
│   └── evaluate.py
└── requirements.txt
```
