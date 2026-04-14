"""
CLI script to ingest documents.
Usage:
    python scripts/ingest.py --path data/raw/contract.pdf --doc_type contract --jurisdiction NY
    python scripts/ingest.py --path data/raw/ --doc_type court_opinion
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.ingestor import Ingestor
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into LexRAG.")
    parser.add_argument("--path", required=True, help="Path to PDF or directory of PDFs")
    parser.add_argument("--doc_type", default="unknown", help="Document type (contract, court_opinion, statute)")
    parser.add_argument("--jurisdiction", default="unknown", help="Jurisdiction (NY, CA, Federal, etc.)")
    args = parser.parse_args()

    metadata = {"doc_type": args.doc_type, "jurisdiction": args.jurisdiction}
    ingestor = Ingestor()
    path = Path(args.path)

    if path.is_dir():
        total = ingestor.ingest_directory(str(path), metadata)
        logger.success(f"Total chunks ingested: {total}")
    elif path.is_file() and path.suffix == ".pdf":
        total = ingestor.ingest_file(str(path), metadata)
        logger.success(f"Chunks ingested: {total}")
    else:
        logger.error("Path must be a PDF file or a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
