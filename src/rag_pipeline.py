import warnings
warnings.filterwarnings("ignore")

import os
import pathlib
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import Retriever

# Initialise components once — reused across all calls
loader = PDFLoader()
chunker = Chunker()
embedder = Embedder()
retriever = Retriever()


def ingest(pdf_path: str) -> int:
    """
    Full ingestion pipeline: PDF → chunks → vectors → ChromaDB.
    Returns number of chunks stored.
    Skips ingestion if paper is already indexed.
    """
    source_file = os.path.basename(pdf_path)

    # Check if already ingested — avoid duplicates
    existing = embedder.collection.get(
        where={"source_file": source_file},
        limit=1
    )
    if existing["ids"]:
        print(f"'{source_file}' already indexed — skipping.")
        return 0

    # Run the pipeline
    pages = loader.load(pdf_path)
    if not pages:
        print(f"No text extracted from '{source_file}'. Skipping.")
        return 0

    chunks = chunker.chunk(pages)
    stored = embedder.store(chunks)

    return stored


def get_indexed_papers() -> list[str]:
    """
    Returns list of unique paper names currently stored in ChromaDB.
    """
    results = embedder.collection.get(include=["metadatas"])
    papers = list({m["source_file"] for m in results["metadatas"]})
    return sorted(papers)
