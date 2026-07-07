import warnings
warnings.filterwarnings("ignore")

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.llm_runner import LLMRunner
from src.generation.prompt_builder import build_prompt

# ── In-memory ChromaDB — resets when app restarts ─────────────
_chroma_client = chromadb.EphemeralClient()
_collection = _chroma_client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)

# ── Shared embedding model — loaded once ───────────────────────
_device = "cuda" if torch.cuda.is_available() else "cpu"
_embed_model = SentenceTransformer(EMBED_MODEL, device=_device)

# ── Shared components ──────────────────────────────────────────
_loader = PDFLoader()
_chunker = Chunker()
_embedder = Embedder(_collection)
_embedder.model = _embed_model          # reuse same model instance
_retriever = Retriever(_collection, _embed_model)
_llm = LLMRunner()

# Track ingested papers in memory
_ingested_papers: set = set()

def ingest(pdf_path: str) -> int:
    """
    Ingests a PDF into in-memory ChromaDB.
    Skips if already ingested this session.
    """
    source_file = os.path.basename(pdf_path)

    if source_file in _ingested_papers:
        return 0

    # Run the pipeline
    pages = _loader.load(pdf_path)
    if not pages:
        print(f"No text extracted from '{source_file}'. Skipping.")
        return 0

    chunks = _chunker.chunk(pages)
    stored = _embedder.store(chunks)
    _ingested_papers.add(source_file)

    return stored

def query(question: str, paper_filter: str = None) -> dict:
    """
    Full RAG query pipeline: question → retrieve → prompt → LLM → answer.
    paper_filter: optionally restrict search to one paper by filename.
    Returns {answer, sources}
    """
    # Retrieve relevant chunks
    chunks = _retriever.retrieve(question, paper_filter=paper_filter)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded papers.",
            "sources": []
        }

    # Build grounded prompt
    prompt = build_prompt(question, chunks)

    # Generate answer
    answer = _llm.generate(prompt)

    # Format sources for UI display
    sources = [
        {
            "file": c["source_file"],
            "page": c["page_num"],
            "snippet": c["text"][:200]
        }
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}

def get_indexed_papers() -> list[str]:
    """
    Returns papers ingested in this session.
    """
    return sorted(list(_ingested_papers))
