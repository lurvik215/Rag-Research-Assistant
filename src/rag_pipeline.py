import warnings
warnings.filterwarnings("ignore")

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, TOP_K

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.llm_runner import LLMRunner
from src.generation.prompt_builder import build_prompt
from src.ingestion.image_extractor import ImageExtractor

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
_image_extractor = ImageExtractor()
# Track ingested papers in memory
_ingested_papers: set = set()
def ingest(pdf_path: str, extract_images: bool = True) -> dict:
    """
    Full ingestion pipeline: PDF → text chunks + image chunks → ChromaDB.
    extract_images: if True, uses Vision LLM to describe figures.
    Returns dict: {text_chunks, caption_chunks, image_chunks, total}
    """
    source_file = os.path.basename(pdf_path)

    if source_file in _ingested_papers:
        return {"text_chunks": 0, "caption_chunks": 0,
                "image_chunks": 0, "total": 0}

    # Safety: delete stale chunks
    try:
        existing = _collection.get(where={"source_file": source_file})
        if existing["ids"]:
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Single PDF read — pages + title
    result = _loader.load(pdf_path)
    pages  = result["pages"]
    title  = result["title"]

    if not pages:
        print(f"No text extracted from '{source_file}'. Skipping.")
        return {"text_chunks": 0, "caption_chunks": 0,
                "image_chunks": 0, "total": 0}

    # Store title chunk
    if title and title != "Unknown":
        title_chunk = {
            "chunk_index": -1,
            "text":        f"The title of this paper is: {title}",
            "source_file": source_file,
            "page_num":    1,
            "chunk_id":    f"{source_file}_title"
        }
        _embedder.store([title_chunk])

    # Store text chunks
    chunks      = _chunker.chunk(pages)
    text_stored = _embedder.store(chunks)

    # Store figure captions
    captions        = _image_extractor.extract_captions(pages)
    caption_stored  = _embedder.store(captions) if captions else 0

    # Store image descriptions (Vision LLM)
    image_stored = 0
    if extract_images:
        print(f"Extracting image descriptions from {source_file}...")
        image_chunks = _image_extractor.describe_page_images(
            pdf_path, pages
        )
        image_stored = _embedder.store(image_chunks) if image_chunks else 0

    _ingested_papers.add(source_file)

    total = text_stored + caption_stored + image_stored + 1
    print(f"Total: {text_stored} text + {caption_stored} captions "
          f"+ {image_stored} image descriptions")

    return {
        "text_chunks":    text_stored,
        "caption_chunks": caption_stored,
        "image_chunks":   image_stored,
        "total":          total
    }

def query(question: str, paper_filter: list = None,
          model: str = None) -> dict:
    """
    RAG query across one or multiple papers.
    paper_filter: list of filenames to search in.
                  None = search all ingested papers.
    """

    counting_words = ["how many", "count", "list all",
                      "how much", "number of", "total", "all the"]
    is_counting = any(w in question.lower() for w in counting_words)
    top_k = 15 if is_counting else TOP_K

    # Detect title/author questions — always fetch title chunk directly
    title_words = ["title", "name of the paper", "paper called",
                   "what is this paper", "paper about"]
    is_title_question = any(w in question.lower() for w in title_words)

    if is_title_question and paper_filter:
        # Directly fetch title chunks for all papers in filter
        title_chunks = []
        for paper in (paper_filter if isinstance(paper_filter, list)
                      else [paper_filter]):
            try:
                result = _collection.get(
                    ids=[f"{paper}_title"],
                    include=["documents", "metadatas"]
                )
                if result["documents"]:
                    title_chunks.append({
                        "text":        result["documents"][0],
                        "source_file": paper,
                        "page_num":    1,
                        "distance":    0.0   # perfect match
                    })
            except Exception:
                pass

        if title_chunks:
            # Merge title chunks with semantic results
            semantic = _retriever.retrieve(
                question,
                top_k=3,
                paper_filter=paper_filter[0] if isinstance(
                    paper_filter, list) else paper_filter
            )
            chunks = title_chunks + semantic
            prompt = build_prompt(question, chunks)
            answer = _llm.generate(prompt, model=model)
            sources = [{"file": c["source_file"], "page": c["page_num"],
                        "snippet": c["text"][:200]} for c in chunks]
            return {"answer": answer, "sources": sources}
    # ── Multi-paper retrieval ─────────────────────────────────
    if paper_filter and len(paper_filter) > 0:
        all_chunks = []
        per_paper_k = max(2, top_k // len(paper_filter))

        for paper in paper_filter:
            chunks = _retriever.retrieve(
                question,
                top_k=per_paper_k,
                paper_filter=paper
            )
            all_chunks.extend(chunks)

        # Sort by relevance and take top_k overall
        all_chunks.sort(key=lambda x: x["distance"])
        chunks = all_chunks[:top_k]
    else:
        chunks = _retriever.retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded papers.",
            "sources": []
        }

    # Build grounded prompt
    prompt = build_prompt(question, chunks)

    # Generate answer
    answer = _llm.generate(prompt,model=model)

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
