from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import Chunker
from config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunks_have_required_keys():
    loader = PDFLoader()
    chunker = Chunker()
    pages = loader.load("data/raw/ICSIT.pdf")
    chunks = chunker.chunk(pages)
    for chunk in chunks:
        assert "text" in chunk
        assert "source_file" in chunk
        assert "page_num" in chunk
        assert "chunk_id" in chunk


def test_chunk_size_within_limit():
    loader = PDFLoader()
    chunker = Chunker()
    pages = loader.load("data/raw/ICSIT.pdf")
    chunks = chunker.chunk(pages)
    for chunk in chunks:
        # Allow 10% overflow from splitter behaviour
        assert len(chunk["text"]) <= CHUNK_SIZE * 1.1


def test_overlap_exists():
    loader = PDFLoader()
    chunker = Chunker()
    pages = loader.load("data/raw/ICSIT.pdf")
    chunks = chunker.chunk(pages)
    # Check that consecutive chunks from the same page share some text
    same_page = [c for c in chunks if c["page_num"] == 1]
    if len(same_page) >= 2:
        end_of_first = same_page[0]["text"][-CHUNK_OVERLAP:]
        start_of_second = same_page[1]["text"][:CHUNK_OVERLAP]
        # They should share at least some characters
        overlap_found = any(
            word in start_of_second
            for word in end_of_first.split()[-3:]
        )
        assert overlap_found, "No overlap detected between consecutive chunks"
