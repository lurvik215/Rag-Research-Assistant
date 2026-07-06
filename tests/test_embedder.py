import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, EMBED_MODEL


def test_vector_dimensions():
    model = SentenceTransformer(EMBED_MODEL)
    vector = model.encode(["test sentence"]).tolist()[0]
    assert len(vector) == 384, f"Expected 384 dims, got {len(vector)}"


def test_chromadb_has_chunks():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("research_papers")
    assert col.count() > 0, "ChromaDB is empty"


def test_metadata_keys_exist():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("research_papers")
    results = col.get(limit=5, include=["metadatas"])
    for meta in results["metadatas"]:
        assert "source_file" in meta
        assert "page_num" in meta
        assert "chunk_index" in meta
