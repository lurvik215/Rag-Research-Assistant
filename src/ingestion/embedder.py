import torch
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHROMA_DIR


class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding device: {self.device}")
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, chunks: list[dict]) -> int:
        """
        Embeds chunks and stores them in ChromaDB.
        Returns number of chunks stored.
        """
        if not chunks:
            print("No chunks to store.")
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "source_file": c["source_file"],
                "page_num": c["page_num"],
                "chunk_index": c["chunk_index"]
            }
            for c in chunks
        ]

        print(f"Embedding {len(texts)} chunks...")
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        ).tolist()

        # Store in batches of 100 to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                embeddings=vectors[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        print(f"Stored {len(texts)} chunks. Total in DB: {self.collection.count()}")
        return len(texts)
