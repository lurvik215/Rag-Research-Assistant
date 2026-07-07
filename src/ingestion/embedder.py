import warnings
warnings.filterwarnings("ignore")

import torch
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


class Embedder:
    def __init__(self, collection):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.collection = collection

    def store(self, chunks: list[dict]) -> int:
        if not chunks:
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

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                embeddings=vectors[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        return len(texts)
