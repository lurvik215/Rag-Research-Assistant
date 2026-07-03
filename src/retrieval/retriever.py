import torch
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHROMA_DIR, TOP_K


class Retriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection("research_papers")

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embeds the question and finds the most similar chunks in ChromaDB.
        Returns list of {text, source_file, page_num, distance}
        """
        query_vector = self.model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source_file": meta["source_file"],
                "page_num": meta["page_num"],
                "distance": round(dist, 4)
            })

        return chunks
