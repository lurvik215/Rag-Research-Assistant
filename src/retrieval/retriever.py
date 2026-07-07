import warnings
warnings.filterwarnings("ignore")

from config import TOP_K


class Retriever:
    def __init__(self, collection, embed_model):
        self.collection = collection
        self.model = embed_model

    def retrieve(self, question: str, top_k: int = TOP_K,
                 paper_filter: str = None) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_vector = self.model.encode([question]).tolist()
        where = {"source_file": paper_filter} if paper_filter else None

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=min(top_k, self.collection.count()),
            where=where,
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
