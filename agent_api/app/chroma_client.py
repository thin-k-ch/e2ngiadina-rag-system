import chromadb
from chromadb.config import Settings

class ChromaClient:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(collection)

    def search(self, query_embedding, top_k: int = 10):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
