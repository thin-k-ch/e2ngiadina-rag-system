import chromadb
from chromadb.config import Settings

class ChromaStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.col = self.client.get_collection(collection)
        except Exception:
            self.col = self.client.create_collection(collection)

    def upsert(self, ids, documents, metadatas, embeddings):
        self.col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def count(self) -> int:
        return self.col.count()
