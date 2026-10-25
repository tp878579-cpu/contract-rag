# src/retrieval/vector_store.py
"""Store chunk vectors in Chroma and search them by meaning."""

from dataclasses import dataclass

import chromadb

from src.config import settings
from src.embeddings.embedder import get_embedder
from src.ingestion.metadata import ChunkRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    score: float          # similarity: higher = more relevant (about 0 to 1)


class VectorStore:
    def __init__(self, collection_name: str = settings.collection_name):
        self.client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine distance
        )
        self.embedder = get_embedder()

    # ---------- write ----------
    def add_records(self, records: list[ChunkRecord], batch_size: int = 64) -> int:
        """Embed and store records. Same IDs overwrite, so re-ingesting is safe."""
        if not records:
            return 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r.text for r in batch]
            self.collection.upsert(
                ids=[r.id for r in batch],
                documents=texts,
                metadatas=[r.metadata for r in batch],
                embeddings=self.embedder.embed_documents(texts),
            )
        logger.info("Stored %d record(s). Collection size: %d", len(records), self.count())
        return len(records)

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from one file."""
        self.collection.delete(where={"source": source})
        logger.info("Deleted chunks from %s", source)

    def reset(self) -> None:
        """Wipe the whole collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection reset")

    # ---------- read ----------
    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_vector,
        source: str | None = None,
    ) -> list[SearchResult]:
        """Return the top_k most similar chunks, optionally limited to one file."""
        if self.count() == 0:
            return []

        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            where={"source": source} if source else None,
        )

        results = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(SearchResult(id=id_, text=doc, metadata=meta, score=1.0 - dist))
        return results

    def get_all(self) -> list[SearchResult]:
        """Return every stored chunk (the BM25 index is built from this)."""
        data = self.collection.get()
        return [
            SearchResult(id=i, text=d, metadata=m, score=0.0)
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def list_sources(self) -> list[str]:
        return sorted({r.metadata["source"] for r in self.get_all()})