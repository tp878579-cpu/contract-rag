# src/retrieval/hybrid.py
"""Hybrid retrieval: merge vector search and BM25 with Reciprocal Rank Fusion."""

from src.config import settings
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import SearchResult, VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = settings.rrf_k,
) -> list[SearchResult]:
    """Merge several ranked lists. Score of a chunk = sum of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    chunks: dict[str, SearchResult] = {}
    extra: dict[str, dict] = {}

    for name, results in enumerate(result_lists):
        label = "vector_score" if name == 0 else "bm25_score"
        for rank, r in enumerate(results, start=1):
            fused[r.id] = fused.get(r.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(r.id, r)
            extra.setdefault(r.id, {})[label] = round(r.score, 4)

    merged = []
    for id_, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
        base = chunks[id_]
        # Copy metadata so we never modify what is stored in the database.
        # vector_score keeps the raw cosine similarity for the guardrails later.
        metadata = {**base.metadata, **extra[id_]}
        merged.append(SearchResult(id=id_, text=base.text, metadata=metadata, score=score))
    return merged


class HybridRetriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.bm25 = BM25Index.from_vector_store(self.store)

    def refresh(self) -> None:
        """Rebuild the keyword index. Call this after ingesting new files."""
        self.bm25 = BM25Index.from_vector_store(self.store)

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_final,
        source: str | None = None,
        mode: str = "hybrid",          # "hybrid" | "vector" | "bm25"
    ) -> list[SearchResult]:
        """Return the top_k chunks. `mode` lets the evaluation compare strategies."""
        vector_hits = (
            self.store.search(query, top_k=settings.top_k_vector, source=source)
            if mode in ("hybrid", "vector") else []
        )
        bm25_hits = (
            self.bm25.search(query, top_k=settings.top_k_bm25, source=source)
            if mode in ("hybrid", "bm25") else []
        )

        if mode == "vector":
            return vector_hits[:top_k]
        if mode == "bm25":
            return bm25_hits[:top_k]

        merged = reciprocal_rank_fusion([vector_hits, bm25_hits])
        logger.info(
            "Hybrid: %d vector + %d keyword -> %d unique", len(vector_hits), len(bm25_hits), len(merged)
        )
        return merged[:top_k]