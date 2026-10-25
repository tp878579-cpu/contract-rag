# src/retrieval/bm25_index.py
"""Keyword search (BM25) over the chunks stored in the vector database."""

import re

from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.vector_store import SearchResult, VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keeps clause numbers like "4.2" and "3.1.1" as single tokens
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "with",
    "is", "are", "be", "as", "at", "this", "that", "it", "its", "from", "can",
    "i", "my", "me", "do", "does", "what", "how", "which",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, records: list[SearchResult] | None = None):
        self.records: list[SearchResult] = []
        self._bm25: BM25Okapi | None = None
        if records:
            self.build(records)

    @classmethod
    def from_vector_store(cls, store: VectorStore | None = None) -> "BM25Index":
        """Build the index from every chunk currently stored."""
        store = store or VectorStore()
        return cls(store.get_all())

    def build(self, records: list[SearchResult]) -> None:
        self.records = records
        corpus = [tokenize(r.text) for r in records]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        logger.info("BM25 index built with %d chunk(s)", len(records))

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_bm25,
        source: str | None = None,
    ) -> list[SearchResult]:
        """Return the top_k chunks with the best keyword match (score > 0 only)."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.records, scores), key=lambda x: x[1], reverse=True)

        results = []
        for rec, score in ranked:
            if score <= 0:
                break
            if source and rec.metadata.get("source") != source:
                continue
            results.append(SearchResult(
                id=rec.id, text=rec.text, metadata=rec.metadata, score=float(score)
            ))
            if len(results) >= top_k:
                break
        return results