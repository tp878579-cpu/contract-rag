# src/retrieval/reranker.py
"""Cross-encoder reranking: re-score candidate chunks by reading query + chunk together."""

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.vector_store import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self, model_name: str = settings.reranker_model):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Load on first use."""
        if self._model is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = settings.top_k_final,
    ) -> list[SearchResult]:
        """Re-order results by relevance. The new score is 0 to 1 (higher = better)."""
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        raw_scores = [float(s) for s in self.model.predict(pairs)]

        # Some sentence-transformers versions already return 0-1 scores, others return
        # raw logits. Only convert when the values are clearly logits.
        if any(s < 0.0 or s > 1.0 for s in raw_scores):
            scores = [_sigmoid(s) for s in raw_scores]
        else:
            scores = raw_scores

        reranked = []
        for r, score in zip(results, scores):
            # Copy metadata so stored data is never modified
            metadata = {**r.metadata, "rerank_score": round(score, 4)}
            reranked.append(SearchResult(id=r.id, text=r.text, metadata=metadata, score=score))

        reranked.sort(key=lambda r: r.score, reverse=True)
        logger.info("Reranked %d candidate(s) -> keeping %d", len(results), min(top_k, len(reranked)))
        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Shared instance, so the model is loaded only once per process."""
    return Reranker()