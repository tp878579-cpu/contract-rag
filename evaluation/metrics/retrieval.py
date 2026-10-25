# evaluation/metrics/retrieval.py
"""Retrieval metrics: is the right clause in the top-k results?"""

import re


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def text_is_relevant(source: str, text: str, qa: dict) -> bool:
    """Relevant = right file AND contains one of the evidence phrases."""
    if qa.get("source") and source != qa["source"]:
        return False
    body = _norm(text)
    return any(_norm(e) in body for e in qa["evidence"])


def first_relevant_rank(results, qa: dict) -> int | None:
    """1-based rank of the first relevant result, or None."""
    for i, r in enumerate(results, start=1):
        if text_is_relevant(r.metadata.get("source", ""), r.text, qa):
            return i
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose right clause appears in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], k: int) -> float:
    """Mean Reciprocal Rank: 1 if first result is right, 0.5 if second, and so on."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)