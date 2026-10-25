# evaluation/metrics/generation.py
"""Answer-quality metrics. Cheap and deterministic: no LLM judge needed."""

import re

from evaluation.metrics.retrieval import text_is_relevant


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def answer_correct(answer: str, qa: dict) -> bool | None:
    """True if every expected keyword appears in the answer. None if none were given."""
    keywords = qa.get("answer_keywords", [])
    if not keywords:
        return None
    body = _norm(answer)
    return all(_norm(k) in body for k in keywords)


def citation_correct(citations, qa: dict) -> bool:
    """True if at least one cited source is the gold clause."""
    return any(text_is_relevant(c.source, c.full_text, qa) for c in citations)