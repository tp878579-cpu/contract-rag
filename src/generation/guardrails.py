# src/generation/guardrails.py
"""Guardrails: refuse when retrieval is weak, and flag unsupported answers."""

import re
from dataclasses import dataclass, field

from src.config import settings
from src.generation.citations import Citation
from src.generation.prompts import NOT_FOUND_MESSAGE
from src.retrieval.vector_store import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Raw cosine similarity from bge-small is high even for unrelated text,
# so it needs a higher threshold than the reranker's 0-1 score. Tune on your data.
MIN_VECTOR_SCORE = 0.45

_MARKER_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class GuardrailResult:
    answer: str
    refused: bool = False
    warnings: list[str] = field(default_factory=list)


def best_relevance(results: list[SearchResult]) -> tuple[float | None, float]:
    """Return (best score, threshold to compare it with), or (None, 0) if unknown."""
    rerank = [r.metadata["rerank_score"] for r in results if "rerank_score" in r.metadata]
    if rerank:
        return max(rerank), settings.min_relevance_score

    vector = [r.metadata["vector_score"] for r in results if "vector_score" in r.metadata]
    if vector:
        return max(vector), MIN_VECTOR_SCORE

    return None, 0.0   # e.g. keyword-only search: no meaningful score to judge


def check_retrieval(results: list[SearchResult]) -> str | None:
    """Return the refusal message if retrieval is too weak, otherwise None."""
    if not results:
        logger.info("Guardrail: no results retrieved")
        return NOT_FOUND_MESSAGE

    score, threshold = best_relevance(results)
    if score is not None and score < threshold:
        logger.info("Guardrail: best relevance %.3f < %.3f, refusing", score, threshold)
        return NOT_FOUND_MESSAGE
    return None


def is_not_found(answer: str) -> bool:
    text = answer.lower().replace("\u2019", "'")
    return "couldn't find this" in text or "could not find this" in text


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(_MARKER_RE.sub("", text))}


def check_answer(
    answer: str,
    citations: list[Citation],
    results: list[SearchResult],
) -> GuardrailResult:
    """Inspect the generated answer. Warnings are for the UI/logs and never block it."""
    if not answer.strip():
        return GuardrailResult(NOT_FOUND_MESSAGE, refused=True, warnings=["Empty answer"])

    if is_not_found(answer):
        return GuardrailResult(NOT_FOUND_MESSAGE, refused=True)

    warnings = []
    if not citations:
        warnings.append("Answer has no valid citations. Verify it against the sources.")

    # Figures in the answer should exist somewhere in the retrieved text
    source_numbers = _numbers(" ".join(r.text for r in results))
    unsupported = sorted(_numbers(answer) - source_numbers)
    if unsupported:
        warnings.append(f"Numbers not found in the sources: {', '.join(unsupported)}")

    for w in warnings:
        logger.warning("Guardrail: %s", w)
    return GuardrailResult(answer, refused=False, warnings=warnings)