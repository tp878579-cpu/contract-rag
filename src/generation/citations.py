# src/generation/citations.py
"""Map [1], [2] markers in an answer back to the retrieved chunks."""

import re
from dataclasses import dataclass

from src.ingestion.metadata import format_citation
from src.retrieval.vector_store import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Matches [1], [2], and also [1, 2] or [1,2]
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SNIPPET_CHARS = 300


@dataclass
class Citation:
    number: int           # the [n] shown in the answer
    label: str            # "sample.pdf, p.1, 2. Termination."
    source: str
    page_number: int
    section: str
    snippet: str          # start of the chunk text, for display
    full_text: str
    score: float


def extract_cited_numbers(answer: str, max_number: int) -> list[int]:
    """Return the valid source numbers used in the answer, in order of first use."""
    seen: list[int] = []
    for match in _MARKER_RE.finditer(answer):
        for part in match.group(1).split(","):
            n = int(part.strip())
            if 1 <= n <= max_number and n not in seen:
                seen.append(n)
    return seen


def strip_invalid_markers(answer: str, max_number: int) -> str:
    """Remove markers that point to sources that don't exist, e.g. [9]."""
    def fix(match: re.Match) -> str:
        nums = [int(p.strip()) for p in match.group(1).split(",")]
        valid = [n for n in nums if 1 <= n <= max_number]
        if len(valid) == len(nums):
            return match.group(0)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = _MARKER_RE.sub(fix, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def build_citations(answer: str, results: list[SearchResult]) -> tuple[str, list[Citation]]:
    """Return (cleaned answer, citations actually used)."""
    answer = strip_invalid_markers(answer, max_number=len(results))
    numbers = extract_cited_numbers(answer, max_number=len(results))

    citations = []
    for n in numbers:
        r = results[n - 1]
        text = r.text.strip()
        citations.append(Citation(
            number=n,
            label=format_citation(r.metadata),
            source=r.metadata.get("source", "unknown"),
            page_number=r.metadata.get("page_number", 0),
            section=r.metadata.get("section", ""),
            snippet=text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
            full_text=text,
            score=r.score,
        ))

    if not citations:
        logger.warning("Answer contains no valid citations")
    return answer, citations