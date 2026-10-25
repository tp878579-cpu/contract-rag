# src/ingestion/chunker.py
"""Clause-aware chunking: split at clause headings first, then split long clauses."""

import bisect
import re
from dataclasses import dataclass

from src.config import settings
from src.ingestion.loader import PageContent
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CLAUSE_CHARS = 40   # clauses shorter than this are merged into the next one
LABEL_MAX_CHARS = 60    # max length of the section label

# Patterns that mark the start of a clause / section heading
_HEADING_PATTERNS = [
    # "Article 3", "Section 2.", "Clause IV", "Schedule 1"
    re.compile(r"^(article|section|clause|schedule|annex|appendix)\s+[\divxl]+[.:)]?(\s.*)?$", re.I),
    # "2.1 Payment Terms", "3.2.1 Notice"
    re.compile(r"^\d+(\.\d+)+[.)]?\s+\S"),
    # "1. Term", "2) Termination"
    re.compile(r"^\d+[.)]\s+\S"),
    # "SERVICE AGREEMENT", "CONFIDENTIALITY"
    re.compile(r"^[A-Z][A-Z0-9 ,&'\-/]{3,70}$"),
]


@dataclass
class Chunk:
    text: str
    source: str        # file name
    page_number: int   # page where the chunk starts
    section: str       # clause heading, e.g. "2. Termination."
    chunk_index: int   # position within the document


def _is_heading(line: str) -> bool:
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _build_document(pages: list[PageContent]) -> tuple[str, list[int], list[int]]:
    """Join pages into one text, remembering where each page starts."""
    parts, offsets, numbers = [], [], []
    pos = 0
    for page in pages:
        offsets.append(pos)
        numbers.append(page.page_number)
        parts.append(page.text)
        pos += len(page.text) + 2          # +2 for the "\n\n" separator
    return "\n\n".join(parts), offsets, numbers


def _page_at(offset: int, offsets: list[int], numbers: list[int]) -> int:
    idx = bisect.bisect_right(offsets, offset) - 1
    return numbers[max(idx, 0)]


def _find_headings(text: str) -> list[tuple[int, str]]:
    heads, pos = [], 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _is_heading(stripped):
            heads.append((pos, stripped[:LABEL_MAX_CHARS]))
        pos += len(line) + 1
    return heads


def _split_clauses(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) for each clause, merging tiny ones forward."""
    heads = _find_headings(text)
    if not heads:
        return [(0, len(text), "")]

    if heads[0][0] > 0 and text[: heads[0][0]].strip():
        heads.insert(0, (0, "Preamble"))

    clauses = []
    for i, (start, label) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        clauses.append((start, end, label))

    merged, carry_start = [], None
    for i, (start, end, label) in enumerate(clauses):
        if carry_start is None:
            carry_start = start
        is_last = i == len(clauses) - 1
        if len(text[carry_start:end].strip()) < MIN_CLAUSE_CHARS and not is_last:
            continue
        merged.append((carry_start, end, label))
        carry_start = None
    return merged


def _split_long(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split text > size into overlapping pieces, preferring natural break points."""
    pieces, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_start = start + size // 2
            for sep in ("\n\n", ". ", "; ", "\n", " "):
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        piece = text[start:end]
        lead = len(piece) - len(piece.lstrip())
        piece = piece.strip()
        if piece:
            pieces.append((start + lead, piece))

        if end >= n:
            break
        nxt = end - overlap
        if nxt > start:
            space = text.find(" ", nxt, end)     # don't start mid-word
            start = space + 1 if space != -1 else nxt
        else:
            start = end
    return pieces


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Turn cleaned pages into clause-aware chunks (works across many files)."""
    by_source: dict[str, list[PageContent]] = {}
    for page in pages:
        by_source.setdefault(page.source, []).append(page)

    chunks: list[Chunk] = []
    for source, doc_pages in by_source.items():
        text, offsets, numbers = _build_document(doc_pages)
        index = 0

        for start, end, label in _split_clauses(text):
            clause = text[start:end]
            if not clause.strip():
                continue

            if len(clause) <= chunk_size:
                pieces = [(start + len(clause) - len(clause.lstrip()), clause.strip())]
            else:
                pieces = [(start + off, p) for off, p in _split_long(clause, chunk_size, overlap)]

            for j, (offset, piece) in enumerate(pieces):
                # Continuation pieces get the heading repeated so they keep their context
                body = f"{label}\n{piece}" if (j > 0 and label) else piece
                chunks.append(Chunk(
                    text=body,
                    source=source,
                    page_number=_page_at(offset, offsets, numbers),
                    section=label or "General",
                    chunk_index=index,
                ))
                index += 1

        logger.info("%s -> %d chunk(s)", source, index)
    return chunks