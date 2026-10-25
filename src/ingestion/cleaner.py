# src/ingestion/cleaner.py
"""Clean raw page text: strip headers/footers/page numbers, fix whitespace."""

import re
from collections import Counter
from dataclasses import replace

from src.ingestion.loader import PageContent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lines that are just a page number, e.g. "Page 3 of 10", "- 3 -", "3/10", "3"
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$"),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
]

EDGE_LINES = 3        # how many lines at the top/bottom of a page to inspect
MIN_REPEAT_RATIO = 0.5  # line must appear on >= 50% of pages to count as header/footer
MIN_PAGES_FOR_DETECTION = 3


def _is_page_number(line: str) -> bool:
    return any(p.match(line) for p in _PAGE_NUMBER_PATTERNS)


def _signature(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' look the same."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_indices(n_lines: int) -> set[int]:
    top = set(range(min(EDGE_LINES, n_lines)))
    bottom = set(range(max(0, n_lines - EDGE_LINES), n_lines))
    return top | bottom


def _find_repeated_lines(pages: list[PageContent]) -> set[str]:
    """Find header/footer lines that repeat across many pages."""
    if len(pages) < MIN_PAGES_FOR_DETECTION:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.text.split("\n")
        sigs = {
            _signature(lines[i])
            for i in _edge_indices(len(lines))
            if lines[i].strip()
        }
        counts.update(sigs)  # set -> each page counts a line only once

    threshold = max(MIN_PAGES_FOR_DETECTION, int(len(pages) * MIN_REPEAT_RATIO))
    repeated = {sig for sig, c in counts.items() if c >= threshold}
    if repeated:
        logger.info("Detected %d repeated header/footer line(s)", len(repeated))
    return repeated


def normalize_text(text: str) -> str:
    """Fix line endings, hyphenation and whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "")   # nbsp, soft hyphen
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)          # termi-\nnation -> termination
    text = re.sub(r"[ \t]+", " ", text)                          # collapse spaces
    text = re.sub(r" *\n *", "\n", text)                         # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)                       # max one blank line
    return text.strip()


def clean_pages(pages: list[PageContent]) -> list[PageContent]:
    """Clean a list of pages. Pages that end up empty are dropped."""
    repeated = _find_repeated_lines(pages)
    cleaned: list[PageContent] = []

    for page in pages:
        lines = page.text.split("\n")
        edge = _edge_indices(len(lines))

        kept = []
        for i, line in enumerate(lines):
            if i in edge and (_is_page_number(line) or _signature(line) in repeated):
                continue
            kept.append(line)

        text = normalize_text("\n".join(kept))
        if text:
            cleaned.append(replace(page, text=text))

    logger.info("Cleaned %d page(s) -> %d kept", len(pages), len(cleaned))
    return cleaned