# src/ingestion/metadata.py
"""Attach IDs and metadata to chunks, and convert them for the vector database."""

import hashlib
from dataclasses import dataclass

from src.ingestion.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkRecord:
    """A chunk ready to be stored: unique ID + text + metadata."""
    id: str
    text: str
    metadata: dict


def make_chunk_id(chunk: Chunk) -> str:
    """Deterministic ID: re-ingesting the same file gives the same IDs,
    so the database can overwrite instead of creating duplicates."""
    raw = f"{chunk.source}|{chunk.chunk_index}|{chunk.text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_to_record(chunk: Chunk) -> ChunkRecord:
    # Vector DBs only accept str / int / float / bool values in metadata
    metadata = {
        "source": chunk.source,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "char_count": len(chunk.text),
    }
    return ChunkRecord(id=make_chunk_id(chunk), text=chunk.text, metadata=metadata)


def chunks_to_records(chunks: list[Chunk]) -> list[ChunkRecord]:
    records = [chunk_to_record(c) for c in chunks]

    # Safety net: drop any accidental duplicate IDs
    unique = {}
    for r in records:
        unique.setdefault(r.id, r)
    if len(unique) != len(records):
        logger.warning("Dropped %d duplicate chunk(s)", len(records) - len(unique))

    logger.info("Prepared %d record(s)", len(unique))
    return list(unique.values())


def to_vector_db_format(records: list[ChunkRecord]) -> tuple[list[str], list[str], list[dict]]:
    """Return (ids, documents, metadatas), the three lists Chroma's add() takes."""
    ids = [r.id for r in records]
    documents = [r.text for r in records]
    metadatas = [r.metadata for r in records]
    return ids, documents, metadatas


def format_citation(metadata: dict) -> str:
    """Human-readable source label, e.g. 'nda.pdf, p.3, 2. Termination'."""
    parts = [metadata.get("source", "unknown")]
    if metadata.get("page_number") is not None:
        parts.append(f"p.{metadata['page_number']}")
    section = metadata.get("section")
    if section and section != "General":
        parts.append(section)
    return ", ".join(parts)