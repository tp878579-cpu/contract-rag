# scripts/ingest_folder.py
"""Ingest every supported file in data/raw into the vector store."""

from src.config import settings
from src.ingestion.chunker import chunk_pages
from src.ingestion.cleaner import clean_pages
from src.ingestion.loader import load_folder
from src.ingestion.metadata import chunks_to_records
from src.retrieval.vector_store import VectorStore


def main() -> None:
    pages = load_folder(settings.raw_dir)
    records = chunks_to_records(chunk_pages(clean_pages(pages)))

    store = VectorStore()
    store.add_records(records)
    print("stored:", store.count(), "| sources:", store.list_sources())


if __name__ == "__main__":
    main()