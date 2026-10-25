# src/config.py
"""Central configuration. Every other module imports settings from here."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = contract-rag/
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # ---------- Paths ----------
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    vector_db_dir: Path = BASE_DIR / "data" / "vector_db"

    # ---------- Chunking ----------
    chunk_size: int = 800          # max characters per chunk
    chunk_overlap: int = 150       # overlap between neighbouring chunks

    # ---------- Embeddings (local and free) ----------
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ---------- Vector store ----------
    collection_name: str = "contracts"

    # ---------- Retrieval ----------
    top_k_vector: int = 10         # candidates from vector search
    top_k_bm25: int = 10           # candidates from keyword search
    top_k_final: int = 5           # chunks sent to the LLM
    rrf_k: int = 60                # constant for Reciprocal Rank Fusion
    use_reranker: bool = False
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # ---------- LLM ----------
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")   # anthropic | openai
    llm_model: str = os.getenv("LLM_MODEL", "")                  # set in .env
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = 0.0       # keep answers deterministic for contracts
    max_tokens: int = 1000

    # ---------- Guardrails ----------
    min_relevance_score: float = 0.2   # below this -> "I don't know"

    # ---------- Misc ----------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_cache: bool = True


settings = Settings()

# Make sure data folders exist
for _p in (settings.raw_dir, settings.processed_dir, settings.vector_db_dir):
    _p.mkdir(parents=True, exist_ok=True)