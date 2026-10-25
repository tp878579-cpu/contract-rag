# src/pipeline.py
"""The whole RAG flow behind one call: RAGPipeline().ask("question")."""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.generation.citations import Citation, build_citations
from src.generation.guardrails import check_answer, check_retrieval
from src.generation.llm import get_llm
from src.generation.prompts import NOT_FOUND_MESSAGE, build_messages
from src.ingestion.chunker import chunk_pages
from src.ingestion.cleaner import clean_pages
from src.ingestion.loader import load_document
from src.ingestion.metadata import chunks_to_records
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import SearchResult, VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANDIDATES = 10   # retrieve wide, then keep the best settings.top_k_final


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    refused: bool = False
    sources: list[SearchResult] = field(default_factory=list)   # chunks given to the LLM
    mode: str = "hybrid"
    reranked: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_s: float = 0.0
    llm_s: float = 0.0
    total_s: float = 0.0


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.retriever = HybridRetriever(self.store)

    # ---------- ingestion ----------
    def ingest_file(self, file_path: str | Path) -> int:
        """Load, clean, chunk and store one file. Returns the number of chunks."""
        pages = clean_pages(load_document(file_path))
        records = chunks_to_records(chunk_pages(pages))
        count = self.store.add_records(records)
        self.retriever.refresh()          # so BM25 sees the new chunks
        return count

    def remove_file(self, source: str) -> None:
        self.store.delete_source(source)
        self.retriever.refresh()

    # ---------- retrieval ----------
    def _retrieve(
        self, question: str, mode: str, source: str | None, use_reranker: bool
    ) -> tuple[list[SearchResult], bool]:
        candidates = self.retriever.search(
            question, top_k=CANDIDATES, source=source, mode=mode
        )
        if not candidates:
            return [], False

        if use_reranker:
            try:
                top = get_reranker().rerank(question, candidates, top_k=settings.top_k_final)
                return top, True
            except Exception as e:      # e.g. model download failed
                logger.error("Reranker failed (%s). Falling back to un-reranked results.", e)

        return candidates[: settings.top_k_final], False

    # ---------- main entry point ----------
    def ask(
        self,
        question: str,
        source: str | None = None,
        mode: str = "hybrid",             # "hybrid" | "vector" | "bm25"
        use_reranker: bool | None = None,
    ) -> RAGAnswer:
        t0 = time.perf_counter()
        use_reranker = settings.use_reranker if use_reranker is None else use_reranker
        result = RAGAnswer(question=question, answer="", mode=mode)

        if not question.strip():
            result.answer, result.refused = "Please type a question.", True
            return result

        # 1) retrieve + rerank
        t = time.perf_counter()
        sources, reranked = self._retrieve(question, mode, source, use_reranker)
        result.retrieval_s = time.perf_counter() - t
        result.sources, result.reranked = sources, reranked

        # 2) guardrail before spending an LLM call
        refusal = check_retrieval(sources)
        if refusal:
            result.answer, result.refused = refusal, True
            result.total_s = time.perf_counter() - t0
            return result

        # 3) generate
        system, messages = build_messages(question, sources)
        llm_resp = get_llm().generate(system, messages)
        result.llm_s = llm_resp.latency_s
        result.input_tokens = llm_resp.input_tokens
        result.output_tokens = llm_resp.output_tokens

        # 4) citations (same `sources` list as the prompt, so numbers line up)
        answer, citations = build_citations(llm_resp.text, sources)

        # 5) check the answer
        checked = check_answer(answer, citations, sources)
        result.answer = checked.answer
        result.refused = checked.refused
        result.warnings = checked.warnings
        result.citations = [] if checked.refused else citations

        result.total_s = time.perf_counter() - t0
        logger.info(
            "ask() done in %.2fs (retrieval %.2fs, llm %.2fs, refused=%s)",
            result.total_s, result.retrieval_s, result.llm_s, result.refused,
        )
        return result


def _print_answer(r: RAGAnswer) -> None:
    print("\nANSWER:\n" + r.answer)
    if r.citations:
        print("\nSOURCES:")
        for c in r.citations:
            print(f"  [{c.number}] {c.label}")
    for w in r.warnings:
        print("WARNING:", w)
    print(f"\n{r.total_s:.1f}s total | {r.input_tokens} in / {r.output_tokens} out tokens | "
          f"reranked={r.reranked}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How can I end the contract?"
    _print_answer(RAGPipeline().ask(question))