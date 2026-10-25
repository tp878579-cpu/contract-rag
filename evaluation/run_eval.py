# evaluation/run_eval.py
"""Run the benchmark.

    python -m evaluation.run_eval                    # retrieval + generation
    python -m evaluation.run_eval --retrieval-only   # no LLM calls, free and fast
    python -m evaluation.run_eval --sleep 6          # pause between LLM calls (rate limits)
"""

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from evaluation.metrics.generation import answer_correct, citation_correct
from evaluation.metrics.retrieval import first_relevant_rank, mrr, recall_at_k
from src.pipeline import RAGPipeline
from src.retrieval.reranker import get_reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "datasets" / "qa_pairs.json"
RESULTS_DIR = EVAL_DIR / "results"
K_VALUES = (1, 3, 5)
CANDIDATES = 10

# (name, retrieval mode, use reranker)
CONFIGS = [
    ("Vector only", "vector", False),
    ("BM25 only", "bm25", False),
    ("Hybrid (RRF)", "hybrid", False),
    ("Hybrid + rerank", "hybrid", True),
]


def load_dataset() -> list[dict]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    for qa in data:
        qa.setdefault("answerable", True)
    return data


def eval_retrieval(pipeline: RAGPipeline, dataset: list[dict]) -> list[dict]:
    questions = [q for q in dataset if q["answerable"]]
    rows = []
    for name, mode, rerank in CONFIGS:
        ranks, latencies = [], []
        try:
            for qa in questions:
                t = time.perf_counter()
                results = pipeline.retriever.search(
                    qa["question"], top_k=CANDIDATES, source=qa.get("source"), mode=mode
                )
                if rerank:
                    results = get_reranker().rerank(qa["question"], results, top_k=max(K_VALUES))
                latencies.append(time.perf_counter() - t)
                ranks.append(first_relevant_rank(results[: max(K_VALUES)], qa))
        except Exception as e:
            logger.error("Skipping '%s': %s", name, e)
            continue

        row = {"config": name, "n": len(ranks), "latency_ms": statistics.mean(latencies) * 1000}
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(ranks, k)
        row["mrr@5"] = mrr(ranks, 5)
        rows.append(row)
        logger.info("%s: recall@5=%.2f", name, row["recall@5"])
    return rows


def eval_generation(pipeline: RAGPipeline, dataset: list[dict], sleep_s: float) -> dict:
    answered, unanswerable = [], []
    for i, qa in enumerate(dataset, start=1):
        logger.info("Question %d/%d", i, len(dataset))
        r = pipeline.ask(qa["question"], source=qa.get("source"))
        if qa["answerable"]:
            answered.append((qa, r))
        else:
            unanswerable.append((qa, r))
        if not r.refused:
            time.sleep(sleep_s)       # only LLM calls count against rate limits

    stats: dict = {"n_answerable": len(answered), "n_unanswerable": len(unanswerable)}

    if answered:
        correct = [answer_correct(r.answer, qa) for qa, r in answered]
        scored = [c for c in correct if c is not None]
        stats["answer_accuracy"] = sum(scored) / len(scored) if scored else None
        stats["wrongly_refused"] = sum(r.refused for _, r in answered) / len(answered)
        replied = [(qa, r) for qa, r in answered if not r.refused]
        stats["citation_rate"] = (
            sum(bool(r.citations) for _, r in replied) / len(replied) if replied else None
        )
        stats["citation_accuracy"] = (
            sum(citation_correct(r.citations, qa) for qa, r in replied) / len(replied)
            if replied else None
        )
        stats["avg_latency_s"] = statistics.mean(r.total_s for _, r in answered)
        stats["avg_tokens"] = statistics.mean(r.input_tokens + r.output_tokens for _, r in answered)
        stats["warning_rate"] = sum(bool(r.warnings) for _, r in answered) / len(answered)

    if unanswerable:
        stats["refusal_accuracy"] = sum(r.refused for _, r in unanswerable) / len(unanswerable)

    stats["failures"] = [
        {"question": qa["question"], "answer": r.answer[:200]}
        for qa, r in answered
        if answer_correct(r.answer, qa) is False
    ]
    return stats


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def write_report(retrieval: list[dict], generation: dict | None, n_total: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation report",
        f"Date: {date.today()} | Questions: {n_total}",
        "",
        "## Retrieval (is the right clause in the top-k?)",
        "",
        "| Config | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in retrieval:
        lines.append(
            f"| {r['config']} | {_pct(r['recall@1'])} | {_pct(r['recall@3'])} | "
            f"{_pct(r['recall@5'])} | {r['mrr@5']:.2f} | {r['latency_ms']:.0f} ms |"
        )

    if generation:
        g = generation
        lines += [
            "", "## End-to-end answers", "",
            "| Metric | Value |", "|---|---|",
            f"| Answer accuracy (keyword match) | {_pct(g.get('answer_accuracy'))} |",
            f"| Answers with citations | {_pct(g.get('citation_rate'))} |",
            f"| Citations pointing to the right clause | {_pct(g.get('citation_accuracy'))} |",
            f"| Valid questions wrongly refused | {_pct(g.get('wrongly_refused'))} |",
            f"| Off-topic questions correctly refused | {_pct(g.get('refusal_accuracy'))} |",
            f"| Answers with guardrail warnings | {_pct(g.get('warning_rate'))} |",
        ]
        if "avg_latency_s" in g:
            lines.append(f"| Avg latency | {g['avg_latency_s']:.1f} s |")
            lines.append(f"| Avg tokens per answer | {g['avg_tokens']:.0f} |")
        if g.get("failures"):
            lines += ["", "## Wrong answers (review these)", ""]
            for f in g["failures"]:
                lines.append(f"- **Q:** {f['question']}  \n  **A:** {f['answer']}")

    path = RESULTS_DIR / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "results.json").write_text(
        json.dumps({"retrieval": retrieval, "generation": generation}, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--sleep", type=float, default=4.0, help="seconds between LLM calls")
    args = parser.parse_args()

    dataset = load_dataset()
    pipeline = RAGPipeline()
    if pipeline.store.count() == 0:
        raise SystemExit("Vector store is empty. Run: python -m scripts.ingest_folder")

    retrieval = eval_retrieval(pipeline, dataset)
    generation = None if args.retrieval_only else eval_generation(pipeline, dataset, args.sleep)

    path = write_report(retrieval, generation, len(dataset))
    print("\n" + path.read_text(encoding="utf-8"))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()