# Evaluation report
Date: 2026-10-25 | Questions: 5

## Retrieval (is the right clause in the top-k?)

| Config | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Latency |
|---|---|---|---|---|---|
| Vector only | 100% | 100% | 100% | 1.00 | 1312 ms |
| BM25 only | 67% | 67% | 67% | 0.67 | 0 ms |
| Hybrid (RRF) | 100% | 100% | 100% | 1.00 | 59 ms |
