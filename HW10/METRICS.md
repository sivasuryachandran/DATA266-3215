# HW10 metrics

All numbers come from `results.json` (quality) and `measurements.json` (timing, failures), produced by `rag_pipeline.py` and `measure.py`. Query-level values are in `results.json` under `perquery`.

## Setup

| Item | Value |
|---|---|
| Dataset | `ccdv/arxiv-summarization`, config `document`, first 100 `train` documents |
| Average document length | 6,517 words |
| Vector database | ChromaDB 1.5.9, persistent, cosine HNSW, one collection per chunking |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (normalised) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20 dense candidates |
| Single-turn queries | 14 (S1–S8 reuse the paper's wording, S9–S14 are paraphrased) |
| Dialogues | 6 two-turn dialogues (D1–D6); only turn 2 is scored |
| Relevance | coded rubric: target paper and at least 3 of 3–4 concept patterns (2 for D6) |

## Chunking statistics

| Chunking | Chunks | Mean words | Min words | Max words | Chunk time (s) | Embed time (s) | Chunks per second |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed window (150 / 30 overlap) | 4,724 | 148.8 | 31 | 150 | 0.31 | 2.71 | 1,742 |
| sentence-aware (about 150) | 5,256 | 135.9 | 9 | 300 | 0.38 | 2.28 | 2,307 |

Embedding ran on an NVIDIA GeForce RTX 5090. Building both Chroma collections took 17–24 s across runs.

## Without vs with reranking (14 single-turn queries)

| Chunking | Reranking | P@1 | P@3 | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | no | 0.857 | 0.738 | 0.686 | 0.586 | 0.416 | 0.685 | 0.911 |
| fixed | yes | 1.000 | 0.786 | 0.757 | 0.643 | 0.516 | 0.751 | 1.000 |
| sentence | no | 0.857 | 0.810 | 0.757 | 0.629 | 0.434 | 0.668 | 0.917 |
| sentence | yes | 0.929 | 0.786 | 0.743 | 0.664 | 0.408 | 0.713 | 0.964 |

The chunking comparison is the fixed and sentence rows with the same reranking setting.

## Per-query reciprocal rank for the queries that are not 1.0 everywhere

| Query | fixed, dense | fixed, rerank | sentence, dense | sentence, rerank |
|---|---:|---:|---:|---:|
| S4 (SiC second harmonic) | 1.00 | 1.00 | 1.00 | 0.50 |
| S11 (merging binaries, templates) | 0.50 | 1.00 | 0.33 | 1.00 |
| S14 (accessible astronomy resources) | 0.25 | 1.00 | 0.50 | 1.00 |

All other queries (S1–S3, S5–S10, S12, S13) have reciprocal rank 1.00 in every configuration.

## Without vs with conversational memory (turn 2 of 6 dialogues)

| Chunking | Reranking | Memory | P@1 | P@3 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fixed | no | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.024 |
| fixed | no | yes | 1.000 | 0.722 | 0.667 | 0.460 | 0.639 | 1.000 |
| fixed | yes | no | 0.167 | 0.111 | 0.067 | 0.032 | 0.084 | 0.270 |
| fixed | yes | yes | 1.000 | 0.833 | 0.800 | 0.564 | 0.760 | 1.000 |
| sentence | no | no | 0.167 | 0.111 | 0.067 | 0.039 | 0.088 | 0.246 |
| sentence | no | yes | 0.833 | 0.722 | 0.767 | 0.481 | 0.668 | 0.917 |
| sentence | yes | no | 0.167 | 0.222 | 0.167 | 0.090 | 0.118 | 0.347 |
| sentence | yes | yes | 1.000 | 0.944 | 0.867 | 0.539 | 0.706 | 1.000 |

## Query latency (14 queries, 3 passes, GPU)

| Chunking | Pipeline | Median (ms) | p95 (ms) |
|---|---|---:|---:|
| fixed | dense only | 6.8 | 23.6 |
| fixed | dense + rerank | 17.6 | 36.1 |
| sentence | dense only | 8.4 | 12.3 |
| sentence | dense + rerank | 18.9 | 26.0 |

Reranking 20 candidates adds about 10 ms per query on this GPU. On a CPU it would be much larger; I did not measure that.

## Cost

| Item | Value |
|---|---|
| API calls | 0 |
| Money spent on inference | $0 (all models run locally) |
| One full run of `rag_pipeline.py` | 48.6 s wall clock, including model loading |

## Error analysis

`measure.py` flags every (query, configuration) pair with reciprocal rank below 1 or P@5 below 0.5. That is 43 of the 56 single-turn cases (14 queries × 4 configurations) and 48 dialogue cases (6 dialogues × 8 configurations). They fall into four groups.

| Group | Cases | What happened |
|---|---:|---|
| Few relevant chunks (ceiling) | 10 | S2, S4, S7 and S11 with reranking: the index holds only 2–5 chunks that satisfy the rubric, so P@5 is capped at 0.4–1.0 even when the top result is right. All ten have reciprocal rank 1.00. These are not retrieval failures. |
| Paraphrase miss in dense retrieval | 4 | S11 and S14 without reranking. For S11 the top 5 contained chunks from other papers (documents 25, 81 or 82, 96); for S14 the right paper was found but its general chunks ranked above the ones about accessibility. Reranking moved the first relevant chunk to rank 1 in all four. |
| Reranker regression | 1 | S4, sentence chunks: the cross-encoder placed a chunk from the right paper that fails the rubric above the relevant one (reciprocal rank 0.50). |
| No-memory follow-ups | 24 | Turn 2 alone ("What conditions make it stable?", "How accurate is it compared with numerical simulations?", "What is the role of the Dirac cone there?") retrieved chunks from unrelated papers. 12 of the 24 have reciprocal rank 0.00, including all four configurations of D4 and D5. Reranking does not rescue them because the cross-encoder scores against the same ambiguous query. |
| Memory residual | 4 | D2 and D3 with memory have the right paper in every top-5 slot, but only 2 of 5 are relevant (P@5 0.4), and D3 with sentence chunks and no reranking has reciprocal rank 0.50. Memory fixes the topic, not the ranking inside the paper. |

The groups add to 10 + 4 + 1 + 24 + 4 = 43. The full per-case list is printed by `measure.py` and stored in `measurements.json` under `failures`.

## Caveats

14 queries and 6 dialogues are too few to separate configurations that differ by a few points; one query changing rank moves a single-turn average by about 0.07 and a memory average by about 0.17. Relevance is a keyword rubric tied to one paper per query, so useful chunks from other papers count as wrong and some right-paper chunks that do not contain the keywords count as wrong. Recall is computed against the relevant chunks in each index, so recall values are not directly comparable between chunkings, and S10 (65 relevant chunks) caps recall for any method. The memory test is favourable by design and has no topic switches. Only retrieval is evaluated; no answers are generated. The candidate pool is 20, so reranking cannot recover chunks the dense stage missed. Latency numbers are from a single GPU machine.
