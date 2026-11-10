# HW10 metrics

Numbers in the quality tables are copied from the outputs of the executed `HW10_RAG.ipynb` (GPU run, 2026-11-09). `results.json` comes from a separate run of `rag_pipeline.py` and agrees on every single-turn value; a few no-memory dialogue rows differ by up to about 0.015 (for example sentence-aware, reranked, no memory: P@5 0.240 in the notebook, 0.253 in `results.json`). Timing and failure data are in `measurements.json`, written by `measure.py`.

## Setup

| Item | Value |
|---|---|
| Dataset | `ccdv/arxiv-summarization`, config `document`, first 100 `train` documents (average 6,517 words) |
| Vector database | ChromaDB 1.5.9, persistent, cosine HNSW, one collection per chunking (single-threaded build, `search_ef` 200) |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2`, normalised |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20 dense candidates |
| Single-turn queries | 40 (S1–S8 reuse the paper's wording, S9–S40 are paraphrased) |
| Dialogues | 15 two-turn dialogues (D1–D15); only turn 2 is scored |
| Relevance | coded rubric: chunk is from the target paper and mentions at least 3 of 3–4 concept patterns (2 for D6) |
| Confidence intervals | 95% bootstrap over queries (5,000 resamples, seed 0); paired for differences |

## Chunking statistics

| Chunking | Chunks | Mean words | Min | Max | Chunk time (s) | Embed time (s) | Chunks per second |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed window (150 / 30 overlap) | 4,724 | 148.8 | 31 | 150 | 0.33 | 2.17 | 2,177 |
| sentence-aware (about 150) | 5,256 | 135.9 | 9 | 300 | 0.33 | 2.14 | 2,456 |

Embedding ran on an NVIDIA GeForce RTX 5090. Building both collections took 16 s in the last run.

## Without vs with reranking (40 single-turn queries)

| Chunking | Reranking | P@1 | P@3 | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | no | 0.675 | 0.600 | 0.575 | 0.480 | 0.373 | 0.554 | 0.775 |
| fixed | yes | 0.775 | 0.625 | 0.605 | 0.520 | 0.384 | 0.577 | 0.845 |
| sentence | no | 0.675 | 0.625 | 0.585 | 0.495 | 0.357 | 0.554 | 0.785 |
| sentence | yes | 0.725 | 0.617 | 0.600 | 0.525 | 0.366 | 0.584 | 0.813 |

### 95% confidence intervals

| Chunking | Reranking | P@5 | R@5 | MRR |
|---|---|---|---|---|
| fixed | no | 0.575 [0.475, 0.675] | 0.373 [0.290, 0.464] | 0.775 [0.669, 0.872] |
| fixed | yes | 0.605 [0.515, 0.690] | 0.384 [0.291, 0.483] | 0.845 [0.747, 0.931] |
| sentence | no | 0.585 [0.485, 0.680] | 0.357 [0.280, 0.441] | 0.785 [0.686, 0.879] |
| sentence | yes | 0.600 [0.505, 0.695] | 0.366 [0.291, 0.447] | 0.813 [0.712, 0.906] |

### Paired differences (wins / ties / losses are per query, on reciprocal rank in the last column)

| Comparison | ΔP@5 [CI] | ΔR@5 [CI] | ΔMRR [CI] | W / T / L (MRR) |
|---|---|---|---|---|
| rerank − dense, fixed | +0.030 [−0.025, 0.085] | +0.010 [−0.065, 0.080] | **+0.070 [0.000, 0.144]** | 10 / 27 / 3 |
| rerank − dense, sentence | +0.015 [−0.025, 0.060] | +0.009 [−0.028, 0.041] | +0.028 [−0.031, 0.093] | 6 / 29 / 5 |
| sentence − fixed, dense | +0.010 [−0.045, 0.065] | −0.017 [−0.072, 0.035] | +0.010 [−0.066, 0.088] | 9 / 25 / 6 |
| sentence − fixed, rerank | −0.005 [−0.055, 0.045] | −0.018 [−0.069, 0.034] | −0.032 [−0.115, 0.052] | 6 / 29 / 5 |

Only the fixed-window MRR gain from reranking has a lower bound at (not below) zero, and that is borderline. Nothing else in the reranking or chunking comparisons is distinguishable from zero.

Largest per-query changes in reciprocal rank from reranking: gains S14 (0.25 → 1.00, fixed), S11 (0.33 → 1.00, sentence), S18 (0.33 → 1.00, sentence); losses S37 (1.00 → 0.33, fixed), S4 (1.00 → 0.50, sentence), S39 (0.50 → 0.20, sentence).

## Without vs with conversational memory (turn 2 of 15 dialogues)

| Chunking | Reranking | Memory | P@1 | P@3 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fixed | no | no | 0.200 | 0.156 | 0.160 | 0.090 | 0.171 | 0.234 |
| fixed | no | yes | 0.933 | 0.667 | 0.573 | 0.378 | 0.597 | 0.967 |
| fixed | yes | no | 0.333 | 0.267 | 0.200 | 0.106 | 0.225 | 0.430 |
| fixed | yes | yes | 1.000 | 0.778 | 0.720 | 0.502 | 0.700 | 1.000 |
| sentence | no | no | 0.200 | 0.200 | 0.187 | 0.117 | 0.192 | 0.337 |
| sentence | no | yes | 0.867 | 0.667 | 0.640 | 0.473 | 0.609 | 0.933 |
| sentence | yes | no | 0.267 | 0.267 | 0.240 | 0.144 | 0.215 | 0.428 |
| sentence | yes | yes | 1.000 | 0.756 | 0.707 | 0.520 | 0.654 | 1.000 |

### Paired improvement from memory (all intervals exclude zero)

| Chunking | Reranking | ΔP@5 [CI] | ΔR@5 [CI] | ΔMRR [CI] |
|---|---|---|---|---|
| fixed | no | +0.413 [0.227, 0.613] | +0.288 [0.165, 0.435] | +0.733 [0.510, 0.920] |
| fixed | yes | +0.520 [0.360, 0.667] | +0.396 [0.253, 0.557] | +0.570 [0.349, 0.779] |
| sentence | no | +0.453 [0.280, 0.613] | +0.355 [0.206, 0.527] | +0.596 [0.340, 0.821] |
| sentence | yes | +0.467 [0.267, 0.653] | +0.376 [0.204, 0.551] | +0.572 [0.372, 0.767] |

## Rubric audit (120 chunks read one by one)

`audit.py` pooled the top-5 chunks of fixed|dense and sentence|rerank for 10 randomly sampled single-turn queries, plus sentence|rerank|memory for 4 dialogues. Each chunk was labelled useful or not for answering the query without seeing the rubric's verdict. **The labels were assigned by Claude, not by independent human annotators**; the chunks are in `audit_items.txt` and the labels in `audit_labels.json` so they can be spot-checked.

| | Rubric: relevant | Rubric: not relevant |
|---|---:|---:|
| Judged useful | 72 | 8 |
| Judged not useful | 15 | 25 |

Agreement 80.8%, Cohen's κ 0.55 (moderate), rubric precision 0.83, recall 0.90. The rubric is lenient: its 15 false positives are mostly methods, notation, references and background chunks from the right paper. On the audited pool it overstates P@5 by about 0.06 (fixed dense 0.72 vs 0.66; sentence reranked 0.73 vs 0.67) and MRR by 0.13–0.20 (0.92 vs 0.72; 0.96 vs 0.84). Absolute scores are therefore upper bounds; the comparisons between configurations are more reliable than the levels.

## Query latency (40 queries × 3 passes, GPU)

| Chunking | Pipeline | Median (ms) | p95 (ms) |
|---|---|---:|---:|
| fixed | dense only | 4.9 | 6.4 |
| fixed | dense + rerank | 16.5 | 20.2 |
| sentence | dense only | 5.1 | 7.7 |
| sentence | dense + rerank | 16.2 | 21.8 |

Reranking 20 candidates adds about 11 ms per query on this GPU. I did not obtain a clean CPU timing (the one CPU attempt was interrupted). An earlier GPU measurement while other programs were using the card gave 18–30 ms for reranked queries, so latency varies with load.

## Cost

| Item | Value |
|---|---|
| API calls | 0 |
| Money spent on inference | $0 (all models run locally) |
| `rag_pipeline.py` end to end | 38 s wall clock including model loading |
| Executed notebook | 51 s |

## Error analysis

`measure.py` flags every (query, configuration) case with reciprocal rank below 1 or P@5 below 0.5: **141** cases (40 queries × 4 configurations + 15 dialogues × 8 configurations = 280 cases in total). It sorts them automatically:

| Category | Cases | What happens |
|---|---:|---|
| Follow-up without memory | 52 | Turn 2 alone ("What conditions make it stable?", "Which sites does it prefer?") retrieves unrelated papers; 24 of the 52 have reciprocal rank 0. Reranking cannot rescue them because it scores against the same ambiguous query. |
| Capped by few relevant chunks (top-1 correct) | 27 | S2, S4, S7, S11, S24, S35, S40 and others: only 1–5 chunks in the index satisfy the rubric, so P@5 is capped even though the first result is right. Not retrieval failures. |
| Right paper, answer chunk ranked low | 23 | The correct paper is found but rubric-passing chunks rank below intro, notation or reference text (S14, S16, S19, S20, S33). |
| Other papers outrank the answer | 23 | Paraphrase misses: S11, S22 (supernova typing; top-5 includes the EX Lupi paper), S32 (solar eruption; the EX Lupi "eruption" paper outranks it), S36, S39. Reranking fixes some (S11) and not others (S22, S32, S36). |
| Follow-up with memory: right paper, weak ranking | 10 | Memory fixes the topic, but only 1–2 of the top 5 chunks answer the question (D2, D3, D9, D14). |
| Follow-up with memory: other papers in top 5 | 6 | D12 (cosmic-ray cutoff) and D13 (radio galaxy age) still pull in chunks from other papers after the first relevant hit. |

Per-case detail is in `measurements.json` under `failures`.

## Caveats

- Queries, dialogues, the relevance rubric and the audit labels were all produced by the same assistant; there are no independent human annotators. The audit covers 14 of the 55 queries and dialogues.
- Forty queries still give wide intervals (about ±0.1 on P@5 and MRR), which is why the reranking and chunking differences are inconclusive.
- Recall is computed against the rubric-relevant chunks in each index, so recall is not directly comparable between chunkings, and S10 (65 relevant chunks) caps recall for any method.
- The memory test is favourable by design: the follow-ups cannot be answered alone and no dialogue switches topic. Simple concatenation would pollute retrieval after a topic change.
- Only retrieval is evaluated; no answers are generated. The candidate pool is 20, so reranking cannot recover chunks the dense stage missed.
- Reruns are not bit-identical: one query's rank changed between two runs of the same code (GPU floating-point differences on near-tied chunks). Differences of that size are far below the confidence intervals.
- During development the S37 rubric was rewritten after it returned zero relevant chunks (it required the phrase "charge order", which appears only in the paper's abstract, not its body).
