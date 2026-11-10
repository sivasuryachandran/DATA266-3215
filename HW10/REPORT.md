# HW10 – Improving RAG with Reranking, Chunking and Conversational Memory

**Code:** `rag_pipeline.py` and the executed notebook `HW10_RAG.ipynb` · **Tables:** `METRICS.md` · **Vector DB:** ChromaDB (cosine)

## 1. Setup

| Item | Choice |
|---|---|
| Data | `ccdv/arxiv-summarization`, config `document`, first 100 `train` documents (avg 6,517 words), streamed because the full split did not fit on the C: drive |
| Preprocessing | strip `@xcite`; replace `@xmath<N>` with `[math]`; fix tokenisation artefacts; regex sentence split (each raw line is a whole paragraph); drop fragments under 3 words |
| Chunking 1 – fixed window | 150 words, 30 overlap, cuts mid-sentence: 4,724 chunks |
| Chunking 2 – sentence-aware | whole sentences packed to about 150 words, last sentence carried over, run-on text hard-split at 150 words: 5,256 chunks (mean 136, max 300 words) |
| Index | `all-MiniLM-L6-v2`, one ChromaDB collection per strategy |
| Retrieval | top 20 dense candidates; without reranking the top K are returned as is |
| Reranking | cross-encoder `ms-marco-MiniLM-L-6-v2` re-scores the 20 candidates |
| Memory | window of one turn: a follow-up is sent as "previous question + current question" |

### Evaluation protocol
* 40 single-turn queries (S1–S8 use the paper's wording; S9–S40 are paraphrased) and 15 two-turn dialogues whose second question cannot be answered alone ("What conditions make it stable?"). Only turn 2 is scored.
* Relevance rubric, coded per query: a chunk is relevant if it comes from the target paper and mentions at least 3 of 3–4 concept patterns needed to answer the question.
* Metrics: Precision@K, Recall@K (against all rubric-relevant chunks in that index), MRR (rank searched to 10). 95% bootstrap intervals over queries; differences are paired.
* Rubric audit: 120 pooled chunks from 14 queries were read without seeing the rubric's verdict and labelled useful or not. Labels were assigned by the assistant, not independent annotators.

## 2. Sample queries

**"How can qubits be protected when every qubit suffers the same error?"** – the top chunks are all from the fully-correlated-noise paper; the relevant ones state that all qubits in the codeword suffer the same error operator and how many data qubits can be encoded.

**"How can online astronomy resources be made usable for people with disabilities?"** (paraphrase) – dense rank of the first relevant chunk 4 (fixed) / 2 (sentence); after reranking 1 for both.

**"How are signals from merging binary stars found in noisy interferometer data using templates?"** – dense rank 2 / 3, reranked rank 1 for both chunkings.

**Multi-turn D1.** Turn 1: "What learning rates are obtained for regularized kernel based methods for additive models?" Turn 2: "Why do they compare favourably in high dimensions?" Without memory the third result is from an unrelated category-theory paper ("higher dimensional analogues…"); with memory all top results come from the additive-models paper.

**Multi-turn D2.** "What conditions make it stable?" alone retrieves a Fourier-transform chunk from a swarming paper; with the first question prefixed, all top results are from the coupled-quadratic-maps paper.

## 3. Results (full tables with intervals in `METRICS.md`)

| | P@5 | R@5 | MRR |
|---|---:|---:|---:|
| fixed, dense | 0.575 | 0.373 | 0.775 |
| fixed, reranked | 0.605 | 0.384 | 0.845 |
| sentence, dense | 0.585 | 0.357 | 0.785 |
| sentence, reranked | 0.600 | 0.366 | 0.813 |

Turn 2 of the dialogues (P@5 / MRR): fixed dense 0.160 / 0.234 → with memory 0.573 / 0.967; fixed reranked 0.200 / 0.430 → 0.720 / 1.000; sentence dense 0.187 / 0.337 → 0.640 / 0.933; sentence reranked 0.240 / 0.428 → 0.707 / 1.000.

## 4. Findings

1. **Reranking helps a little, mostly at rank 1.** Fixed windows: MRR +0.070 (95% CI [0.000, 0.144]), P@1 0.675 → 0.775. Sentence chunks: MRR +0.028 (CI [−0.031, 0.093]). P@5 and R@5 move by 0.03 or less, inside their intervals. Per query the effect is mixed: 10 wins / 27 ties / 3 losses (fixed) and 6 / 29 / 5 (sentence). The largest gains are on paraphrased queries (S14: 0.25 → 1.00; S11: 0.33 → 1.00); reranking also hurts some (S37 1.00 → 0.33 with fixed chunks, S4 1.00 → 0.50 with sentence chunks).
2. **Chunking makes no detectable difference.** Every paired interval for sentence minus fixed contains zero (dense P@5 +0.010, CI [−0.045, 0.065]; reranked MRR −0.032, CI [−0.115, 0.052]). Sentence chunks win 9 and lose 6 queries without reranking; with reranking 6 and 5.
3. **Memory is essential for follow-ups.** Turn-2 MRR rises from 0.23–0.43 to 0.93–1.00, and P@5 by 0.41–0.52, with every paired interval excluding zero. Reranking and memory are complementary (best: P@5 0.71–0.72, MRR 1.00). Memory fixes which paper is searched, not the ranking inside it: 16 flagged cases remain with memory, mostly the right paper with weak chunk ranking.
4. **The keyword rubric is lenient.** On 120 audited chunks it agrees with chunk-by-chunk judgments 80.8% of the time (κ 0.55); it marks 15 chunks relevant that were judged not useful (methods, notation, references, background) against 8 the other way. It overstates P@5 by about 0.06 and MRR by 0.13–0.20 on the audited pool, so absolute scores are upper bounds.
5. **Error analysis.** Of 141 flagged cases, 52 are follow-ups without memory (24 with reciprocal rank 0), 27 are capped because only 1–5 chunks qualify, 23 are right-paper-wrong-chunk, 23 are other papers outranking the answer (S22, S32, S36 are not fixed by reranking), and 16 are follow-ups with memory that still rank weakly.

## 5. Limitations
* Queries, dialogues, the rubric and the audit labels all come from one assistant; there are no independent human annotators, and the audit covers 14 of 55 queries/dialogues.
* Forty queries give intervals of about ±0.1, so small reranking and chunking differences are inconclusive. An earlier 14-query version of this evaluation suggested clearer effects (MRR 0.911 → 1.000 with reranking, better early precision for sentence chunks); they did not hold up at 40 queries.
* Recall is against each index's own relevant chunks, so it is not directly comparable across chunkings.
* The memory test is favourable by design and has no topic switches; concatenation would fail after one.
* Only retrieval is evaluated, with a fixed candidate pool of 20.
* Reruns agree to within about 0.015 on a few dialogue rows (GPU floating-point differences).
