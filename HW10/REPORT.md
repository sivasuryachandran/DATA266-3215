# HW10 – Improving RAG with Reranking, Chunking and Conversational Memory

**Code:** `rag_pipeline.py` (single script, reproduces everything) · **Raw numbers:** `results.json` · **Vector DB:** ChromaDB (persistent, `chroma_db/`)

## 1. Setup

| Item | Choice |
|---|---|
| Data | `ccdv/arxiv-summarization`, config `document`, first 100 `train` docs (`article` field; avg 6,517 words). Loaded with `streaming=True` because the full split did not fit on the C: drive. |
| Preprocessing | Strip `@xcite`; replace `@xmath<N>` with `[math]` and collapse runs; fix tokenisation artefacts (` .`, `( `); normalise whitespace; split into sentences with a regex (the raw text has one very long line per paragraph); drop fragments under 3 words. |
| Chunking 1 – **fixed window** | 150-word sliding window, 30-word overlap, cuts anywhere (mid-sentence). 4,724 chunks. |
| Chunking 2 – **sentence-aware** | Pack whole sentences up to ~150 words, carry the last sentence over as overlap; run-on "sentences" (tables, reference lists) are hard-split at 150 words. 5,256 chunks (mean 136 words, max 300). |
| Embedding / index | `all-MiniLM-L6-v2` (normalised), one ChromaDB collection per strategy, cosine HNSW. |
| Retrieval | Top-20 candidates from the vector DB; **without reranking** the top-K are returned as-is. |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` re-scores the 20 candidates, then top-K. |
| Memory | `ConversationMemory` (window = 1 turn): a follow-up is rewritten as `"<previous user question> <current question>"` before embedding and reranking, so "they / it / that" is resolved by the earlier topic. |

### Evaluation protocol
* **Queries:** 14 single-turn queries (S1–S8 use the paper's own vocabulary; S9–S14 are paraphrased with little exact-term overlap, and many nearby physics documents compete) and 6 two-turn dialogues (D1–D6) whose second question is unanswerable alone (e.g. *"What conditions make it stable?"*).
* **Relevance (manual rubric, applied identically to every chunk):** a chunk is relevant if it comes from the target document **and** mentions ≥3 (≥2 for D6) of 3–4 hand-chosen concepts needed to answer the question, i.e. it is useful evidence. The rubric is coded in `QUERIES` / `DIALOGS`.
* **Metrics:** Precision@K, Recall@K (relevant retrieved ÷ all relevant chunks in that index), MRR (reciprocal rank of the first relevant chunk, searched to rank 10). Averaged over queries. For memory, only the turn-2 query is scored.

## 2. Sample queries (top-3, relevant = ✔)

**Q: "How does synchronization occur in coupled quadratic maps?"** (sentence-aware chunks)

| Rank | Without reranking | With reranking |
|---|---|---|
| 1 | ✔ "by coupling … an arbitrarily large number of orbits may become stable" | ✔ "rather than merely focusing on multistability, we also study the phenomenology of two coupled quadratic maps…" |
| 2 | ✔ "rather than merely focusing on multistability…" | ✔ "by coupling, as we will show…" |
| 3 | ✘ "the dynamics commutes with the symmetry…" (generic) | ✘ "the condition [math] is not necessary…" |

**Q: "How can squeezed vacuum be used for non-invasive measurement of cavity parameters?"** – dense and reranked both return only chunks from the right paper; reranking promotes the chunk stating the proposed method ("we have proposed and experimentally demonstrated a method for non-invasive measurements…", score 8.4) and the chunk explaining how linewidth is read from the correlation, over a generic apparatus description.

**Q: "How can online astronomy resources be made usable for people with disabilities?"** (paraphrase, hardest case) – dense rank of first relevant chunk: 4 (fixed) / 2 (sentence); after reranking: **1** for both.

**Multi-turn (D1).** Turn 1: *"What learning rates are obtained for regularized kernel based methods for additive models?"* Turn 2: *"Why do they compare favourably in high dimensions?"*
* No memory: 2 of top-3 are right by luck of the keyword "high dimensions", the third is from an unrelated category-theory paper ("higher dimensional analogues…").
* With memory: all top-3 come from the additive-models paper.

**Multi-turn (D2).** Turn 2 alone: *"What conditions make it stable?"* → top-3 contains a Fourier-transform chunk from a swarming paper (doc 10) twice. With memory the query becomes "How does synchronization occur in coupled quadratic maps? What conditions make it stable?" → all top-3 are from the coupled-maps paper (doc 36) and relevant.

## 3. Results

### 3.1 Without vs. with reranking (14 single-turn queries)

| Chunking | Reranking | P@1 | P@3 | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| fixed | no | 0.857 | 0.738 | 0.686 | 0.586 | 0.416 | 0.685 | 0.911 |
| fixed | **yes** | **1.000** | 0.786 | 0.757 | 0.643 | 0.516 | 0.751 | **1.000** |
| sentence | no | 0.857 | 0.810 | 0.757 | 0.629 | 0.434 | 0.668 | 0.917 |
| sentence | **yes** | 0.929 | 0.786 | 0.743 | 0.664 | 0.408 | 0.713 | 0.964 |

### 3.2 Chunking 1 vs. chunking 2
Read from the same table (rows with the same reranking setting): see "fixed" vs "sentence".

### 3.3 Without vs. with conversational memory (turn 2 of 6 dialogues)

| Chunking | Reranking | Memory | P@1 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|
| fixed | no | no | 0.000 | 0.000 | 0.000 | 0.028 | 0.024 |
| fixed | no | **yes** | 1.000 | 0.667 | 0.460 | 0.639 | 1.000 |
| fixed | yes | no | 0.167 | 0.067 | 0.032 | 0.084 | 0.270 |
| fixed | yes | **yes** | 1.000 | 0.800 | 0.564 | 0.760 | 1.000 |
| sentence | no | no | 0.167 | 0.067 | 0.039 | 0.088 | 0.246 |
| sentence | no | **yes** | 0.833 | 0.767 | 0.481 | 0.668 | 0.917 |
| sentence | yes | no | 0.167 | 0.167 | 0.090 | 0.118 | 0.347 |
| sentence | yes | **yes** | 1.000 | 0.867 | 0.539 | 0.706 | 1.000 |

## 4. Findings

1. **Reranking mainly improves the *top* of the list, as expected.** With fixed chunks every metric improved (MRR 0.911 → 1.000, P@1 0.857 → 1.000, R@5 0.416 → 0.516). With sentence chunks MRR rose 0.917 → 0.964 and P@10/R@10 improved, but P@5 and R@5 were flat to slightly lower (0.757 → 0.743, 0.434 → 0.408). The gain comes almost entirely from the hard paraphrased queries: S14 first-relevant rank went 4 → 1 (fixed) and 2 → 1 (sentence), S11 2 → 1 and 3 → 1. The bi-encoder already finds the right *document* for easy queries (S1–S8 have MRR ≈ 1.0 without reranking), so the cross-encoder has little to fix there; it occasionally hurts (S4 with sentence chunks: MRR 1.0 → 0.5).
2. **Chunking matters less than reranking, and the trade-off is precision vs. recall.** Without reranking, sentence-aware chunks beat fixed windows on early precision (P@3 0.810 vs 0.738, P@5 0.757 vs 0.686): chunks are coherent units, so their embeddings are cleaner and relevant sentences are not split across a boundary. With reranking, fixed windows score slightly higher on R@5/R@10 and MRR (0.516 vs 0.408 R@5), probably because the cross-encoder is robust to ragged boundaries and fixed windows are uniform in length; sentence chunks vary from 9 to 300 words, so the reranker's score is less comparable. I would not call either strategy the winner on this evidence.
3. **Conversational memory is essential for follow-ups.** Without it, the second turn of the dialogues retrieved almost nothing relevant (MRR 0.02–0.35, P@5 ≤ 0.17) because "they / it / how accurate is it" carries no topic. Prefixing the previous question lifted MRR to 0.92–1.00 and P@5 to 0.67–0.87. The best configuration overall is **sentence-aware chunking + reranking + memory** (P@5 0.867, MRR 1.0). Reranking and memory are complementary: reranking with memory gave +0.13 P@5 over dense with memory for fixed chunks and +0.10 for sentence chunks.

## 5. Limitations (please read before trusting the numbers)
* **Small, hand-built evaluation**: 14 queries, 6 dialogues; one query flipping changes averages by ~0.07 (single) or ~0.17 (memory). Most differences in 3.1/3.2 (a few points) are within noise; only the memory effect and the S11/S14 reranking gains are large and consistent.
* **Relevance is a keyword rubric over a single target document**, not human reading of every chunk. It is stricter than "from the right paper" but can mark a genuinely useful chunk non-relevant (or the reverse), and chunks from *other* papers are always counted wrong, even if partly useful.
* **Recall is not directly comparable across chunkings**: the number of relevant chunks (the denominator) differs per index, and queries with many relevant chunks (S10: 65) cap R@K at low values for any method.
* **The memory test is deliberately favourable**: the follow-ups are designed to be unanswerable alone, so "no memory" is a weak baseline, and the dialogues never switch topic. Naïve concatenation would pollute retrieval after a topic change; production systems typically use an LLM to rewrite the query (or detect topic shifts). No LLM was used here, so none of the answers are generated; only retrieval is evaluated.
* Candidate pool is fixed at 20, so reranking cannot recover relevant chunks the dense stage missed.

## 6. Reproduce
```
pip install chromadb sentence-transformers datasets
python fetch_data.py          # streams first 100 train docs -> docs100.json
python rag_pipeline.py        # builds both Chroma collections, runs all evaluations -> results.json
```
