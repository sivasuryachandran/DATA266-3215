# DATA 266 — HW10

Improving RAG with reranking, chunking and conversational memory, on the first 100 training documents of `ccdv/arxiv-summarization` (`document` config).

## Contents

- `HW10_RAG.ipynb` — the submission notebook, executed end to end with outputs saved: preprocessing, two chunkers, ChromaDB index, retrieval with and without reranking, memory demo, evaluation tables with bootstrap confidence intervals, rubric audit, findings.
- `rag_pipeline.py` — the same pipeline as a script; writes `results.json`. The notebook is generated from this file by `build_nb.py`.
- `fetch_data.py` — streams the first 100 train documents into `docs100.json`.
- `measure.py` — timing and failure-analysis measurements; writes `measurements.json`.
- `audit.py` — builds the 120-chunk audit pool and scores the relevance rubric against chunk-by-chunk labels (`audit_items.txt`, `audit_items.json`, `audit_labels.json`, `audit_results.json`).
- `build_nb.py` — builds `HW10_RAG.ipynb` from `rag_pipeline.py`.
- `REPORT.md` — written comparison and findings.
- `METRICS.md` — every table (quality, confidence intervals, timing, cost, failure analysis) with caveats.
- `RUN_LOG.txt` — execution log.
- `AI_USE.md` — AI-use appendix.
- `docs100.json`, `results.json`, `measurements.json`, `pipeline_stdout.txt`, `measure_stdout.txt` — data and raw outputs behind the tables.

The persisted ChromaDB folder (`chroma_db/`, about 218 MB) is not committed. `rag_pipeline.py` and the notebook rebuild both collections (`fixed_window`, `sentence_aware`) in about 20 seconds.

## Setup choices

Embedder `all-MiniLM-L6-v2`; reranker cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 20 dense candidates. Chunking 1 is a 150-word window with 30 words of overlap. Chunking 2 packs whole sentences up to about 150 words and carries the last sentence into the next chunk. Memory prefixes the previous user question to a follow-up. Everything runs locally; there is no API cost.

Evaluation uses 40 single-turn queries and 15 two-turn dialogues. Relevance is a rubric I coded per query: a chunk counts as relevant if it is from the target paper and mentions at least three of the concepts needed to answer the question. I then read 120 pooled chunks without seeing the rubric's verdict and compared (agreement 80.8%, κ 0.55). The rubric is lenient, so absolute scores are upper bounds. Those audit labels were assigned by the assistant, not by independent human annotators.

## Main findings

Reranking helps a little and mostly at the top of the list. With fixed windows MRR goes from 0.775 to 0.845 (paired difference +0.070, 95% CI [0.000, 0.144], borderline); with sentence-aware chunks from 0.785 to 0.813 (CI [−0.031, 0.093]). P@5 and R@5 change by 0.03 or less and are inside their intervals. Per query, reranking wins 10 and loses 3 with fixed windows, wins 6 and loses 5 with sentence chunks. The two chunking strategies are not distinguishable on this evidence. Conversational memory is the one large, clearly significant effect: turn-2 MRR rises from 0.23–0.43 to 0.93–1.00 and P@5 by 0.41–0.52. The best configuration is reranking plus memory (P@5 0.71–0.72, MRR 1.00), with either chunking. The memory test is favourable by design (the follow-ups cannot be answered alone, and no dialogue changes topic). An earlier 14-query version of this evaluation suggested clearer reranking and chunking effects; they did not survive the larger query set. See `METRICS.md` for the failure analysis.

## Run

```bash
pip install chromadb sentence-transformers datasets pandas jupyter
python fetch_data.py                      # docs100.json
python rag_pipeline.py                    # builds both collections, writes results.json
python measure.py                         # timings and failure analysis -> measurements.json
python audit.py score                     # rubric audit (labels are in audit_labels.json)
jupyter nbconvert --to notebook --execute --inplace HW10_RAG.ipynb
```

On Windows set `HF_HOME` to a drive with space before running; a full non-streaming download of the dataset is several GB. Reruns reproduce the tables to within about 0.015 on a few dialogue rows (GPU floating-point differences on near-tied chunks).
