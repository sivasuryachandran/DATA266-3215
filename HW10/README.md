# DATA 266 — HW10

Improving RAG with reranking, chunking and conversational memory, on the first 100 training documents of `ccdv/arxiv-summarization` (`document` config).

## Contents

- `HW10_RAG.ipynb` — the submission notebook, executed end to end with outputs saved (preprocessing, two chunkers, ChromaDB index, retrieval with and without reranking, memory demo, all evaluation tables, findings).
- `rag_pipeline.py` — the same pipeline as a script; writes `results.json`. The notebook cells are generated from this file by `build_nb.py`.
- `fetch_data.py` — streams the first 100 train documents into `docs100.json`.
- `measure.py` — timing and failure-analysis measurements; writes `measurements.json`.
- `build_nb.py` — builds `HW10_RAG.ipynb` from `rag_pipeline.py`.
- `REPORT.md` — longer written comparison and findings.
- `METRICS.md` — every table (quality, timing, cost, failure analysis) with caveats.
- `RUN_LOG.txt` — execution log.
- `AI_USE.md` — AI-use appendix.
- `results.json`, `measurements.json`, `pipeline_stdout.txt` — raw outputs behind the tables.
- `docs100.json` — the 100 streamed documents, so the notebook runs without re-downloading.

The persisted ChromaDB folder (`chroma_db/`, about 218 MB) is not committed. `rag_pipeline.py` and the notebook rebuild both collections (`fixed_window`, `sentence_aware`) in about 20 seconds.

## Setup choices

Embedder `all-MiniLM-L6-v2`; reranker cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 20 dense candidates. Chunking 1 is a 150-word window with 30 words of overlap. Chunking 2 packs whole sentences up to about 150 words and carries the last sentence into the next chunk. Memory prefixes the previous user question to a follow-up. The pipeline has no random components, so reruns give identical numbers. Everything runs locally; there is no API cost.

Relevance is a rubric I coded per query: a chunk counts as relevant if it is from the target paper and mentions at least three of the three or four concepts needed to answer the question.

## Main findings

Reranking helps most on paraphrased queries: MRR 0.911 to 1.000 with fixed windows and 0.917 to 0.964 with sentence chunks. On queries that reuse the paper's wording the dense retriever is already at MRR 1.0, and reranking once made things worse (S4, sentence chunks). The two chunking strategies are close; sentence-aware chunks win on early precision without reranking (P@5 0.757 vs 0.686) and fixed windows win slightly on recall with reranking. With 14 queries I would not call a winner. Conversational memory is the largest effect: turn-2 MRR goes from 0.02 to 0.35 without memory to 0.92 to 1.00 with it. The best configuration is sentence-aware chunking, reranking and memory (P@5 0.867, MRR 1.000). The memory test is favourable by design (the follow-ups cannot be answered alone, and no dialogue changes topic). See `METRICS.md` for the failure analysis.

## Run

```bash
pip install chromadb sentence-transformers datasets pandas jupyter
python fetch_data.py                      # docs100.json
python rag_pipeline.py                    # builds both collections, writes results.json
python measure.py                         # timings and failure analysis -> measurements.json
jupyter nbconvert --to notebook --execute --inplace HW10_RAG.ipynb
```

On Windows set `HF_HOME` to a drive with space before running; the model cache and the streamed data are small, but a full non-streaming download of the dataset is several GB.
