# DATA 266 - HW11 (GPTQ + knowledge distillation), submitted from the HW12 folder

Siva Surya Chandran

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

Only SEED is used in this homework (HP_ID adds no temperature in HW11).

## Status
`HW12.ipynb` was executed top to bottom on a Colab Tesla T4 with outputs saved. Results: `METRICS.md` (Tables HW11.1 and HW11.2 plus the sweeps), `REPORT.md` (written findings, error analysis, quantization vs distillation), `RUN_LOG.txt`, `figures/` (loss trends, sweeps), `AI_USE.md`. The notebook's two findings cells were left as placeholders on purpose (the notebook was not edited after the run); the findings are in `REPORT.md`. `outputs/*.json` and the checkpoints exist only on the Colab machine.

Headline: GPTQ 4-bit on `facebook/opt-125m` -> perplexity 27.66 to 31.18, size 250.5 to 129.3 MB, but about 1.9x slower. Distillation -> DistilBERT student matches the BERT teacher (94.45% vs 94.43% on AG News test) at 1.95x lower batch-1 latency and 38.7% smaller size.

## Choices and assumptions
1. `LLM_ID` (cell 1) defaults to `facebook/opt-125m`. Replace it with the instructor-provided model; the notebook records the exact revision hash.
2. The pasted brief lost its math: the temperature set, the alpha set and the formula symbol were blank. I assumed T in {1, 2, 4, 8} and alpha in {0.3, 0.5, 0.7} (alpha = weight on cross-entropy, KL weight = 1 - alpha, KL scaled by T^2), with alpha = 0.5 during the temperature sweep. Edit `TEMPS`, `ALPHAS`, `SWEEP_ALPHA` in cell 1 if the brief says otherwise.
3. Teacher logits are extracted for all 120k training samples as required; 5,000 of them are held out as validation so T and alpha are chosen without using the test set. Students train on the other 115k.
4. Fixed training settings: 2 epochs, batch 64, lr 5e-5, max length 128, fp16 autocast; every student is re-initialised from `distilbert-base-uncased` with `SEED`.
5. The "Findings" markdown cells and `AI_USE.md` must be written after the run from the real outputs.

## Commit / tags
Repository `data266-3215`, tag `hw11` (the assignment is HW11), with incremental commits.
