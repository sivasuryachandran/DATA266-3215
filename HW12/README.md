# DATA 266 - HW11 (GPTQ + knowledge distillation), submitted from the HW12 folder

Siva Surya Chandran

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

Only SEED is used in this homework (HP_ID adds no temperature in HW11).

## Status
`HW12.ipynb` is built by `scripts/build_notebook.py` but **has not been executed yet**: it needs a CUDA GPU, the instructor's `teacher_model/` folder, and the instructor-approved GPTQ base LLM, none of which are on this Mac. Run it top to bottom on Colab (T4 or better) with `teacher_model/` uploaded next to it. The run writes `RUN_LOG.txt`, `METRICS.md`, `outputs/*.json` and `figures/*.png`; commit the executed notebook with outputs.

## Things to confirm before running
1. `LLM_ID` (cell 1) defaults to `facebook/opt-125m`. Replace it with the instructor-provided model; the notebook records the exact revision hash.
2. The pasted brief lost its math: the temperature set, the alpha set and the formula symbol were blank. I assumed T in {1, 2, 4, 8} and alpha in {0.3, 0.5, 0.7} (alpha = weight on cross-entropy, KL weight = 1 - alpha, KL scaled by T^2), with alpha = 0.5 during the temperature sweep. Edit `TEMPS`, `ALPHAS`, `SWEEP_ALPHA` in cell 1 if the brief says otherwise.
3. Teacher logits are extracted for all 120k training samples as required; 5,000 of them are held out as validation so T and alpha are chosen without using the test set. Students train on the other 115k.
4. Fixed training settings: 2 epochs, batch 64, lr 5e-5, max length 128, fp16 autocast; every student is re-initialised from `distilbert-base-uncased` with `SEED`.
5. The "Findings" markdown cells and `AI_USE.md` must be written after the run from the real outputs.

## Commit / tags
Repository `data266-3215`, tag `hw11` (the assignment is HW11), with incremental commits.
