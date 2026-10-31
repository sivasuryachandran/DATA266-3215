# GPU Assignment 2 (HW4.5) — Training Throughput and Memory Engineering

Student: Sivasurya Chandran (019130215). Tag: `hw4-5`.

This assignment takes the BERT sentiment fine-tune from HW4.a and measures how its training throughput and
GPU memory use change with batch size and with four memory techniques, then re-measures LLM serving with
vLLM. Everything is run on real hardware and logged against the GPU's UUID.

**Start with [`METRICS.md`](METRICS.md)** — it has the summary table and all results. This README explains
what is in the folder, how it was produced and how to reproduce it.

---

## 1. What was done

| Part | What | Hardware | Status |
|---|---|---|---|
| HW4.a baseline | Fine-tune `bert-base-uncased` on 3 classes (Negative / Neutral / Positive); per-epoch curves, confusion matrix, per-class precision/recall, five misclassified examples, peak memory and wall-clock time, save and reload for inference | RTX 4090 | Done |
| A | Batch size scaled by powers of two until out-of-memory; throughput, peak memory, time per epoch | RTX 4090 | Done (single card) |
| B | Mixed precision, gradient checkpointing, gradient accumulation, 8-bit optimiser, each applied alone, plus a best combination; ranking by memory saved per throughput lost | RTX 4090 | Done |
| C | Two-GPU distributed training | — | **Skipped (optional)**, as permitted by the instructor |
| D | vLLM vs Hugging Face pipeline at batch 1/8/32, p50/p95 latency, tokens/s, KV-cache vs context length at batch 8 | Colab Tesla T4 | Done (T4 only; vLLM does not run on native Windows) |
| E | GPU-hours for a full fine-tune, recommended configuration, HW4.5 summary | RTX 4090 | Done |

Limits to be aware of (also stated in `METRICS.md`):
- **One card per part.** The assignment's "both cards" comparisons (two-card plot, max-batch ratio vs VRAM
  ratio, table once per card) are not possible; training is on the RTX 4090 and serving on a Colab T4.
- The HP_ID / CLS_A / CLS_B values in my parameter table were not applied; notebook defaults were used.
- Single run per configuration (no error bars). Accuracy differences between techniques (at most 0.4
  points) are within sampling noise (about ±1.7% on 750 test samples).

## 2. Data and setup

- Dataset: Kaggle `twitter-sentiment-dataset`, file `Twitter_Data.csv` (`clean_text`, `category` ∈ {-1, 0, 1};
  162,980 raw rows, 162,969 after dropping nulls). Labels: -1→0 Negative, 0→1 Neutral, 1→2 Positive.
- Working subset: after dropping nulls and resetting the index, **rows 215–5214** (SLICE=215, 5,000
  consecutive rows). Split 70/15/15 with `train_test_split(random_state=SEED)`, **SEED=3215**: 3,500 train /
  750 validation / 750 test. Training initialisation and shuffling also use SEED.
- Model and hyperparameters carried over from the HW4.a notebook: `bert-base-uncased`, MAX_LEN 256 (fixed
  padding), batch 32, learning rate 2e-5, eps 1e-8, 3 epochs, linear schedule with no warmup, gradient clip
  1.0. Two deliberate differences from that notebook: the 70/15/15 split (it used 90/10, no test set) and
  `torch.optim.AdamW` (the `transformers` class was removed).
- Training machine: SJSU HPC GPU lab, Room ISB 838, Machine 17 — NVIDIA GeForce RTX 4090 24 GB, driver
  610.60, Windows 11, PyTorch 2.6.0+cu124. Serving: Google Colab Tesla T4 15 GB, vLLM 0.32.0, fp16.

## 3. Headline results (full detail in `METRICS.md`)

**HW4 baseline** (batch 32, fp32): 49.2 s for 3 epochs, peak memory 6.18 GB, test accuracy 0.708.
Validation loss and accuracy were still improving at epoch 3, so training past the best epoch cost nothing.
Worst class: Negative (49 of 176 predicted Positive); the five misclassified examples are sarcasm or
rhetorical-question tweets.

**Table HW4.5.1 (RTX 4090, throughput and peak memory at batch 32)**

| Technique | Max batch | Samples/s | Peak GB | Test accuracy |
|---|---|---|---|---|
| Baseline (HW4 settings) | 128 | 231.2 | 6.18 | 0.708 |
| Mixed precision (bf16) | 128 | 481.8 | 4.58 | 0.704 |
| Gradient checkpointing | 512 | 176.4 | 2.08 | 0.708 |
| Gradient accumulation (8 × 4) | — | 57.8 s vs 49.2 s wall | 2.91 | 0.704 |
| 8-bit optimiser | 128 | 222.5 | 5.57 | 0.705 |
| Best combination | 512 | 372.5 (AMP+ckpt+8-bit) | 1.34 | 0.707 (AMP+8-bit) |

- Throughput stops improving at about batch 64; the baseline wall is batch 128 (256 runs out of memory).
- Ranking by memory saved per throughput lost: mixed precision first (it is faster *and* smaller), then
  accumulation, checkpointing, 8-bit optimiser. I would apply mixed precision first and not bother with the
  8-bit optimiser on a 32 GB card (BERT-base optimiser state is under 1 GB).
- Recommended full fine-tune: mixed precision, batch 32. One full-dataset fine-tune (3 epochs over 114,078
  training rows) is about 0.21 GPU-hours with mixed precision and 0.45 with fp32.

**Table HW4.1 (Colab T4, 128 new tokens per request)**

| Configuration | Tokens/s | p50 (ms) | p95 (ms) |
|---|---|---|---|
| HF pipeline, batch 1 | 22.5 | 5,482 | 6,635 |
| vLLM, batch 1 | 65.9 | 1,893 | 1,981 |
| vLLM, batch 8 | 501.9 | 2,042 | 2,045 |
| vLLM, batch 32 | 1,399.5 | 2,928 | 2,952 |

Per-request latency never exceeds the HF baseline up to batch 32, so the "latency worse than baseline"
crossover is not reached in the range tested. For an interactive application p95 per-request latency is
what matters, not total throughput. KV cache costs 28 KiB per token (7.0 GB at batch 8 × 32,768 tokens);
vLLM allocated room for about 312,672 tokens, and batch 8 was verified at a 32,760-token context, so the
T4's longest context at batch 8 is limited by the model's position limit, not by memory.

## 4. Files

| File | What it is |
|---|---|
| `METRICS.md` | HW4.5 summary table (Table HW4.5.2), Parts A–E results, caveats |
| `GPU_Reservation.md` | Lab reservation, GPU provenance and GPU-hour record |
| `AI_USE.md` | AI-use appendix |
| `RUN_LOG.txt` | UUID-prefixed log derived from the raw results (no timestamps were recorded) |
| `RUN_LOG_raw.txt` | Original console log from the benchmark sweep |
| `nvidia-smi-q_GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39.txt` | Full `nvidia-smi -q` capture, RTX 4090 |
| `nvidia-smi-L_RTX4090.txt`, `nvidia-smi-L_ColabT4.txt` | `nvidia-smi -L` UUID captures |
| `uuid_captures_4090_and_T4.png`, `uuid_capture_colab_T4_cell1.png` | Screenshots of both UUID captures |
| `part_a_b_scaling.csv` | Per technique and batch size: status (ok/OOM), samples/s, peak GB, epoch time |
| `part_b_full_runs.csv` | Full 3-epoch runs: wall time, peak GB, per-epoch curves, test accuracy |
| `part_d_serving.csv`, `part_d_kv_cache.csv`, `partD_results.json` | vLLM serving, KV-cache sizes, raw Part D results incl. my prompt and its answer |
| `fig_throughput_vs_batch.png` | Throughput and peak memory vs batch size, all techniques |
| `fig_confusion.png` | Baseline test confusion matrix |
| `hw45.py` | Training and benchmark harness (HW4.a fine-tune with technique switches) |
| `sweep.py` | Runs the batch-doubling sweep (one process per configuration) and the full runs |
| `analyze.py` | Builds the tables and figures from the raw results |
| `infer.py` | Reloads the saved model from `model/` and classifies custom text |
| `hw45_partD_colab.ipynb` | Part D notebook, executed on the Colab T4 (log noise trimmed) |
| `scale.jsonl`, `full.jsonl` | Raw benchmark and full-run records |

**Not included** (too large or not mine to redistribute): `Twitter_Data.csv` (download from Kaggle), the
cached tensors, and the saved model weights (about 440 MB each in `model/`).

## 5. How to reproduce

Requirements: Windows or Linux with an NVIDIA GPU (24 GB reproduces my batch limits), Python 3.11,
PyTorch with CUDA, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `bitsandbytes`.

```bash
# put Twitter_Data.csv in the working directory, then:
python hw45.py --mode full --tag baseline     # HW4 baseline: curves, test metrics, saves model/
python sweep.py                               # Parts A and B: batch scaling and full technique runs (~15 min)
python analyze.py                             # tables and figures
python infer.py "some tweet text"             # reload model/ and classify custom text
```

Single technique flags for `hw45.py`: `--amp` (bf16 autocast), `--ckpt` (gradient checkpointing),
`--accum N` (gradient accumulation, with `--bs` as the physical batch), `--opt8` (bitsandbytes AdamW8bit).
`--mode bench --bs N` times 20 steps at batch size N and prints samples/s and peak memory.

Part D: open `hw45_partD_colab.ipynb` in Google Colab with a GPU runtime and Run all (about 20 minutes; the
final 32k-context verification cell is slow on a T4).

## 6. Measurement notes

- **Memory cap.** On Windows the driver silently spills GPU memory into system RAM instead of raising an
  out-of-memory error, which first produced a bogus throughput collapse at batch 128. `hw45.py` therefore
  caps PyTorch's allocator at 93% of VRAM, so the reported maximum batch sizes are slightly conservative and
  are powers of two.
- **Benchmark method.** Each batch size runs in its own process; 3 warm-up plus 20 timed optimiser steps;
  epoch time is extrapolated from the mean step time. Batches in benchmark mode are drawn at random with
  replacement so batch sizes can exceed the 3,500-sample training set. Peak memory is
  `torch.cuda.max_memory_allocated`.
- **Mixed precision** is bf16 autocast without a gradient scaler.
- **Part D definitions.** vLLM batch B submits B requests together, so per-request latency is the batch
  completion time; tokens/s is generated tokens divided by batch wall time; EOS is ignored so every request
  generates exactly 128 tokens. The HF baseline is unoptimised eager generation, which exaggerates vLLM's
  advantage.
- **The model's answer to my own prompt** (saved in `partD_results.json`) is partly wrong; `METRICS.md`
  explains where. The measured tables, not the model's text, are the evidence.
