# METRICS — HW4.5 GPU Assignment II (Training Throughput and Memory Engineering)

GPU (training, Parts A, B, E): NVIDIA GeForce RTX 4090

UUID: `GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39`

Driver version: 610.60 (Windows, WDDM)

PyTorch CUDA build: 12.4 (torch 2.6.0+cu124)

CUDA reported by nvidia-smi -q: 13.3

GPU (serving, Part D only): Tesla T4 on Google Colab — UUID `GPU-6cb18f34-3c2f-33e8-fdfb-31b18ae8a8df` (vLLM does not run on native Windows)

## Vendor specifications (RTX 4090)

- Architecture: Ada Lovelace (AD102); VRAM 24 GB GDDR6X (24564 MiB reported); power limit 450 W
- Specified memory bandwidth: 1008 GB/s; 4th-generation tensor cores (BF16/FP16/TF32/FP8/INT8)

## HW4.5 summary (best configurations) — not an HW5 baseline

Part C skipped (optional)

**Table HW4.5.2 — Summary (RTX 4090, bert-base-uncased, 5,000 rows 215–5214, SEED=3215)**

| Measurement | Result | Notes |
|---|---:|---|
| HW4 baseline (bs 32, fp32): peak memory | 6.18 GB | `torch.cuda.max_memory_allocated`, 3 epochs |
| HW4 baseline: wall-clock training time | 49.2 s | 3 epochs incl. per-epoch validation; test acc 0.708 |
| Largest batch that runs (baseline fp32) | 128 | 256 raises OOM; allocator capped at 93% of VRAM |
| **Best wall-clock** | **22.9 s** | Mixed precision (bf16), bs 32, full 3-epoch run; test acc 0.704; peak 4.58 GB |
| **Best throughput** | **549.4 samples/s** | Mixed precision, bs 128 (benchmark, 14.0 GB); bs 64 gives 526.2 samples/s at 7.7 GB |
| **Best peak memory** | **1.34 GB** | AMP + checkpointing + 8-bit optimiser, bs 32 (benchmark only, accuracy not trained); smallest accuracy-validated: checkpointing alone 2.08 GB, acc 0.708 |
| Largest batch with any technique | 512 | Gradient checkpointing, and AMP+ckpt+8-bit (1024 OOMs) |
| Recommended config for a full fine-tune | AMP (bf16), bs 32 | 2.1x faster than baseline, 26% less memory, accuracy within noise |
| GPU-hours, one full-dataset fine-tune (3 epochs, 114,078 train rows) | 0.207 GPU-h (AMP) / 0.446 GPU-h (fp32) | See Part E below and `GPU_Reservation.md` |
| Part D serving (Colab T4, fp16) | vLLM b32 1,399 tok/s; HF b1 22.5 tok/s | p50/p95 and KV-cache in Part D below |

All Part A/B/E measurements are associated with the RTX 4090 UUID above; Part D with the T4 UUID.

---


**Scope and honesty notes (read first)**
- **Single card only:** all measurements are on **one NVIDIA GeForce RTX 4090 (24 GB)**, UUID `GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39` (see `nvidia-smi-L.txt`). The assignment's "both cards" comparisons (Part A plot for two cards, max-batch ratio vs VRAM ratio, "same on both cards", Table HW4.5.1 once per card, Part D on both cards) **could not be done**; the instructor clarified one GPU is sufficient.
- **Part C skipped (optional)**
- **Part D (vLLM serving, KV-cache sweep) was run on a Colab Tesla T4, not on the RTX 4090.** vLLM does not run on native Windows and no Linux/WSL2 was available (no admin rights). So Part D numbers are for the T4 (UUID `GPU-6cb18f34-3c2f-33e8-fdfb-31b18ae8a8df`), fp16. The comparison against my HW4.a serving numbers is **not done** (see Part D).
- Hyperparameter-ID / class-pair parameters (HP_ID, CLS_A, CLS_B from my parameter table) were **not applied**; the notebook defaults were used. SEED=3215 and SLICE=215 were applied.
- This summary is the **HW4.5 summary**, not an HW5 baseline.

## Data (identical for baseline and every technique)
- Dataset: Kaggle `twitter-sentiment-dataset`, file `Twitter_Data.csv` (downloaded as `saurabhshahane/twitter-sentiment-dataset`; owner slug not independently verified) (`clean_text`, `category` ∈ {-1,0,1}; 162,980 raw rows, 162,969 after dropping null `category`/`clean_text`). Label map: -1→0 Negative, 0→1 Neutral, 1→2 Positive.
- **Row range:** after dropping nulls and resetting the index, **rows 215–5214 (0-indexed, 5,000 consecutive rows, SLICE=215)**. These are positions in the cleaned frame, not raw CSV line numbers.
- Split 70/15/15 with `train_test_split(random_state=SEED=3215)`: 3,500 train / 750 val / 750 test. Training init, shuffling and `torch` seeds all use SEED=3215.
- Model/settings carried over unchanged from the HW4.a notebook (pulled from Kaggle `praveengovi/bert-twitter-sentiment-classifier`): `bert-base-uncased`, MAX_LEN 256 (fixed padding), bs 32, lr 2e-5, eps 1e-8, 3 epochs, linear schedule no warmup, grad-clip 1.0. **Two deliberate deviations from that notebook:** the split (it used 90/10 with seed 41, no test set; the assignment requires 70/15/15 with SEED) and the optimiser class (`torch.optim.AdamW`, since `transformers.AdamW` was removed; no `correct_bias=False`).

## HW4 baseline (bs 32, fp32) — kept as the HW4.5 baseline
| Epoch | Train loss | Val loss | Val acc |
|---|---|---|---|
| 1 | 0.9932 | 0.8877 | 0.5933 |
| 2 | 0.7494 | 0.7797 | 0.6773 |
| 3 | 0.5609 | 0.7529 | 0.6987 |

- Validation loss and accuracy were still improving at epoch 3, so **no epoch past the best one was trained; overtraining cost = 0**. (Improvement was slowing, so more epochs would give small gains at best — untested.)
- Test accuracy **0.708** (loss 0.7005). **Peak GPU memory 6.18 GB** (`torch.cuda.max_memory_allocated`), **wall-clock 49.2 s** for 3 epochs including per-epoch validation.
- Confusion matrix (rows = true, cols = pred; Neg/Neu/Pos): `[[103,24,49],[33,198,45],[28,40,230]]` (`fig_confusion.png`).

| Class | Precision | Recall | Support |
|---|---|---|---|
| Negative | 0.628 | 0.585 | 176 |
| Neutral | 0.756 | 0.717 | 276 |
| Positive | 0.710 | 0.772 | 298 |

- **Worst class: Negative** (lowest recall and precision); 49 of its 176 test examples are predicted Positive. Five misclassified Negative→Positive examples, with diagnosis (the first 5 in test order; diagnoses are my reading of the text):
  1. "india has achieved all round progress under modi now selling shirts and cups…unemploymentno farmers suicideno…" — sarcasm; surface words are all positive ("progress", "bouncing leaps and bounds").
  2. "any other work mumbaiwho hell shout like dogbe like modi than dream for biopic" — garbled, no clear sentiment cue; "dream"/"biopic" read as positive.
  3. "when modi gave 2000rs then wasnt disastrous idea … which really great idea" — sarcasm via repeated "wasnt disastrous" and "really great idea".
  4. "there difference pakistani state dont support such crime but your narindra modi state support lynching…" — strong negative content but expressed as comparison/accusation; negations ("dont support") confuse it.
  5. "and even print this seriously whats this election modi entire opposition whats their agenda only defeat modi…" — negativity expressed through rhetorical questions with no explicit negative words.
- Caveat: single run, single seed; with 750 test samples one accuracy point ≈ ±1.7% (95% CI), so differences below ~1.5 points between techniques below are **not distinguishable**.

## Part A — batch scaling to the memory wall (RTX 4090 only)
Method: each batch size runs in a separate process; 3 warm-up + 20 timed optimiser steps; epoch time = steps/epoch × mean step time (extrapolated, 3,500 train samples). Batches are drawn at random with replacement so bs may exceed the train-set size. PyTorch allocator capped at 93% of VRAM (~22.3 GB) because Windows otherwise silently spills to system RAM (an uncapped run showed throughput collapsing at bs 128/256 instead of an OOM). Max batch is therefore conservative, and is a power of two (not the true integer maximum). Peak memory = max allocated.

Baseline fp32 (HW4 settings):
| Batch | Samples/s | Peak GB | Epoch (s) |
|---|---|---|---|
| 8 | 162.4 | 2.49 | 21.6 |
| 16 | 208.8 | 3.77 | 16.8 |
| 32 | 231.2 | 6.18 | 15.2 |
| 64 | 244.8 | 11.12 | 14.4 |
| 128 | 256.5 | 20.88 | 14.0 |
| 256 | OOM | – | – |

- **Largest batch that runs: 128.** Throughput gains are <6% per doubling beyond bs 32 and about +5% from 64→128; practically **throughput stops improving around bs 64**. Plot: `fig_throughput_vs_batch.png` (one card, so "same on both cards?" and the max-batch-ratio vs VRAM-ratio comparison are **not answerable**).
- Memory grows ~linearly in batch (≈0.16 GB/sample at fp32 for 256 tokens), which is why the wall comes at 128 on 24 GB.

## Part B — one technique at a time (RTX 4090, bs 32 unless noted)
Throughput/peak memory from the scaling runs at bs 32; accuracy is test accuracy from a full 3-epoch run at the same HW4 settings (bs 32, same seed, 110 optimiser steps).

**Table HW4.5.1 (RTX 4090)**
| Technique | Max batch | Throughput (samp/s) @bs32 | Peak memory (GB) @bs32 | Eval (test) accuracy |
|---|---|---|---|---|
| Baseline (HW4 settings) | 128 | 231.2 | 6.18 | 0.708 |
| Mixed precision (bf16 autocast) | 128 | 481.8 | 4.58 | 0.704 |
| Gradient checkpointing | 512 | 176.4 | 2.08 | 0.708 |
| Gradient accumulation (bs 8 × 4 = eff. 32) | 128 (physical, same as baseline; bs-8 steps) | wall 57.8 s vs 49.2 s | 2.91 (full run) | 0.704 |
| 8-bit optimiser (bnb AdamW8bit) | 128 | 222.5 | 5.57 | 0.705 |
| Best combination — AMP + 8-bit (+ checkpointing for bs 512) | 512 (with ckpt) | 372.5 (AMP+ckpt+8bit) / 481.8 AMP only | 1.34 (AMP+ckpt+8bit @bs32); 3.97 (AMP+8bit full run) | 0.707 (AMP+8bit) |

Notes: (a) for accumulation the comparison is wall time for the same 330 optimiser steps (110 per epoch × 3, effective batch 32): **57.8 s vs 49.2 s, +17.5% wall time**; effective batch achieved 32. Accumulation was only tested with physical bs 8, i.e. it reproduces eff. batch 32 on a small physical batch; a larger effective batch was not tried. (b) Checkpointing costs **−23.7% throughput at bs 32** (and +30% full-run wall time, 64.1 s vs 49.2 s); at bs 8 the cost is −21%. (c) Mixed precision *increases* throughput **+108%** (482 vs 231 samples/s; full run 22.9 s vs 49.2 s) since the 4090's bf16 tensor cores are far faster than fp32, while saving 26% memory — it did not cost accuracy within noise. AMP was bf16 autocast (no GradScaler), not fp16. (d) 8-bit optimiser: −3.7% throughput, −0.61 GB (−10%); its validation accuracy was slightly lower at epoch 3 (0.685 vs 0.699) but test accuracy is within noise. (e) The "best combination" full-accuracy run was AMP+8-bit without checkpointing; AMP+ckpt+8-bit was only benchmarked for throughput/memory (not trained to accuracy). Max batch for ckpt-based configs (512) is limited by activations at 24 GB; all max batches use the same 93% cap.

**Ranking by memory saved per % throughput lost (bs 32)**
| Rank | Technique | Memory saved | Throughput change | GB saved per 1% throughput lost |
|---|---|---|---|---|
| 1 | Mixed precision | 1.60 GB (26%) | **+108%** (gain) | ∞ — saves memory *and* speeds up |
| 2 | Gradient accumulation (bs 8×4) | 3.27 GB (53%) | −14.9% (wall +17.5%) | 0.22 |
| 3 | Gradient checkpointing | 4.10 GB (66%) | −23.7% | 0.17 |
| 4 | 8-bit optimiser | 0.61 GB (10%) | −3.7% | 0.16 |

- **Apply first: mixed precision** (free speedup and memory). Then checkpointing or accumulation only if you actually hit the wall.
- **Not bother with on a 32 GB card: the 8-bit optimiser.** BERT-base's optimiser state is only ~0.9 GB in fp32 (110M params × 8 B), so it saves 0.6 GB for a small throughput loss and a hint of accuracy risk. (I measured on a 24 GB card; 32 GB makes the argument stronger, since baseline bs 128 would already fit.)
- Caveat on ranking: checkpointing's memory benefit scales with batch size; at bs 32 the baseline only uses 6 GB so memory "saved" is not capacity gained. Its real value is the max batch (128 → 512 at 190 samples/s) and at long sequences, which weren't tested.

## Part C
**Part C skipped (optional)**

## Part D — Serving on a Colab Tesla T4 (not the 4090)
Notebook: `hw45_partD_colab.ipynb`; raw results `partD_results.json`. Model `Qwen/Qwen2.5-1.5B-Instruct` (weights cast bf16→fp16 since the T4 has no bf16), vLLM 0.32.0, 128 new tokens per request (EOS ignored), 20 repetitions per configuration after warm-up. vLLM batch B = B requests submitted together with `llm.generate`, so **per-request latency = batch completion time**; tokens/s = generated tokens / batch wall time. HF baseline = `transformers` text-generation pipeline, batch 1, greedy, fp16.

**Table HW4.1 (Tesla T4)**
| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|
| HF pipeline, batch 1 | 22.5 | 5,482 | 6,635 |
| vLLM, batch 1 | 65.9 | 1,893 | 1,981 |
| vLLM, batch 8 | 501.9 | 2,042 | 2,045 |
| vLLM, batch 32 | 1,399.5 | 2,928 | 2,952 |

**Latency vs throughput question.** In this measurement per-request latency never becomes worse than the HF baseline: even at batch 32, p50 is 2.9 s against 5.5 s for HF batch 1, because vLLM (CUDA graphs, fused kernels, no Python per-token loop) is already 2.9x faster than the HF pipeline at batch 1. Against vLLM's own batch 1, latency rises only +8% at batch 8 and +55% at batch 32 while throughput grows 7.6x and 21x, so the "latency worse than baseline while throughput still improves" crossover is not reached in the range tested (batches above 32 were not tried, so I can't say where it would appear). For an interactive application the per-request latency (p95) is what the user feels; total throughput matters for batch/offline serving and cost per token. Caveat: the HF baseline here is unoptimised eager generation, which exaggerates vLLM's advantage.

**KV cache vs context length (batch 8, T4).** KV bytes/token = 2 x 28 layers x 2 KV heads x 128 head-dim x 2 B (fp16) = 28,672 B (28 KiB).
| Context | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---|---|---|---|---|---|---|---|
| KV at batch 8 (GB) | 0.055 | 0.109 | 0.219 | 0.438 | 0.875 | 1.75 | 3.5 | 7.0 |

(Computed from the model config, not measured per context.) With `gpu_memory_utilization=0.85` vLLM allocated a KV cache of 312,672 tokens (about 8.4 GiB; a later re-created engine reported 316,400), so memory alone would allow about **39,000 tokens per request at batch 8**. The model's position limit (32,768) is reached first: batch 8 was **verified at a context of 32,760 tokens** without running out of memory. Longest context at batch 8 on the T4 is therefore **model-limited at ~32.7k, not memory-limited**. The RTX 4090 was not measured for Part D (no Linux), and neither was a second card.

**Comparison with HW4 numbers.** There was no separate earlier HW4.a serving run: this single Colab T4 run (fp16, same model, same definitions) is my HW4.a Table HW4.1, and it is also the HW4.5 Part D measurement. So the HW4-vs-HW4.5 serving comparison is degenerate (identical numbers, nothing differs in hardware, precision or method). The RTX 4090 has **no** vLLM numbers (no Linux), so no cross-card serving comparison exists.

**Own prompt and model.** Model: `Qwen/Qwen2.5-1.5B-Instruct`. Own prompt (supplied by me; saved with its answer in `partD_results.json` under `my_prompt` / `my_prompt_output`, and shown in the last cell output of `hw45_partD_colab.ipynb`): "Explain in simple terms why increasing GPU batch size can improve training throughput at first, but eventually gives smaller speed gains and can run out of GPU memory. Include how GPU utilization and memory limits contribute to this behavior." Generated greedily with a 400-token limit; the answer ends naturally.

Assessment of the answer (my reading, checked against my own Part A/B measurements): it is partly right and partly weak. It correctly says memory use grows with batch size and that the GPU can run out of memory. But it never mentions GPU utilization or why throughput gains flatten (the compute units are already saturated), which the prompt explicitly asked for, and it wrongly says a bigger batch gives "more updates … with each iteration" and faster convergence, which confuses samples/second with optimiser updates (a larger batch gives fewer updates per epoch). It also says larger batches "slow overall performance", which contradicts my measured throughput gains from batch 8 to 64. So a 1.5B model's answer here is a useful illustration but not a reliable explanation; my measured tables are the authoritative evidence.

## Part E — Analysis and summary
**GPU-hours for one full fine-tune on this GPU.** Interpreting "full fine-tune" as one 3-epoch run over the entire cleaned dataset (162,969 rows) with the same 70/15/15 split: train rows = 0.70 × 162,969 = 114,078; samples processed = 114,078 × 3 = 342,234. Measured full-run rates (training + per-epoch validation, so the per-epoch validation overhead is included in the same 70:15 ratio): baseline 3,500×3/49.22 s = 213.3 samples/s; AMP 10,500/22.92 s = 458.2 samples/s.
- Baseline fp32 bs 32: 342,234 / 213.3 = 1,604 s = **0.446 GPU-h**.
- AMP bs 32: 342,234 / 458.2 = 747 s = **0.207 GPU-h**.
- (The HW4-sized 5,000-row fine-tune itself took 49.2 s = 0.0137 GPU-h.) Extrapolation assumes constant throughput and ignores model load/tokenisation; tokenisation of the full set is extra.

**Configuration I'd use for a full fine-tune:** **mixed precision (bf16), batch size 32 (or 64 if lr is rescaled), standard fp32 AdamW, no checkpointing, no accumulation.** Justification: AMP gave 2.1× throughput and −26% memory at unchanged accuracy (0.704 vs 0.708, within noise); batch 32 keeps the HW4 optimiser-step schedule; peak 4.6 GB is far below 24 GB, so checkpointing/accumulation/8-bit buy nothing and cost throughput. Throughput plateaus at ~bs 64 (526 samples/s), so larger batches are pointless.

**Best configurations measured (RTX 4090, bench runs unless noted):**
- Best peak memory: AMP + checkpointing + 8-bit, bs 32 → **1.34 GB** (372.5 samples/s); smallest at bs 8 was 1.08 GB.
- Best throughput: AMP, bs 128 → **549.4 samples/s** (14.0 GB); AMP bs 64 gets 526.2 at 7.7 GB.
- Best wall-clock (full 3-epoch run, HW4 settings): AMP → **22.9 s** (vs 49.2 s baseline), peak 4.58 GB, test acc 0.704.

## Method caveats
Single seed/single run per configuration (no error bars); bench throughput from 20 timed steps; epoch time extrapolated; max batch is a power of two capped at 93% VRAM; accuracy differences (≤0.4 points) are well within sampling noise (±1.7%); environment: Windows (WDDM), driver 610.60, torch 2.6.0+cu124, transformers latest at install, bitsandbytes latest at install.

## Files
`hw45.py` (training/bench harness), `infer.py` (reloads the saved `model/` for custom inference; tested on 3 custom sentences), `model/` (saved HW4 baseline fine-tune), `sweep.py`, `analyze.py`, `scale.jsonl`/`full.jsonl` (raw results), `RUN_LOG.txt` (UUID-prefixed, derived; `RUN_LOG_raw.txt` is the original), `hw45_partD_colab.ipynb` + `partD_results.json` + `part_d_*.csv` (Part D, Colab T4), `part_a_b_scaling.csv`, `part_b_full_runs.csv`, `fig_throughput_vs_batch.png`, `fig_confusion.png`, `nvidia-smi-q_GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39.txt` + `nvidia-smi-L_RTX4090.txt` + `nvidia-smi-L_ColabT4.txt` (UUID captures), `GPU_Reservation.md` (reservation and GPU-hour record), `AI_USE.md`, `README.md`. Not included: the `hw4-5` git tag (not a git repo yet), and the saved model weights / dataset (too large).
