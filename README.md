# DATA266 - HW1

Author: sivasurya.chandran@sjsu.edu

## Personal parameters (Section 0.1)

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|------|------|-------|-------|-------|-------|
| 3215 | 3215 | 215   | 5     | 5     | 2     |

**Note on SID4.** My SJSU ID is 019130215, so the last four digits are `0215`. Taken literally that
has a leading zero, which collapses to the 3-digit number 215 when used as a number and makes SID4
ambiguous with SLICE. I used **3215** instead - the last four digits with the preceding digit `3`
in place of the leading zero - so that SID4 is a genuine 4-digit value and SEED/SLICE stay distinct.
All parameters in this repo are derived from SID4 = 3215.

HP_ID = 5 -> **Schedule-long** -> hidden layers `[64, 32]`, learning rate `0.001`, **60 epochs**
(baseline is the same architecture and learning rate at 30 epochs). CLS_A/CLS_B are derived per the
standing requirements but are not referenced by any HW1 task.

## How to run on Google Colab

1. Copy this whole folder into your Google Drive (anywhere under `MyDrive`).
2. Open `neural_networks.ipynb` in Colab (right-click -> Open with -> Google Colaboratory).
3. `Runtime` -> `Change runtime type` -> **Hardware accelerator: T4 GPU**.
4. `Runtime` -> `Run all`.

Cell 0 mounts Drive, finds the folder containing `diabetes.csv`, and `cd`s into it, so every later
cell uses plain relative paths. If `diabetes.csv` is not found in Drive it falls back to Colab's
upload widget.

`neural_networks.ipynb` and `cuda.ipynb` are **identical combined notebooks** - both filenames are
kept because the assignment lists both, but you only need to run one.

## Contents

| File | What it is |
|------|------------|
| `neural_networks.ipynb` | Combined notebook: autoregressive-models answer, diabetes preprocessing/EDA, PyTorch + TensorFlow baseline vs. HP_ID=5 models, 3-seed measurement, loss curves, and the CUDA section. |
| `cuda.ipynb` | Identical copy of the above (assignment lists both filenames). |
| `matmul.cu` | Tiled CUDA matrix-multiplication kernel + CPU baseline + `cudaEvent` timing. Also embedded in the notebook via `%%writefile`. |
| `diabetes.csv` | Dataset (759 rows, pre-scaled to ~[-1, 1], no header). |
| `METRICS.md` | Required measurement tables. |
| `RUN_LOG.txt` | Console output from the run that produced the reported numbers. |
| `AI_USE.md` | AI-use appendix (Section 0.5). |
| `HW1_writeup.docx` | Document deliverable. |
| `figures/` | Correlation matrix, feature distributions, and per-framework loss curves. |

## Status

**Complete.** The whole notebook has been executed end to end on Google Colab with a Tesla T4
runtime, and all outputs are saved in `neural_networks.ipynb`.

- PyTorch/TensorFlow sections were run both locally (Windows, CPU) and on Colab; all 12 reported
  accuracies are bit-identical across the two environments.
- The CUDA section built with `nvcc` at `-arch=sm_75` and ran at N = 256, 1024, and 4096, with
  correctness verified against the CPU baseline (`max abs diff` ~1e-5, printed `(OK)`).
- Profiler: **`nvprof`**. Nsight Systems (`nsys`) is not installed in the current Colab image;
  `nvprof` ran successfully and supplied the kernel-vs-transfer breakdown. Nsight Compute also ran
  (`matmul_1024_ncu.ncu-rep`), but its reported kernel time is inflated ~300× by 9-pass counter
  replay and is flagged as such rather than used.

`METRICS.md`, `RUN_LOG.txt`, and `HW1_writeup.docx` all contain the real measured numbers.
