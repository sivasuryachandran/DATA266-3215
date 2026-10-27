# DATA 266 — HW8

Siva Surya Chandran — sivasurya.chandran@sjsu.edu

## Personal parameters

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

Same convention as the earlier homeworks. HW8 has no assignment-specific HP_ID mapping; the configuration is fixed by the assignment (T=100, linear β from 1e-4 to 0.02, batch size 128, 50 epochs).

## Contents

- `HW8.ipynb` — executed notebook: (a) forward diffusion visualization, (b) 5-conv-layer time-conditioned noise predictor and training-loss plot, (c) DDPM sampling, (d) DDIM sampling at 50 and 20 steps, comparison, and discussion.
- `HW8_supplement.ipynb` — same code with β_end=0.1 and 30 epochs. **Not the assignment configuration**; it tests why the specified configuration samples poorly (see below).
- `figures/`, `outputs/results.json`, `checkpoints/ddpm_mnist.pt` — main run. `figures_supp/`, `outputs_supp/`, `checkpoints_supp/` — supplementary run.
- `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md` — submission artifacts.

## Main finding (read this)

With the prescribed schedule, ᾱ_T = 0.364, so the forward process does **not** reach pure noise at t=100. Sampling from N(0,I) is therefore off-distribution, and the generated images from DDPM and DDIM are **not recognisable digits** (blobs and stroke fragments). Training itself is fine (loss 0.158 → 0.060). In the supplementary run with β_end=0.1 (ᾱ_T≈0) the samples are clearly more digit-like, which supports this explanation, but they are still imperfect and I tested only that one alternative. DDIM-20 is ~5.5x faster than DDPM-100 and DDIM is deterministic (repeat runs differ by exactly 0).

## Run

Local Apple MPS, using the HW6 environment:

```bash
../HW6/.venv/bin/jupyter nbconvert --to notebook --execute HW8.ipynb \
  --output HW8.ipynb --ExecutePreprocessor.timeout=3600
```

Training takes about 39 minutes for 50 epochs on MPS. MNIST downloads into `data/` on the first run. All seeds are 3215.
