# DATA 266 - HW7

Siva Surya Chandran — sivasurya.chandran@sjsu.edu
019130215

## Personal parameters

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

The same convention as the earlier homeworks is used: the SJSU ID is `019130215`, and `3215` is retained as the four-digit SID4 used throughout this repository. HW7 has no assignment-specific HP_ID mapping. The two required architectures—FC-AE and VAE—are the two models trained here; no extra hyperparameter model was added.

## Contents

- `HW7.ipynb` — executed notebook containing the GAN answers, FashionMNIST loading, model definitions, training, evaluation, visualizations, and interpretation.
- `outputs/results.json` — machine-readable metrics.
- `figures/` — loss curves, reconstructions, VAE prior samples, and the six worst FC-AE reconstructions.
- `checkpoints/fc_ae.pt`, `checkpoints/vae.pt` — model state dictionaries.
- `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md` — submission artifacts required by the standing instructions.

## Run

I ran the notebook locally on Apple MPS using the same environment I used for HW6:

```bash
../HW6/.venv/bin/jupyter nbconvert --to notebook --execute HW7.ipynb \
  --output HW7.ipynb --ExecutePreprocessor.timeout=3600
```

It downloads FashionMNIST into `data/` on the first run. All Python, NumPy, and PyTorch seeds are set to `3215`.
