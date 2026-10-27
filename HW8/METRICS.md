# HW8 metrics

Numbers come from `outputs/results.json` (main run, assignment configuration) and `outputs_supp/results.json` (supplementary run). Device: Apple MPS, seed 3215, batch size 128, Adam lr 2e-4, 416,641 parameters, 5 conv layers.

## Training

| Run | β schedule | ᾱ_T | Epochs | Final epoch loss | Train time (s) |
|---|---|---:|---:|---:|---:|
| Main (assignment) | 1e-4 → 0.02 | 0.364 | 50 | 0.0599 | 2345 |
| Supplementary | 1e-4 → 0.1 | ≈0 | 30 | 0.0382 | 1395 |

The two losses are not directly comparable because the noise levels differ.

## Sampling comparison (256 images, same starting noise for every sampler)

Classifier: small CNN trained 3 epochs on MNIST (98.6% on 2,000 test images). "Confidence" is the mean max-softmax; "diff vs DDPM" is mean absolute pixel difference on a 0–1 scale.

### Main run (assignment configuration)

| Sampler | Network evals | Time (s) | Confidence | Distinct classes | Pixel mean | Diff vs DDPM |
|---|---:|---:|---:|---:|---:|---:|
| DDPM | 100 | 6.03 | 0.836 | 9 | 0.261 | — |
| DDIM η=0, 100 | 100 | 6.22 | 0.787 | 8 | 0.306 | 0.177 |
| DDIM η=0, 50 | 50 | 2.79 | 0.794 | 8 | 0.303 | 0.172 |
| DDIM η=0, 20 | 20 | 1.09 | 0.796 | 8 | 0.300 | 0.166 |
| Real MNIST | — | — | 0.993 | — | 0.119 | — |

### Supplementary run (β_end=0.1, not the assignment configuration)

| Sampler | Network evals | Time (s) | Confidence | Distinct classes | Pixel mean | Diff vs DDPM |
|---|---:|---:|---:|---:|---:|---:|
| DDPM | 100 | 5.69 | 0.795 | 10 | 0.138 | — |
| DDIM η=0, 100 | 100 | 5.38 | 0.822 | 10 | 0.195 | 0.209 |
| DDIM η=0, 50 | 50 | 2.71 | 0.807 | 10 | 0.176 | 0.189 |
| DDIM η=0, 20 | 20 | 1.08 | 0.805 | 10 | 0.165 | 0.179 |
| Real MNIST | — | — | 0.988 | — | 0.119 | — |

## Determinism

DDIM-50 repeated from the same noise: max absolute difference 0.0. DDPM from the same noise with two different noise seeds: mean absolute difference 0.177 (main), 0.190 (supplementary).

## Caveats

In the main run the generated images are not recognisable digits, so classifier confidence there is not a quality measure. Pixel mean (real 0.119) is the more telling number: the main run is 2–2.5x too bright. The supplementary run is closer but still has DDIM brighter than real. Confidence differences between samplers are small and from one seed. No FID was computed.
