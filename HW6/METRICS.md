# METRICS - Self-Supervised and Contrastive Learning (STL-10)

Everything below comes from the runs listed in `RUN_LOG.txt` (Apple-silicon Mac, MPS GPU), SEED = 3215 (SID4 = 3215, see the note on SID4
in the README). Raw numbers: `outputs/results.json` (main run), `outputs/robustness.json` (extra experiments), `ablation_strongcrop/results.json` (ablation run).

## Data

- STL-10: 5,000 labeled train, 8,000 test, 100,000 unlabeled (96x96 RGB).
- Labeled subset: 500 images = 10% of train, stratified 50 per class, drawn with SEED. Same 500 images in Parts A, B, C.
- All test numbers are on the full 8,000-image test set.

## Setup

| | Part A | Part B | Part C |
|:--|:--|:--|:--|
| Backbone | ResNet-18, random init, fc removed (512-d) | same | same |
| Pretraining data | - | all 100,000 unlabeled | 20,000 unlabeled (random subset, SEED) |
| Objective | cross-entropy, 10 classes | 4-way rotation (0/90/180/270) | NT-Xent, cosine sim, tau = 0.2 |
| Head | linear 512->10 | linear 512->4 (removed) | MLP 512->512->128 (removed) |
| Epochs | 15 | 15 (+ 20 linear) | 20 (+ 20 linear) |
| Optimizer | AdamW 1e-3, one-cycle, batch 64 | AdamW 1e-3, cosine, batch 256 | AdamW 1e-3, cosine, batch 256 |
| Augmentation | crop (0.5-1) + flip | crop (0.5-1) + flip, then rotation | the 4 demo augmentations: random resized crop (0.6-1), flip, colour jitter (0.4, 0.4, 0.4, hue 0.1), grayscale (0.2); two views |
| Linear probe | - | Adam 1e-2, wd 1e-4, batch 32, 20 ep, standardised features, encoder frozen | same |

The augmentations in Part C are the ones in the course demo (`Demo_6_Self_Supervised_Learning.ipynb`), re-implemented as batched GPU ops (the hue shift is a YIQ rotation, not torchvision's HSV shift).

The encoder is frozen in the probes: eval mode, no gradients, parameters never given to the optimizer, and an MD5 hash of the
encoder weights is checked before/after (`encoder hash unchanged` lines in the log). Linear-probe hyperparameters were fixed in advance and not tuned on the test set.

## Test accuracy (main run, SEED 3215)

| Model | Test acc | Top-5 NN precision | 1-NN acc | Queries with 0/5 correct neighbours (of 8,000) |
|:--|--:|--:|--:|--:|
| A - Supervised (500 labels) | 43.95% | 39.66% | 42.40% | 1,825 |
| B - Rotation SSL + linear | **47.85%** | **40.97%** | **44.74%** | 1,591 |
| C - SimCLR + linear (demo augmentations) | 44.62% | 39.89% | 43.22% | 1,671 |
| (ablation) C' - SimCLR, crop 0.2-1, no hue | 51.16% | 44.96% | 49.44% | 1,516 |
| (reference) random encoder + linear | 28.59% | - | - | - |

## Five subsets / seeds (`scripts/robustness.py`, seeds 3215-3219, test accuracy %)

Each seed draws a different stratified 500-image subset; Part A is retrained from scratch for each, the probes are retrained. The pretrained encoders (B, B', C, C') are the same in every run.

| Model | Seed 3215 | 3216 | 3217 | 3218 | 3219 | Mean +- std |
|:--|--:|--:|--:|--:|--:|--:|
| A - Supervised | 43.95 | 44.36 | 46.25 | 43.81 | 45.86 | 44.85 +- 1.13 |
| B - Rotation, 100k images | 47.85 | 46.60 | 46.64 | 47.69 | 46.55 | 47.06 +- 0.65 |
| B' - Rotation, 20k images (same as SimCLR's) | 43.00 | 42.66 | 42.59 | 43.21 | 42.59 | 42.81 +- 0.28 |
| C - SimCLR, demo augmentations | 44.62 | 43.66 | 45.21 | 44.61 | 43.60 | 44.34 +- 0.69 |
| C' - SimCLR, crop 0.2-1, no hue (ablation) | 51.16 | 50.85 | 51.71 | 49.46 | 48.60 | 50.36 +- 1.29 |
| random encoder + linear | 28.59 | 28.20 | 32.06 | 29.44 | 28.81 | 29.42 +- 1.54 |

## Pretext-task results

| | Final value |
|:--|--:|
| Rotation accuracy, train (epoch 15) / on test images | 83.66% / 89.24% |
| Rotation-20k (B') train accuracy, epoch 15 | 73.70% |
| SimCLR C (demo aug): NT-Xent epoch 1 -> 20 | 4.28 -> 2.03 |
| SimCLR C: positive-pair top-1 epoch 1 -> 10 -> 20 | 25.4% -> 97.4% -> 99.0% |
| SimCLR C' (ablation): NT-Xent, positive-pair top-1 at epoch 20 | 2.68, 81.5% |
| Wall-clock: A / B pretext / C pretext / whole notebook | ~1 / ~43 / ~22 / ~66 min |

## Per-class test accuracy (%, main run)

| Class | A | B | C | random enc. |
|:--|--:|--:|--:|--:|
| airplane | 60 | 51 | 69 | 51 |
| bird | 43 | 39 | 35 | 23 |
| car | 58 | 62 | 48 | 37 |
| cat | 23 | 45 | 30 | 18 |
| deer | 41 | 31 | 41 | 24 |
| dog | 21 | 28 | 27 | 18 |
| horse | 48 | 59 | 38 | 22 |
| monkey | 35 | 52 | 52 | 24 |
| ship | 54 | 66 | 56 | 43 |
| truck | 56 | 45 | 51 | 27 |

Top confusions (count, true -> predicted): A car->truck 261, ship->truck 209; B truck->car 172, truck->ship 166; C car->truck 221, truck->car 184.

## Fixed queries in Part D (test index, class)

6224 dog (CLS_A = 5), 1842 car (CLS_B = 2), 3770 airplane, 2209 deer. Same four images for all three encoders. Neighbours are top-5 by cosine similarity,
query excluded. Number of same-class neighbours out of 5:

| Query | Supervised | Rotation | SimCLR (demo aug) |
|:--|--:|--:|--:|
| dog | 1 | 2 | 1 |
| car | 5 | 2 | 5 |
| airplane | 3 | 4 | 2 |
| deer | 4 | 2 | 4 |

## Caveats

- The five-subset study varies the labeled subset and the supervised/probe training, but each SSL method was pretrained once; pretraining-seed variance is not measured.
- Part A's test accuracy fluctuated between epochs (14-45%); the reported number is the final epoch, not the best.
- B used 100k unlabeled images and C used 20k (as the assignment specifies); B' is the data-matched control.
- The ablation C' differs from C in two settings at once (crop scale 0.2-1 vs 0.6-1, and no hue vs hue 0.1), so it cannot say which one matters.
- Ranking between A and C is within noise; B > A and B > C and C' > everything are larger than the five-subset spread.
