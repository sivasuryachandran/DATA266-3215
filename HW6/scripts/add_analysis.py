import json, nbformat as nbf
nb = nbf.read("HW6.ipynb", 4)
nb.cells.append(nbf.v4.new_markdown_cell(r"""
## Results and analysis

Main results (single run, SEED = 3215, SimCLR with the **demo's augmentations**):

| Model | Training | Test acc (8,000 imgs) | Top-5 NN precision | 1-NN acc |
|:--|:--|--:|--:|--:|
| A - Supervised | 500 labels, end-to-end, 15 ep | 43.95% | 39.66% | 42.40% |
| B - Rotation SSL + linear | 100k unlabeled, 15 ep; linear on 500 labels | **47.85%** | **40.97%** | **44.74%** |
| C - SimCLR + linear | 20k unlabeled, tau=0.2, demo aug, 20 ep; linear on 500 labels | 44.62% | 39.89% | 43.22% |
| (ablation) C' - SimCLR, stronger crop (0.2-1) and no hue | same as C | 51.16% | 44.96% | 49.44% |
| (reference) random encoder + linear | none | 28.59% | - | - |

Extra experiments (`scripts/robustness.py`, 5 different 500-image subsets and seeds 3215-3219; the pretrained encoders are the same in every run, only the labeled subset, the supervised training and the probe change):

| Model | Mean +- std (%) | Min - max |
|:--|--:|--:|
| A - Supervised | 44.85 +- 1.13 | 43.81 - 46.25 |
| B - Rotation, 100k images | 47.06 +- 0.65 | 46.55 - 47.85 |
| B' - Rotation, 20k images (same images as SimCLR) | 42.81 +- 0.28 | 42.59 - 43.21 |
| C - SimCLR, demo augmentations | 44.34 +- 0.69 | 43.60 - 45.21 |
| C' - SimCLR, stronger crop / no hue (ablation) | 50.36 +- 1.29 | 48.60 - 51.71 |
| random encoder + linear | 29.42 +- 1.54 | 28.20 - 32.06 |

**Which model performed best.** With the augmentations the assignment prescribes (the demo's), **rotation SSL (B) is best: 47.85%** in the main run and 47.06 +- 0.65 over five subsets, about 2-3 points above both
supervised (44.85 +- 1.13) and SimCLR (44.34 +- 0.69). SimCLR and the supervised baseline are statistically indistinguishable (their ranges overlap). Both SSL encoders are far above a random
frozen encoder (29%), so the pretraining clearly learned something. The nearest-neighbour precision tells the same story (B 41.0% > C 39.9% > A 39.7%), though the gaps are small.

**The ranking depends on the SimCLR augmentations, and on how much unlabeled data each method sees.** Two controls I added change how the headline should be read:
- *Data-matched:* B used 100k unlabeled images, C only 20k. When rotation gets the same 20k images (B'), it drops to 42.8 +- 0.3, *below* SimCLR with the demo augmentations (44.3 +- 0.7). So rotation's lead over demo-SimCLR comes from seeing 5x more data, not from a better objective.
- *Augmentation strength:* my first SimCLR run used a stronger crop (scale 0.2-1.0) and no hue jitter. That variant reaches **51.16%** (50.36 +- 1.29 over five subsets) and beats everything, including rotation with 5x more data, by about 3 points.
  The demo uses crop scale 0.6-1.0 and adds hue; the two settings differ in both, so I cannot say which one matters, only that the combination does. Both are one pretraining run each.

**Why SimCLR with the demo augmentations underperformed.** Its positive-pair top-1 accuracy reached 99% by epoch 20 (97% by epoch 10) and the NT-Xent loss fell to 2.03, versus 81.5% and 2.68 with the stronger augmentation. The contrastive
task had become too easy: two crops that each cover 60-100% of the image overlap heavily, so the network can match them from low-level cues (colour layout, texture, the shared pixels) without learning anything object-level, and it
stops improving. That is my explanation, supported by the ablation but not isolated by it. The demo's settings were designed for 32x32 CIFAR images; on 96x96 STL-10 images with the same crop scale, the two views share more content.
*Why rotation did well:* the pretext task was learned strongly (89.2% rotation accuracy on test images). Rotation cannot be solved from colour or position alone, and it needs a sense of what is upright (ships 66%, horses 59%, monkeys 52%,
versus 43%, 22%, 24% for the random encoder). *Why supervised was weakest:* 500 images for an 11M-parameter ResNet-18 from scratch overfit (train loss 0.53, test accuracy about 44%, very unstable early: 14% at epoch 4, 38% at epoch 6). It is weakest on
cat (23%) and dog (21%), the classes that need more than 50 examples each, and its errors are mostly look-alike classes (car/truck 261, ship/truck 209).

**What could improve each model.**
- *Supervised (A):* stronger augmentation (RandAugment, mixup/cutmix), label smoothing, more weight decay, early stopping on a slice of the 500; or initialise from the B/C' encoder and fine-tune, the standard way to use few labels.
- *Rotation (B):* the pretext accuracy was still rising slowly at epoch 15 (83.7%), so train longer; combine with a second pretext task (jigsaw, colorization); and probe mid-network layers, where rotation features are usually strongest.
- *SimCLR (C):* make the task harder: crop scale (0.2, 1.0) as in C' (+6.5 points), stronger colour jitter, more negatives (batch 256 gives only 510; use a larger batch or a memory queue), 100+ epochs and all 100k unlabeled images (we used 20k).
- *Linear probes:* only 500 labels; the probe hyperparameters were fixed in advance and not tuned, and more labels or tuning would help all of them.

**What the nearest-neighbour visualisations showed** (`outputs/nn_query*.png`, `outputs/nn_failures.png`; green frame = same class, number of correct neighbours out of 5):

| Query | Supervised | Rotation | SimCLR |
|:--|--:|--:|--:|
| dog (CLS_A) | 1 | 2 | 1 |
| car (CLS_B) | 5 | 2 | 5 |
| airplane | 3 | 4 | 2 |
| deer | 4 | 2 | 4 |

- No encoder wins everywhere: SimCLR and supervised retrieve 5/5 cars while rotation mixes in trucks (its main confusion, truck/car 172); rotation is best on the airplane (4/5). Four queries cannot rank the models, which is why I also measured precision@5 over all 8,000 test images. There the three are close (39.7 / 41.0 / 39.9%), with rotation slightly ahead and the strong-crop SimCLR clearly ahead (45.0%).
- The mistakes are informative. For the floatplane query the wrong neighbours are boats on water (and for SimCLR a truck and a horse): the encoders match *scene and background* (water, grass, tarmac) as much as the object. Dog queries return horses, deer and monkeys: four-legged animals on grass. So the embeddings capture
  coarse semantics (animal vs vehicle) well but confuse look-alike classes within a group.
- The retrieval-failure gallery (all 5 neighbours wrong) shows unusual poses, cluttered or distracting backgrounds, and objects filling the frame (a monkey on a bench whose nearest neighbour is a truck, a parked jet in a black-and-white photo). Counts of queries with 0/5 correct neighbours: supervised 1,825, SimCLR 1,671, rotation 1,591 (of 8,000), strong-crop SimCLR 1,516.
- Cosine similarities are very high for supervised and rotation (0.94-0.99) and noticeably lower and more spread for SimCLR (0.87-0.94), consistent with its loss pushing different images apart. The absolute values are not comparable across encoders; only the ranking within an encoder is meaningful.

**Overall.** Unlabeled data helped: rotation SSL beat the 500-label supervised model by about 2-3 points, and SimCLR with a stronger crop beat it by about 5.5 points, both with the encoder frozen. The demo's SimCLR augmentations
were too weak for this dataset and only matched supervised. The conclusions are limited by one pretraining run per method (the five-subset study varies the labeled subset, not the pretraining seed), a short training schedule, and an untuned linear probe.
"""))
nbf.write(nb, "HW6.ipynb")
