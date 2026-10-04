# Self-Supervised and Contrastive Representation Learning (STL-10)

**Siva Surya Chandran**
Email: sivasurya.chandran@sjsu.edu

**Personal Parameters** (Section 0.1 of the standing requirements):

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|------|------|-------|-------|-------|-------|
| 3215 | 3215 | 215   | 5     | 5     | 2     |

**Note on SID4.** Same as in the earlier homeworks: my SJSU ID is 019130215, so the last four digits are 0215, and I use
3215 (leading zero replaced by 3) so SID4 is a genuine 4-digit value. All parameters are derived from SID4 = 3215.

HW6 has no HP_ID mapping and no data slice. SEED = 3215 is used for every random seed, and CLS_A = 5 (dog) and CLS_B = 2 (car) are the
classes of the first two query images in Part D.

---

## What's in here

| File / folder | Purpose |
|:--|:--|
| `HW6.ipynb` | The assignment, executed top to bottom with outputs, including the written analysis at the end |
| `outputs/` | Plots (augmentations, losses, per-class accuracy, nearest-neighbour figures per query, retrieval failures) and `results.json` with every number |
| `scripts/` | `build_notebook.py` (generates the notebook code), `add_analysis.py` (appends the analysis cell after the run), `robustness.py` + `plot_robustness.py` (five-subset study and data-matched rotation) |
| `ablation_strongcrop/` | The first full run (SimCLR with crop 0.2-1, no hue): notebook, log, results and encoder |
| `HW6.pdf` | The document (.pdf) that contains my findings |
| `METRICS.md` | All numbers, setup table, per-class accuracy, query results |
| `RUN_LOG.txt` | Console output of the run that produced the numbers, with timestamps |
| `AI_USE.md` | Note on AI assistant use |

## What it does

ResNet-18 with no pretrained weights is the backbone everywhere (96x96 STL-10 images, 512-d features). The labeled subset is 500
images (50 per class, 10% of train), the same in every part.

1. **Part A** - supervised, end to end, 15 epochs.
2. **Part B** - rotation prediction on all 100,000 unlabeled images for 15 epochs, then the rotation head is removed, the encoder is frozen, and a linear layer is trained on the 500 labels for 20 epochs.
3. **Part C** - SimCLR-style: 20,000 unlabeled images, two views from the 4 augmentations in the course demo (random resized crop 0.6-1, flip, colour jitter 0.4/0.4/0.4/hue 0.1, grayscale), MLP projection head, NT-Xent with cosine similarity and tau = 0.2, 20 epochs; then head removed, encoder frozen, linear probe for 20 epochs.
4. **Part D** - top-5 cosine nearest neighbours on the test set for the three encoders, same four query images (dog, car, airplane, deer), plus precision@5 over all 8,000 test queries and a gallery of retrieval failures.
5. **Extra** - two controls I added: five different 500-image subsets/seeds, and rotation SSL on the same 20k images SimCLR uses.

Results:

| Model | Test accuracy | Five-subset mean +- std | Top-5 NN precision |
|:--|--:|--:|--:|
| A - Supervised, 500 labels | 43.95% | 44.85 +- 1.13 | 39.66% |
| B - Rotation SSL + linear (100k images) | **47.85%** | 47.06 +- 0.65 | **40.97%** |
| C - SimCLR + linear (demo augmentations, 20k images) | 44.62% | 44.34 +- 0.69 | 39.89% |
| (control) B' - rotation on the same 20k images | - | 42.81 +- 0.28 | - |
| (ablation) C' - SimCLR with crop 0.2-1 and no hue | 51.16% | 50.36 +- 1.29 | 44.96% |
| (reference) random encoder + linear | 28.59% | 29.42 +- 1.54 | - |

With the demo's augmentations, rotation is best, and SimCLR is no better than the supervised baseline. This is not because contrastive learning is weaker: SimCLR's positive-pair
accuracy hit 99%, so the task was too easy with 60-100% crops, and when I made the crop stronger (and dropped hue) it became the best model by about 3 points. Rotation's
lead over demo-SimCLR also disappears when it gets the same 20k images (42.8 vs 44.3). The nearest-neighbour pictures show coarse semantics
(vehicle vs animal) but also matching by background (a floatplane retrieves boats on water). The full explanation and improvement ideas are at the end of the notebook.

Things to know:

- **Main result uses the demo's augmentations**, as the assignment says. The stronger-crop run is kept as an ablation in `ablation_strongcrop/` (its own notebook, log, results and loss plot), and it differs in two settings at once (crop scale and hue), so I can't say which one matters.
- **Seeds:** the five-subset study varies the labeled subset, not the pretraining; each SSL method was pretrained once.
- **Linear probes** use precomputed frozen features (eval mode, no gradients) with standardisation fit on the 500 labeled images, and the encoder weight hash is verified unchanged.
- **Part A** is the final-epoch accuracy, not the best epoch, and it was noisy across epochs.
- Parts A and B came out identical in two separate full runs of the notebook, so the pipeline is deterministic on this machine.

## Running it

I ran it on a Mac (16 GB, MPS GPU) in about 66 minutes; it uses CUDA, MPS or CPU, whichever it finds (CPU would be far slower).

```
pip install torch torchvision matplotlib numpy jupyter
jupyter nbconvert --to notebook --execute HW6.ipynb
python scripts/robustness.py   # optional extra experiments, ~11 min
```

STL-10 (2.6 GB) downloads into `data/` on first run and is not committed. SEED = 3215 is set for Python, NumPy and torch; the GPU augmentations are random, so a rerun on different hardware
will not reproduce the exact numbers (MPS/CUDA kernels are not fully deterministic), but should land close.

Encoder checkpoints are written to `checkpoints/` by the notebook (and `encoder_rotation_20k.pt` by `scripts/robustness.py`); they are not committed because of their size.
