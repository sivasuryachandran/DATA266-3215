# DATA 266 — HW11

Siva Surya Chandran — sivasurya.chandran@sjsu.edu

## Personal parameters

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

Same convention as the earlier homeworks. This assignment has no training and no hyperparameters to pick, so only SEED is used (it fixes the random picks in one figure). The Part 1 caption is "a photo of a dog"; dog is CIFAR-10 class index 5, which happens to equal CLS_A.

## Contents

- `HW11.ipynb` — executed notebook. Part 1: load CLIP, define the caption, preprocess 1,000 images (100 per class), extract and normalize embeddings, compute cosine similarity, show the images with their scores. Part 2: zero-shot CIFAR-10 classification, per-class accuracy and a confusion matrix. The analysis is at the end.
- `figures/` — top and bottom 32 images by score, random samples per class, score box plot, caption-vs-class heatmap, per-class accuracy, and two confusion matrices (single prompt and 4-template ensemble).
- `outputs/results.json`, `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md` — submission artifacts.

## Setup choices

I used `openai/clip-vit-base-patch32` and the CIFAR-10 test set, because its 10 classes give a clear ground truth for whether an image matches a caption. Part 1 takes the first 100 test images of each class. CIFAR-10 images are 32x32 and the CLIP processor upsamples them to 224x224. Part 2 classifies by the highest cosine similarity among the 10 prompts "a photo of a {class}". I also tried an average of 4 prompt templates for comparison.

## Main findings

In Part 1 dog images score highest (mean cosine 0.259) and ships and trucks lowest (about 0.195). The ranking separates dogs from the rest well (ROC-AUC 0.962), but only 70 of the 100 top-ranked images are dogs, and the top 32 already include cats, deer and a bird. For every class, its own caption has the highest mean similarity.

In Part 2 zero-shot accuracy is 88.8% on the 10,000 test images (89.05% with the template ensemble, a gain too small to matter). Automobile and horse reach 97.6%, while deer (76.5%) and frog (77.4%) are worst. The largest confusion is deer predicted as horse (154 images), then truck as automobile (89) and cat as dog (79). The small, blurry images probably hurt the animal classes most, but I did not test that.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision transformers matplotlib scikit-learn jupyter nbconvert nbformat ipykernel pillow
.venv/bin/jupyter nbconvert --to notebook --execute HW11.ipynb \
  --output HW11.ipynb --ExecutePreprocessor.timeout=-1
```

CIFAR-10 goes into `data/` and the CLIP weights come from Hugging Face on the first run. CLIP inference is deterministic. The full run took about 21 minutes on Apple MPS, mostly embedding the 10,000 test images.
