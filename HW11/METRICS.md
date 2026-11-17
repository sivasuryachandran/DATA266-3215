# HW11 metrics

Numbers come from `outputs/results.json` (one run, Apple MPS, torch 2.15.0, transformers 5.23.0).

| Item | Value |
|---|---|
| Model | `openai/clip-vit-base-patch32`, 151.3M params, 512-d embeddings, frozen |
| Part 1 images | 1,000 (100 per CIFAR-10 class), caption "a photo of a dog" |
| Part 1 cosine range | 0.152 to 0.285 |
| Part 1 mean cosine, dog / cat / deer | 0.259 / 0.239 / 0.234 |
| Part 1 mean cosine, ship / truck | 0.195 / 0.195 |
| Part 1 ROC-AUC (dog vs not), top-100 precision | 0.962, 0.70 |
| Matching caption is best for N of 10 classes | 10 |
| Part 2 zero-shot accuracy, single prompt | 88.80% (10,000 test images) |
| Part 2 zero-shot accuracy, 4-template ensemble | 89.05% |
| Part 2 accuracy on the 1,000-image Part 1 subset | 90.4% |

## Per-class accuracy (single prompt / ensemble)

| Class | Single | Ensemble |
|---|---:|---:|
| airplane | 0.901 | 0.907 |
| automobile | 0.976 | 0.963 |
| bird | 0.888 | 0.874 |
| cat | 0.860 | 0.844 |
| deer | 0.765 | 0.785 |
| dog | 0.884 | 0.884 |
| frog | 0.774 | 0.792 |
| horse | 0.976 | 0.978 |
| ship | 0.955 | 0.964 |
| truck | 0.901 | 0.914 |

Top confusions (single prompt): deer to horse 154, truck to automobile 89, cat to dog 79, frog to cat 79, frog to bird 75, dog to cat 69.
