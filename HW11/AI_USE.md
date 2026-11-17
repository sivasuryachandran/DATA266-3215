# AI Use - CLIP Similarity and Zero-Shot Classification

## What did you use an assistant for, and what did you write yourself?

I used Claude Code to build and review most of this homework. I wrote the notebook code (CLIP loading, normalized embeddings, similarity scores, figures, and the zero-shot evaluation), ran it, and drafted the README, metrics and analysis. I gave it the assignment, checked the outputs and figures, edited the wording, and I am responsible for understanding everything in the submission.

## Give one specific thing it got wrong.

The first run crashed at the box-plot cell because the code used `boxplot(labels=...)`, which this matplotlib version no longer accepts. Its fix was then reported as a rerun when the edit had silently failed (a macOS `sed -i` error), so the notebook was unchanged. A draft of the analysis also said the non-dog images in the top 100 were mostly cats and deer, which the saved results do not show.

## How did you find out?

The traceback showed the matplotlib error, and the `sed` error message showed the edit had not been applied. I noticed the analysis sentence had no number behind it: `outputs/results.json` stores only the 0.70 top-100 precision, not what the misses are.

## What did you change, and why does your version work now?

I switched to `tick_labels`, reapplied the edit and reran the whole notebook to the end. I limited the top-100 claim to what the top-32 figure actually shows (cats, deer and a bird). The numbers in the write-up now come directly from `outputs/results.json` and the notebook output.
