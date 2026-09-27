# AI Use - Fine-Tuning an LLM with LoRA

## What did you use an assistant for, and what did you write yourself?

I wrote the main parts of the assignment myself - the data prep (including grouping the test
split into 167 unique dialogues with 3 references each), the LoRA config, the PyTorch training
loop with gradient accumulation, the before/after comparison, and the r=4 vs r=16 experiment.
I also did all the analysis of the outputs myself, since reading the summaries and figuring out
what actually changed was the point of the assignment.

Where I did use an assistant was mostly setup and debugging: figuring out why `peft` wouldn't
import on Colab (the preinstalled `torchao` conflict), checking how the PEFT API wants things
passed in (`TaskType.SEQ_2_SEQ_LM`, `target_modules` names for T5), and the matplotlib code for
the loss plots, since I always forget that syntax.

## Give one specific thing it got wrong.

When I was setting up the rank experiment, I asked how to train a second adapter with a smaller
rank. The suggestion was to just change `r=16` to `r=4` and leave everything else the same,
including `lora_alpha=32`. That runs fine and trains fine, which is what made it easy to miss -
but it means the r=4 run isn't really a fair comparison anymore.

## How did you find out?

I went back to how LoRA actually applies the update, `ΔW = (alpha / r) · BA`. With alpha fixed
at 32, r=16 gets scaled by 2 but r=4 gets scaled by 8, so the r=4 updates would be 4x bigger for
the same A and B. That's basically changing the effective learning rate at the same time as the
rank, so if r=4 came out better or worse I wouldn't know whether it was because of the rank or
because of the scaling.

## What did you change, and why does your version work now?

I changed `make_lora_model` so that `lora_alpha = 2 * r`, which keeps alpha/r at 2 for both
runs (alpha=32 for r=16, alpha=8 for r=4). I also call `set_seed(SEED)` (SEED = 3215) inside it and inside the
training function, so both runs get the same LoRA init and the same batch order. Now the only
thing that differs between the two runs is the rank, so the comparison in Part 7 actually
measures what it says it does. I explained this in the notebook too (end of Part 3).
