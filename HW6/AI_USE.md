# AI Use - Self-Supervised and Contrastive Learning (STL-10)

## What did you use an assistant for, and what did you write yourself?

I used Claude Code (an AI coding assistant) to assist me in building this homework. I wrote the main parts of the assignment myself (data loading, the GPU augmentations, the rotation and SimCLR training loops, the linear probe, the nearest-neighbour plots), ran it, and drafted the analysis text, the extra experiments (five subsets, data-matched rotation, the stronger-crop ablation)
and the README/METRICS/PDF write-up from the logged results, following the structure of my HW1-HW5 submissions. I chose the setup from the assignment text and the course demo, and I am responsible for checking and understanding
the submitted code and conclusions.

## Give one specific thing it got wrong.

The assignment says to use "the 4 augmentations utilized in the demo", and the assistant first guessed them (the standard SimCLR set, with a crop scale of 0.2-1.0 and no hue) without having the demo notebook, and wrote an analysis around the resulting 51.16%
and "SimCLR is best". When the demo turned out to use a crop scale of 0.6-1.0 plus hue, the re-run gave 44.62% and the conclusion reversed (rotation best, SimCLR no better than supervised). Its first draft of that analysis also had two small number errors
(it said SimCLR was best on 8 of 10 classes when it was 6 of 10, and quoted the wrong random-encoder baselines for ship/horse/monkey).

## How did you find out?

I pointed out that the demo notebook was available. The assistant read it, saw the two differences, re-ran the whole notebook with the demo's augmentations, and kept the first run as a labelled ablation. The wrong numbers in the first analysis were found by checking each claim against `outputs/results.json` after the run.

## What did you change, and why does your version work now?

Part C now uses the demo's augmentations (`simclr_aug` in the notebook), the analysis was rewritten around the new numbers, and the first run is reported honestly as an ablation (`ablation_strongcrop/`) with the caveat that it changes two settings at once.
I also added the five-subset and data-matched experiments so that the ranking is not judged from one run of one 500-image subset.
