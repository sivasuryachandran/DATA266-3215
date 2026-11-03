# AI Use - Latent Diffusion and Stable Diffusion

## What did you use an assistant for, and what did you write yourself?

I used Claude Code to build most of this homework. It wrote the notebook code (VAE encoding, latent forward diffusion, the U-Net with attention and classifier-free guidance, the training loop, and sampling), ran it, and drafted the README, metrics and analysis. I gave it the assignment, checked the outputs and figures, edited the wording, and I am responsible for understanding everything in the submission.

## Give one specific thing it got wrong.

It first estimated about 2 hours of training from a short benchmark of a few steps. The real run took about 2.7 hours (9620 s), because the speed over a full run was slower than in the first few steps. Also, the guidance check showed 100% class match at w=3, which sounds better than it is: the classifier I used is only 90.3% accurate and I only tested 100 images.

## How did you find out?

I compared the benchmark against the per-epoch times and the train_seconds saved in the checkpoint. For the guidance result I reread the classifier accuracy and sample size in `outputs/results.json` before writing it up.

## What did you change, and why does your version work now?

I put the real training time in the README and metrics, and described the guidance result as a rough sanity check, with its limits stated (small sample, imperfect classifier, no diversity or FID measured). The numbers in the write-up now come directly from `outputs/results.json` and the notebook output.
