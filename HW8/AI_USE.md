# AI Use - Mini Diffusion Model (DDPM to DDIM)

## What did you use an assistant for, and what did you write yourself?

I wrote most of this homework myself, including the MNIST setup, forward diffusion process, noise-prediction CNN, training loop, DDPM and DDIM sampling code, plots, and discussion. I ran the notebook and checked the results myself.

I used Claude Code mainly as a reviewer. It helped me check the diffusion explanation, identify a problem in the discussion of the final timestep, and rewrite some of the text in clearer plain English. It also helped me check the notebook for consistency. I did not use the assistant to write the main homework code, and I am responsible for understanding and checking everything in the submission.

## Give one specific thing it got wrong.

The assistant initially described the t=100 image as pure noise and said that DDIM with eta=1 exactly matched the DDPM implementation. Those statements were too strong for this notebook. With the assigned schedule, abar_T is about 0.364 and sqrt(abar_T) is about 0.60, so the final state is heavily corrupted but not exactly standard Gaussian.

## How did you find out?

I checked the schedule values printed by the notebook and looked at the forward-diffusion visualization. A faint trace of the digit can remain at t=100, which is consistent with the calculated value of sqrt(abar_T). I also compared the DDPM and DDIM update formulas and saw that the clipping used in the DDIM implementation means the two samplers should not be called exactly identical.

## What did you change, and why does your version work now?

I changed the wording to say that the image is heavily corrupted rather than pure noise, added a sentence explaining why the final state is not exactly standard Gaussian, and corrected the DDIM comparison. I also checked that the notebook code is syntactically valid and that the explanations agree with the formulas used in the implementation.
