# AI Use - GAN Concepts and FashionMNIST AutoEncoders

## What did you use an assistant for, and what did you write yourself?

I used Claude Code, an AI coding assistant, to help me build this homework. I wrote the main parts of the assignment myself, including the FashionMNIST setup, the FC-AE and VAE model definitions, the training and evaluation code, the plots, and the final interpretation. I also ran the notebook, checked the saved metrics and figures, and prepared the README, METRICS file, notebook, and PDF write-up from the logged results.

I used the assistant mainly to help organize the notebook, explain the GAN concepts in plain English, draft some of the PyTorch scaffolding, and check that the required submission files were present. I chose the two models from the assignment text and I am responsible for checking and understanding the submitted code and conclusions.

## Give one specific thing it got wrong.

The assistant's first VAE implementation used mean-squared reconstruction error averaged over all image pixels together with a KL term averaged over the images and latent dimensions. The code ran, but the scales of the two loss terms were not comparable. The first run printed a VAE KL loss of only `0.000007` and a test PSNR of `10.616 dB`, which meant the model was barely using the variational part of the objective.

## How did you find out?

I checked the separate reconstruction and KL losses printed for every VAE epoch instead of looking only at the total loss. The KL value was almost zero throughout training, so I compared the reduction and averaging used by the two terms. That showed that the reconstruction loss was averaged over many pixels while the KL term was not being given a comparable scale.

## What did you change, and why does your version work now?

I changed the VAE reconstruction term to mean binary cross-entropy and kept the KL term as a mean over the batch and latent dimensions. This puts the two parts on more comparable scales and makes the KL regularizer contribute to training. After rerunning, the final KL loss was `0.048201`, the reconstruction loss was `0.427910`, and the test PSNR was `12.042 dB`.

I also checked that the executed notebook runs without errors, saves both model checkpoints, writes the JSON metrics, and produces the reconstruction, sampling, loss, and error-analysis figures used in the submission.
