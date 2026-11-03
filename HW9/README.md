# DATA 266 — HW9

Siva Surya Chandran — sivasurya.chandran@sjsu.edu

## Personal parameters

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---:|---:|---:|---:|---:|---:|
| 3215 | 3215 | 215 | 5 | 5 | 2 |

Same convention as the earlier homeworks. The assignment says I can skip the second hyperparameter configuration, since the pretrained VAE and the diffusion U-Net already make two models, so the configuration is fixed by the assignment.

## Contents

- `HW9.ipynb` — executed notebook: VAE compression and reconstruction check, forward diffusion in latent space, the conditional U-Net, training, and classifier-free-guidance sampling, with an analysis at the end.
- `figures/` — VAE reconstructions, forward diffusion, latent channel, training loss, the required 64-image sample grid (`samples_w3.png`), and a w=0 vs w=3 comparison.
- `outputs/results.json`, `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md` — submission artifacts.
- `checkpoints/ldm_unet.pt` — trained U-Net. The cached latents (about 250 MB) are left out of the repo and get rebuilt on the first run.

## Setup choices

Fashion-MNIST is 28x28, but the SD VAE shrinks images by 8x and the assignment asks for attention at 16x16. So I resize the images to 128x128 and repeat the channel three times, which gives a 4x16x16 latent. Latents are the VAE posterior mean times 0.18215. Sampling runs all 1000 reverse steps with `eps = (1+w)·eps_cond − w·eps_uncond` and w = 3.

## Main findings

The VAE reconstructs well (MSE 1.9e-4 on a 0–1 scale, about 37.6 dB), so it isn't the bottleneck. With the prescribed schedule abar_T is about 4e-5, so the forward process really does end in noise (decoded images are unrecognisable by t=500), and sampling from N(0, I) makes sense, unlike in HW8. The U-Net has 14.5M parameters and trained for 2.7 hours on Apple MPS; loss went from 0.393 to 0.094 and was still slowly falling at the specified learning rate of 2e-5.

The w=3 samples cover all ten classes and look like Fashion-MNIST items, with some texture artefacts on bags and shirts. A quick check with a small CNN classifier gave 100% class match at w=3 versus 83% at w=0 on 100 images. That is a rough sanity check only: the classifier is 90.3% accurate, the sample is small, and I did not measure diversity.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision diffusers safetensors transformers accelerate matplotlib jupyter nbconvert nbformat ipykernel
.venv/bin/jupyter nbconvert --to notebook --execute HW9.ipynb \
  --output HW9.ipynb --ExecutePreprocessor.timeout=-1
```

The VAE downloads from Hugging Face and Fashion-MNIST goes into `data/` on the first run. If `checkpoints/ldm_unet.pt` exists the notebook loads it instead of retraining; otherwise training takes about 2.7 hours on MPS. All seeds are 3215.
