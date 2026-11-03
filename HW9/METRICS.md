# HW9 metrics

Numbers come from `outputs/results.json` (one run, seed 3215, Apple M5 MPS, torch 2.15.0, diffusers 0.42.1).

| Item | Value |
|---|---|
| VAE | `stabilityai/sd-vae-ft-mse` (frozen), 83.7M params |
| Input to VAE | Fashion-MNIST resized to 128x128, 3 channels, range [-1, 1] |
| Latent | 4 x 16 x 16, posterior mean times 0.18215 |
| Scaled latent mean / std | -0.039 / 1.265 |
| Encoding time (60,000 images) | 1445 s |
| Reconstruction MSE (1,024 train images) | 1.92e-4 on [0,1] (7.68e-4 on [-1,1]), about 37.6 dB PSNR |
| Diffusion | T=1000, linear beta 1e-4 to 0.02, abar_T = 4.04e-5 |
| U-Net | 14,494,340 params; base 64, multipliers (1,2,4), 2 res blocks per level, attention at 16x16 and the bottleneck |
| Training | 50 epochs, batch 128, Adam, LR 2e-5, label drop 0.1, 468 steps per epoch |
| Loss, epoch 1 / epoch 50 | 0.3926 / 0.0941 |
| Training time | 9620 s (about 2.7 h) |
| Sampling 64 images (1000 steps, w=3) | 123 s |

## Guidance check (100 generated images, 10 per class, same noise)

The classifier is a small CNN trained for 3 epochs (90.3% on 2,000 real test images). This is a sanity check, not part of the assignment.

| Guidance | Class match | Mean confidence |
|---|---:|---:|
| w = 0 | 0.83 | 0.834 |
| w = 3 | 1.00 | 0.958 |

Only 100 images and an imperfect classifier, and I did not measure diversity or FID.
