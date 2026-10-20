# HW7 metrics

I ran both models on Apple MPS using the standard FashionMNIST split: 60,000 training images and 10,000 test images. I used batches of 256, Adam with a learning rate of `1e-3`, 8 epochs, and SEED `3215`. The numbers below come from `outputs/results.json`.

## Main comparison

| Model | Latent dim | Trainable parameters | Final train loss | Test MSE | Test PSNR (dB) | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| FC-AE | 32 | 476,720 | 0.014989 | 0.014938 | 18.257 | 15.86 |
| VAE | 16 | 474,672 | 0.476111 | 0.062484 | 12.042 | 18.77 |

For the VAE, the training loss is binary cross-entropy plus the KL loss. I use test MSE for both models so their pixel reconstruction results can be compared directly. The VAE ended with a reconstruction loss of `0.427910` and a KL loss of `0.048201`.

## Class-wise test MSE

| Class | FC-AE | VAE |
|---|---:|---:|
| T-shirt/top | 0.013731 | 0.062027 |
| Trouser | 0.008246 | 0.052524 |
| Pullover | 0.014251 | 0.062361 |
| Dress | 0.013937 | 0.056608 |
| Coat | 0.013255 | 0.059924 |
| Sandal | 0.022046 | 0.064541 |
| Shirt | 0.014488 | 0.056522 |
| Sneaker | 0.011815 | 0.045424 |
| Bag | 0.022543 | 0.090844 |
| Ankle boot | 0.015065 | 0.074066 |

The FC-AE did best on trousers and sneakers. Sandals and bags were the hardest classes for it to reconstruct. The VAE followed a similar pattern, but its errors were higher because its KL regularizer encourages a smoother latent representation instead of focusing only on pixel-perfect reconstruction.

## Error analysis

The notebook reports the error for every class and saves `figures/fc_worst6.png`, which shows six of the FC-AE's highest-error test examples. These are mostly thin, unusual, or ambiguously shaped garments. Pixel MSE can be high even when the reconstruction still looks like the right type of clothing, so these results should not be treated as classification accuracy.
