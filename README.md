# DATA 266

**Siva Surya Chandran**
Email: sivasurya.chandran@sjsu.edu
San José State University - DATA 266

Homework for this course, one folder per assignment. Each folder has its own README with the
actual writeup.

| Folder | Assignment |
|:--|:--|
| [`HW1/`](HW1/) | Neural networks (PyTorch/TensorFlow) + CUDA matmul |
| [`HW2/`](HW2/) | Word2Vec transfer learning, RAG pipeline, training-time optimizations |
| [`HW3/`](HW3/) | Prompt engineering + self-attention from scratch |
| [`HW4/`](HW4/) | Mini GPT from scratch (multi-head attention, decoding strategies) |
| [`HW5/`](HW5/) | LoRA fine-tuning of flan-t5-base on DialogSum (r=4 vs r=16) |
| [`HW6/`](HW6/) | Self-supervised and contrastive learning on STL-10 (supervised vs rotation vs SimCLR) |
| [`HW7/`](HW7/) | GAN questions + FashionMNIST autoencoders: fully connected AE vs VAE (reconstructions, prior samples, worst-case reconstructions) |
| [`HW8/`](HW8/) | Diffusion on MNIST: forward process, time-conditioned noise predictor, DDPM vs DDIM sampling (T=100, 50 epochs) |
| [`GPU_assignment_1/`](GPU_assignment_1/) | GPU Assignment I (HW2.5): matmul TFLOPS across precisions, memory bandwidth, naive vs fused attention OOM boundary, sustained-load thermals on an RTX 4090 |
| [`GPU_assignment_2/`](GPU_assignment_2/) | GPU Assignment II (HW4.5): BERT fine-tune throughput and memory engineering on an RTX 4090 (batch scaling to the memory wall, mixed precision, gradient checkpointing, gradient accumulation, 8-bit optimizer) plus vLLM serving on a Colab T4. Tagged `hw4-5` |
| [`HW9/`](HW9/) | Latent diffusion on Fashion-MNIST: SD VAE latents, class-conditional U-Net, classifier-free guidance |
