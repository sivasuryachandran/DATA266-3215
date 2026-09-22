# METRICS — HW2.5 GPU Assignment I

GPU: NVIDIA GeForce RTX 4090

UUID: `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c`

Driver version: 570.86.16

PyTorch CUDA build: 12.4

CUDA reported by nvidia-smi -q: 12.8

## Vendor specifications

- Architecture: Ada Lovelace (AD102)
- Memory type: GDDR6X
- Specified memory bandwidth: 1008 GB/s
- Tensor cores: 4th generation
- Supported/reported reduced precisions: TF32, FP16, BF16, FP8, INT8
- Sources: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/, https://vast.ai/pricing/gpu/RTX-4090

## Part B — FP8 result

- FP8 achieved throughput: 313.06 TFLOPS
- Matrix size: N=4096
- Repetitions: 30
- GPU UUID: `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c`

## Table HW2.5.1 — Summary

| Measurement | Your GPU | Notes |
|---|---:|---|
| Peak achieved TFLOPS (BF16) | 154.55 | N=8192, UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| % of theoretical peak (BF16) | 93.6% | Dense BF16 reference, UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Effective bandwidth (GB/s) | 949.41 | 94.2% of specified bandwidth, UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Naive attention OOM length | Largest successful=65536; smallest failing=81920 | UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Fused attention OOM length | No OOM through 131072 | UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Steady-state / peak throughput | 91.4% | First-30-second peak=163.56 TFLOPS; final-five-minute mean=149.44 TFLOPS; UUID `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Throttle onset | None observed | 450 W software power cap active; no thermal slowdown |

All measurements are associated with the GPU UUID recorded above.
