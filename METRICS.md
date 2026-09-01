# METRICS - HW1

Personal parameters: SID4=3215, SEED=3215, SLICE=215, HP_ID=5 (Schedule-long), CLS_A=5, CLS_B=2

SID4 note: my SJSU ID is 019130215, whose last four digits are `0215`; the leading zero reduces it
to 3 digits, so I used 3215 as SID4 to keep it a true 4-digit value. See README.md.
(CLS_A/CLS_B are not used by any HW1 task).

Baseline config (both frameworks): hidden layers `[64, 32]`, lr `0.001`, 30 epochs.
Modified config (HP_ID=5): hidden layers `[64, 32]`, lr `0.001`, **60 epochs**.
Split: 70/15/15 train/val/test, `random_state=SEED`, identical split reused for every model.

## 1. Test accuracy - mean ± std over 3 training seeds (SEED, SEED+1, SEED+2)

| Framework  | Model    | Seed 3215 | Seed 3216 | Seed 3217 | Mean   | Std    |
|------------|----------|-----------|-----------|-----------|--------|--------|
| PyTorch    | Baseline | 0.7632    | 0.7544    | 0.7105    | 0.7427 | 0.0230 |
| PyTorch    | Modified | 0.7719    | 0.7719    | 0.7719    | 0.7719 | 0.0000 |
| TensorFlow | Baseline | 0.7193    | 0.7544    | 0.7632    | 0.7456 | 0.0189 |
| TensorFlow | Modified | 0.7719    | 0.7895    | 0.7807    | 0.7807 | 0.0072 |

Both frameworks show the same direction of effect: the HP_ID=5 modified model (60 epochs) improves
mean test accuracy over the 30-epoch baseline and reduces seed-to-seed variance.

**Reproducibility note:** these 12 numbers are bit-identical between the original local run
(Windows 11, CPU-only, torch 2.13.0+cpu / tensorflow-cpu 2.21.0) and the Colab run (Ubuntu, Tesla
T4 runtime). Fixing the split with `random_state=SEED` and seeding each training run with
`torch.manual_seed` / `tf.keras.utils.set_random_seed` was sufficient for exact reproducibility
across both operating systems and hardware.

## 2. Loss curves (SEED=3215 run)

See `neural_networks.ipynb` Section 6 and `figures/loss_curves_pytorch.png` /
`figures/loss_curves_tensorflow.png`. Train and validation loss track closely throughout in both
frameworks - no divergence - indicating the modified model is still converging, not overfitting, at
60 epochs.

## 3. CUDA matrix multiplication - CPU vs. GPU timing

Hardware: **NVIDIA Tesla T4** (compute capability sm_75, 15360 MiB), driver 580.82.07, CUDA 13.0;
compiled with `nvcc` release 12.8 V12.8.93 at `-O3 -arch=sm_75`.
Kernel: tiled `matMulTiled`, TILE = 16 (16×16 = 256 threads per block).
Timing: `cudaEvent` timers, with one warm-up kernel launch excluded from the measurement.

| Matrix size | Grid × Block (total threads) | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | End-to-end GPU (ms) | Speedup (CPU / end-to-end) |
|---|---|---|---|---|---|---|
| 256  | (16,16) × (16,16) = 65,536       | 23.236   | 0.113   | 1.473  | 1.586   | **14.65×** |
| 1024 | (64,64) × (16,16) = 1,048,576    | 3663.505 | 5.808   | 5.034  | 10.842  | **337.90×** |
| 4096 | (256,256) × (16,16) = 16,777,216 | skipped¹ | 196.829 | 76.655 | 273.484 | ~857× ² |

¹ The naive single-threaded O(N³) CPU baseline is skipped at N=4096 in `matmul.cu`
(`ranCpu = N <= 2048`); at 4096 it would take roughly 234 s (≈64× the N=1024 time, since
4096³/1024³ = 64).
² Extrapolated from that estimate, **not measured** - reported as an estimate only.

Correctness was verified at every size against the CPU result: max absolute difference 2.29e-05
(N=256) and 9.16e-05 (N=1024), both printed as `(OK)`, consistent with float32 accumulation-order
differences rather than a logic error.

### Profiler output separating kernel time from transfer time

**Profiler used: `nvprof`** (legacy CUDA profiler). Nsight Systems (`nsys`) is **not installed** in
the current Colab image (`nsys: command not found`), so the notebook fell back to `nvprof`, which
ran successfully and produced the required kernel-vs-transfer breakdown. Nsight Compute (`ncu`)
also ran; its report is saved as `matmul_1024_ncu.ncu-rep`.

`nvprof ./matmul 1024`:

```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   78.80%  11.586ms         2  5.7929ms  5.7892ms  5.7965ms  matMulTiled(float const *, float const *, float*, int)
                   10.71%  1.5741ms         1  1.5741ms  1.5741ms  1.5741ms  [CUDA memcpy DtoH]
                   10.50%  1.5432ms         2  771.61us  740.21us  803.02us  [CUDA memcpy HtoD]
      API calls:   90.58%  182.89ms         3  60.962ms  59.553us  182.76ms  cudaMalloc
                    2.87%  5.8001ms         2  2.9000ms  2.7600us  5.7973ms  cudaEventSynchronize
                    2.87%  5.7913ms         1  5.7913ms  5.7913ms  5.7913ms  cudaDeviceSynchronize
                    2.40%  4.8534ms         3  1.6178ms  969.02us  2.9039ms  cudaMemcpy
                    0.79%  1.5986ms       114  14.022us      88ns  858.19us  cuDeviceGetAttribute
                    0.26%  517.73us         3  172.58us  115.72us  203.56us  cudaFree
                    0.12%  252.20us         2  126.10us  13.353us  238.84us  cudaLaunchKernel
```

This cleanly separates the two components: the kernel accounts for **78.80%** of GPU activity
(11.586 ms over 2 launches - the warm-up plus the timed launch, ≈5.79 ms each, matching the
5.808 ms `cudaEvent` measurement in the table), while the memory copies account for the remaining
**21.21%** (DtoH 1.5741 ms + HtoD 1.5432 ms = 3.117 ms of pure device-side copy time). Note that
`cudaMalloc` dominates the *API-call* column at 182.89 ms; that is one-time CUDA context and
allocation setup, not per-operation cost, and is excluded from the `cudaEvent` timings.

**Caveat on the `ncu` run:** Nsight Compute reports a kernel time of 1714.722 ms for the same
N=1024 kernel. That is *not* the real kernel time - `ncu` replays the kernel 9 times per launch to
collect hardware counters (`Profiling "matMulTiled" - 0: ... - 9 passes`), inflating the measurement
by roughly 300×. The un-instrumented `./matmul 1024` and `nvprof` figures (≈5.8 ms) are the correct
ones and are what the table reports.

## 4. Crossover discussion

The GPU is already the better choice at the smallest size measured: at **N = 256** the end-to-end
GPU path (1.586 ms, including both transfers) beats the 23.236 ms CPU baseline by 14.65×, so the
crossover lies somewhere below N = 256 in these measurements. The crossover is not at size zero
because the GPU path pays costs the CPU path does not: at N=256, 1.473 ms of H2D+D2H PCIe transfer
is **93% of the entire 1.586 ms end-to-end time**, while the kernel itself is only 0.113 ms - plus
per-launch overhead and one-time context/`cudaMalloc` setup that `nvprof` measures at 182.89 ms.
Those costs are fixed or grow only as O(N²) with data volume, whereas the compute they displace
grows as O(N³), so at very small N the transfer and launch overhead dominates and the GPU cannot
win. By N=1024 the arithmetic has grown 64× faster than the transfers and the speedup jumps from
14.65× to 337.90×. The gap between kernel-only and end-to-end speedup at N=256 (206× vs 14.65×) is
exactly this effect: the kernel is enormously faster, but at that size the result spends most of its
wall-clock time crossing the PCIe bus rather than being computed.
