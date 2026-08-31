# AI-Use Appendix — Sivasurya Chandran

## 1. Which parts did you use an assistant for, and which did you write yourself?

I wrote the majority of the code myself. This includes the `neural_networks.ipynb` pipeline (data
loading, the 70/15/15 split, the PyTorch `TorchFFN` model and training loop, the parallel
TensorFlow `Sequential` model, and the multi-seed measurement loop), the CUDA `matmul.cu` tiled
kernel with its CPU baseline and `cudaEvent`-based timing harness, and the written answers in
`cuda.ipynb`. I derived my personal parameters from my SID4 (3215; my SJSU ID is 019130215, and its
last four digits `0215` have a leading zero that collapses to 3 digits, so I used 3215) — SEED, SLICE, HP_ID, CLS_A,
CLS_B — and decided to use the pre-scaled `diabetes.csv` supplied for this course as-is rather than
substituting the raw Pima dataset.

I used Google Gemini for two things. First, debugging: I ran everything in Google Colab because my
laptop has no NVIDIA GPU (Intel UHD 620 integrated graphics only), and when cells failed there I
pasted the tracebacks into Gemini to help interpret the errors and suggest fixes. Second, some
routine boilerplate: parts of the matplotlib plotting code for the correlation matrix, feature
distributions, and loss curves, and a few setup/scaffolding cells. I reviewed and edited everything
it produced, and I checked every generated figure and accuracy table against the printed console
output before accepting it.

## 2. Give one specific thing it produced that was wrong. Paste the wrong output.

The profiling cell gave me a kernel time that was way off - about 300x slower than it should be:

```
==PROF== Profiling "matMulTiled" - 0: 0%....50%....100% - 9 passes
GPU kernel time:       1714.722 ms
Speedup (CPU / GPU kernel only):       2.16x
```

The same kernel at the same size had just run in 5.808 ms without the profiler, which came out to
630.81x.

## 3. How did you find out? What did the failure look like?

The 2.16x number looked wrong straight away. A tiled GPU matmul should beat a plain single-threaded
CPU loop by far more than 2x at N=1024, and I had just seen it do 630x a minute earlier. The answer
was sitting in the output itself: `9 passes`. Nsight Compute runs the kernel nine times over to
collect its counters, so what it reports is mostly profiler overhead. My `cudaEvent` timer wasn't
broken - it was correctly timing a run that happened to be nine times longer than a normal one.

## 4. What did you change, and why does your version work?

Nothing in the code needed fixing. I just used the right numbers: the timing table uses the
`cudaEvent` results from the normal run, not the profiled one. I checked those against `nvprof`,
which measured the kernel at 5.7929 ms on its own and showed the split between compute and memory
transfer (78.80% kernel, 21.21% transfer). Getting ~5.8 ms from two different tools is what
convinced me the 1714 ms was just overhead. I kept the `ncu` output in my notes with a comment
explaining why it's high, since someone re-running it would hit the same thing and wonder.

I checked correctness separately. `matmul.cu` compares the GPU result against the CPU one and
printed `(OK)` at both sizes, with a max difference of 2.29e-05 at N=256 and 9.16e-05 at N=1024 -
small enough to just be float rounding.
