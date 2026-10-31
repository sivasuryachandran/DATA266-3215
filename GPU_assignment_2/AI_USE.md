# AI-Use Appendix - Sivasurya Chandran (HW4.5 / GPU Assignment II)

## 1. Which parts did you use an assistant for, and which did you write yourself?

I did the experiment setup, configuration, execution, monitoring, and result validation for this assignment. I selected `Twitter_Data.csv`, used my assigned `SEED=3215` and `SLICE=215`, set up Python and the required environment, downloaded and prepared the data, ran the GPU experiments, monitored GPU utilization and memory behavior, ran the Part D notebook in Colab, and wrote the final prompt used for the vLLM benchmark.

I used Claude Code mainly to help implement and iterate on the experiment scripts from my instructions and the assignment requirements. This included generating initial versions of `hw45.py`, `sweep.py`, `analyze.py`, `infer.py`, and the Part D Colab notebook, as well as helping format drafts of `METRICS.md`, `GPU_Reservation.md`, and this appendix. The BERT model, preprocessing, and original hyperparameter setup came from my HW4.a Kaggle notebook (`praveengovi/bert-twitter-sentiment-classifier`) and were adapted for the HW4.5 experiments.

I was responsible for actually running the experiments and checking whether the resulting behavior made sense. This mattered because several initial generated implementations did not work correctly on my hardware, and I had to use the observed GPU behavior, logs, and benchmark results to identify those problems and rerun the affected experiments.

For Part D, the notebook initially contained an AI-drafted placeholder for `MY_PROMPT`, but I replaced it with my own prompt before running the benchmark. The prompt and its generated output are saved in `partD_results.json`.

I did not apply `HP_ID`, `CLS_A`, or `CLS_B` from the parameter table because I did not understand what those fields represented. Claude also helped draft the one-line diagnoses for the five misclassified examples in `METRICS.md`.

## 2. Give one specific thing it produced that was wrong. Paste the wrong output.

One important issue was the first Part A batch-size sweep. The generated implementation did not account for the memory behavior of my Windows GPU setup. It produced:

```text
baseline 64 ok 246.31808696687875 11.119388103485107
baseline 128 ok 106.0403946516284 20.875064849853516
(batch 256 never returned; nvidia-smi: memory.used 24064 MiB, utilization.gpu 100 %)
```

This result was suspicious because throughput dropped from about 246 samples/s at batch 64 to only 106 samples/s at batch 128. Batch 256 then stayed running instead of completing or producing the expected CUDA out-of-memory error.

There were also two implementation bugs during the experiments. The first sweep initially failed because the cached dataset was loaded using the default `torch.load` behavior and produced a `weights_only` unpickling error. Gradient checkpointing at batch 256 was also initially reported as a crash, but the actual exception was `StopIteration` because the benchmark exhausted the available batches.

## 3. How did you find out? What did the failure look like?

I caught the main problem by comparing the measured throughput, GPU memory, and GPU utilization rather than accepting the benchmark output directly.

At batch 64 I measured about 246 samples/s, while batch 128 suddenly dropped to about 106 samples/s even though GPU memory increased from roughly 11.1 GB to 20.9 GB. When I tried batch 256, the process remained at 100% GPU utilization with approximately 24 GB of GPU memory in use but never returned a benchmark result.

That did not look like the expected memory-wall behavior. Instead of receiving an OOM and stopping the batch-size sweep, the system became extremely slow. The behavior was consistent with Windows/WDDM allowing GPU-memory pressure to spill into system memory rather than immediately producing the CUDA OOM that the experiment expected.

I diagnosed the other failures from `RUN_LOG.txt`. The first was an `UnpicklingError` related to the `weights_only` behavior of `torch.load`. The checkpointing failure was a `StopIteration` at `x, m, y = next(it)`, showing that it was a benchmark-loop problem rather than a GPU-memory failure.

## 4. What did you change, and why does your version work?

After identifying that the sweep was not reaching a clean CUDA memory wall, I changed the experiment so the PyTorch process was limited to 93% of available GPU memory using:

```python
torch.cuda.set_per_process_memory_fraction(0.93)
```

This forced PyTorch to raise an actual `OutOfMemoryError` before Windows could turn the experiment into an extremely slow memory-spill case.

After making that change and rerunning the experiment, batch 128 achieved about 256.5 samples/s, which was consistent with the throughput plateau around batch 64. Batch 256 then produced a clean OOM. This gave me a reproducible baseline maximum batch size of 128.

I also changed the cache loading to:

```python
torch.load(..., weights_only=False)
```

because the cache was created by my own experiment and needed to load the complete stored objects.

For the checkpointing benchmark, I changed the benchmark sampling so it could draw batches with replacement rather than exhausting the 3,500-example training subset. I then reran the affected checkpointing sweep starting at batch 256. It successfully reached batch 512 and produced an OOM at batch 1024.

The 93% process-memory cap means the reported maximum batch sizes are intentionally a little conservative, but it gave me a consistent and measurable CUDA memory wall instead of Windows silently spilling memory and corrupting the throughput comparison.