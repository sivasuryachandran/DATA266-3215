# HW11 metrics (GPU: Tesla T4; SID4=3215, SEED=3215)

Part 1 model: `facebook/opt-125m` revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`; GPTQ 4-bit, group 128, 128 Wikitext2-train calibration windows. Quality metric = Wikitext2-test perplexity. Latency = 64 new tokens, batch 1 (generation) and 512-token forward (prefill).

**Table HW11.1 - GPTQ comparison**

| Configuration | Wikitext2 perplexity | p50 / p95 latency, generate-64 (ms) | p50 / p95 prefill-512 (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---:|---:|---:|---:|---:|
| Base LLM (fp16) | 27.661 | 593.2 / 815.1 | 11.4 / 14.3 | 1.366 | 250.5 |
| GPTQ 4-bit LLM | 31.183 | 1156.3 / 1542.9 | 20.8 / 25.4 | 1.234 | 129.3 |

**Table HW11.2 - distillation comparison (AG News test, 7,600 samples)**

| Configuration | Accuracy | p50 / p95 latency, batch 1 (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---:|---:|---:|---:|
| Teacher (BERT-base) | 0.9443 | 8.63 / 10.42 | 1.432 | 438.0 |
| Best distilled student (T=8, alpha=0.3) | 0.9445 | 4.42 / 5.28 | 1.700 | 268.6 |

Temperature sweep (alpha=0.5); selection by validation accuracy on 5,000 held-out train samples:

| T | val acc | test acc |
|---:|---:|---:|
| 1 | 0.9434 | 0.9457 |
| 2 | 0.9442 | 0.9454 |
| 4 | 0.9446 | 0.9470 |
| 8 | 0.9454 | 0.9461 |

Loss-weighting sweep (T=8; alpha = weight on cross-entropy):

| alpha | val acc | test acc |
|---:|---:|---:|
| 0.3 | 0.9472 | 0.9445 |
| 0.5 | 0.9454 | 0.9461 |
| 0.7 | 0.9462 | 0.9464 |

Per-class test accuracy (teacher / student):

| Class | Teacher | Student |
|---|---:|---:|
| World | 0.9489 | 0.9526 |
| Sports | 0.9884 | 0.9879 |
| Business | 0.9137 | 0.9111 |
| Sci/Tech | 0.9263 | 0.9263 |

## Measurement notes (read before comparing rows)

- **Student peak memory in Table HW11.2 is inflated.** The teacher was still on the GPU when the student was benchmarked, so the student's 1.700 GB includes the teacher's ~0.44 GB fp32 weights. `reset_peak_memory_stats()` resets the peak but not what is already allocated. A rough corrected figure is about 1.26 GB, but that is an estimate, not a measurement; the notebook was not re-run.
- All Table HW11.2 numbers use fp32 inference on the same T4, batch size 128 for accuracy/memory and batch size 1 (200 samples, 20 warm-up, p50/p95) for latency.
- Table HW11.1 latency: 64 new tokens, batch 1, greedy (50 runs, 5 warm-up); prefill: one 512-token forward pass (100 runs, 10 warm-up).
- AG News test has 7,600 samples, so one standard error of an accuracy near 0.944 is about 0.26 points. All test-accuracy differences in the sweeps (0.9445 to 0.9470) are smaller than that.
- Sweep selection used validation accuracy (5,000 held-out train samples): T=8 won the temperature sweep and alpha=0.3 won the weighting sweep, even though their test accuracies are not the highest in each sweep (T=4: 0.9470; alpha=0.7: 0.9464).
- Sweep values (T in {1,2,4,8}, alpha in {0.3,0.5,0.7}, alpha=0.5 during the T sweep) were my assumption because the assignment text lost its math symbols; alpha is the weight on cross-entropy, and the KL term is scaled by T^2.
- Training settings fixed for all 7 students: 2 epochs, batch 64, lr 5e-5, max length 128, fp16 autocast, 115,000 training samples, SEED=3215.
