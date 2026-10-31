# GPU Reservation and GPU-Hour Record — HW4.5

> Reservation details are from the lab's approval email (credentials deliberately not recorded here).
> GPU-time figures are estimates (not metered). UUID screenshots: `uuid_captures_4090_and_T4.png` and
> `uuid_capture_colab_T4_cell1.png`.

## Reservation

| Field | Value |
|---|---|
| Provider / how obtained | SJSU HPC GPU lab (instructor-approved reservation request; approval email dated Thu Oct 1, from the lab staff) |
| Location / machine | Room ISB 838, Machine No. 17 (final allocation; check-in at ISB 836) |
| Reservation window | Start Thu Oct 30 2026, 12:00 pm; end 11:59 AM after 1 day (i.e. Fri Oct 31 2026); result collection 12:00–12:30 PM on the end date |
| Reservation ID | None issued; the approval email is the reservation record (keep it with the submission) |
| GPU | 1 × NVIDIA GeForce RTX 4090 |
| VRAM | 24 GB (24564 MiB) |
| Host OS | Windows 11 Pro 10.0.26200 (WDDM driver) |
| GPU UUID | `GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39` |
| Serving GPU (Part D) | Tesla T4, Google Colab free tier, UUID `GPU-6cb18f34-3c2f-33e8-fdfb-31b18ae8a8df` (1 × 15 GB) |
| Second GPU / Part C | None — Part C skipped (optional) |

## GPU provenance

| Field | Value |
|---|---|
| GPU name | NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39` |
| Driver version | 610.60 |
| CUDA reported by driver | 13.3 |
| PyTorch | 2.6.0+cu124 |
| VRAM capacity | 24564 MiB |
| Reported power limit | 450 W |

The captures are `nvidia-smi-q_GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39.txt` and `nvidia-smi-L_RTX4090.txt`
(4090), and `nvidia-smi-L_ColabT4.txt` (T4; take a screenshot of the Colab cell 1 output as well).

## Benchmark usage

Runs were on 2026-10-30 (local time). Per-run timings are in `part_b_full_runs.csv`.

| Activity | GPU time (approx.) |
|---|---|
| HW4 baseline + five technique fine-tunes (3 epochs each) | 269 s of training in total (49.2 + 22.9 + 64.1 + 50.9 + 24.1 + 57.8) plus model loads |
| 34 short batch-scaling benchmark runs (23 steps each) | roughly 10–15 minutes including process and model start-up (estimate) |
| One stalled run (WDDM memory spill, discarded) | about 10 minutes (estimate) |
| **Total, RTX 4090** | **about 0.4 GPU-hours (estimate, not metered)** |
| Colab T4 session (Part D) | about 1 hour (free-tier T4, estimate) |

## GPU-hour estimate for one full fine-tune

Samples processed = 0.70 × 162,969 rows × 3 epochs = 342,234. Measured full-run rates (training plus
per-epoch validation): baseline fp32 213.3 samples/s, AMP 458.2 samples/s.

- Baseline fp32, bs 32: 342,234 / 213.3 = 1,604 s = **0.446 GPU-h**
- Mixed precision, bs 32: 342,234 / 458.2 = 747 s = **0.207 GPU-h**
