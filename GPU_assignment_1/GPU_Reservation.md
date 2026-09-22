# GPU Reservation Record - HW2.5

This assignment needed an RTX-series GPU, and my Mac doesn't have one. The TA (Shriansh Chari)
posted an update saying we could use any NVIDIA RTX GPU this round instead of just the campus lab,
since the lab was going to be short on capacity because of the Edge AI Hackathon - so I rented a
pod instead of waiting on lab time.

## Provider and instance

| Field | Value |
|---|---|
| Provider | Vast.ai |
| Listing | Type #49545596 - Thailand, TH |
| GPU | 1x RTX 4090, 24 GB VRAM |
| Host machine | 440BX Desktop Reference Platform, AMD EPYC 7C13, PCIe 4.0 x16 |
| Instance ID | 52008951 (later reconnected as 52010579 after an IP change) |
| Price | $0.260/hr plus bandwidth |
| Reliability (listed) | 99.79% |

## GPU actually used (from the run itself)

| Field | Value |
|---|---|
| GPU name | NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-f7f152a8-7ef2-fa17-4745-38a374331d8c` |
| Driver version | 570.86.16 |
| CUDA (driver-reported) | 12.8 |
| PyTorch build used | 2.6.0+cu124 (had to reinstall - the default `pip install torch` grabbed a cu130 build that the driver couldn't run) |

## Timing (from RUN_LOG.txt)

| Event | Timestamp (UTC) |
|---|---|
| Notebook run started (Part A) | 2026-09-22 05:53:40 |
| Part E (20-min sustained load) started | 2026-09-22 06:27:18 |
| Part E finished | 2026-09-22 06:47:23 |
| Full notebook run finished | 2026-09-22 06:47:23 |

Measured run time (Part A through the end of Part E): **~54 minutes** of actual GPU compute.

## GPU-hours billed - TODO, fill in from the Vast.ai billing page

The timestamps above only cover the notebook's own execution, not the whole time the pod was
rented (I also spent time before/after that debugging the CUDA driver mismatch, setting up SSH
keys, and copying files back down). Need to check the **Vast.ai billing / instance history page**
for the actual instance start and stop time, then fill in below:

| Field | Value |
|---|---|
| Pod rented at (UTC) | _(fill in)_ |
| Pod terminated at (UTC) | _(fill in)_ |
| Total wall-clock time rented | _(fill in)_ |
| GPU-hours billed | _(fill in)_ |
| Total cost | _(fill in, should be close to hours x $0.260/hr plus any bandwidth charges)_ |

Once filled in, this table plus `RUN_LOG.txt` and `METRICS.md` (all tagged with the same GPU UUID
above) is the full provenance trail for this assignment.
