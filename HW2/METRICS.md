# METRICS — HW2

Three independent parts. Parts 1 and 2 ran locally on CPU (Windows 11, Python 3.11.0);
Part 3 ran on Google Colab with a Tesla T4, since the techniques it measures are GPU memory
and throughput techniques that are meaningless on CPU. Global seed 42 everywhere.

---

## Part 1 — Word2Vec transfer learning on IMDB

Pretrained model: `word2vec-google-news-300` (3,000,000 words, 300 dims).
Corpus: IMDB 50,000 reviews (25k train + 25k test), 10,974,387 tokens after tokenization.
Fine-tuning: skip-gram, `window=5`, `min_count=5`, `negative=10`, `sample=1e-4`, 5 epochs.
Transfer step: 32,066 / 39,156 IMDB vocabulary words (81.9%) seeded from the pretrained
vectors; the remaining 7,090 IMDB-only words keep their random initialization.

### Top-3 nearest neighbours, before vs. after fine-tuning

| word   | pre top1         | post top1         | pre top2          | post top2          | pre top3           | post top3        |
|:-------|:-----------------|:------------------|:------------------|:-------------------|:-------------------|:-----------------|
| cast   | casts (0.722)    | actors (0.653)    | casting (0.719)   | supporting (0.645) | Cast (0.664)       | ensemble (0.593) |
| score  | scoring (0.720)  | morricone (0.675) | scores (0.660)    | music (0.671)      | scored (0.638)     | ennio (0.665)    |
| plot   | plots (0.762)    | storyline (0.750) | Plot (0.652)      | story (0.711)      | plotting (0.633)   | plotline (0.678) |
| screen | screens (0.773)  | screens (0.597)   | onscreen (0.612)  | onscreen (0.523)   | LCD_screen (0.560) | stage (0.471)    |
| review | reviewed (0.663) | comment (0.660)   | reviewing (0.661) | reviews (0.617)    | reviews (0.638)    | nixflix (0.581)  |

Every word except `screen` replaces its morphological neighbours (casts/casting,
scoring/scores/scored, plots/plotting) with semantic, movie-domain ones. `score` is the
clearest case: it moves off the sports-numbers sense entirely and lands next to *morricone*,
*music*, *ennio* — the film-composer sense.

### Vector shift (original vs. fine-tuned, same word)

| word   |   cos(original, fine-tuned) |   shift (1 - cos) |   L2 distance |
|:-------|----------------------------:|------------------:|--------------:|
| score  |                      0.5299 |            0.4701 |        3.6714 |
| review |                      0.5384 |            0.4616 |        3.4615 |
| plot   |                      0.5929 |            0.4071 |        2.8017 |
| cast   |                      0.6138 |            0.3862 |        2.7176 |
| screen |                      0.6497 |            0.3503 |        2.6774 |

- **Most shifted: `score`** (cos = 0.5299) — the news sense (sports scoring) and the movie
  sense (soundtrack) are almost unrelated meanings, so the vector has the furthest to travel.
- **Least shifted: `screen`** (cos = 0.6497) — it keeps `screens` and `onscreen` as its top-2
  in both spaces. The news sense (a physical display) and the movie sense (where a film is
  shown) are close enough that fine-tuning mostly just weakens the similarities rather than
  replacing the neighbours. Note the cosines drop sharply (0.773 → 0.597) even though the
  ranking holds — the word ends up in a less crowded, more diffuse neighbourhood after training.

Timings: pretrained model load 45.9s, tokenization 25.0s, 5 fine-tuning epochs 174.7s.
Figures in `Part1_Word2Vec_IMDB.ipynb` (t-SNE sections); artifacts in `results/`
(`neighbors_before_after.csv`, `neighbors_long.csv`, `vector_shift.csv`,
`imdb_finetuned_w2v.kv`).

---

## Part 2 — RAG pipeline over movie Wikipedia pages

Corpus: 10 movie Wikipedia pages via `WikipediaLoader`, 20,000 chars each.
Splitter: `RecursiveCharacterTextSplitter`, 500 chars / 50 overlap → **601 chunks**.
Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Vector store: Chroma (601 vectors).
LLM: `google/flan-t5-base`, local CPU. Retriever: top-k = 3.

### Retrieval evaluation (5 questions)

| question                                                              | ground truth movie       | top-3 contains answer | rank of first relevant chunk |
|:----------------------------------------------------------------------|:-------------------------|:----------------------|:-----------------------------|
| What crime is Andy Dufresne convicted of in The Shawshank Redemption? | The Shawshank Redemption | Yes                   | 1                            |
| What is the name of the spinning top object used in Inception?        | Inception (film)         | Yes                   | 1                            |
| Who directed The Godfather?                                           | The Godfather            | Yes                   | 1                            |
| In Pulp Fiction, what do Vincent and Jules do for a living?           | Pulp Fiction             | No                    | -                            |
| Who plays the Joker in The Dark Knight?                               | The Dark Knight          | Yes                   | 1                            |

**Retrieval Success Rate = 4/5 = 80%.**

The Yes/No column started as a keyword match (does the chunk contain a hint word like
"murder" or "coppola"), but that is a weak test — a hint word can appear in an unrelated
sentence — so every retrieved chunk was read by hand against the ground-truth passage before
the column was trusted.

### Generated answers (500/50)

| question | answer |
|:---|:---|
| What crime is Andy Dufresne convicted of in The Shawshank Redemption? | murdering his wife and her lover |
| What is the name of the spinning top object used in Inception? | Cobb |
| Who directed The Godfather? | Francis Ford Coppola |
| In Pulp Fiction, what do Vincent and Jules do for a living? | crime |
| Who plays the Joker in The Dark Knight? | Ledger |

Retrieval at 80% overstates end-to-end quality. Only 2 of the 5 answers are fully correct.
"Cobb" names the character rather than the object, "crime" is vague where "hitmen" was
wanted, and "Ledger" is a surname-only answer drawn from a correctly retrieved chunk — all
generation-side shortfalls sitting on top of retrieval that mostly worked.

### Chunk-size comparison — 500/50 vs. 1000/100

Re-split at 1000 chars / 100 overlap → **330 chunks** (down from 601), first two questions rerun.

| question | answer (500/50) | answer (1000/100) | top-1 movie (both) | top-1 chars 500/50 | top-1 chars 1000/100 |
|:---|:---|:---|:---|---:|---:|
| What crime is Andy Dufresne convicted of...? | murdering his wife and her lover | murdering his wife and her lover | The Shawshank Redemption | 443 | 443 |
| What is the name of the spinning top object used in Inception? | Cobb | I don't know | Inception (film) | 383 | 550 |

Bigger chunks changed nothing for Shawshank — the whole fact sits inside one 500-char chunk
either way, and the top-1 chunk came back byte-identical at 443 chars. For Inception it got
*worse*: the larger chunks pulled a different section of the page (box-office gross) into the
top slot and the model fell back to "I don't know". Larger chunks are not uniformly better;
they change which chunk wins the similarity contest, and that can go either direction.

### Two failure cases

**1. Chunk boundary separated important information — the Inception "totem" question.**
At 500/50 the top-1 chunk is a generic film-intro paragraph, while the sentence that actually
answers the question ("Cobb uses a top that spins indefinitely in a dream to verify that he
is in the real world") sits at rank 2. The model leaned on the top chunk and answered "Cobb",
the character rather than the object. At 1000/100 the relevant sentence dropped out of the
top slot altogether and the answer degraded to "I don't know" — one failure mode traded for
another, not fixed. The fact is present in the corpus at both chunk sizes and never lands
where the model relies on it most.

**2. Relevant chunk ranked too low — the Pulp Fiction occupation question.**
All three retrieved chunks are correctly from Pulp Fiction's Plot section, so retrieval found
the right *document*. But none of them state the job title: rank 1 shows Jules and Vincent at
a bar with Marsellus, rank 2 mentions Jules' "life of crime", rank 3 is an unrelated shooting
scene. The sentence naming them as hitmen sits in a chunk that missed the top-3 cut. Given
"life of crime" as its best clue, the model answered "crime". This is the one question scored
as a retrieval miss, and it shows that "right document" and "right chunk containing the
specific fact" are different bars — similarity over short chunks can favour generic
contextual overlap over the single sentence that answers the question.

Artifacts in `results/`: `answers_v1.csv`, `retrieved_chunks_v1.csv`,
`retrieval_evaluation.csv`, `chunking_comparison.csv`, `rag_failures.csv`, `part2_summary.md`.

---

## Part 3 — Training-time optimization techniques

Hardware: Google Colab, **Tesla T4**. Every experiment below uses the identical model, data,
batch size, and step count; only the flag under test changes.

Shared config: MLP with 8 layers × 2048 hidden units, in_dim 1024, batch 256, 200 steps,
SGD with momentum 0.9, synthetic data, `torch.manual_seed(0)` reset before each run.
Fixed memory floor: 120.1 MB of parameters, rising to 360.4 MB once gradients and momentum
buffers are counted. Checkpointing can never reclaim any of that floor, which matters in §3.

### Summary table (as written to `results/optimization_techniques_summary.csv`)

| technique                | variant                     | time (s) | peak GPU mem (MB) | final loss |
|:-------------------------|:-----------------------------|---------:|------------------:|-----------:|
| Tensor creation          | CPU                          |  117.155 |               n/a |     2.3028 |
| Tensor creation          | GPU                          |    3.597 |             380.7 |     2.3033 |
| Weight init              | default (PyTorch)            |    3.055 |             500.8 |     2.3033 |
| Weight init              | Xavier uniform               |    3.098 |             500.8 |     2.2998 |
| Weight init              | large-normal (std=1.0, bad)  |    2.667 |             500.8 |        NaN |
| Activation checkpointing | off (bs=16384)               |   21.582 |            1843.6 |     2.3026 |
| Activation checkpointing | on (2 segments, bs=16384)    |   31.058 |            1459.6 |     2.3026 |
| Gradient accumulation    | none (bs=256)                |    4.017 |             622.0 |     2.3033 |
| Gradient accumulation    | 4× accum (micro_bs=64)       |    5.519 |             880.0 |     2.3025 |
| Mixed precision          | fp32                         |    3.976 |            1102.5 |     2.3033 |
| Mixed precision          | fp16 (autocast + GradScaler) |    2.174 |            1224.7 |     2.3033 |

### 1. Tensor creation — CPU vs. GPU

**117.16s → 3.60s, a 32.6× speedup**, loss unchanged (2.3028 vs. 2.3033). The model is deep
and wide enough (8 × 2048, batch 256) to keep the GPU occupied. On a much smaller model the
gap would narrow, because kernel-launch and host-to-device overhead is roughly fixed and
starts to dominate as the actual work shrinks.

### 2. Weight initialization

| scheme | time (s) | final loss | mean of last 10 steps |
|:---|---:|---:|---:|
| default (PyTorch) | 3.055 | 2.3033 | 2.3025 |
| Xavier uniform | 3.098 | 2.2998 | 2.3028 |
| large-normal (std=1.0) | 2.667 | NaN | NaN |

Default and Xavier are indistinguishable here — both sit near 2.30 for the whole run, which
is expected: 200 steps at this learning rate on random synthetic data is not enough for real
learning, and the labels carry no signal to learn anyway. The bad init is the informative one.
With std=1.0 through 8 layers the activations blow up on the very first forward pass,
gradients follow, and the loss is NaN almost immediately with no recovery. That is precisely
the failure mode Xavier/Kaiming initialization exists to prevent — holding activation scale
roughly constant as depth grows. Worth noting that it also finished *fastest* (2.667s): NaN
arithmetic is not slower, so wall-clock time alone would have crowned this run the winner.

### 3. Activation checkpointing

Measured as a sweep, with the no-checkpointing baseline re-measured at each batch size:

| batch | config | peak mem (MB) | time (s) | mem saved | time cost |
|------:|:---|---:|---:|---:|---:|
|  4096 | no checkpointing        |  838.1 |  5.28 | — | — |
|  4096 | checkpoint, 2 segments  |  775.1 |  7.21 |  7.5% | +36.5% |
|  4096 | checkpoint, 4 segments  |  743.1 |  7.35 | 11.3% | +39.1% |
| 16384 | no checkpointing        | 1843.6 | 21.58 | — | — |
| 16384 | checkpoint, 2 segments  | 1459.6 | 31.06 | 20.8% | +43.9% |
| 16384 | checkpoint, 4 segments  | 1459.6 | 34.31 | 20.8% | +59.0% |

**Best result: batch 16384, 2 segments — 20.8% less memory for 43.9% more time.** The loss is
identical to four decimals with and without checkpointing (2.3026), which is the correctness
check that matters: recomputing activations during backward is mathematically a no-op, so any
loss change would mean gradients were being broken rather than recomputed.

The sweep is what makes the mechanism legible. At batch 4096 activations are about 57% of
peak and checkpointing saves 7.5%; at 16384 they are about 80% of peak and it saves 20.8%.
The 360.4 MB floor of weights, gradients, and momentum is untouchable either way, so the
technique's value scales directly with how much of peak memory is activations — which is why
it is a large-batch/large-model tool and does essentially nothing at batch 256. Going from 2
to 4 segments at batch 16384 bought no extra memory at all but cost another 15 points of
time; more segments is not automatically better.

### 4. Gradient accumulation

4 micro-batches of 64 vs. one batch of 256 — identical examples per optimizer step.

Loss lands in the same place (2.3025 vs. 2.3033), the expected result: both versions see the
same 256 examples before each update, whether in one chunk or four. Time went up as expected
(5.52s vs. 4.02s), from four separate forward/backward passes plus Python loop overhead
instead of one.

Peak memory, however, came out **higher** for the accumulated run (880.0 MB vs. 622.0 MB),
which is the opposite of the technique's whole purpose. Reported as measured rather than
smoothed over: at this size, holding a 64-example batch instead of a 256-example one saves
only a few MB of activations, so the number is dominated by PyTorch's caching allocator —
`max_memory_allocated` reflects fragmentation and block reuse across the extra
forward/backward/allocation cycles, not the intrinsic footprint of the technique. The
memory-saving claim for gradient accumulation holds in the regime where a single batch's
activations genuinely do not fit; this model at batch 256 is not that regime, and the
measurement says so.

### 5. Mixed precision

**3.98s → 2.17s, a 1.83× speedup**, loss identical (2.3033 both). A solid result for the T4's
fp16 tensor cores.

Peak memory did **not** improve — it rose 11.1% (1102.5 MB → 1224.7 MB). Autocast keeps a
float32 master copy of the weights and `GradScaler` adds state of its own, so at this model
size that fp32 overhead outweighs the savings from fp16 activations. On larger matmul-heavy
models (transformers, large CNNs) the activation savings dominate and mixed precision wins on
both axes; here only the speed axis pays off.

### What this all says together

The two techniques that were unambiguous wins — GPU tensors (32.6×) and mixed precision
(1.83×) — both buy speed, and both were free in terms of loss. The two memory techniques
(checkpointing, accumulation) only pay off in the regime where activations dominate peak
memory, and at the shared 256-batch configuration neither does: checkpointing had to be moved
to batch 16384 before its 20.8% saving appeared, and accumulation's memory number came out
backwards. Weight init is the odd one out — it costs nothing either way, and the only thing
separating the schemes here is that a bad one destroys training outright.

Figures in `Part3_Training_Optimizations.ipynb`; summary artifact
`results/optimization_techniques_summary.csv`.
