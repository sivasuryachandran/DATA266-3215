# DATA 266 - Homework 2

**Siva Surya Chandran**
SJSU ID: 019130215
Email: sivasurya.chandran@sjsu.edu
San José State University - DATA 266, Fall 2026

---

## What's in here

Three separate parts, one notebook each.

| Part | Notebook | Topic |
|:--|:--|:--|
| 1 | `Part1_Word2Vec_IMDB.ipynb` | Word2Vec transfer learning - fine-tuning Google News vectors on IMDB reviews |
| 2 | `Part2_RAG_LangChain.ipynb` | RAG pipeline over movie Wikipedia pages using LangChain + Chroma |
| 3 | `Part3_Training_Optimizations.ipynb` | Five training-time optimization techniques, measured on a GPU |

Supporting files:

- `METRICS.md` - all numbers from all three parts, with the discussion for each
- `RUN_LOG.txt` - console transcripts of the actual runs, plus environment details
- `AI_USE.md` - AI-use appendix (what I used an assistant for, and a specific thing it got wrong)
- `results/` - CSVs and saved artifacts written by the notebooks
- `HW2.pdf` - assignment handout

---

## Part 1 - Word2Vec on IMDB

Takes Google's pretrained `word2vec-google-news-300` vectors and keeps training them on the
50,000 IMDB movie reviews, then measures how far five words moved: `cast`, `score`, `plot`,
`screen`, `review`.

The transfer step seeds 32,066 of the 39,156 IMDB vocabulary words (81.9%) with their
pretrained vectors before fine-tuning; IMDB-only words start random.

**Result:** `score` moved the most (cosine 0.5299 between its original and fine-tuned vector).
In the news-trained space its nearest neighbours are *scoring*, *scores*, *scored* - the
sports sense. After fine-tuning they're *morricone*, *music*, *ennio* - the film-composer
sense. `screen` moved the least (0.6497), since a physical display and a cinema screen aren't
far apart to begin with.

## Part 2 - RAG over movie Wikipedia pages

10 movie Wikipedia pages → 601 chunks (500 chars, 50 overlap) → MiniLM embeddings → Chroma →
`flan-t5-base` running locally. The chain is wired by hand (retriever → prompt → LLM → output
parser) rather than using a prebuilt LangChain class, so each stage is inspectable.

**Retrieval Success Rate: 4/5 = 80%.** Though only 2 of the 5 generated answers are fully
correct - retrieval finding the right chunk and the model producing the right answer are
different things, and the gap between those two numbers is most of what Part 2 is about.

Also reruns two questions at 1000/100 chunking. Bigger chunks didn't help: one question was
unchanged, the other got *worse* (a correct-ish answer became "I don't know").

## Part 3 - Training optimizations

Five techniques, each measured with everything else held fixed - same model, data, batch size,
step count, seed. Run on a Colab Tesla T4.

| Technique | Result |
|:--|:--|
| CPU vs GPU tensors | **32.6× faster** on GPU (117.2s → 3.6s) |
| Weight initialization | Xavier ≈ default; std=1.0 init → NaN within a few steps |
| Activation checkpointing | **20.8% less memory for 43.9% more time** (at batch 16384) |
| Gradient accumulation | Same loss, more time, and memory went *up* - see note below |
| Mixed precision | **1.83× faster** (4.0s → 2.2s), memory up 11.1% |

Two results came out against expectation and are reported as measured rather than smoothed
over: gradient accumulation used *more* peak memory than the baseline, and mixed precision
*increased* memory while still delivering its speedup. Both are explained in `METRICS.md` -
short version is that this model is too small for either technique's memory savings to beat
the fixed overhead they add.

Activation checkpointing also needed the batch size raised to 16384 before it did anything
useful. At the default batch of 256 the model's fixed 360 MB of weights/gradients/momentum is
nearly the entire memory footprint, leaving almost no activation memory to reclaim.

---

## Running it

Parts 1 and 2 run on CPU. Part 3 needs a CUDA GPU - it measures GPU memory and fp16 tensor
cores, so there's nothing to see on CPU.

```
pip install gensim datasets numpy pandas matplotlib scikit-learn
pip install langchain langchain-community langchain-huggingface langchain-chroma \
            wikipedia chromadb sentence-transformers transformers accelerate
pip install torch
```

Then run the notebooks top to bottom. Part 1 downloads the ~1.6GB Google News vectors on
first run, so give it a few minutes.

**Environments used for the committed results:**

- Parts 1 & 2 - local, Windows 11, Python 3.11.0, CPU only
- Part 3 - Google Colab, Tesla T4

Seed is 42 throughout. Exact package versions are recorded at the top of `RUN_LOG.txt`.
