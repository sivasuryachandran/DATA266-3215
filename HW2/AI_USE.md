# AI-Use Appendix — Sivasurya Chandran

## 1. Which parts did you use an assistant for, and which did you write yourself?

I wrote the majority of the code myself. That includes the Part 1 Word2Vec pipeline
(`Part1_Word2Vec_IMDB.ipynb`) — the IMDB tokenizer that strips the literal `<br />` tags,
the transfer step that copies Google News vectors into a fresh skip-gram model before
training, and the cosine/L2 vector-shift measurement — the Part 2 RAG pipeline
(`Part2_RAG_LangChain.ipynb`), which I deliberately wired by hand (retriever →
`PromptTemplate` → LLM → `StrOutputParser`) instead of using one of LangChain's prebuilt
chain classes so every stage stayed inspectable, and the Part 3 experiment harness
(`Part3_Training_Optimizations.ipynb`) — the shared `train_loop` that every technique reuses
so only the one flag under test changes, the segmented-forward implementation for activation
checkpointing, and the gradient-accumulation loop. I also chose the setup decisions myself:
holding batch size, step count, model shape, and seed fixed across every Part 3 experiment;
running Parts 1 and 2 locally on CPU and Part 3 on a Colab T4 because the techniques being
measured are GPU memory/throughput techniques and mean nothing on CPU; and picking the five
Part 1 target words (`cast`, `score`, `plot`, `screen`, `review`) because each has a distinct
general-news sense that movie-review text should visibly pull away from.

I used an AI assistant for two things. First, debugging: when cells failed I pasted the
tracebacks in to help interpret them and suggest fixes — the `datasets` library refusing the
legacy `imdb` loader script, the `wikipedia` package's 403, and the Part 3 checkpointing
section not showing any memory benefit at first. Second, routine boilerplate: parts of the
matplotlib code for the t-SNE before/after plot, the loss-curve panels, and the summary bar
charts, plus some scaffolding cells. I reviewed and edited everything it produced, and I
checked every generated figure and table against the printed console output before accepting
it.

## 2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first version of the activation-checkpointing section ran checkpointing at the shared
batch size of 256 and reported that the technique saved essentially nothing — the assistant's
draft commentary explained this away as checkpointing "not being worth it for MLPs." The
numbers it was reasoning from looked like this:

```
model parameters alone :    120.1 MB
+ gradients + momentum :    360.4 MB

batch=   256  no checkpointing :    380.7 MB    3.42s
batch=   256  checkpoint, 2 segments:    379.9 MB    4.88s   memory +0.2%  time +42.7%
```

A 0.2% memory saving for 43% more time is the technique looking useless. The conclusion drawn
from it — that checkpointing doesn't help here — was the wrong conclusion.

## 3. How did you find out? What did the failure look like?

The failure wasn't a crash; it was a result that didn't match the mechanism. Activation
checkpointing reclaims memory from *stored activations* and nothing else. The printout above
says the fixed floor — weights plus gradients plus SGD momentum buffers — is 360.4 MB out of a
380.7 MB peak, so activations were roughly 20 MB of the total. Checkpointing could not have
saved much, because there was almost nothing there to save. The experiment wasn't measuring
the technique; it was measuring a batch size too small for the technique to apply.

The tell is that activations scale with batch size while parameters don't. That's a knob, so
I turned it.

## 4. What did you change, and why does your version work?

I rewrote the section to sweep batch size and segment count instead of testing one
configuration, and to re-measure the no-checkpointing baseline at each batch size so the
comparison stays honest. At batch 16384 the activation share rises to about 80% of peak, and
the technique does what it's supposed to:

```
batch= 16384  no checkpointing :   1843.6 MB   21.58s   (~80% of peak is activations)
batch= 16384  checkpoint, 2 segments:   1459.6 MB   31.06s   memory +20.8%  time +43.9%
```

20.8% of peak memory traded for 43.9% more wall-clock time. My version works because it puts
the model in the regime the technique was designed for — fitting a batch that otherwise
wouldn't fit — rather than in a regime where the answer is dominated by a fixed cost
checkpointing can never touch. I kept the batch=4096 row in the sweep on purpose: it shows the
saving climbing from 7.5% to 20.8% as activations grow, which makes the mechanism visible
instead of asking the reader to take one number on faith.

I verified it separately rather than trusting the memory number alone. Recomputing activations
during backward is mathematically a no-op, so the loss must be unchanged — and it is, 2.3026
either way to four decimals. If checkpointing had been silently breaking gradients, that
number would have moved.

One related thing I checked rather than assumed: going from 2 segments to 4 at batch 16384
saved no additional memory (1459.6 MB both times) but cost another 15 percentage points of
time. I left that row in as-measured instead of trimming it, since "more segments is strictly
better" is the intuitive guess and it isn't true here.
