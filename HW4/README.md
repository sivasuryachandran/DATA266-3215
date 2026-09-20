# Mini GPT from Scratch

**Siva Surya Chandran**
Email: sivasurya.chandran@sjsu.edu

---

## What's in here

One notebook, one small dataset.

| File | Purpose |
|:--|:--|
| `HW4.ipynb` | The assignment - a decoder-only GPT I built from scratch in PyTorch |
| `shakespeare.txt` | Training data - the Romeo and Juliet balcony scene (1,738 characters) |
| `Demo_4_Mini_GPT.ipynb` | The single-head demo we went over in class, which this builds on |

Also in here:

- `METRICS.md` - all the numbers from the notebook (vocab, params, loss curve, generated text) with some discussion
- `RUN_LOG.txt` - the actual console output from running the notebook, plus what environment I ran it on
- `AI_USE.md` - short note on where I used an AI assistant and something it got wrong

---

## What it does

A character-level GPT, decoder-only, written by hand - no `nn.Transformer`, no
`nn.MultiheadAttention`, no HuggingFace, just `nn.Linear`, `nn.LayerNorm`, `nn.GELU` and basic
tensor ops. It's built on top of the single-head demo from class but I extended it to:

- multi-head attention (4 heads instead of 1)
- 4 decoder blocks stacked instead of 1
- longer sequence length (128 instead of 64)
- actual training with a loss curve
- three ways of generating text (greedy, temperature, top-k), compared against each other

Setup: `d_model=128`, `n_heads=4`, `n_layers=4`, `seq_len=128`, pre-norm blocks (norm before
attention/feedforward, not after), 822,321 parameters total.

Trained with Adam, lr `3e-4`, cross-entropy loss, batch size 16, 8 epochs. Loss goes from
2.74 down to 1.05.

All three decoding methods work and I compared them directly. Greedy is the most "sensible"
looking but gets stuck repeating itself ("or or or or", "nor nor nor"). Temperature at 1.5 is
the most random but stops looking like real words pretty fast. Top-k with k=15 ends up as a
decent middle ground - more variety than k=3 without totally falling apart. Worth saying up
front: the training text is only about 1,700 characters, so the model is basically just
memorizing pieces of this one passage rather than learning English in general - "ROMEO:",
"JULIET:", "name" show up in almost everything it generates no matter which decoding method I
use. More on that in `METRICS.md`.

---

## Running it

Just needs CPU, the model's small enough (822K params) that GPU isn't necessary.

```
pip install torch numpy matplotlib
```

Then run `HW4.ipynb` from top to bottom. Seed is 42 everywhere, so rerunning
it should give you the same loss curve, and greedy decoding should give the exact same text
every time.

I ran the committed version locally on CPU - see `RUN_LOG.txt` for exact package versions.
