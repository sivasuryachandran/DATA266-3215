# METRICS - Mini GPT from Scratch

Everything below came from one run of the notebook, on CPU, seed 42.

---

## Data and tokenization

- Source file: `shakespeare.txt`, the Romeo and Juliet balcony scene, 1,738 characters total.
- Character-level vocab: 49 unique characters (letters, punctuation, the apostrophe, newline, space).
- Sliding window dataset, seq_len=128, stride 1, which gives 1,609 overlapping training windows.

Since the source text is so short, a 128-char window means the windows overlap a lot - most
of what the model is training on is the same handful of sentences shifted by a character or
two each time. That's basically why it ends up memorizing instead of generalizing (more on
that below).

## Model

| Hyperparameter | Value |
|:--|--:|
| d_model | 128 |
| n_heads | 4 |
| head_dim | 32 |
| n_layers | 4 |
| seq_len | 128 |
| feedforward multiplier | 4x |
| dropout | 0.1 |
| Total parameters | 822,321 |

Structure: token embedding + learnable positional embedding, then 4 decoder blocks
(pre-norm: LayerNorm before attention, LayerNorm before feedforward, each with a residual
connection), then a final LayerNorm, then a linear layer back to vocab size.

Before plugging the attention module into the full model I ran a quick shape check on it by
itself - fed it a `[2, 10, 32]` tensor and checked the output came back the same shape. It did.

## Training

Adam, lr=3e-4, cross-entropy loss, batch size 16, 8 epochs.

| Epoch | Loss |
|--:|--:|
| 1 | 2.7393 |
| 2 | 2.2435 |
| 3 | 2.1113 |
| 4 | 1.9756 |
| 5 | 1.7870 |
| 6 | 1.5540 |
| 7 | 1.2897 |
| 8 | 1.0450 |

Loss drops the whole way through, no weird spikes or plateaus. Roughly cut in half (2.7 to
1.0) over 8 epochs. Curve is plotted in the notebook under Part 3.

## Generation - comparing the three decoding methods

Same prompt ("ROMEO:") fed into the trained model each time.

**Greedy** (always picks the highest-probability character, no randomness):

```
ROMEO:
I takee thy word word:
Call me ne be wet;

JULIET:
What that thu thy namat thus Moueld, not, not,
And, nor nor nor nor nor nor bear be ny or or or or be,
```

You can see it get stuck looping on "nor nor nor" and "or or or or" - once it locks onto a
pattern that looks locally likely it just keeps repeating it.

**Temperature sampling**, tried at 0.5, 1.0, 1.5:

- T=0.5: pretty close to greedy, still repetitive but a bit more varied
- T=1.0: noticeably more broken up, starts losing real words
- T=1.5: mostly just noise at this point, barely any recognizable words left

**Top-k sampling**, tried at k=3 and k=15 (temperature fixed at 1.0):

- k=3: stays close to greedy-ish behavior
- k=15: more variety than k=3, but doesn't fall apart the way high-temperature sampling does,
  since it never gets to consider the really unlikely characters in the first place

### Quick comparison

| Method | Coherence | Diversity | Deterministic? |
|:--|:--|:--|:--|
| Greedy | highest | lowest | yes |
| Temperature, T=0.5 | high | low-ish | no |
| Temperature, T=1.5 | low | highest | no |
| Top-k, k=3 | high | low-ish | no |
| Top-k, k=15 | medium | high | no |

Most coherent: greedy, it always takes the safest option. Most diverse: temperature at 1.5,
but at the cost of mostly not looking like English anymore. Best tradeoff if you want some
variety without total nonsense: top-k with k=15.

### Why it's mostly memorizing, not "writing"

`shakespeare.txt` is only about 1,700 characters, so no matter which decoding method I use,
I keep seeing the same fragments come back - "ROMEO:", "JULIET:", "name", "Romeo". The model
hasn't really learned English, it's learned this one specific passage and is recombining bits
of it. I'd expect the same general pattern to hold on a bigger dataset (greedy = safe and
repetitive, high temperature = diverse but messy, top-k = somewhere in between), but the
actual text would probably look a lot more like real writing instead of echoing the training
data back at you.

Full transcripts and the loss plot are in `Mini_GPT_Assignment.ipynb`, Parts 3 and 4.
