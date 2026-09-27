# Fine-Tuning an LLM with LoRA

**Siva Surya Chandran**
Email: sivasurya.chandran@sjsu.edu

**Personal Parameters** (Section 0.1 of the standing requirements):

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|------|------|-------|-------|-------|-------|
| 3215 | 3215 | 215   | 5     | 5     | 2     |

**Note on SID4.** My SJSU ID is 019130215, so the last four digits are 0215. Taken literally that
has a leading zero, which collapses to the 3-digit number 215 when used as a number and makes SID4
ambiguous with SLICE. I used 3215 instead - the last four digits with the preceding digit 3 in place
of the leading zero - so that SID4 is a genuine 4-digit value and SEED/SLICE stay distinct. All
parameters in this repo are derived from SID4 = 3215.

HW5 uses a fixed sweep (r=4 vs r=16) for everyone, so HP_ID isn't used, and it uses the full
DialogSum splits with no classes, so SLICE, CLS_A and CLS_B aren't used either. SEED = 3215 is used
for every random seed in the notebook.

---

## What's in here

| File / folder | Purpose |
|:--|:--|
| `HW5.ipynb` | The assignment - LoRA fine-tuning of flan-t5-base on DialogSum, run on Colab (T4) |
| `outputs/` | Baseline outputs (`baseline_outputs.json`), all results (`results.json`), and the two loss plots |
| `adapters/` | The two saved LoRA adapters (r=16 and r=4). Only the adapter weights, not the full model |
| `HW5.pdf` | The document (.pdf) that contains my findings. |

Also in here:

- `METRICS.md` - all the numbers from the notebook (params, loss, ROUGE, example outputs) with some discussion
- `RUN_LOG.txt` - the console output from running the notebook, plus what environment I ran it on
- `AI_USE.md` - short note on where I used an AI assistant and something it got wrong

---

## What it does

I took `google/flan-t5-base` (about 248M params) and fine-tuned it to summarize conversations using
LoRA through HuggingFace PEFT, on the DialogSum dataset (`neil-code/dialogsum-test`: 1,999 train,
499 validation, 499 test).

Steps, in the same order as the assignment:

1. **Data prep** - each dialogue becomes the input (with "Summarize the following conversation." in
   front) and its human summary becomes the target. The test split lists every dialogue 3 times
   (once per human summary), so I group those and end up with 167 unique test dialogues, each with 3 references.
2. **Baseline** - plain flan-t5-base summarizes 2 test dialogues (`test_0`, `test_1`), saved to
   `outputs/baseline_outputs.json`.
3. **LoRA** - r=16, alpha=32, dropout 0.05, on the `q` and `v` attention projections. That's
   1,769,472 trainable params out of 249,347,328 total (0.71%).
4. **Fine-tuning** - full train split, 3 epochs, AdamW lr 1e-3, effective batch 8. Final training
   loss 1.10, validation loss went from 1.83 before training to 1.08 after. About 11 minutes on a T4.
5. **After fine-tuning** - same 2 dialogues again.
6. **Comparison** - before vs after, plus ROUGE on all 167 test dialogues.
7. **Experiment** - trained a second adapter with r=4 (alpha=8) and compared it to r=16.

Short version of the results:

| | ROUGE-1 | ROUGE-L | trainable params |
|:--|--:|--:|--:|
| flan-t5-base, no fine-tuning | 28.38 | 24.92 | - |
| LoRA r=4 | 48.84 | 41.40 | 442,368 |
| LoRA r=16 | 49.06 | 41.25 | 1,769,472 |

Fine-tuning makes a big difference. Before, the model writes one short sentence about one detail
and usually misses the point of the conversation. After, it writes DialogSum-style summaries
("#Person1# suggests #Person2# ...") that cover the whole conversation. Rank doesn't matter here:
r=4 and r=16 are within half a point of each other on every ROUGE score (r=4 even wins ROUGE-L),
so r=4 gets the same result with 4x fewer trainable parameters. Both models still sometimes
mix up who said what. More detail in `METRICS.md` and at the end of Parts 6 and 7 in the notebook.

---

## Running it

I ran it on a Google Colab T4 GPU runtime, connected from VS Code (the whole notebook took about
26 minutes, 11 of them per training run). In Colab, open `HW5.ipynb`, choose Runtime → Change
runtime type → T4 GPU, then run all. The first cell installs `peft`, `evaluate` and `rouge_score`,
and uninstalls Colab's old `torchao`, which breaks the newest `peft`.

It also runs outside Colab (it uses CUDA, Apple MPS or CPU, whichever it finds):

```
pip install torch transformers datasets peft evaluate rouge_score matplotlib
```

I first tried it on my Mac (16 GB), but training used up all the memory and it slowed to a
crawl, so that's why I switched to Colab. SEED = 3215 is used everywhere and decoding is greedy,
so the example outputs should come out the same when you rerun it on the same kind of GPU.

The two adapters in `adapters/` are the checkpoints. They were trained by running `HW5.ipynb`
top to bottom, and the exact training calls in the notebook are:

```python
lora16 = make_lora_model(r=16)       # Part 3
log16 = train_lora(lora16, "r=16")   # Part 4 -> adapters/flan-t5-base-dialogsum-lora-r16
lora4 = make_lora_model(r=4)         # Part 7
log4 = train_lora(lora4, "r=4")      # Part 7 -> adapters/flan-t5-base-dialogsum-lora-r4
```

To load a saved adapter later:

```python
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base, "adapters/flan-t5-base-dialogsum-lora-r16")
```
