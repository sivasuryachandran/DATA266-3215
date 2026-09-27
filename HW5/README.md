# Fine-Tuning an LLM with LoRA

**Siva Surya Chandran**
Email: sivasurya.chandran@sjsu.edu

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
   loss 1.14, validation loss went from 1.83 before training to 1.08 after. About 11 minutes on a T4.
5. **After fine-tuning** - same 2 dialogues again.
6. **Comparison** - before vs after, plus ROUGE on all 167 test dialogues.
7. **Experiment** - trained a second adapter with r=4 (alpha=8) and compared it to r=16.

Short version of the results:

| | ROUGE-1 | ROUGE-L | trainable params |
|:--|--:|--:|--:|
| flan-t5-base, no fine-tuning | 28.37 | 24.99 | - |
| LoRA r=4 | 49.50 | 41.32 | 442,368 |
| LoRA r=16 | 50.38 | 42.34 | 1,769,472 |

Fine-tuning makes a big difference. Before, the model writes one short sentence about one detail
and usually misses the point of the conversation. After, it writes DialogSum-style summaries
("#Person1# suggests #Person2# ...") that cover the whole conversation. Rank barely matters here:
r=4 gets almost the same result with 4x fewer trainable parameters. Both models still sometimes
mix up who said what. More detail in `METRICS.md` and at the end of Parts 6 and 7 in the notebook.

---

## Running it

I ran it on Google Colab with a T4 GPU (both training runs together took about 21 minutes). Open
`HW5.ipynb` in Colab, choose Runtime → Change runtime type → T4 GPU, then run all. The first cell
installs `peft`, `evaluate` and `rouge_score`, and uninstalls Colab's old `torchao`, which breaks the
newest `peft`. The Google Drive mount cell near the top is optional and only works on Colab.

It also runs outside Colab (it uses CUDA, Apple MPS or CPU, whichever it finds):

```
pip install torch transformers datasets peft evaluate rouge_score matplotlib
```

I first tried it on my Mac (16 GB), but training used up all the memory and it slowed to a
crawl, so that's why I switched to Colab. Seed is 42 everywhere and decoding is greedy, so the
example outputs should come out the same when you rerun it on the same kind of GPU.

To load a saved adapter later:

```python
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base, "adapters/flan-t5-base-dialogsum-lora-r16")
```
