# DATA 266 - Homework 3

**Siva Surya Chandran**
SJSU ID: 019130215
Email: sivasurya.chandran@sjsu.edu
San Jose State University - DATA 266, Fall 2026

---

## What's in here

Two separate parts, one notebook each.

| Part | Notebook | Topic |
|:--|:--|:--|
| A | `hw3_part_a_prompt_engineering.ipynb` | Six prompt engineering techniques (zero-shot, few-shot, CoT, zero-shot CoT, meta-prompting, tree of thoughts) run through LangChain against a local Ollama model |
| B | `hw3_part_b_self_attention.ipynb` | Scaled dot-product self-attention built from scratch, trained with and without causal masking |

Supporting files:

- `METRICS.md` - all numbers from both parts, with the discussion for each
- `RUN_LOG.txt` - console transcripts of the actual notebook runs
- `AI_USE.md` - AI-use appendix (what I used an assistant for, and a specific thing it got wrong)
- `figures/` - PNGs written by the Part B notebook (loss curves, attention heatmaps)
- `scripts/` - the notebook-generator scripts and the findings-report builder
- `HW3_Findings.pdf` - the final findings report

---

## Part A - Prompt engineering techniques

Backend is a local Ollama `llama3.2` model via `langchain-ollama`, since no Anthropic API key was
available in this environment. Each technique gets the same two test problems: an average-speed
question (correct answer 52.5 mph) and a syllogism (correct answer "cannot be determined").

Tree of Thoughts originally called the same prompt three times and got back three identical
branches, since Ollama's sampling was fully deterministic under a fixed seed. Fixed by giving each
branch a different persona (direct-solve, skeptical, counterexample-seeking) so the three
reasoning paths actually differ before the judge step compares them.

## Part B - Self-attention and causal masking

Single-head scaled dot-product attention implemented with just `nn.Embedding`, `nn.Linear`, and
plain matrix multiplication/softmax - no `nn.MultiheadAttention`, `nn.Transformer`, or HuggingFace
classes. Trained on a fixed 5-sentence text (46 word-level tokens, 39 unique words) with a
next-token objective, once without masking and once with causal masking.

**Result:** the causal-masked model reached a lower, more stable training loss than the unmasked
one for most of training. The unmasked attention heatmap shows real structure (not flat/random);
the masked heatmap is confirmed strictly lower-triangular both visually and with an assertion in
the notebook.

---

## Running it

```
pip install langchain langchain-ollama torch numpy matplotlib
```

Ollama needs to be running locally with the `llama3.2` model pulled for Part A. Part B is pure
PyTorch and runs on CPU in well under a minute (300 epochs over 46 tokens).

Then run each notebook top to bottom, or regenerate them from the scripts in `scripts/`:

```
python scripts/build_prompting_notebook.py
python scripts/build_attention_notebook.py
jupyter nbconvert --to notebook --execute --inplace hw3_part_a_prompt_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace hw3_part_b_self_attention.ipynb
```

Seed is 215 throughout (derived from SID4 per the course's standing instructions). Exact package
versions are recorded at the top of `METRICS.md`.
