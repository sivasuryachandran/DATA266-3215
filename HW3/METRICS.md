# METRICS - HW3

**SID4 = 0215 | SEED = 215 | SLICE = 215 | HP_ID = 5 | CLS_A = 5 | CLS_B = 9**

## Part A - Prompt Engineering Techniques

Backend: local Ollama `llama3.2` via `langchain-ollama` (no Anthropic API key was available in the
environment; substituted per instructor-context tradeoff of a working, code-driven LangChain
pipeline over a blocked cloud call). Temperature 0.0 for all techniques except Tree-of-Thoughts
branch generation (temperature 0.7, with distinct personas per branch to ensure genuine diversity).

| Technique | Example 1 (avg speed; correct = 52.5 mph) - final answer | Example 2 (syllogism; correct = "Cannot be determined") - final answer | Correct? (Ex.1 / Ex.2) |
|---|---|---|---|
| Zero-Shot | 90 | No | ✗ / ~✓ |
| Few-Shot | 52.5 | Cannot be determined | ✓ / ✓ |
| Chain-of-Thought | 52.5 | Cannot be determined | ✓ / ✓ |
| Zero-Shot CoT | 52.5 | "Maria must be an engineer" (affirming the consequent) | ✓ / ✗ |
| Meta-Prompting | 52.5 | No, not necessarily an engineer | ✓ / ✓ |
| Tree of Thoughts | 52.5 | "Not necessarily an engineer" | ✓ / ✓ |

Model calls per technique per example: Zero-Shot/Few-Shot/CoT/Zero-Shot-CoT/Meta-Prompting = 1 each;
Tree of Thoughts = 4 (3 branches + 1 judge).

## Part B - Self-Attention and Causal Masking

Dataset: fixed 5-sentence text specified by the assignment, word-level tokenized (alphabetic words
only - punctuation is dropped rather than kept as its own token).

| Quantity | Value |
|---|---|
| Sequence length (tokens) | 46 |
| Vocabulary size | 39 |
| Embedding dimension (`D_MODEL`) | 32 |
| Training epochs | 300 |
| Optimizer | Adam, lr = 1e-2 |
| Final training loss (unmasked) | ≈ 0.17 (noisy, several spikes after epoch 150) |
| Final training loss (causal-masked) | ≈ 0.154 (lower and much more stable than unmasked) |
| Upper-triangular attention mass (masked model, sanity check) | 0.0 (exactly zero, assert passes) |

Figures produced (`figures/`):
- `part1_training_loss.png` - unmasked model training curve
- `part1_unmasked_attention_heatmap.png` - learned unmasked attention weights, full 46×46 matrix
- `part2_training_loss_comparison.png` - unmasked vs. causal-masked training curves
- `part2_masked_attention_heatmap.png` - learned causal attention weights, strictly lower-triangular

## Environment

- Python 3.11.0, PyTorch 2.13.0, NumPy 2.4.6, LangChain 1.4.0 / langchain-ollama 1.1.1, Ollama
  (local), matplotlib 3.10.8.
- Platform: Windows 11 Pro, no GPU used (CPU-only training; dataset and model are small enough that
  this is not a bottleneck - 300 epochs over 46 tokens completes in well under a minute).
