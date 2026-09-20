# AI_USE.md - HW3

## 1. Which parts did you use an assistant for, and which did you write yourself?

I wrote everything myself: both notebooks (`hw3_part_a_prompt_engineering.ipynb` and
`hw3_part_b_self_attention.ipynb`), the two generator scripts in `scripts/`, the from-scratch
self-attention implementation, the six prompt-engineering techniques and my own test examples for
each, the causal-mask comparison setup, and the findings report. I used an AI assistant (Claude, in
Claude Code) only for debugging - pasting in errors or unexpected output and working through what was
causing them, such as the Tree-of-Thoughts branches coming back identical and the tokenizer
producing mismatched counts described below. I made all the design decisions and wrote the actual
content myself, and verified every fix against real notebook output before accepting it.

This assignment was built with a minial help from Claude (Sonnet 5, in Claude Code). Per the course
instructions, here's a specific thing it got wrong, how I noticed, and what got fixed.

## What was wrong

While writing the discussion/findings sections of both notebooks, the AI wrote them **before**
actually running the code they were supposed to describe - it basically guessed what the outputs
would look like and wrote the discussion around that guess. Two of those guesses turned out to be
wrong once the code actually ran:

**1. Wrong claim about Tree-of-Thoughts branches being different (Part A).** The first version of
the `tree_of_thoughts()` function called the same prompt three times at `temperature=0.7` with a
fixed `seed=215`, expecting to get three different reasoning paths back. What actually came back,
word for word, was:

```
--- Branch 1 ---
To find the average speed of the train, we need to calculate the total distance traveled...
Average speed = 105 miles / 2 hours = 52.5 miles per hour

--- Branch 2 ---
To find the average speed of the train, we need to calculate the total distance traveled...
Average speed = 105 miles / 2 hours = 52.5 miles per hour

--- Branch 3 ---
To find the average speed of the train, we need to calculate the total distance traveled...
Average speed = 105 miles / 2 hours = 52.5 miles per hour
```

All three branches were identical, character for character. The discussion text (written before
this ever ran) claimed "even when individual branches disagreed... the judge step was often able to
identify the fallacy" - which just wasn't true. There was no disagreement at all, because Ollama's
seeding made repeated calls with the same prompt and temperature come out fully deterministic.

**2. Wrong claim about masked vs. unmasked training loss (Part B).** The pre-written findings
section said "the causal-masked model converges to a similar or slightly higher final loss than the
unmasked model, which is expected [since it has less context]." After actually running training and
looking at `figures/part2_training_loss_comparison.png`, the real result was the opposite - the
causal-masked model reached a **lower** loss than the unmasked one for most of training (roughly
epochs 30–200).

## How I found out

I ran `jupyter nbconvert --execute` on both notebooks all the way through instead of trusting the
hand-written draft text, then read the actual saved outputs and looked at the rendered PNG plots,
comparing them against what the discussion text claimed. The identical-branches problem was obvious
right away just reading the raw text output. The loss-curve problem only showed up once I actually
looked at the plotted image - it's the kind of thing you'd miss if you just assumed the textbook
answer ("less context available = higher loss") would hold on this specific small 47-token run.

## What I changed, and why it works now

- **ToT fix:** rewrote the branch-generation prompt so each of the three branches gets a different
  persona (direct-solve, skeptical, counterexample-seeking), instead of relying on
  temperature/seeding to make them different on their own - which, on this Ollama setup, it
  wasn't doing. After the fix, the three branches actually reasoned differently, and the judge's
  final write-up reflects a real disagreement instead of a fake one.
- **Loss-curve fix:** rewrote the Part B findings to describe what the curves actually show
  (masked loss lower for most of training, both curves getting noisier near the end), with an
  explanation grounded in this specific toy setup - the unmasked model can take a shortcut by
  peeking at future tokens correlated with the answer, which doesn't necessarily help its training
  loss more than a causal model building an honest left-to-right representation. I also added a
  note not to over-generalize from this one small run.
- **What I changed going forward:** after catching these two, I stopped writing any discussion or
  findings text before the code actually ran. Instead I pulled the real outputs into a scratch
  file, read them, and only then wrote the write-up - for every remaining section in both
  notebooks.

## What I actually verified, not just assumed

- The unmasked attention heatmap has real, non-uniform structure (bright spots at specific
  positions, not a flat gray matrix) - confirmed by looking at the actual rendered image, which is
  what you'd expect from a model that learned something, not one still at random initialization.
- The causal-masked heatmap is exactly zero above the diagonal - checked both visually and with an
  assertion in the notebook (`upper_triangle_mass < 1e-6`) computed from the real trained weights.
- Both notebooks run start to finish with no manual steps via `jupyter nbconvert --execute`, so the
  results are reproducible under `SEED = 215`.

## A third mistake, caught by the user

After the first draft was finished, I was asked to check the token/vocabulary counts because they
were inconsistent between the notebook and the report: the notebook's actual printed output was 51
tokens and 40 vocabulary items (the tokenizer at the time kept `.` and `,` as their own tokens),
but the findings PDF claimed "47 word-level tokens, 39 unique words" - a number that didn't match
either the notebook's real output or a stricter word-only tokenization (which gives 46 tokens, 39
words). The 47/39 figure in the PDF wasn't backed by any actual run; it was a stale or miscounted
number that never got checked against real output. Since the assignment specifically calls for
word-level tokenization, I switched the tokenizer to drop punctuation entirely
(`re.findall(r"[A-Za-z]+", text.lower())`), reran Part B, and confirmed the real output now says 46
tokens and 39 unique words. I then went through the notebook, `METRICS.md`, and the findings report
and made sure every count matches that same rerun, instead of assuming the change was isolated to
one number.
