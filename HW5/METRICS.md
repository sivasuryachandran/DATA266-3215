# METRICS - Fine-Tuning an LLM with LoRA

Everything below comes from one run of the notebook, on Google Colab (T4 GPU), SEED = 3215
(SID4 = 3215, see the note on SID4 in the README). The console output of that run is in `RUN_LOG.txt`.

---

## Data

- Dataset: `neil-code/dialogsum-test` - 1,999 train, 499 validation, 499 test rows.
- The test split only has **167 unique dialogues**. Each one appears 3 times with a different human
  summary (`test_0_1`, `test_0_2`, `test_0_3`). I grouped them, so each test dialogue is
  used once and scored against its 3 references.
- Input = "Summarize the following conversation.\n\n{dialogue}\n\nSummary:", target = the summary.
- Tokenized lengths (train): input median 213 tokens, only 45 of 1,999 dialogues hit the 512 cap;
  summary median 37 tokens, cap 128.
- Human reference summaries average 19.3 words.

## Model and LoRA setup

| Setting | Value |
|:--|:--|
| Base model | `google/flan-t5-base` |
| LoRA target modules | `q`, `v` (all attention blocks, encoder and decoder) |
| lora_dropout | 0.05 |
| bias | none |
| r / alpha (main run) | 16 / 32 |
| r / alpha (experiment) | 4 / 8 |

| | Total params | Trainable params | Trainable % | Adapter file |
|:--|--:|--:|--:|--:|
| r=16 | 249,347,328 | 1,769,472 | 0.7096% | 7.1 MB |
| r=4 | 248,020,224 | 442,368 | 0.1784% | 1.8 MB |

I kept alpha/r = 2 for both ranks, so the only thing that changes between the two runs is the rank.
If alpha stayed at 32 for r=4, its updates would be scaled 4x more and I'd be changing the effective
learning rate at the same time.

## Training

AdamW, lr 1e-3, 5% warmup then linear decay, gradient clipping at 1.0, micro-batch 4 × 2
accumulation steps = effective batch 8, 3 epochs = 750 optimizer steps. Full train split.

| | r=16 | r=4 |
|:--|--:|--:|
| Val loss before training | 1.8295 | 1.8295 |
| Val loss after epoch 1 | 1.1105 | 1.1221 |
| Val loss after epoch 2 | 1.0843 | 1.0989 |
| Val loss after epoch 3 | **1.0781** | 1.0948 |
| Final training loss (avg of last 25 steps) | **1.1038** | 1.1619 |
| Training time | 668 s | 667 s |

Training loss drops fast in the first ~75 steps (1.76 to 1.34 for r=16) and after that it
slowly goes down with a lot of step-to-step noise. The r=4 curve follows the r=16 one almost
exactly, just a bit higher at every logged point. Validation loss was still going down at epoch 3
for both, so neither one was overfitting yet. Plots: `outputs/loss_curve_r16.png` and
`outputs/loss_curve_r4_vs_r16.png`.

## ROUGE on all 167 test dialogues

Greedy decoding, max 128 new tokens, each prediction scored against the best of its 3 references.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | avg words |
|:--|--:|--:|--:|--:|--:|
| flan-t5-base, no fine-tuning | 28.38 | 11.06 | 24.92 | 24.92 | 15.5 |
| LoRA r=4 | 48.84 | 23.73 | **41.40** | **41.38** | 21.8 |
| LoRA r=16 | **49.06** | **24.17** | 41.25 | 41.18 | 23.8 |

## Before vs after fine-tuning (the 2 demo dialogues)

**test_0** - #Person1# dictates a memo to Ms. Dawson banning Instant Messaging at work.

| | Summary |
|:--|:--|
| Reference | Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore. |
| Base | The memo is to be distributed to all employees by this afternoon. |
| LoRA r=4 | Ms. Dawson asks #Person1# to take a dictation for #Person1#. #Person1# tells #Person2# that all office communications are restricted to email correspondence and official memos. The memo also restricts external communications. |
| LoRA r=16 | #Person1# asks Ms. Dawson to take a dictation for #Person1#. Ms. Dawson tells #Person1# the new policy applies to all communications, including intra-office communications, and will be placed on probation. |

**test_1** - #Person2# is late because of traffic and gets talked into taking public transport.

| | Summary |
|:--|:--|
| Reference | #Person2# arrives late because of traffic jam. #Person1# persuades #Person2# to use public transportations to keep healthy and to protect the environment. |
| Base | The traffic jam at the Carrefour intersection has caused a lot of congestion. |
| LoRA r=4 | #Person2# got stuck in traffic again. #Person1# suggests #Person2# start taking public transport system to work. #Person2# feels bad about the pollution problem in this city. #Person1# suggests biking to work. |
| LoRA r=16 | #Person2# got stuck in traffic again. #Person1# suggests taking public transport to work. #Person2# thinks it's better for the environment and biking to work will give #Person2# more freedom. |

**What changed:** the base model grabs one detail (the memo deadline, the traffic jam) and writes
one sentence about it, missing the actual point. After LoRA, the summaries use the DialogSum style
(#Person1#/#Person2#) and go through the whole conversation. It's not perfect though:

- In test_0 the r=16 model gets the roles backwards (#Person1# is the one dictating the policy, not
  Ms. Dawson), never says the policy bans Instant Messaging, and garbles the probation part.
- In test_1 both models stop before the ending, where #Person2# decides to quit driving to work.

## r=4 vs r=16

- ROUGE is basically a tie: r=16 is ahead by 0.2 (ROUGE-1) and 0.4 (ROUGE-2), and r=4 is ahead
  by 0.15 on ROUGE-L. r=4 gets the whole jump from the base model (ROUGE-1 28.4 → 48.8) with 4x fewer
  trainable parameters.
- Loss favors r=16: its train and val loss are lower at every checkpoint. That just doesn't show up
  as better summaries.
- r=16 writes slightly longer summaries (23.8 vs 21.8 words; humans average 19.3).
- Training time is identical, because almost all the compute goes into the frozen base model.
- The two give different summaries on 145/167 test dialogues, but a lot of it is wording. Neither
  is clearly better on the examples: on test_0, r=4 says what the policy actually is but gets the
  opening backwards; r=16 gets the opening right but stays vague.
- Same weakness in both, mixing up speakers:
  - test_2: #Person1# tells Kate about the divorce → r=4 says "Kate tells #Person2#", r=16 says "Kate tells #Person1#".
  - test_3: #Person1# asks Brian to dance → both say Brian asks/invites #Person2# (who is Brian himself).

  Going up in rank didn't fix this, so my guess is it's more about the base model's size than the adapter's capacity.

Caveat: this is one seed and 167 test dialogues. When I first ran the notebook with seed 42 (before
switching to SEED = 3215), r=16 was ahead by about 1 ROUGE point on every metric. With SEED = 3215
the gap is 0.2 and ROUGE-L flips to r=4, so the difference between the ranks is within seed-to-seed
noise. With val loss still going down at epoch 3, I think training longer would help more than
raising the rank.
