# AI Use - Mini GPT from Scratch

## What did you use an assistant for, and what did you write yourself?

I wrote the actual model code myself - the attention math, the decoder block, stacking it
into the full GPT, the training loop, all three decoding functions. That's the core of the
assignment and I wanted to actually understand it, not just have something spit it out.

Where I did lean on an assistant was mainly two things: debugging shape errors in the
attention code, and some of the matplotlib plotting boilerplate for the loss curve, since I
always forget the exact syntax for that.

## Give one specific thing it got wrong.

When I was first splitting Q/K/V into multiple heads, I asked for help fixing a
`RuntimeError` from a shape mismatch in the attention scores. The suggested fix reshaped the
tensors but did the transpose in the wrong order, so heads and sequence positions ended up
swapped. It ran without crashing, which is what made it sneaky - the model trained, the loss
went down, but the attention output didn't actually match the expected `[batch, seq, d_model]`
layout, and when I ran my shape-check cell it wasn't matching up the way it should.

## How did you find out?

I already had a sanity-check cell that feeds a small dummy tensor through the attention
module and checks the output shape against the input shape. The shapes matched, so at first
it looked fine. What actually gave it away was printing the intermediate tensor shapes after
the reshape/transpose step and comparing them against what I expected by hand for a small
example (2 heads, seq len 4) - the head dimension and sequence dimension were transposed
relative to each other, so it happened to still produce the "right" final shape while mixing
up which position was attending to which.

## What did you change, and why does your version work now?

I rewrote the reshape/transpose step myself, working through the dimensions on paper first
(`[batch, seq, d_model] -> [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]`)
instead of trusting the suggested code, and re-ran the small dummy-tensor check by hand to
confirm each head was actually looking at the right positions before I trusted it with the
real training run. Everything downstream (training loss, generation) was run only after that
was fixed.
