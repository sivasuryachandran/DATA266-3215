# HW11 written findings (GPTQ and distillation) - SID4 3215

All numbers come from `METRICS.md` and the executed `HW12.ipynb`. One T4 GPU, one run, SEED 3215.

## Part 1 - GPTQ on `facebook/opt-125m` (revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6), Wikitext2

4-bit GPTQ (group size 128, 128 calibration windows of 1,024 tokens from Wikitext2 train) raised Wikitext2-test perplexity from 27.66 to 31.18 (+12.7%) and cut the on-disk size from 250.5 MB to 129.3 MB (-48.4%). Peak GPU memory fell only from 1.366 GB to 1.234 GB (-9.7%). Latency got worse, not better: generating 64 tokens at batch 1 went from p50/p95 593/815 ms to 1156/1543 ms (about 1.95x slower), and a 512-token forward pass from 11.4/14.3 ms to 20.8/25.4 ms (about 1.8x slower). The likely reasons are that a 125M-parameter model on a T4 is not memory-bandwidth bound, so the extra dequantization work in the 4-bit kernels costs more than the saved weight reads, and that the embeddings and output layer stay in fp16 (not tested; I did not profile the kernels). Quantization took 169 s.

## Part 2 - distillation, BERT-base teacher to DistilBERT student, AG News

The teacher (BERT-base, 109.5M parameters) reaches 94.43% test accuracy. The best student by validation accuracy (T=8, alpha=0.3, 67.0M parameters) reaches 94.45%, so on accuracy it matches the teacher (the difference of 0.02 points is far inside the 0.26-point standard error), while being 1.95x faster at batch size 1 (p50/p95 4.42/5.28 ms vs 8.63/10.42 ms) and 38.7% smaller on disk (268.6 MB vs 438.0 MB). The student's reported peak memory (1.70 GB vs 1.43 GB) is a measurement artifact: the teacher was still loaded on the GPU. See the measurement notes in `METRICS.md`.

**Temperature sweep (alpha=0.5).** Test accuracy was 0.9457, 0.9454, 0.9470, 0.9461 for T = 1, 2, 4, 8, and validation accuracy rose slowly with T (0.9434 to 0.9454), so T=8 was selected. I would not claim temperature matters here: the whole spread in test accuracy is 0.16 points, smaller than the standard error. **Loss weighting (T=8).** alpha = 0.3, 0.5, 0.7 gave test accuracy 0.9445, 0.9461, 0.9464 (validation 0.9472, 0.9454, 0.9462), again indistinguishable. Selecting by validation picked alpha=0.3, which has the lowest test accuracy of the three; that is selection noise, not evidence against it. In the loss curves the cross-entropy term and the T^2-scaled KL term both fall steadily and flatten after about one epoch; the KL term is larger at high T because of the T^2 scaling.

## Where compression hurts

Per-class test accuracy, teacher / student: World 0.9489 / 0.9526, Sports 0.9884 / 0.9879, Business 0.9137 / 0.9111, Sci/Tech 0.9263 / 0.9263. The class that degrades most is **Business**, but only by 0.26 points (about 5 of 1,900 articles), and the student is better on World, so there is essentially no class-level loss. There are 103 test articles the teacher gets right and the student gets wrong, and 104 the other way round. Of the 103, 28 are Business predicted as Sci/Tech and 29 are Sci/Tech predicted as Business; the two models differ mainly across the Business / Sci/Tech boundary.

Five Business examples the teacher classifies correctly and the student does not:

1. "MetroPCS obtains Cingular air com ... Wired amp; Wireless continues its reign as the top IT priority among IT managers ..." (student: Sci/Tech)
2. "Woman's death probed. Public health officials are investigating why a 38-year-old woman died two weeks after undergoing gastric bypass surgery ..." (student: Sci/Tech)
3. "EBay gets into rentals. EBay plans to buy the apartment and home rental service Rent.com for $415 million ..." (student: Sci/Tech)
4. "Microsoft Settles Novell Antitrust Suit (Reuters) ..." (student: Sci/Tech)
5. "Botswana miners 'face dismissal'. Botswana's giant Debswana diamond mining firm says it will sack workers ..." (student: World)

What they have in common: four of the five are predicted Sci/Tech, and three of those (telecom/IT spending, eBay, Microsoft) are corporate news about technology companies, which sits right on the Business / Sci/Tech boundary. The other two (a hospital investigation and a mining stoppage) read like general or world news with business only implied by the source, so the labels are arguably ambiguous. The flipped articles are about as long as the average test article (35.6 words vs 37.7), so length does not explain it. This is five hand-picked examples from one class; I did not test any explanation beyond looking at them.

## Quantization vs distillation (4-6 sentences)

Quantization (GPTQ) keeps the architecture and shrinks the weights: here it halved the on-disk size and changed peak memory only a little, at a cost of 12.7% higher perplexity and, on this small model and GPU, about 1.9x slower generation. Distillation trains a smaller architecture: DistilBERT has 39% fewer parameters, roughly halved batch-1 latency, and kept the teacher's accuracy on AG News, but it needed 10 minutes of training per student and a labelled teacher pipeline, whereas GPTQ needed only a 169-second calibration run and no training. The two results are on different tasks, so I do not compare their accuracy numbers. For a latency-bound deployment I would try distillation (or another smaller architecture) first, because in my measurements only distillation made inference faster. For a memory-bound deployment I would try quantization first, since it gave the larger drop in stored model size with no retraining, though on a model this small the memory-at-runtime savings were only about 10%, so the gain would be larger for a model whose weights dominate memory.
