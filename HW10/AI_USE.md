# AI Use - RAG with Reranking, Chunking and Conversational Memory

## What did you use an assistant for, and what did you write yourself?

I used Claude Code to build this homework. I gave it the assignment text and it wrote the data-loading, preprocessing, both chunkers, the ChromaDB indexing, the reranking step, the memory class, the evaluation code, the notebook and the report files. It also chose the models (`all-MiniLM-L6-v2` and the MS MARCO cross-encoder), wrote the 14 queries and 6 dialogues, and wrote the relevance rubric for each. I supplied the task, the requirement for a notebook that is already run, and the request for these files; I did not write the code myself.

The assistant ran everything in this environment, so every number in the tables comes from code that was actually executed. The written interpretation is the assistant's reading of those numbers, and I should be explicit that the relevance judgments are a coded keyword rubric the assistant defined, not chunks I read one by one.

## Give one specific thing it got wrong.

The first version of the sentence-aware chunker assumed the dataset stores one sentence per line, so it split on newlines. The dataset actually stores whole paragraphs or whole articles on one line. The result was only 100 chunks for strategy 2, one per document, with a mean of 5,686 words and a maximum of 29,688 words, which is far beyond what the embedding model can read. After the regex sentence splitter was added, one run-on "sentence" (a table or reference list with no punctuation) still produced a chunk of 1,820 words. The first query set was also too easy: reciprocal rank was 1.0 in every configuration, so it could not tell reranking apart from no reranking.

## How did you find out?

The index step prints the number of chunks and the mean, minimum and maximum words per chunk. A hundred chunks for a hundred documents, and a maximum near 30,000 words, were obviously wrong next to 4,724 chunks for the fixed window. The 1,820-word chunk showed up in the same printout after the first fix. The too-easy queries showed up in the per-query table, where every query had reciprocal rank 1.0 with and without reranking.

## What did you change, and why does your version work now?

The preprocessing now splits each line on sentence-final punctuation with a regular expression, and the chunker hard-splits any sentence longer than the chunk size into 150-word pieces. Strategy 2 now has 5,256 chunks with a mean of 136 words and a maximum of 300, in line with strategy 1. I added six paraphrased queries (S9–S14) whose wording differs from the papers, which produced real differences between dense retrieval and reranking (S11 and S14 improved to rank 1 with reranking; S4 got worse with sentence chunks). The results are deterministic, so rerunning `rag_pipeline.py` or the notebook reproduces the tables.
