import re, nbformat as nbf
src = open("d:/HW10/rag_pipeline.py", encoding="utf-8").read()
src = src.replace('ROOT = os.path.dirname(os.path.abspath(__file__))', 'ROOT = os.getcwd()')
parts = re.split(r"^# -{10,} ", src, flags=re.M)
head, secs = parts[0], parts[1:]
head = head.split('"""', 2)[2]                       # drop module docstring
secs = [s for s in secs if not s.startswith("main")]
code = lambda s: nbf.v4.new_code_cell(s.strip())
md = lambda s: nbf.v4.new_markdown_cell(s.strip())
c = []
c.append(md("""# HW10 – Improving RAG with Reranking, Chunking and Conversational Memory
**Dataset:** `ccdv/arxiv-summarization` (document config), first 100 train docs · **Vector DB:** ChromaDB · **Embedder:** all-MiniLM-L6-v2 · **Reranker:** ms-marco-MiniLM-L-6-v2

Pipeline: preprocess → 2 chunking strategies → embed + index → retrieve (dense) → rerank → conversational memory → evaluate (Precision@K, Recall@K, MRR) → compare."""))
c.append(md("## 0. Setup & data"))
c.append(code("# !pip install chromadb sentence-transformers datasets pandas\n" + head))
c.append(code('''import os, json
if not os.path.exists("docs100.json"):
    from datasets import load_dataset
    ds = load_dataset("ccdv/arxiv-summarization", "document", split="train", streaming=True)  # streaming: avoids multi-GB download
    docs_ = [{"id": i, "article": x["article"], "abstract": x["abstract"]} for i, x in zip(range(100), ds)]
    json.dump(docs_, open("docs100.json", "w"))
docs = json.load(open("docs100.json"))
print(len(docs), "documents; avg words =", int(np.mean([len(d["article"].split()) for d in docs])))
'''))
titles = {"1": "## 1. Preprocessing", "2": "## 2. Two chunking strategies\n1. **fixed window** – 150 words, 30 overlap (cuts mid-sentence)\n2. **sentence-aware** – whole sentences packed to ~150 words, 1-sentence overlap",
          "3": "## 3. Embed and index in ChromaDB", "4": "## 4. Retrieval (dense) and reranking (cross-encoder)",
          "5": "## 5. Conversational memory\nA follow-up is rewritten as `<previous question> <current question>` before retrieval/reranking.",
          "6": "## 6. Evaluation setup\nRelevance rubric: chunk is relevant iff it is from the target document **and** mentions ≥3 (D6: ≥2) of the hand-chosen concepts needed to answer the query. Queries S1–S8 use paper vocabulary; S9–S14 are paraphrased (harder). D1–D6 are two-turn dialogues whose 2nd turn is unanswerable alone."}
for s in secs:
    n = s[0]
    body = s.split("\n", 1)[1]
    c.append(md(titles[n])); c.append(code(body))
    if n == "1":
        c.append(code('print("RAW  :", docs[0]["article"][:300].replace("\\n", " "))\nprint("CLEAN:", " ".join(preprocess(docs[0]["article"])[:2])[:300])'))
c.append(md("## 7. Build the indexes"))
c.append(code('''embedder = SentenceTransformer(EMB_NAME)
reranker = CrossEncoder(RERANK_NAME)
client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
cols, corpus = build_index(docs, embedder, client)
R = Retriever(cols, embedder, reranker)'''))
c.append(md("## 8. Sample queries: without vs with reranking (top-3, ✔ = relevant)"))
c.append(code('''def show(qs, chunking):
    print("="*110); print(f"[{qs['id']}] {qs['q']}   (chunking = {chunking})")
    for rr in (False, True):
        print(f"  --- {'WITH reranking' if rr else 'WITHOUT reranking'}")
        for i, h in enumerate(R.search(qs["q"], chunking, k=3, rerank=rr), 1):
            print(f"   {i}. doc {h['doc']:>2} score {h['score']:7.3f} {'✔' if is_relevant(h['text'], h['doc'], qs) else '✘'}  {snippet(h['text'], 150)}")
for qs in [QUERIES[1], QUERIES[2], QUERIES[13]]:
    for ch in CHUNKERS: show(qs, ch)'''))
c.append(md("## 9. Evaluation: without vs with reranking, chunking 1 vs 2"))
c.append(code('''import pandas as pd
rows, perq = [], []
for ch in CHUNKERS:
    for rr in (False, True):
        ms = []
        for qs in QUERIES:
            tot = n_relevant(corpus[ch], qs)
            m = metrics(R.search(qs["q"], ch, k=10, rerank=rr), qs, tot); ms.append(m)
            perq.append({"query": qs["id"], "chunking": ch, "rerank": rr, "RR": m["RR"], "P@5": m["P@5"], "n_relevant_chunks": tot})
        rows.append({"chunking": ch, "rerank": rr, **avg(ms)})
single = pd.DataFrame(rows)
cols_ = ["chunking","rerank","P@1","P@3","P@5","P@10","R@5","R@10","RR"]
single[cols_].rename(columns={"RR":"MRR"}).round(3)'''))
c.append(code('''pq = pd.DataFrame(perq)
pq.pivot_table(index="query", columns=["chunking","rerank"], values="RR").round(2).loc[[q["id"] for q in QUERIES]]  # per-query reciprocal rank'''))
c.append(md("## 10. Conversational memory – multi-turn demo (turn 2 depends on turn 1)"))
c.append(code('''def run_dialog(dlg, ch, rr, use_mem, k=3):
    mem = ConversationMemory(window=1)
    for ti, q in enumerate(dlg["turns"]):
        query = mem.contextualise(q) if use_mem else q
        hits = R.search(query, ch, k=10, rerank=rr); mem.add(q, hits)
        if ti == 1:
            print(f"  Turn 1: {dlg['turns'][0]}\\n  Turn 2: {q}\\n  Query actually sent: {query}")
            for i, h in enumerate(hits[:k], 1):
                print(f"   {i}. doc {h['doc']:>2} {'✔' if is_relevant(h['text'], h['doc'], dlg['spec2']) else '✘'}  {snippet(h['text'], 140)}")
for d in [DIALOGS[0], DIALOGS[1]]:
    for use_mem in (False, True):
        print("="*100); print(f"{d['id']} – {'WITH memory' if use_mem else 'WITHOUT memory'} (sentence-aware + rerank)")
        run_dialog(d, "sentence_aware", True, use_mem)'''))
c.append(md("## 11. Evaluation: without vs with conversational memory (turn-2 queries, 6 dialogues)"))
c.append(code('''rows = []
for ch in CHUNKERS:
    for rr in (False, True):
        for use_mem in (False, True):
            ms = []
            for dlg in DIALOGS:
                mem = ConversationMemory(window=1)
                for ti, q in enumerate(dlg["turns"]):
                    query = mem.contextualise(q) if use_mem else q
                    hits = R.search(query, ch, k=10, rerank=rr); mem.add(q, hits)
                    if ti == 1: ms.append(metrics(hits, dlg["spec2"], n_relevant(corpus[ch], dlg["spec2"])))
            rows.append({"chunking": ch, "rerank": rr, "memory": use_mem, **avg(ms)})
memdf = pd.DataFrame(rows)
memdf[["chunking","rerank","memory","P@1","P@3","P@5","R@5","R@10","RR"]].rename(columns={"RR":"MRR"}).round(3)'''))
c.append(md("""## 12. Findings

**Reranking vs none.** Reranking mainly improves the top of the list: with fixed chunks MRR 0.911 → 1.000, P@1 0.857 → 1.000, R@5 0.416 → 0.516; with sentence chunks MRR 0.917 → 0.964 and R@10 0.668 → 0.713 (P@5/R@5 flat to slightly lower). Gains come from the paraphrased queries (S11, S14: first relevant chunk moves to rank 1); on easy queries the dense retriever is already at MRR≈1, and reranking occasionally hurts (S4, sentence chunks).

**Chunking 1 vs 2.** Without reranking, sentence-aware chunks are better at early precision (P@3 0.810 vs 0.738, P@5 0.757 vs 0.686) because chunks are coherent units. With reranking, fixed windows score slightly higher on R@5/R@10/MRR. Differences are small relative to the noise of 14 queries; neither is a clear winner.

**Memory vs none.** Large and consistent: without memory the second turn ("Why do they…", "What conditions make it stable?") gets MRR 0.02–0.35 and P@5 ≤ 0.17; with memory MRR 0.92–1.00 and P@5 0.67–0.87. Best overall: sentence-aware + reranking + memory (P@5 0.867, MRR 1.0). Reranking and memory are complementary.

**Limitations.** Tiny hand-built evaluation (14 queries / 6 dialogues); relevance is a keyword rubric tied to one target document, not every chunk read by hand; recall denominators differ between chunkings; the memory test is favourable by design (follow-ups are unanswerable alone, no topic switches) and simple concatenation would fail after a topic change; only retrieval is evaluated (no LLM generation); candidate pool of 20 bounds what reranking can recover."""))
nb = nbf.v4.new_notebook(cells=c)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "d:/HW10/HW10_RAG.ipynb")
