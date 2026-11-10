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
          "6": "## 6. Evaluation setup\nRelevance rubric: chunk is relevant iff it is from the target document **and** mentions ≥3 (D6: ≥2) of the hand-chosen concepts needed to answer the query. Queries 40 single-turn queries (S1–S8 use paper vocabulary; S9–S40 are paraphrased and harder) and 15 two-turn dialogues (D1–D15). Section 11b audits this rubric against chunk-by-chunk judgments. Dialogues whose 2nd turn is unanswerable alone."}
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
            perq.append({"query": qs["id"], "chunking": ch, "rerank": rr, "RR": m["RR"], "P@5": m["P@5"], "R@5": m["R@5"], "n_relevant_chunks": tot})
        rows.append({"chunking": ch, "rerank": rr, **avg(ms)})
single = pd.DataFrame(rows)
cols_ = ["chunking","rerank","P@1","P@3","P@5","P@10","R@5","R@10","RR"]
single[cols_].rename(columns={"RR":"MRR"}).round(3)'''))
c.append(code('''pq = pd.DataFrame(perq)
pq.pivot_table(index="query", columns=["chunking","rerank"], values="RR").round(2).loc[[q["id"] for q in QUERIES]]  # per-query reciprocal rank'''))
c.append(md("### 9b. Bootstrap 95% confidence intervals (resampling queries) and paired differences\nWith 40 queries, a difference is only treated as real when its paired-difference interval excludes 0."))
c.append(code('''rng = np.random.default_rng(0); B = 5000
def boot(x):
    x = np.asarray(x, float); bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def piv(df, met): return df.pivot_table(index="query", columns=["chunking","rerank"], values=met)
out_rows = []
for met in ("P@5", "R@5", "RR"):
    P = piv(pq, met)
    for (ch, rr) in P.columns:
        m, lo, hi = boot(P[(ch, rr)]); out_rows.append({"metric": met, "chunking": ch, "rerank": rr, "mean": m, "ci_low": lo, "ci_high": hi})
display(pd.DataFrame(out_rows).round(3))
d_rows = []
for met in ("P@5", "R@5", "RR"):
    P = piv(pq, met)
    cmp = {"rerank - dense (fixed)": P[("fixed_window", True)] - P[("fixed_window", False)],
           "rerank - dense (sentence)": P[("sentence_aware", True)] - P[("sentence_aware", False)],
           "sentence - fixed (dense)": P[("sentence_aware", False)] - P[("fixed_window", False)],
           "sentence - fixed (rerank)": P[("sentence_aware", True)] - P[("fixed_window", True)]}
    for name, d in cmp.items():
        m, lo, hi = boot(d); d_rows.append({"metric": met, "comparison": name, "mean_diff": m, "ci_low": lo, "ci_high": hi,
                                            "excludes_0": bool(lo > 0 or hi < 0), "wins": int((d > 1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()), "losses": int((d < -1e-9).sum())})
display(pd.DataFrame(d_rows).round(3))'''))
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
c.append(md("## 11. Evaluation: without vs with conversational memory (turn-2 queries, 15 dialogues)"))
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
c.append(code('''# bootstrap CIs and paired differences for memory (per-dialogue turn-2 metrics)
mrows = {}
for ch in CHUNKERS:
    for rr in (False, True):
        for use_mem in (False, True):
            v = []
            for dlg in DIALOGS:
                mem = ConversationMemory(window=1)
                for ti, q in enumerate(dlg["turns"]):
                    query = mem.contextualise(q) if use_mem else q
                    hits = R.search(query, ch, k=10, rerank=rr); mem.add(q, hits)
                    if ti == 1: v.append(metrics(hits, dlg["spec2"], n_relevant(corpus[ch], dlg["spec2"])))
            mrows[(ch, rr, use_mem)] = pd.DataFrame(v)
rows = []
for (ch, rr) in [(a, b) for a in CHUNKERS for b in (False, True)]:
    for met in ("P@5", "R@5", "RR"):
        d = mrows[(ch, rr, True)][met] - mrows[(ch, rr, False)][met]
        m, lo, hi = boot(d); rows.append({"chunking": ch, "rerank": rr, "metric": met, "mem - nomem": m, "ci_low": lo, "ci_high": hi, "excludes_0": bool(lo > 0 or hi < 0)})
pd.DataFrame(rows).round(3)'''))
c.append(md("""## 11b. Auditing the keyword rubric against chunk-by-chunk judgments
`audit.py` pooled the top-5 chunks from two configurations for 10 randomly sampled single-turn queries and 4 dialogue turn-2 queries (120 distinct chunks). Each chunk was read **blind to the rubric's verdict** and labelled *useful for answering the query* (1) or not (0). The labeller was Claude, not the student; the 120 chunks are in `audit_items.txt` and the labels in `audit_labels.json` so they can be spot-checked."""))
c.append(code('''items = json.load(open("audit_items.json")); lab = json.load(open("audit_labels.json"))
y = np.array([lab[str(i)] for i in range(len(items))]); r_ = np.array([it["rubric"] for it in items])
tp, tn = int(((y==1)&(r_==1)).sum()), int(((y==0)&(r_==0)).sum()); fp, fn = int(((y==0)&(r_==1)).sum()), int(((y==1)&(r_==0)).sum())
po = (tp+tn)/len(y); pe = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp))/len(y)**2
print(f"chunks={len(y)}  agreement={po:.3f}  Cohen kappa={(po-pe)/(1-pe):.3f}")
print(f"rubric precision={tp/(tp+fp):.3f} recall={tp/(tp+fn):.3f}  (rubric marked {r_.sum()} relevant, judged {y.sum()} useful)")
display(pd.DataFrame([[tp, fn], [fp, tn]], index=["judged useful", "judged not useful"], columns=["rubric relevant", "rubric not relevant"]))
print(json.dumps({k: v for k, v in json.load(open("audit_results.json")).items() if k.startswith("mean|")}, indent=1))'''))
c.append(md("""## 12. Findings (40 single-turn queries, 15 dialogues, bootstrap 95% CIs)

**Reranking vs none.** Reranking helps a little, and mostly at the top of the list. With fixed chunks MRR is 0.775 → 0.845 (paired difference +0.070, 95% CI [0.000, 0.144]: the lower bound sits at zero, so this is borderline) and P@1 0.675 → 0.775; with sentence chunks MRR is 0.785 → 0.813 (+0.028, CI [−0.034, 0.093]). P@5 and R@5 move by ≤ 0.03 with intervals straddling 0. Per query (reciprocal rank): fixed chunks 10 wins / 27 ties / 3 losses, sentence chunks 6 / 29 / 5. So the 14-query result that reranking clearly helped does not hold up at 40 queries: the effect is positive but not statistically distinguishable from zero, and reranking does sometimes hurt (e.g. S4, S21, S24 with sentence chunks).

**Chunking 1 vs 2.** No detectable difference: every paired interval for sentence minus fixed contains 0 (e.g. dense P@5 +0.010, CI [−0.045, 0.070]; reranked MRR −0.032, CI [−0.116, 0.047]). The earlier 14-query hint that sentence-aware chunks had better early precision is within noise.

**Memory vs none.** The one large, significant effect. Turn-2 MRR rises from 0.23–0.43 to 0.93–1.00, and the paired improvement excludes 0 in all four configurations (e.g. P@5 +0.41 to +0.52, MRR +0.57 to +0.73). Best configuration: reranking + memory (P@5 0.71–0.72, MRR 1.00). Memory fixes *which paper*; it does not fix ranking inside the paper.

**Rubric audit.** On 120 chunks read one by one, the keyword rubric agrees with the judgments 80.8% of the time (Cohen's κ = 0.55, moderate); precision 0.83, recall 0.90. It is lenient: it accepts methods, background and notation chunks from the right paper that do not answer the question (15 false positives vs 8 false negatives). On the audited pool this inflates P@5 by about 0.06 and MRR by 0.10–0.20 in absolute terms. Absolute numbers should therefore be read as upper bounds, and the comparisons between configurations as more trustworthy than the levels.

**Limitations.** Queries, dialogues, rubric and the audit labels were all produced by the same assistant, not by independent human annotators; the audit covers 14 of 55 queries; recall denominators differ between chunkings; the memory test is favourable by design (follow-ups are unanswerable alone, no topic switches) and simple concatenation would fail after a topic change; only retrieval is evaluated (no LLM generation); the candidate pool of 20 bounds what reranking can recover."""))
nb = nbf.v4.new_notebook(cells=c)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "d:/HW10/HW10_RAG.ipynb")
