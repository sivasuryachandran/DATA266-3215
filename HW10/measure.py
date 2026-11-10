"""Timing / cost / failure-analysis measurements for HW10 -> measurements.json (+ stdout)."""
import json, os, time, platform, sys
import numpy as np, torch, chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import rag_pipeline as rp

ROOT = rp.ROOT
docs = json.load(open(os.path.join(ROOT, "docs100.json")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| python", platform.python_version(), "| torch", torch.__version__)
emb = SentenceTransformer(rp.EMB_NAME); rr_model = CrossEncoder(rp.RERANK_NAME)
client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
M = {"device": dev}

# ---- indexing timings per chunking strategy
M["index"] = {}
for name, fn in rp.CHUNKERS.items():
    t = time.perf_counter()
    recs = [c for d in docs for c in fn(rp.preprocess(d["article"]))]
    t_chunk = time.perf_counter() - t
    t = time.perf_counter()
    E = emb.encode(recs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    t_emb = time.perf_counter() - t
    wc = [len(x.split()) for x in recs]
    M["index"][name] = dict(chunks=len(recs), mean_words=float(np.mean(wc)), min_words=min(wc), max_words=max(wc),
                            chunk_s=t_chunk, embed_s=t_emb, chunks_per_s=len(recs) / t_emb)
    print(name, M["index"][name])

cols, corpus = rp.build_index(docs, emb, client)
R = rp.Retriever(cols, emb, rr_model)

# ---- query latency (median of 3 passes over the 14 queries), dense vs dense+rerank
M["latency_ms"] = {}
for ch in rp.CHUNKERS:
    for rr in (False, True):
        ts = []
        for _ in range(3):
            for qs in rp.QUERIES:
                t = time.perf_counter(); R.search(qs["q"], ch, k=10, rerank=rr); ts.append((time.perf_counter() - t) * 1000)
        M["latency_ms"][f"{ch}|{'rerank' if rr else 'dense'}"] = dict(median=float(np.median(ts)), p95=float(np.percentile(ts, 95)))
print(json.dumps(M["latency_ms"], indent=1))

# ---- failure analysis: every (query, config) with RR<1 or P@5<0.5
fails = []
def analyse(qid, spec, query, ch, rr, tag=""):
    hits = R.search(query, ch, k=10, rerank=rr)
    m = rp.metrics(hits, spec, rp.n_relevant(corpus[ch], spec))
    if m["RR"] < 1 or m["P@5"] < 0.5:
        wrong = next((h for h in hits if not rp.is_relevant(h["text"], h["doc"], spec)), None)
        wrong_docs = sorted({h["doc"] for h in hits[:5] if h["doc"] not in spec["docs"]})
        fails.append(dict(id=qid + tag, chunking=ch, rerank=rr, RR=m["RR"], P5=m["P@5"], n_rel=rp.n_relevant(corpus[ch], spec),
                          off_doc_in_top5=wrong_docs,
                          top1_relevant=rp.is_relevant(hits[0]["text"], hits[0]["doc"], spec),
                          first_wrong=dict(doc=wrong["doc"], text=rp.snippet(wrong["text"], 160)) if wrong else None))
for qs in rp.QUERIES:
    for ch in rp.CHUNKERS:
        for rr in (False, True): analyse(qs["id"], qs, qs["q"], ch, rr)
for d in rp.DIALOGS:
    for ch in rp.CHUNKERS:
        for rr in (False, True):
            analyse(d["id"], d["spec2"], d["turns"][1], ch, rr, "-t2-nomem")
            analyse(d["id"], d["spec2"], d["turns"][0] + " " + d["turns"][1], ch, rr, "-t2-mem")
M["failures"] = fails
print(len(fails), "failing (query, config) cases")
for f in fails: print(f["id"], f["chunking"], "rerank" if f["rerank"] else "dense", "RR=%.2f P@5=%.1f nrel=%d off_docs=%s top1_ok=%s" % (f["RR"], f["P5"], f["n_rel"], f["off_doc_in_top5"], f["top1_relevant"]))
def category(f):
    if "-t2-nomem" in f["id"]: return "follow-up without memory"
    if "-t2-mem" in f["id"]:   return "follow-up with memory: right paper, weak ranking" if not f["off_doc_in_top5"] else "follow-up with memory: other papers in top 5"
    if f["top1_relevant"]:     return "capped: few relevant chunks (top-1 correct)"
    if f["off_doc_in_top5"]:   return "other papers outrank the answer"
    return "right paper, answer chunk ranked low"
from collections import Counter
M["failure_categories"] = dict(Counter(category(f) for f in fails))
for f in fails: f["category"] = category(f)
print(M["failure_categories"])

# ---- per-query wins / ties / losses (reciprocal rank), from results.json
res = json.load(open(os.path.join(ROOT, "results.json")))["perquery"]
ids = [q["id"] for q in rp.QUERIES]
def wtl(a, b, key="RR"):
    d = [res[a(i)][key] - res[b(i)][key] for i in ids]
    return dict(wins=sum(x > 1e-9 for x in d), ties=sum(abs(x) <= 1e-9 for x in d), losses=sum(x < -1e-9 for x in d))
M["wtl"] = {
  "rerank vs dense (fixed)":    wtl(lambda i: f"{i}|fixed_window|rerank",    lambda i: f"{i}|fixed_window|dense"),
  "rerank vs dense (sentence)": wtl(lambda i: f"{i}|sentence_aware|rerank",  lambda i: f"{i}|sentence_aware|dense"),
  "sentence vs fixed (dense)":  wtl(lambda i: f"{i}|sentence_aware|dense",   lambda i: f"{i}|fixed_window|dense"),
  "sentence vs fixed (rerank)": wtl(lambda i: f"{i}|sentence_aware|rerank",  lambda i: f"{i}|fixed_window|rerank"),
}
print(json.dumps(M["wtl"], indent=1))
json.dump(M, open(os.path.join(ROOT, "measurements.json"), "w"), indent=1)
