"""
HW10 - Improving RAG with Reranking, Chunking and Conversational Memory
Dataset : ccdv/arxiv-summarization ('document' config), first 100 train docs
Vector DB: ChromaDB (persistent, cosine)   Embedder: all-MiniLM-L6-v2
Reranker : cross-encoder/ms-marco-MiniLM-L-6-v2
Run     : python rag_pipeline.py   (needs docs100.json, produced by fetch step)
"""
import json, re, os, sys, time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf"))
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS, OVERLAP_WORDS = 150, 30
N_CANDIDATES = 20          # retrieved from vector DB before reranking
KS = (1, 3, 5, 10)

# ----------------------------------------------------------------- 1. preprocess
def preprocess(text: str):
    """Return a list of clean sentences. The dataset stores one sentence per line,
    with LaTeX placeholders (@xmath12, @xcite) and ' .' tokenisation artefacts."""
    text = re.sub(r"@xcite", "", text)
    text = re.sub(r"@xmath\d+", "[math]", text)
    text = re.sub(r"(\[math\]\s*){2,}", "[math] ", text)       # collapse runs
    sents = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([.,;:!?)])", r"\1", line)          # ' .' -> '.'
        line = re.sub(r"\(\s+", "(", line)
        # lines can be whole paragraphs -> split on sentence-final punctuation
        for s in re.split(r"(?<=[.?!])\s+(?=[A-Za-z\[(\"'])", line):
            if len(s.split()) >= 3:                             # drop fragments
                sents.append(s.strip())
    return sents

# ----------------------------------------------------------------- 2. chunking
def chunk_fixed(sents, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Strategy 1: fixed-size sliding word window (cuts mid-sentence)."""
    words = " ".join(sents).split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]

def chunk_sentence(sents, size=CHUNK_WORDS, overlap_sents=1):
    """Strategy 2: sentence-aware packing up to ~size words, carrying the last
    sentence into the next chunk as overlap. Never splits a sentence."""
    chunks, cur, n = [], [], 0
    flat = []                       # hard-split run-on "sentences" (tables, ref lists)
    for s in sents:
        ws = s.split()
        flat += [" ".join(ws[i:i + size]) for i in range(0, len(ws), size)] if len(ws) > size else [s]
    for s in flat:
        w = len(s.split())
        if cur and n + w > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            n = sum(len(x.split()) for x in cur)
        cur.append(s); n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks

CHUNKERS = {"fixed_window": chunk_fixed, "sentence_aware": chunk_sentence}

# ----------------------------------------------------------------- 3. index
def build_index(docs, embedder, client):
    cols, corpus = {}, {}
    for name, fn in CHUNKERS.items():
        recs = []
        for d in docs:
            for j, c in enumerate(fn(preprocess(d["article"]))):
                recs.append({"id": f"{name}-d{d['id']}-c{j}", "doc": d["id"], "text": c})
        corpus[name] = recs
        try: client.delete_collection(name)
        except Exception: pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        embs = embedder.encode([r["text"] for r in recs], batch_size=64,
                               normalize_embeddings=True, show_progress_bar=False)
        for i in range(0, len(recs), 2000):
            b = recs[i:i + 2000]
            col.add(ids=[r["id"] for r in b], embeddings=embs[i:i + 2000].tolist(),
                    documents=[r["text"] for r in b],
                    metadatas=[{"doc": r["doc"]} for r in b])
        cols[name] = col
        wc = [len(r["text"].split()) for r in recs]
        print(f"[index] {name:15s} chunks={len(recs):5d} mean_words={np.mean(wc):.0f} "
              f"min={min(wc)} max={max(wc)}")
    return cols, corpus

# ----------------------------------------------------------------- 4. retrieval
class Retriever:
    def __init__(self, cols, embedder, reranker):
        self.cols, self.emb, self.rr = cols, embedder, reranker

    def search(self, query, chunking, k=10, rerank=False):
        q = self.emb.encode([query], normalize_embeddings=True)[0].tolist()
        res = self.cols[chunking].query(query_embeddings=[q], n_results=N_CANDIDATES)
        hits = [{"id": i, "text": t, "doc": m["doc"], "score": 1 - d}
                for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                                      res["metadatas"][0], res["distances"][0])]
        if rerank:
            s = self.rr.predict([(query, h["text"]) for h in hits])
            for h, sc in zip(hits, s): h["score"] = float(sc)
            hits.sort(key=lambda h: -h["score"])
        return hits[:k]

# ----------------------------------------------------------------- 5. memory
class ConversationMemory:
    """Rolling window of previous turns. A follow-up is rewritten by prefixing the
    previous user question(s), so pronouns/ellipsis ('they', 'it', 'that') are
    resolved by the retriever through the earlier topical context."""
    def __init__(self, window=1):
        self.window, self.turns = window, []

    def contextualise(self, query):
        if not self.turns:
            return query
        prev = " ".join(t["query"] for t in self.turns[-self.window:])
        return f"{prev} {query}"

    def add(self, query, hits):
        self.turns.append({"query": query, "top_doc": hits[0]["doc"] if hits else None})

# ----------------------------------------------------------------- 6. evaluation
def is_relevant(text, doc, spec):
    """Rubric (documented in report): chunk is relevant iff it comes from a target
    document AND mentions >= spec['min_hits'] distinct topical concepts needed to
    answer the question, i.e. it is useful evidence for the answer."""
    if doc not in spec["docs"]:
        return False
    t = text.lower()
    return sum(bool(re.search(p, t)) for p in spec["terms"]) >= spec["min_hits"]

def metrics(hits, spec, n_rel_total, ks=KS):
    rel = [is_relevant(h["text"], h["doc"], spec) for h in hits]
    out = {}
    for k in ks:
        r = sum(rel[:k])
        out[f"P@{k}"] = r / k
        out[f"R@{k}"] = r / n_rel_total if n_rel_total else 0.0
    rr = next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0)
    out["RR"] = rr
    return out

QUERIES = [  # single-turn queries ----------------------------------------------
 dict(id="S1", q="What learning rates are obtained for regularized kernel based methods for additive models?",
      docs=[0], terms=[r"learning rate", r"additive", r"kernel", r"regulari"], min_hits=3),
 dict(id="S2", q="How does synchronization occur in coupled quadratic maps?",
      docs=[36], terms=[r"synchroniz", r"coupl", r"quadratic|logistic", r"map"], min_hits=3),
 dict(id="S3", q="How can squeezed vacuum be used for non-invasive measurement of cavity parameters?",
      docs=[33], terms=[r"squeez", r"cavity", r"measur", r"vacuum"], min_hits=3),
 dict(id="S4", q="What causes the enhanced mid-infrared second harmonic generation in SiC nanopillars?",
      docs=[6], terms=[r"second.harmonic|shg", r"nanopillar", r"resonan", r"sic"], min_hits=3),
 dict(id="S5", q="How is bend loss predicted for large mode area photonic crystal fibers?",
      docs=[37], terms=[r"bend", r"loss", r"photonic crystal|pcf", r"mode area|lma"], min_hits=3),
 dict(id="S6", q="Why does magnetoresistance become linear in the surface state of a topological insulator?",
      docs=[20], terms=[r"magnetoresistance", r"linear", r"topological insulator", r"surface state"], min_hits=3),
 dict(id="S7", q="How do phylogenetic networks model reticulate evolution?",
      docs=[58], terms=[r"phylogenetic", r"network", r"reticulat", r"evolution"], min_hits=3),
 dict(id="S8", q="What was observed during the 1994 outburst of the T Tauri star EX Lupi?",
      docs=[48], terms=[r"ex lup", r"outburst", r"t tauri", r"spectr|photomet"], min_hits=3),
 # harder: paraphrased, little exact jargon overlap, many topically-near documents
 dict(id="S9", q="How is the spin structure of the proton and neutron probed through form factors?",
      docs=[4], terms=[r"spin", r"nucleon", r"form factor", r"quark|gluon"], min_hits=3),
 dict(id="S10", q="What do microwave background anisotropy maps tell us about open universe models?",
      docs=[12], terms=[r"open", r"dmr|cobe|microwave", r"normali[sz]ation|amplitude|quadrupole", r"model"], min_hits=3),
 dict(id="S11", q="How are signals from merging binary stars found in noisy interferometer data using templates?",
      docs=[27], terms=[r"matched.filter|template", r"gravitational", r"compact|binary|merger|inspiral", r"detector|noise|data"], min_hits=3),
 dict(id="S12", q="How does hot asymmetric nuclear matter separate into liquid and vapour phases?",
      docs=[49], terms=[r"liquid.gas|phase transition", r"neutron.rich|asymmetr", r"nuclear matter", r"temperature|thermal|spinodal"], min_hits=3),
 dict(id="S13", q="Why does a chaotic ratchet move particles in the opposite direction?",
      docs=[2], terms=[r"ratchet", r"current", r"revers", r"chaotic|bifurcation"], min_hits=3),
 dict(id="S14", q="How can online astronomy resources be made usable for people with disabilities?",
      docs=[40], terms=[r"access", r"librar", r"disab|impair|challenged|blind", r"web|online|resource"], min_hits=3),
]

DIALOGS = [  # multi-turn: turn 2 is meaningless without turn 1 -------------------
 dict(id="D1", turns=["What learning rates are obtained for regularized kernel based methods for additive models?",
                      "Why do they compare favourably in high dimensions?"],
      spec2=dict(docs=[0], terms=[r"high.dimension", r"learning rate|rate", r"additive", r"nonparametric|optimal"], min_hits=3)),
 dict(id="D2", turns=["How does synchronization occur in coupled quadratic maps?",
                      "What conditions make it stable?"],
      spec2=dict(docs=[36], terms=[r"stab", r"synchroniz", r"condition|threshold|critical", r"coupl"], min_hits=3)),
 dict(id="D3", turns=["What was observed during the 1994 outburst of the T Tauri star EX Lupi?",
                      "How bright did it get and how long did it last?"],
      spec2=dict(docs=[48], terms=[r"outburst", r"mag|bright", r"days?|weeks?|months?|duration|decline", r"ex lup"], min_hits=3)),
 dict(id="D4", turns=["How is bend loss predicted for large mode area photonic crystal fibers?",
                      "How accurate is it compared with numerical simulations?"],
      spec2=dict(docs=[37], terms=[r"accura|agree|good", r"numerical|simulation|multipole|full.vector", r"bend", r"loss"], min_hits=3)),
 dict(id="D5", turns=["Why does magnetoresistance become linear in the surface state of a topological insulator?",
                      "What is the role of the Dirac cone there?"],
      spec2=dict(docs=[20], terms=[r"dirac", r"linear", r"magnetoresistance", r"surface"], min_hits=3)),
 dict(id="D6", turns=["How do phylogenetic networks model reticulate evolution?",
                      "How can they be visualised?"],
      spec2=dict(docs=[58], terms=[r"visuali[sz]", r"network", r"phylogenetic|reticulat|evolution"], min_hits=2)),
]

def n_relevant(corpus_recs, spec):
    return sum(is_relevant(r["text"], r["doc"], spec) for r in corpus_recs)

def avg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

def snippet(t, n=170): return t[:n].replace("\n", " ") + ("..." if len(t) > n else "")

# ----------------------------------------------------------------- main
def main():
    docs = json.load(open(os.path.join(ROOT, "docs100.json")))
    print(f"[data] {len(docs)} documents, avg words={np.mean([len(d['article'].split()) for d in docs]):.0f}")
    embedder = SentenceTransformer(EMB_NAME)
    reranker = CrossEncoder(RERANK_NAME)
    client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
    t0 = time.time()
    cols, corpus = build_index(docs, embedder, client)
    print(f"[index] done in {time.time()-t0:.0f}s")
    R = Retriever(cols, embedder, reranker)
    out = {"index": {n: len(c) for n, c in corpus.items()}, "samples": {}, "single": {}, "memory": {}, "perquery": {}}

    # ---- sample queries: without vs with reranking
    for qs in QUERIES[:3]:
        out["samples"][qs["id"]] = {"query": qs["q"]}
        for ch in CHUNKERS:
            for rr in (False, True):
                hits = R.search(qs["q"], ch, k=3, rerank=rr)
                out["samples"][qs["id"]][f"{ch}|{'rerank' if rr else 'dense'}"] = [
                    {"doc": h["doc"], "score": round(h["score"], 3),
                     "relevant": is_relevant(h["text"], h["doc"], qs), "text": snippet(h["text"], 260)} for h in hits]

    # ---- single-turn evaluation: 2 chunkers x {dense, rerank}
    for ch in CHUNKERS:
        for rr in (False, True):
            rows = []
            for qs in QUERIES:
                tot = n_relevant(corpus[ch], qs)
                hits = R.search(qs["q"], ch, k=10, rerank=rr)
                m = metrics(hits, qs, tot); m["n_rel_total"] = tot; rows.append(m)
                out["perquery"][f"{qs['id']}|{ch}|{'rerank' if rr else 'dense'}"] = m
            a = avg(rows); out["single"][f"{ch}|{'rerank' if rr else 'dense'}"] = a

    # ---- conversational memory: turn-2 of each dialog, with/without memory
    demo = []
    for ch in CHUNKERS:
        for rr in (False, True):
            for use_mem in (False, True):
                rows = []
                for dlg in DIALOGS:
                    mem = ConversationMemory(window=1)
                    for ti, q in enumerate(dlg["turns"]):
                        query = mem.contextualise(q) if use_mem else q
                        hits = R.search(query, ch, k=10, rerank=rr)
                        # (with memory the cross-encoder also scores against the contextualised query)
                        mem.add(q, hits)
                        if ti == 1:
                            spec = dlg["spec2"]; tot = n_relevant(corpus[ch], spec)
                            m = metrics(hits, spec, tot); rows.append(m)
                            out["perquery"][f"{dlg['id']}-t2|{ch}|{'rerank' if rr else 'dense'}|{'mem' if use_mem else 'nomem'}"] = m
                            if dlg["id"] in ("D1", "D2") and ch == "sentence_aware" and rr:
                                demo.append({"dialog": dlg["id"], "memory": use_mem, "turn1": dlg["turns"][0],
                                             "turn2": q, "sent_query": query,
                                             "top3": [{"doc": h["doc"], "relevant": is_relevant(h["text"], h["doc"], spec),
                                                       "text": snippet(h["text"], 200)} for h in hits[:3]]})
                out["memory"][f"{ch}|{'rerank' if rr else 'dense'}|{'mem' if use_mem else 'nomem'}"] = avg(rows)
    out["memory_demo"] = demo
    json.dump(out, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
    print("[done] results.json written")

if __name__ == "__main__":
    main()
