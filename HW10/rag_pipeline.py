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
        # single-threaded build + wider search beam => reproducible, near-exact ANN results across runs
        col = client.create_collection(name, metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1,
                                                        "hnsw:construction_ef": 200, "hnsw:search_ef": 200})
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
 # --- expansion set (S15-S40): one paraphrased question per additional paper
 dict(id="S15", q="How precisely can the decay constant of a charmed meson be measured from leptonic decays?",
      docs=[1], terms=[r"decay constant", r"leptonic", r"cleo", r"systematic|statistical"], min_hits=3),
 dict(id="S16", q="How does a moving random phase screen change the flicker of laser light after long propagation?",
      docs=[3], terms=[r"scintillation", r"phase (diffuser|screen)|diffuser", r"temporal|time", r"index|fluctuation"], min_hits=3),
 dict(id="S17", q="How can a nanoelectronic device mimic short-term and long-term synaptic memory?",
      docs=[7], terms=[r"synap", r"volatile|short.term", r"long.term|plasticity", r"neuromorphic|memristor|device"], min_hits=3),
 dict(id="S18", q="How can an object's pose be found in a 3D medical image in one shot using a shape prior?",
      docs=[8], terms=[r"b.scale|ball scale", r"model", r"pose|position|orientation", r"3d|image"], min_hits=3),
 dict(id="S19", q="Do stars with very eccentric giant planets also host further undiscovered planets?",
      docs=[9], terms=[r"eccentric", r"additional|extra|further|second planet", r"stab", r"hd\s*\d+|radial velocity"], min_hits=3),
 dict(id="S20", q="What forces determine the steady shapes of a swarm of interacting organisms?",
      docs=[10], terms=[r"swarm", r"equilibri", r"energy|minimi[sz]er", r"integral|density|continuum"], min_hits=3),
 dict(id="S21", q="How is the bound molecular state of two cold atoms near a magnetic resonance described analytically?",
      docs=[13], terms=[r"feshbach", r"binding energy|bound state", r"two.channel", r"wave ?function|scattering length"], min_hits=3),
 dict(id="S22", q="Can exploding stars be typed from their brightness curves instead of spectra?",
      docs=[14], terms=[r"supernova|\bsne?\b", r"classif", r"photometr|light curve", r"spectr"], min_hits=3),
 dict(id="S23", q="What is the scattering cross section between two bound electron-hole pairs at low energy?",
      docs=[16], terms=[r"exciton", r"scattering", r"cross section", r"monte carlo|phase shift"], min_hits=3),
 dict(id="S24", q="How can the skewness of primordial fluctuations be tracked using a transport equation?",
      docs=[17], terms=[r"non.gaussian", r"inflation", r"transport|moment", r"skewness|variance|super.horizon"], min_hits=3),
 dict(id="S25", q="What exotic decay channels does a light pseudoscalar Higgs open for the Z boson?",
      docs=[19], terms=[r"higgs", r"cp.odd|pseudoscalar", r"decay", r"ilc|gigaz|2hdm|doublet"], min_hits=3),
 dict(id="S26", q="How do pulsating stars help determine distances to nearby faint irregular galaxies?",
      docs=[21], terms=[r"variable|cepheid|rr lyr", r"dwarf", r"local group|leo a", r"distance"], min_hits=3),
 dict(id="S27", q="How does a pulsed particle or light beam make sound in water?",
      docs=[22], terms=[r"acoustic|pressure|sound", r"water", r"proton|laser|beam", r"pulse|thermoacoustic"], min_hits=3),
 dict(id="S28", q="How many very distant bright galaxies were found in archival Hubble pointings?",
      docs=[23], terms=[r"borg|dropout", r"galax", r"hubble|hst|wfc3", r"number density|z\s*~?\s*8|luminosity"], min_hits=3),
 dict(id="S29", q="How does magnetic flux appear in the quiet solar atmosphere at the smallest scales?",
      docs=[25], terms=[r"quiet sun|internetwork", r"flux", r"emergence|emerg", r"photosphere|chromosphere"], min_hits=3),
 dict(id="S30", q="How can dark energy be probed with diagnostic parameters in a geometry with torsion?",
      docs=[26], terms=[r"statefinder", r"torsion", r"acceler", r"cosmolog"], min_hits=3),
 dict(id="S31", q="How can qubits be protected when every qubit suffers the same error?",
      docs=[28], terms=[r"correlated", r"error correct|noise", r"qubit", r"codeword|encod"], min_hits=3),
 dict(id="S32", q="How can a solar eruption and the darkening it causes in extreme ultraviolet be modelled together?",
      docs=[29], terms=[r"cme|coronal mass", r"dimming", r"euv", r"model|expansion"], min_hits=3),
 dict(id="S33", q="How can journeys of transit riders between stations be inferred from counts and surveys?",
      docs=[32], terms=[r"origin.destination|\bod\b", r"passenger", r"station", r"survey|count"], min_hits=3),
 dict(id="S34", q="How does a toxic gas molecule stick to a copper monolayer in quantum chemistry simulations?",
      docs=[34], terms=[r"\bno\b|nitrogen monoxide", r"adsor", r"cu|copper", r"density functional|gga|generalized gradient"], min_hits=3),
 dict(id="S35", q="Why can the highest energy cosmic rays not simply be extragalactic protons?",
      docs=[35], terms=[r"cosmic ray", r"proton", r"cutoff|gzk|photopion", r"composition|ankle"], min_hits=3),
 dict(id="S36", q="What do the spectra of a dead star contaminated with rocky debris reveal?",
      docs=[41], terms=[r"white dwarf", r"calcium|magnesium|iron|metal", r"pollut|accret|debris", r"spectr"], min_hits=3),
 dict(id="S37", q="How do nuclear resonance spectra of an iron arsenide superconductor change with temperature?",
      docs=[43], terms=[r"nqr|nmr", r"superconduct|iron.based", r"relaxation|broadening|linewidth|spectr", r"temperature|transition|\bk\b"], min_hits=3),
 dict(id="S38", q="How old is a compact radio-loud galaxy as judged from its jet components?",
      docs=[45], terms=[r"1946", r"kinematic age|age", r"jet", r"vlba|vlbi|cso|compact symmetric"], min_hits=3),
 dict(id="S39", q="Can a controller update parameter estimates through their second derivative rather than their first?",
      docs=[53], terms=[r"tuner", r"second derivative", r"adaptive", r"stability|parameter estimat"], min_hits=3),
 dict(id="S40", q="How can a Carleman weight make the cost function convex when recovering a medium from wave data?",
      docs=[55], terms=[r"carleman", r"inverse", r"scattering", r"convex|cost functional|weight"], min_hits=3),
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
 dict(id="D7", turns=["How is primordial non-Gaussianity computed from super-horizon evolution during inflation?",
                      "What equations describe how its moments evolve?"],
      spec2=dict(docs=[17], terms=[r"moment|variance|skewness", r"evolution|transport", r"non.gaussian", r"equation"], min_hits=3)),
 dict(id="D8", turns=["How are two atoms near a Feshbach resonance described by a two-channel model?",
                      "How does its predicted binding energy compare with experiment?"],
      spec2=dict(docs=[13], terms=[r"binding energy", r"measure|experiment|agree", r"two.channel|model", r"feshbach"], min_hits=3)),
 dict(id="D9", turns=["What irregularities does the sunspot cycle show in amplitude and duration?",
                      "Why do the stronger ones rise faster?"],
      spec2=dict(docs=[57], terms=[r"waldmeier|rise time", r"strong", r"cycle", r"amplitude"], min_hits=3)),
 dict(id="D10", turns=["How does quantum error correction work for fully correlated noise?",
                       "How many data qubits can it encode?"],
      spec2=dict(docs=[28], terms=[r"qubit", r"encod|codeword", r"data qubit", r"correlated"], min_hits=3)),
 dict(id="D11", turns=["How was the coronal mass ejection launch modelled together with EUV dimming?",
                       "How many free parameters does it have and how were they constrained?"],
      spec2=dict(docs=[29], terms=[r"free parameter|parameters", r"fit|constrain|observ", r"model", r"cme|dimming"], min_hits=3)),
 dict(id="D12", turns=["What is the proposed origin of ultra high energy cosmic rays?",
                       "What cutoff does it predict and why is that a problem?"],
      spec2=dict(docs=[35], terms=[r"cutoff|gzk", r"photopion|microwave background", r"proton", r"cosmic ray"], min_hits=3)),
 dict(id="D13", turns=["What did VLBI observations reveal about the radio galaxy 1946+708?",
                       "How old is it estimated to be?"],
      spec2=dict(docs=[45], terms=[r"kinematic age|age", r"1946", r"jet", r"vlba|vlbi"], min_hits=3)),
 dict(id="D14", turns=["How does the adaptive tuner differ from standard parameter estimators?",
                       "How does its stability compare?"],
      spec2=dict(docs=[53], terms=[r"stab", r"tuner|second derivative", r"adaptive", r"performance|compar"], min_hits=3)),
 dict(id="D15", turns=["How does nitrogen monoxide adsorb on copper atomic structures?",
                       "Which sites does it prefer?"],
      spec2=dict(docs=[34], terms=[r"site|step", r"adsorb", r"\bno\b|nitrogen monoxide", r"cu|copper"], min_hits=3)),
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

    # ---- bootstrap 95% CIs over queries, and paired differences between configurations
    rng = np.random.default_rng(0)
    B = 5000
    def vec(prefix_ids, suffix, metric):
        return np.array([out["perquery"][f"{i}{suffix}"][metric] for i in prefix_ids])
    def ci(x):
        idx = rng.integers(0, len(x), (B, len(x)))
        bs = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    def dci(a, b):
        d = a - b; idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1)
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    sid = [q["id"] for q in QUERIES]; did = [d["id"] + "-t2" for d in DIALOGS]
    cfg = lambda ch, rr: f"|{ch}|{'rerank' if rr else 'dense'}"
    mcfg = lambda ch, rr, m: f"|{ch}|{'rerank' if rr else 'dense'}|{'mem' if m else 'nomem'}"
    out["ci"], out["delta"] = {}, {}
    for met in ("P@5", "R@5", "RR"):
        for ch in CHUNKERS:
            for rr in (False, True):
                out["ci"][f"single|{ch}|{'rerank' if rr else 'dense'}|{met}"] = ci(vec(sid, cfg(ch, rr), met))
                for m in (False, True):
                    out["ci"][f"memory|{ch}|{'rerank' if rr else 'dense'}|{'mem' if m else 'nomem'}|{met}"] = ci(vec(did, mcfg(ch, rr, m), met))
        for ch in CHUNKERS:   # rerank - dense
            out["delta"][f"rerank-dense|{ch}|{met}"] = dci(vec(sid, cfg(ch, True), met), vec(sid, cfg(ch, False), met))
        for rr in (False, True):  # sentence - fixed
            out["delta"][f"sentence-fixed|{'rerank' if rr else 'dense'}|{met}"] = dci(vec(sid, cfg("sentence_aware", rr), met), vec(sid, cfg("fixed_window", rr), met))
        for ch in CHUNKERS:   # memory - nomemory
            for rr in (False, True):
                out["delta"][f"mem-nomem|{ch}|{'rerank' if rr else 'dense'}|{met}"] = dci(vec(did, mcfg(ch, rr, True), met), vec(did, mcfg(ch, rr, False), met))
    out["n_queries"], out["n_dialogs"] = len(sid), len(did)
    json.dump(out, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
    print("[done] results.json written")

if __name__ == "__main__":
    main()
