"""Rubric audit. `python audit.py dump`  -> audit_items.json + audit_items.txt (blind: no rubric flag shown)
                `python audit.py score` -> reads audit_labels.json {item_id: 0/1}, writes audit_results.json
Pool: top-5 of fixed_window|dense and sentence_aware|rerank for a seeded random sample of 10 single-turn
queries, plus top-5 of sentence_aware|rerank|memory for 4 dialogue turn-2 queries."""
import json, os, sys, random
import numpy as np, chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import rag_pipeline as rp

ROOT = rp.ROOT
def setup():
    emb = SentenceTransformer(rp.EMB_NAME); rr = CrossEncoder(rp.RERANK_NAME)
    client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
    cols = {n: client.get_collection(n) for n in rp.CHUNKERS}
    return rp.Retriever(cols, emb, rr)

def pools():
    rnd = random.Random(1)
    singles = rnd.sample(rp.QUERIES, 10)
    dlgs = rnd.sample(rp.DIALOGS, 4)
    jobs = []
    for q in singles:
        jobs.append((q["id"], q, q["q"], [("fixed_window", False), ("sentence_aware", True)]))
    for d in dlgs:
        jobs.append((d["id"] + "-t2", d["spec2"], d["turns"][0] + " " + d["turns"][1], [("sentence_aware", True)]))
    return jobs

def dump():
    R = setup(); items = []
    for qid, spec, query, cfgs in pools():
        seen = {}
        for ch, rr in cfgs:
            for rank, h in enumerate(R.search(query, ch, k=5, rerank=rr), 1):
                seen.setdefault(h["id"], dict(item_id=f"{qid}::{h['id']}", qid=qid, query=query, chunk_id=h["id"],
                                              doc=h["doc"], text=h["text"], rubric=int(rp.is_relevant(h["text"], h["doc"], spec))))
        items += list(seen.values())
    json.dump(items, open(os.path.join(ROOT, "audit_items.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "audit_items.txt"), "w", encoding="utf-8") as f:   # blind view (no rubric flag)
        cur = None
        for i, it in enumerate(items):
            if it["qid"] != cur:
                cur = it["qid"]; f.write(f"\n##### {cur}: {it['query']}\n")
            f.write(f"[{i}] (doc {it['doc']}) {it['text']}\n\n")
    print(len(items), "items,", len({i['qid'] for i in items}), "queries")

def score():
    items = json.load(open(os.path.join(ROOT, "audit_items.json")))
    lab = json.load(open(os.path.join(ROOT, "audit_labels.json")))       # {index_as_str: 0/1}
    y = np.array([lab[str(i)] for i in range(len(items))]); r = np.array([it["rubric"] for it in items])
    tp = int(((y == 1) & (r == 1)).sum()); tn = int(((y == 0) & (r == 0)).sum())
    fp = int(((y == 0) & (r == 1)).sum()); fn = int(((y == 1) & (r == 0)).sum())
    po = (tp + tn) / len(y); pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / len(y) ** 2
    out = dict(n_items=len(y), n_queries=len({i["qid"] for i in items}), agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else 1.0,
               rubric_precision=tp / (tp + fp) if tp + fp else None, rubric_recall=tp / (tp + fn) if tp + fn else None,
               tp=tp, tn=tn, fp=fp, fn=fn, judged_relevant=int(y.sum()), rubric_relevant=int(r.sum()))
    # metrics on the audited pool under rubric vs judged labels (P@5 and RR of the two audited configs)
    R = setup(); per = []
    byid = {it["item_id"]: (lab[str(i)], it["rubric"]) for i, it in enumerate(items)}
    for qid, spec, query, cfgs in pools():
        for ch, rr in cfgs:
            hits = R.search(query, ch, k=5, rerank=rr)
            J = [byid[f"{qid}::{h['id']}"][0] for h in hits]; Rb = [byid[f"{qid}::{h['id']}"][1] for h in hits]
            rr_ = lambda v: next((1 / (i + 1) for i, x in enumerate(v) if x), 0.0)
            per.append(dict(qid=qid, cfg=f"{ch}|{'rerank' if rr else 'dense'}", P5_rubric=sum(Rb) / 5, P5_judged=sum(J) / 5,
                            RR_rubric=rr_(Rb), RR_judged=rr_(J)))
    out["per_config"] = per
    for cfg in sorted({p["cfg"] for p in per}):
        rows = [p for p in per if p["cfg"] == cfg]
        out[f"mean|{cfg}"] = {k: float(np.mean([p[k] for p in rows])) for k in ("P5_rubric", "P5_judged", "RR_rubric", "RR_judged")}
        out[f"mean|{cfg}"]["n"] = len(rows)
    json.dump(out, open(os.path.join(ROOT, "audit_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_config"}, indent=1))

if __name__ == "__main__":
    {"dump": dump, "score": score}[sys.argv[1]]()
