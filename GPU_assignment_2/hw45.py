"""HW4.5 BERT fine-tune harness (HW4.a settings: bert-base-uncased, MAX_LEN 256, bs 32, lr 2e-5, 3 epochs, no warmup, clip 1.0).

Data: Twitter_Data.csv, drop null category/clean_text, then rows [SLICE, SLICE+5000) of the cleaned frame,
split 70/15/15 with random_state=SEED.

Modes:
  bench : fixed number of timed steps at a batch size (Part A/B scaling); one JSON line to results.jsonl
  full  : full 3-epoch fine-tune with per-epoch metrics and test evaluation
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

SEED, SLICE, N_ROWS, MAX_LEN = 3215, 215, 5000, 256
CSV = os.environ.get("TWITTER_CSV", "Twitter_Data.csv")
CACHE = "data_cache.pt"


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_data():
    if os.path.exists(CACHE):
        return torch.load(CACHE, weights_only=False)  # our own cache file
    df = pd.read_csv(CSV)
    df = df[~df["category"].isnull()]
    df = df[~df["clean_text"].isnull()].reset_index(drop=True)
    first, last = SLICE, min(SLICE + N_ROWS, len(df))
    df = df.iloc[first:last].reset_index(drop=True)
    # category values are -1/0/1 -> labels 0/1/2 (LabelEncoder order, as in HW4.a)
    df["label"] = df["category"].astype(float).map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    tok = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    enc = tok(list(df.clean_text.astype(str)), add_special_tokens=True, max_length=MAX_LEN,
              padding="max_length", truncation=True, return_tensors="pt")
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED)
    d = dict(ids=enc["input_ids"], mask=enc["attention_mask"].float(), y=torch.tensor(df.label.values),
             tr=tr, va=va, te=te, text=list(df.clean_text.astype(str)), rows=(first, last - 1))
    torch.save(d, CACHE)
    return d


def batches(d, ix, bs, shuffle, gen=None):
    ix = np.array(ix)
    if shuffle:
        ix = ix[torch.randperm(len(ix), generator=gen).numpy()]
    for i in range(0, len(ix), bs):
        b = ix[i:i + bs]
        yield d["ids"][b], d["mask"][b], d["y"][b]


@torch.no_grad()
def evaluate(model, d, ix, bs=64, amp=False):
    model.eval(); P, Y, L = [], [], 0.0
    for x, m, y in batches(d, ix, bs, False):
        x, m, y = x.cuda(), m.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(x, attention_mask=m, labels=y)
        L += o.loss.item() * len(y); P.append(o.logits.argmax(-1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return L / len(Y), accuracy_score(Y, P), Y, P


def make_opt(model, a):
    if a.opt8:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=2e-5, eps=1e-8)
    return torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench"); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--amp", action="store_true"); p.add_argument("--ckpt", action="store_true")
    p.add_argument("--opt8", action="store_true"); p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tag", default="baseline"); p.add_argument("--out", default="results.jsonl")
    a = p.parse_args()

    # Windows (WDDM) silently spills VRAM into system RAM; cap the allocator so a true wall raises OOM instead.
    torch.cuda.set_per_process_memory_fraction(0.93)
    seed_all(); d = load_data()
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).cuda()
    if a.ckpt:
        model.gradient_checkpointing_enable()
    opt = make_opt(model, a)
    n_tr = len(d["tr"]); steps_per_epoch = (n_tr + a.bs - 1) // a.bs
    sched = get_linear_schedule_with_warmup(opt, 0, max(1, steps_per_epoch // a.accum) * a.epochs)
    gen = torch.Generator().manual_seed(SEED)
    cfg = dict(tag=a.tag, mode=a.mode, bs=a.bs, amp=a.amp, ckpt=a.ckpt, opt8=a.opt8, accum=a.accum,
               eff_bs=a.bs * a.accum, gpu=torch.cuda.get_device_name(0))

    def step_fn(x, m, y, last):
        x, m, y = x.cuda(), m.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            loss = model(x, attention_mask=m, labels=y).loss
        (loss / a.accum).backward()
        if last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        return loss.item()

    torch.cuda.reset_peak_memory_stats()
    if a.mode == "bench":
        try:
            model.train(); times = []
            for s in range(a.steps + 3):
                # random batch (with replacement) so bs may exceed the 3,500-sample train set
                b = d["tr"][torch.randint(0, n_tr, (a.bs,), generator=gen).numpy()]
                x, m, y = d["ids"][b], d["mask"][b], d["y"][b]
                torch.cuda.synchronize(); t = time.time()
                step_fn(x, m, y, True)
                torch.cuda.synchronize(); times.append(time.time() - t)
            t = np.mean(times[3:])  # skip warmup
            cfg.update(status="ok", samples_per_s=a.bs / t, step_s=t,
                       epoch_s=steps_per_epoch * t, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        except torch.cuda.OutOfMemoryError:
            cfg.update(status="oom")
    else:
        hist, t0 = [], time.time(); best = (-1, 0)
        for ep in range(1, a.epochs + 1):
            model.train(); tl, n = 0.0, 0
            for i, (x, m, y) in enumerate(batches(d, d["tr"], a.bs, True, gen)):
                tl += step_fn(x, m, y, ((i + 1) % a.accum == 0) or (i + 1) * a.bs >= n_tr) * len(y); n += len(y)
            vl, va, _, _ = evaluate(model, d, d["va"], amp=a.amp)
            hist.append(dict(epoch=ep, train_loss=tl / n, val_loss=vl, val_acc=va))
            if va > best[0]: best = (va, ep)
            print(hist[-1], flush=True)
        torch.cuda.synchronize(); wall = time.time() - t0
        tel, tea, Y, P = evaluate(model, d, d["te"], amp=a.amp)
        cm = confusion_matrix(Y, P, labels=[0, 1, 2])
        rep = classification_report(Y, P, target_names=["Negative", "Neutral", "Positive"], output_dict=True, zero_division=0)
        rec = [rep[c]["recall"] for c in ["Negative", "Neutral", "Positive"]]
        worst = int(np.argmin(rec)); te = np.array(d["te"])
        mis = [(d["text"][te[i]], int(Y[i]), int(P[i])) for i in range(len(Y)) if Y[i] == worst and P[i] != Y[i]][:5]
        cfg.update(status="ok", wall_s=wall, peak_gb=torch.cuda.max_memory_allocated() / 2**30, history=hist,
                   best_epoch=best[1], test_acc=tea, test_loss=tel, confusion=cm.tolist(), report=rep,
                   worst_class=worst, misclassified=mis, rows=d["rows"],
                   n=dict(train=len(d["tr"]), val=len(d["va"]), test=len(d["te"])),
                   samples_per_s=a.epochs * n_tr / wall)
        os.makedirs("model", exist_ok=True)
        if a.tag == "baseline":
            model.save_pretrained("model"); torch.save(model.state_dict(), "model/fineTuneModel.pt")
    with open(a.out, "a") as f:
        f.write(json.dumps(cfg) + "\n")
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("history", "report", "misclassified")}))


if __name__ == "__main__":
    main()
