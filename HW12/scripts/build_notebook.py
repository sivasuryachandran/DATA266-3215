"""Builds HW12.ipynb (the assignment's "HW11": GPTQ + knowledge distillation). Run: python scripts/build_notebook.py"""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# HW11 - GPTQ quantization and knowledge distillation (submitted in the HW12 folder)

Run on a CUDA GPU (Colab T4 or better). Part 1 and Part 2 use different models and tasks and are kept separate.
Cell 1 states the personal parameters (Standing Requirement 0.1). Every number in `METRICS.md` is written by the last code cell from the saved results.
""")

code(r'''
# ---- Cell 0: install (run once). It restarts the runtime when done; then run from Cell 1 on. ----
# gptqmodel may replace torch with a newer CUDA build; Colab's preinstalled torchaudio/torchvision then mismatch
# and break transformers' imports. This notebook needs neither, so remove them.
%pip uninstall -y -q torchaudio torchvision
%pip install -q "transformers>=4.45" datasets accelerate optimum gptqmodel matplotlib scikit-learn
import os; os._exit(0)   # forces a runtime restart so the new packages are picked up (Colab shows "session crashed": expected)
''')

code(r'''
# ---- Cell 1: personal parameters (Standing Requirements 0.1) + setup ----
import os, sys, time, json, random, shutil, datetime, platform
import numpy as np, torch

SID4 = 3215
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10
print(f"SID4={SID4} SEED={SEED} SLICE={SLICE} HP_ID={HP_ID} CLS_A={CLS_A} CLS_B={CLS_B}")
# HW11 uses only SEED; HP_ID adds no extra temperature (assignment text).

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

assert torch.cuda.is_available(), "This notebook needs a CUDA GPU (latency / peak-memory / GPTQ kernels)."
DEV = "cuda:0"
OUT, FIG = "outputs", "figures"
for d in (OUT, FIG): os.makedirs(d, exist_ok=True)

LOG_PATH = "RUN_LOG.txt"
def log(*a):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] " + " ".join(str(x) for x in a)
    print(line)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

import transformers, datasets
log("=== HW11 run start ===")
log(f"SID4={SID4} SEED={SEED} SLICE={SLICE} HP_ID={HP_ID} CLS_A={CLS_A} CLS_B={CLS_B}")
log(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} transformers={transformers.__version__} python={platform.python_version()}")

# ---- configuration (edit here only) ----
LLM_ID = "facebook/opt-125m"      # <- replace with the instructor-provided/approved base LLM
TEACHER_DIR = "teacher_model"     # <- folder with the fine-tuned BERT teacher
TEACHER_REPO = None               # <- set to the git URL of the repo that holds teacher_model/ (cloned automatically if the folder is missing)
STUDENT_ID = "distilbert/distilbert-base-uncased"
TEMPS = [1, 2, 4, 8]              # temperature sweep (see README: values were lost in the pasted brief)
ALPHAS = [0.3, 0.5, 0.7]          # weight on cross-entropy; KL weight = 1 - alpha
SWEEP_ALPHA = 0.5                 # alpha held fixed during the temperature sweep
EPOCHS, BATCH, LR, MAXLEN, VAL_N = 2, 64, 5e-5, 128, 5000
''')

md("""
## Part 1 - GPTQ: base vs 4-bit quantized LLM (Wikitext2)
Same base model for both runs; Wikitext2 train is used for calibration only, Wikitext2 test for perplexity.
""")

code(r'''
# ---- Part 1a: model revision, data, calibration, helpers ----
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset

LLM_REV = HfApi().model_info(LLM_ID).sha          # exact revision, pinned for both runs
log(f"Base LLM: {LLM_ID} revision={LLM_REV}")
tok = AutoTokenizer.from_pretrained(LLM_ID, revision=LLM_REV)
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
SEQ = 2048 if "opt" in LLM_ID else 1024   # window for perplexity evaluation
CALIB = 1024    # calibration length: decode->re-tokenize adds BOS, so 2048-token windows overflow OPT's 2048 positions (CUDA assert)

# calibration: 128 random SEQ-token windows from the TRAIN split (seeded)
train_ids = tok("\n\n".join(wt["train"]["text"]), return_tensors="pt").input_ids[0]
g = np.random.RandomState(SEED)
starts = g.randint(0, len(train_ids) - CALIB - 1, size=128)
calib_texts = [tok.decode(train_ids[s:s+CALIB]) for s in starts]
test_ids = tok("\n\n".join(wt["test"]["text"]), return_tensors="pt").input_ids
log(f"calibration windows=128 x {CALIB} tokens; Wikitext2 test tokens={test_ids.shape[1]}")

@torch.no_grad()
def wikitext_ppl(model):
    model.eval(); nll, n = 0.0, 0
    for i in range(0, test_ids.shape[1] - 1, SEQ):
        ids = test_ids[:, i:i+SEQ].to(DEV)
        if ids.shape[1] < 2: break
        loss = model(ids, labels=ids).loss
        nll += loss.item() * (ids.shape[1] - 1); n += ids.shape[1] - 1
    return float(np.exp(nll / n)), nll / n

def bench(fn, warmup=10, runs=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))

def dir_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6

def measure_llm(model, name, path):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ppl, nll = wikitext_ppl(model)
    prompt = test_ids[:, :128].to(DEV)
    gen = lambda: model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=64, min_new_tokens=64, do_sample=False)
    pre = lambda: model(test_ids[:, :512].to(DEV))
    g50, g95 = bench(gen, 5, 50)
    p50, p95 = bench(pre, 10, 100)
    r = dict(name=name, ppl=ppl, nll=nll, gen_p50_ms=g50, gen_p95_ms=g95, prefill_p50_ms=p50, prefill_p95_ms=p95,
             peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, size_mb=dir_mb(path))
    log(name, json.dumps(r)); return r
''')

code(r'''
# ---- Part 1b: base (fp16) measurement ----
BASE_DIR, Q_DIR = "ckpt/llm_base", "ckpt/llm_gptq4"
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
base = AutoModelForCausalLM.from_pretrained(LLM_ID, revision=LLM_REV, dtype=torch.float16).to(DEV)
base.save_pretrained(BASE_DIR)                       # fp16 safetensors -> on-disk size
res_base = measure_llm(base, "Base LLM (fp16)", BASE_DIR)
del base; torch.cuda.empty_cache()
''')

code(r'''
# ---- Part 1c: GPTQ quantization (4-bit, group 128), then reload from disk and measure ----
set_seed()
t0 = time.time()
qcfg = GPTQConfig(bits=4, group_size=128, dataset=calib_texts, tokenizer=tok, desc_act=False, model_seqlen=CALIB)
qm = AutoModelForCausalLM.from_pretrained(LLM_ID, revision=LLM_REV, quantization_config=qcfg,
                                          dtype=torch.float16, device_map=DEV)
quant_s = time.time() - t0
qm.save_pretrained(Q_DIR); tok.save_pretrained(Q_DIR)
log(f"GPTQ quantization time {quant_s:.0f}s")
del qm; torch.cuda.empty_cache()
qm = AutoModelForCausalLM.from_pretrained(Q_DIR, dtype=torch.float16, device_map=DEV)
res_q = measure_llm(qm, "GPTQ 4-bit LLM", Q_DIR); res_q["quant_time_s"] = quant_s
del qm; torch.cuda.empty_cache()
json.dump(dict(model=LLM_ID, revision=LLM_REV, base=res_base, gptq=res_q), open(f"{OUT}/part1.json", "w"), indent=1)
import pandas as pd
display(pd.DataFrame([res_base, res_q]).set_index("name").round(3))
''')

md("""
### Part 1 findings
*(Write after the run, from the table above: perplexity change, p50/p95 generation and prefill latency, peak memory, on-disk size, and why latency did or did not improve for a small model on this GPU.)*
""")

md("""
## Part 2 - Knowledge distillation on AG News (BERT teacher -> DistilBERT student)
Teacher logits are extracted for **all 120,000** training samples. To select T and alpha without touching the test set, 5,000 training samples (seeded) are held out as validation; the students train on the remaining 115,000 using the cached teacher logits. Test accuracy is reported for every run, but selection uses validation accuracy.
""")

code(r'''
# ---- Part 2a: data + teacher logits for ALL training samples ----
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
ag = load_dataset("fancyzhx/ag_news"); CLASSES = ag["train"].features["label"].names
tr_text, tr_y = list(ag["train"]["text"]), np.array(ag["train"]["label"])
te_text, te_y = list(ag["test"]["text"]), np.array(ag["test"]["label"])
log(f"AG News train={len(tr_text)} test={len(te_text)} classes={CLASSES}")

def encode(tk, texts):
    return tk(texts, truncation=True, max_length=MAXLEN, return_token_type_ids=False)["input_ids"]

def collate_ids(tk):
    def f(batch):
        ids = [b[0] for b in batch]
        out = tk.pad({"input_ids": ids}, return_tensors="pt")
        return out["input_ids"], out["attention_mask"], torch.tensor([b[1] for b in batch])
    return f

if not os.path.isfile(os.path.join(TEACHER_DIR, "config.json")) and TEACHER_REPO:
    import subprocess, glob   # clone the repo (git-lfs needed for large weights) and locate teacher_model/
    subprocess.run(["git", "lfs", "install"], check=False); subprocess.run(["git", "clone", "--depth", "1", TEACHER_REPO, "teacher_repo"], check=True)
    hits = glob.glob("teacher_repo/**/teacher_model", recursive=True)
    if hits: TEACHER_DIR = hits[0]
    log("teacher_model located at", TEACHER_DIR, os.listdir(TEACHER_DIR))
if not os.path.isfile(os.path.join(TEACHER_DIR, "config.json")):
    raise FileNotFoundError(f"'{TEACHER_DIR}/config.json' not found on this machine. Upload the instructor's teacher_model folder "
        "to the Colab filesystem (Files panel -> upload, or drive.mount + copy) and set TEACHER_DIR in cell 1 to its path, then re-run from here.")
t_tok = AutoTokenizer.from_pretrained(TEACHER_DIR)
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_DIR).to(DEV).eval()
log("teacher:", teacher.config.architectures, "params", sum(p.numel() for p in teacher.parameters()))

@torch.no_grad()
def predict_logits(model, tk, texts, bs=256):
    ids = encode(tk, texts); order = np.argsort([len(x) for x in ids]); out = np.zeros((len(ids), 4), np.float32)
    for i in range(0, len(ids), bs):
        idx = order[i:i+bs]
        b = tk.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = model(**b).logits.float().cpu().numpy()
    return out

t0 = time.time()
T_train_logits = predict_logits(teacher, t_tok, tr_text)
T_test_logits = predict_logits(teacher, t_tok, te_text)
np.save(f"{OUT}/teacher_train_logits.npy", T_train_logits)
tacc = float((T_test_logits.argmax(1) == te_y).mean())
log(f"teacher logits extracted for {len(tr_text)} train samples in {time.time()-t0:.0f}s; teacher test acc={tacc:.4f}; train acc={(T_train_logits.argmax(1)==tr_y).mean():.4f}")
assert tacc > 0.85, "Teacher label order/tokenizer mismatch? accuracy is too low."
''')

code(r'''
# ---- Part 2b: distillation loss + training function ----
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
s_tok = AutoTokenizer.from_pretrained(STUDENT_ID)
s_tr_ids, s_te_ids = encode(s_tok, tr_text), encode(s_tok, te_text)

perm = np.random.RandomState(SEED).permutation(len(tr_text))
val_idx, fit_idx = perm[:VAL_N], perm[VAL_N:]

def distill_loss(s_logits, t_logits, y, T, alpha):
    ce = F.cross_entropy(s_logits, y)
    kd = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * (T * T)
    return alpha * ce + (1 - alpha) * kd, ce, kd

@torch.no_grad()
def eval_ids(model, ids, y, bs=256):
    model.eval(); order = np.argsort([len(x) for x in ids]); pred = np.zeros(len(ids), int)
    for i in range(0, len(ids), bs):
        idx = order[i:i+bs]
        b = s_tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            pred[idx] = model(**b).logits.argmax(-1).cpu().numpy()
    return float((pred == y).mean()), pred

def train_student(T, alpha, tag):
    set_seed()                                           # same seed -> same init of the classifier head and same batch order
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_ID, num_labels=4).to(DEV)   # fresh from pretrained
    ds = [(s_tr_ids[i], int(tr_y[i]), T_train_logits[i]) for i in fit_idx]
    def coll(batch):
        o = s_tok.pad({"input_ids": [b[0] for b in batch]}, return_tensors="pt")
        return o["input_ids"], o["attention_mask"], torch.tensor([b[1] for b in batch]), torch.tensor(np.stack([b[2] for b in batch]))
    gen = torch.Generator().manual_seed(SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=coll, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = EPOCHS * len(dl); sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(); hist = []; step = 0; t0 = time.time()
    run = dict(tot=0., ce=0., kd=0., n=0)
    for ep in range(EPOCHS):
        model.train()
        for ids, am, y, tl in dl:
            ids, am, y, tl = ids.to(DEV), am.to(DEV), y.to(DEV), tl.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                sl = model(input_ids=ids, attention_mask=am).logits
            loss, ce, kd = distill_loss(sl.float(), tl, y, T, alpha)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); step += 1
            for k, v in (("tot", loss), ("ce", ce), ("kd", kd)): run[k] += v.item()
            run["n"] += 1
            if step % 200 == 0:
                rec = dict(step=step, epoch=ep + step / len(dl) - ep, loss=run["tot"]/run["n"], ce=run["ce"]/run["n"], kd=run["kd"]/run["n"])
                hist.append(rec); run = dict(tot=0., ce=0., kd=0., n=0)
                if step % 1000 == 0: log(f"[{tag}] step {step}/{total} loss={rec['loss']:.4f} ce={rec['ce']:.4f} kd={rec['kd']:.4f}")
        vacc, _ = eval_ids(model, [s_tr_ids[i] for i in val_idx], tr_y[val_idx])
        log(f"[{tag}] epoch {ep+1}/{EPOCHS} val_acc={vacc:.4f}")
    tacc_s, _ = eval_ids(model, s_te_ids, te_y)
    r = dict(tag=tag, T=T, alpha=alpha, val_acc=vacc, test_acc=tacc_s, train_s=time.time() - t0, history=hist)
    log(f"[{tag}] DONE T={T} alpha={alpha} val_acc={vacc:.4f} test_acc={tacc_s:.4f} time={r['train_s']:.0f}s")
    return model, r

BEST = dict(val=-1)
def run_one(T, alpha, tag):
    global BEST
    model, r = train_student(T, alpha, tag)
    if r["val_acc"] > BEST["val"]:                       # keep only the best-by-validation student on disk
        shutil.rmtree("ckpt/student_best", ignore_errors=True); model.save_pretrained("ckpt/student_best"); s_tok.save_pretrained("ckpt/student_best")
        BEST = dict(val=r["val_acc"], T=T, alpha=alpha, tag=tag)
    del model; torch.cuda.empty_cache(); return r
''')

code(r'''
# ---- Part 2c: temperature sweep (alpha fixed), each student re-initialised from the same checkpoint ----
sweepT = [run_one(T, SWEEP_ALPHA, f"T={T},a={SWEEP_ALPHA}") for T in TEMPS]
bestT = max(sweepT, key=lambda r: r["val_acc"])["T"]
log(f"best temperature by validation accuracy: T={bestT}")
display(pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in sweepT]).round(4))
''')

code(r'''
# ---- Part 2d: loss-weighting sweep at the best temperature ----
sweepA = [run_one(bestT, a, f"T={bestT},a={a}") for a in ALPHAS]
bestA = max(sweepA, key=lambda r: r["val_acc"])["alpha"]
log(f"best alpha at T={bestT}: {bestA}; overall best student: {BEST}")
display(pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in sweepA]).round(4))
json.dump(dict(sweepT=sweepT, sweepA=sweepA, best=BEST, teacher_test_acc=tacc), open(f"{OUT}/part2_sweeps.json", "w"), indent=1)
''')

code(r'''
# ---- Part 2e: sample training log, loss trends, sweep figures ----
import matplotlib; import matplotlib.pyplot as plt
print("Sample training log (first 6 logged windows of the first run):")
for h in sweepT[0]["history"][:6]: print({k: round(v, 4) for k, v in h.items()})
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for r in sweepT + sweepA:
    st = [h["step"] for h in r["history"]]
    ax[0].plot(st, [h["loss"] for h in r["history"]], label=r["tag"]); ax[1].plot(st, [h["ce"] for h in r["history"]]); ax[2].plot(st, [h["kd"] for h in r["history"]])
for a_, t_ in zip(ax, ["total loss", "CE (hard labels)", "KL x T^2 (teacher)"]): a_.set_title(t_); a_.set_xlabel("step")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{FIG}/loss_trends.png", dpi=130); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].bar([str(r["T"]) for r in sweepT], [r["test_acc"] for r in sweepT]); ax[0].set_title(f"Test acc vs T (alpha={SWEEP_ALPHA})")
ax[1].bar([str(r["alpha"]) for r in sweepA], [r["test_acc"] for r in sweepA]); ax[1].set_title(f"Test acc vs alpha (T={bestT})")
for a_ in ax: a_.set_ylim(min(r["test_acc"] for r in sweepT + sweepA) - 0.01, None)
plt.tight_layout(); plt.savefig(f"{FIG}/sweeps.png", dpi=130); plt.show()
''')

code(r'''
# ---- Part 2f: teacher vs best student - accuracy, latency, memory, size (same test set, same GPU) ----
def bench_cls(model, tk, texts, name, path):
    ids = encode(tk, texts); model.eval()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    acc = None; pred = np.zeros(len(ids), int); order = np.argsort([len(x) for x in ids])
    with torch.no_grad():
        for i in range(0, len(ids), 128):
            idx = order[i:i+128]; b = tk.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(DEV)
            pred[idx] = model(**b).logits.argmax(-1).cpu().numpy()          # fp32 inference for fairness
    peak = torch.cuda.max_memory_allocated() / 1e9
    one = [tk.pad({"input_ids": [ids[j]]}, return_tensors="pt").to(DEV) for j in range(200)]
    k = iter(range(10**9))
    with torch.no_grad(): p50, p95 = bench(lambda: model(**one[next(k) % 200]), 20, 200)
    r = dict(name=name, acc=float((pred == te_y).mean()), p50_ms=p50, p95_ms=p95, peak_mem_gb=peak, size_mb=dir_mb(path),
             params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    log(name, json.dumps(r)); return r, pred

teacher.save_pretrained("ckpt/teacher_copy")
rt, pred_t = bench_cls(teacher.float(), t_tok, te_text, "Teacher (BERT-base)", "ckpt/teacher_copy")
student = AutoModelForSequenceClassification.from_pretrained("ckpt/student_best").to(DEV)
rs, pred_s = bench_cls(student.float(), s_tok, te_text, f"Best distilled student (T={BEST['T']}, alpha={BEST['alpha']})", "ckpt/student_best")
json.dump(dict(teacher=rt, student=rs), open(f"{OUT}/part2_final.json", "w"), indent=1)
display(pd.DataFrame([rt, rs]).set_index("name").round(4))
''')

code(r'''
# ---- Part 2g: where compression hurts - per-class accuracy and 5 teacher-right / student-wrong examples ----
from sklearn.metrics import confusion_matrix
pc = lambda p: np.array([(p[te_y == c] == c).mean() for c in range(4)])
at, as_ = pc(pred_t), pc(pred_s); drop = at - as_; worst = int(drop.argmax())
display(pd.DataFrame({"teacher": at, "student": as_, "drop": drop}, index=CLASSES).round(4))
log(f"largest degradation: class '{CLASSES[worst]}' (teacher {at[worst]:.4f} -> student {as_[worst]:.4f})")
cand = np.where((pred_t == te_y) & (pred_s != te_y) & (te_y == worst))[0]
pick = np.random.RandomState(SEED).choice(cand, size=min(5, len(cand)), replace=False)
examples = [dict(idx=int(i), true=CLASSES[te_y[i]], student_pred=CLASSES[pred_s[i]], n_words=len(te_text[i].split()), text=te_text[i]) for i in pick]
for e in examples: print(f"[{e['true']} -> student said {e['student_pred']}] ({e['n_words']} words) {e['text']}\n")
cm = confusion_matrix(te_y[(pred_t == te_y) & (pred_s != te_y)], pred_s[(pred_t == te_y) & (pred_s != te_y)], labels=range(4))
print("teacher-right/student-wrong confusion (rows=true, cols=student pred):\n", pd.DataFrame(cm, CLASSES, CLASSES))
allw = (pred_t == te_y) & (pred_s != te_y)
print("mean words: flipped examples %.1f vs all test %.1f" % (np.mean([len(te_text[i].split()) for i in np.where(allw)[0]]), np.mean([len(t.split()) for t in te_text])))
print("number of flipped (teacher right, student wrong):", int(allw.sum()), "| reverse (student right, teacher wrong):", int(((pred_s == te_y) & (pred_t != te_y)).sum()))
json.dump(dict(per_class=dict(zip(CLASSES, zip(at.tolist(), as_.tolist()))), worst=CLASSES[worst], examples=examples), open(f"{OUT}/part2_errors.json", "w"), indent=1)
''')

md("""
### Part 2 findings, error analysis and compression comparison
*(Write after the run, from the outputs above: best T and alpha and what the sweeps show; student vs teacher on accuracy, latency, memory and size; the class that degrades most and what the five examples have in common; then four to six sentences comparing quantization with distillation - do not compare accuracy values across the two tasks - and which technique you would try first for a latency-bound and for a memory-bound deployment.)*
""")

code(r'''
# ---- Final: write METRICS.md from saved results ----
p1, p2 = json.load(open(f"{OUT}/part1.json")), json.load(open(f"{OUT}/part2_final.json"))
sw, er = json.load(open(f"{OUT}/part2_sweeps.json")), json.load(open(f"{OUT}/part2_errors.json"))
b, q, t, s = p1["base"], p1["gptq"], p2["teacher"], p2["student"]
L = [f"# HW11 metrics (GPU: {torch.cuda.get_device_name(0)}; SID4={SID4}, SEED={SEED})", "",
     f"Part 1 model: `{p1['model']}` revision `{p1['revision']}`; GPTQ 4-bit, group 128, 128 Wikitext2-train calibration windows. Quality metric = Wikitext2-test perplexity. Latency = 64 new tokens, batch 1 (generation) and 512-token forward (prefill).", "",
     "**Table HW11.1 - GPTQ comparison**", "",
     "| Configuration | Wikitext2 perplexity | p50 / p95 latency, generate-64 (ms) | p50 / p95 prefill-512 (ms) | Peak memory (GB) | Size on disk (MB) |", "|---|---:|---:|---:|---:|---:|"]
for r in (b, q): L.append(f"| {r['name']} | {r['ppl']:.3f} | {r['gen_p50_ms']:.1f} / {r['gen_p95_ms']:.1f} | {r['prefill_p50_ms']:.1f} / {r['prefill_p95_ms']:.1f} | {r['peak_mem_gb']:.3f} | {r['size_mb']:.1f} |")
L += ["", "**Table HW11.2 - distillation comparison (AG News test, 7,600 samples)**", "",
      "| Configuration | Accuracy | p50 / p95 latency, batch 1 (ms) | Peak memory (GB) | Size on disk (MB) |", "|---|---:|---:|---:|---:|"]
for r in (t, s): L.append(f"| {r['name']} | {r['acc']:.4f} | {r['p50_ms']:.2f} / {r['p95_ms']:.2f} | {r['peak_mem_gb']:.3f} | {r['size_mb']:.1f} |")
L += ["", f"Temperature sweep (alpha={SWEEP_ALPHA}); selection by validation accuracy on 5,000 held-out train samples:", "", "| T | val acc | test acc |", "|---:|---:|---:|"]
L += [f"| {r['T']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} |" for r in sw["sweepT"]]
L += ["", f"Loss-weighting sweep (T={sw['sweepA'][0]['T']}; alpha = weight on cross-entropy):", "", "| alpha | val acc | test acc |", "|---:|---:|---:|"]
L += [f"| {r['alpha']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} |" for r in sw["sweepA"]]
L += ["", "Per-class test accuracy (teacher / student):", "", "| Class | Teacher | Student |", "|---|---:|---:|"]
L += [f"| {c} | {v[0]:.4f} | {v[1]:.4f} |" for c, v in er["per_class"].items()]
open("METRICS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
log("=== HW11 run end ===")
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, Path(__file__).resolve().parent.parent / "HW12.ipynb")
print("wrote HW12.ipynb with", len(cells), "cells")
