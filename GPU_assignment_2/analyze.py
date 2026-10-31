import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

scale = [json.loads(l) for l in open("scale.jsonl")]
full = {r["tag"]: r for r in map(json.loads, open("full.jsonl"))}
by = collections.defaultdict(dict)
for r in scale:
    by[r["tag"]][r["bs"]] = r
names = {"baseline": "Baseline (fp32)", "amp": "Mixed precision (bf16)", "ckpt": "Gradient checkpointing",
         "opt8": "8-bit AdamW", "combo": "AMP + ckpt + 8-bit"}

# Part A / B scaling figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t, rows in by.items():
    ok = sorted((b, r) for b, r in rows.items() if r["status"] == "ok")
    ax[0].plot([b for b, _ in ok], [r["samples_per_s"] for _, r in ok], "o-", label=names[t])
    ax[1].plot([b for b, _ in ok], [r["peak_gb"] for _, r in ok], "o-", label=names[t])
for a_, yl in zip(ax, ["samples / s", "peak allocated GB"]):
    a_.set_xscale("log", base=2); a_.set_xlabel("batch size"); a_.set_ylabel(yl); a_.grid(alpha=.3)
ax[1].axhline(24 * 0.93, ls="--", c="gray"); ax[0].legend(fontsize=7)
fig.suptitle("RTX 4090 (24 GB): throughput and peak memory vs batch size"); fig.tight_layout()
fig.savefig("fig_throughput_vs_batch.png", dpi=150)

print("### Scaling")
for t, rows in by.items():
    print(names[t])
    for b in sorted(rows):
        r = rows[b]
        print(f"  bs {b:5d} {r['status']:4s}", "" if r["status"] != "ok" else
              f"{r['samples_per_s']:7.1f} samp/s  {r['peak_gb']:5.2f} GB  epoch {r['epoch_s']:5.1f}s")

# Table HW4.5.1
def mx(t):
    ok = [b for b, r in by[t].items() if r["status"] == "ok"]
    return max(ok), by[t][max(ok)]
def at(t, b): return by[t][b]
print("\n### Table (throughput/memory at bs32 unless noted)")
base = at("baseline", 32)
rows = [("Baseline (HW4 settings)", "baseline", "baseline"), ("Mixed precision", "amp", "amp"),
        ("Gradient checkpointing", "ckpt", "ckpt"), ("Gradient accumulation (8x4)", None, "accum"),
        ("8-bit optimiser", "opt8", "opt8"), ("Best combination (AMP+8bit; +ckpt for max batch)", "combo", "combo")]
for lab, st, ft in rows:
    f = full.get(ft)
    acc = f["test_acc"] if f else None
    if st:
        m, mr = mx(st); r = at(st, 32)
        print(f"{lab} | max bs {m} | {r['samples_per_s']:.1f} | {r['peak_gb']:.2f} | acc {acc}")
    else:
        print(f"{lab} | eff 32 | wall {f['wall_s']:.1f}s | peak {f['peak_gb']:.2f} | acc {acc}")

print("\n### Ranking: GB saved per % throughput lost (bs32)")
for t in ["amp", "ckpt", "opt8"]:
    r = at(t, 32); saved = base["peak_gb"] - r["peak_gb"]; lost = (1 - r["samples_per_s"] / base["samples_per_s"]) * 100
    print(f"{names[t]}: saved {saved:.2f} GB ({saved/base['peak_gb']*100:.0f}%), throughput change {-lost:+.1f}%, "
          f"GB per %lost = {saved/lost if lost>0 else float('inf'):.2f}")
r = full["accum"]; print(f"accum: peak {r['peak_gb']:.2f} GB vs {base['peak_gb']:.2f}; wall {r['wall_s']:.1f}s vs {full['baseline']['wall_s']:.1f}s")

b = full["baseline"]
print("\n### baseline report"); print(json.dumps(b["report"], indent=1)); print(b["confusion"], b["best_epoch"], b["wall_s"], b["peak_gb"])
for m in b["misclassified"]: print(m)
print("\nfull-run wall/peak/acc:")
for t, f in full.items(): print(t, round(f["wall_s"], 1), round(f["peak_gb"], 2), f["test_acc"], [round(h["val_acc"], 3) for h in f["history"]])

# confusion matrix figure
import numpy as np
cm = np.array(b["confusion"]); fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(3)); ax.set_yticks(range(3))
labs = ["Neg", "Neu", "Pos"]; ax.set_xticklabels(labs); ax.set_yticklabels(labs)
for i in range(3):
    for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Baseline test confusion"); fig.tight_layout()
fig.savefig("fig_confusion.png", dpi=150)
