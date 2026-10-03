import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("outputs/robustness.json"))
names = [("A", "A supervised\n(500 labels)"), ("B rotation (100k)", "B rotation\n(100k imgs)"), ("B' rotation (20k, data-matched)", "B' rotation\n(20k imgs)"),
         ("C SimCLR (20k, demo aug)", "C SimCLR\n(demo aug)"), ("C' SimCLR (20k, strong crop, no hue; ablation)", "C' SimCLR\n(strong crop,\nablation)"), ("random", "random\nencoder")]
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (k, lab) in enumerate(names):
    v = np.array(R[k]) * 100
    ax.bar(i, v.mean(), yerr=v.std(ddof=1), capsize=4, color="#9bb7d4" if "C'" not in lab and "random" not in lab else ("#d9a066" if "C'" in lab else "#bbbbbb"))
    ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color="k", s=10, zorder=3)
    ax.text(i, v.mean() + v.std(ddof=1) + 1.2, f"{v.mean():.1f}", ha="center", fontsize=9)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[1] for n in names], fontsize=8); ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 60)
ax.set_title("5 different 500-image subsets / seeds (bar = mean, whisker = std, dots = runs)", fontsize=9); plt.tight_layout(); plt.savefig("outputs/robustness.png", dpi=130)
