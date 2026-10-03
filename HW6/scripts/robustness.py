"""Extra experiments (run after HW6.ipynb): (1) 5 seeds / 5 different 500-image subsets, (2) data-matched rotation SSL (same 20k images as SimCLR).
Reuses the helper cells of the executed notebook, loads the saved encoders from checkpoints/."""
import json, time, datetime, numpy as np, nbformat as nbf
nb = nbf.read("HW6.ipynb", 4)
g = {}
want = ["SID4 = 3215", "def load(split)", "def rand_resized_crop_flip", "def make_encoder"]
for c in nb.cells:
    if c.cell_type == "code" and any(w in c.source for w in want):
        src = c.source.split("# sanity picture")[0]           # skip the plotting at the end of the augmentation cell
        exec(src, g)
torch, F, nn, np = g["torch"], g["F"], g["nn"], g["np"]
stamp, set_seed, DEVICE = g["stamp"], g["set_seed"], g["DEVICE"]
make_encoder, norm, to_float, light_aug = g["make_encoder"], g["norm"], g["to_float"], g["light_aug"]
xtr, ytr, xte, yte, xun = g["xtr"], g["ytr"], g["xte"], g["yte"], g["xun"]
SEED = g["SEED"]

def subset(seed):
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.where(ytr.numpy()==c)[0], 50, replace=False) for c in range(10)]); rng.shuffle(idx)
    return xtr[idx], ytr[idx]

def load_enc(path):
    e = make_encoder(); e.load_state_dict(torch.load(path, map_location=DEVICE)); return e

def train_supervised(x_lab, y_lab, seed, EP=15, BS=64):
    set_seed(seed)
    class Clf(nn.Module):
        def __init__(s): super().__init__(); s.enc = make_encoder(); s.head = nn.Linear(512, 10)
        def forward(s, x): return s.head(s.enc(x))
    m = Clf().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP*int(np.ceil(500/BS)), pct_start=0.15)
    for ep in range(EP):
        m.train(); perm = torch.randperm(500)
        for i in range(0, 500, BS):
            idx = perm[i:i+BS]
            loss = F.cross_entropy(m(norm(light_aug(to_float(x_lab[idx])))), y_lab[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval(); acc, _ = g["accuracy_from_logits_fn"](m, xte, yte); return acc

# ---- (2) data-matched rotation SSL: same 20k images as SimCLR, 15 epochs, same recipe as Part B
simclr_idx = torch.from_numpy(np.random.RandomState(SEED).choice(len(xun), 20000, replace=False))
x20 = xun[simclr_idx]
set_seed(); enc_r20 = make_encoder(); head = nn.Linear(512, 4).to(DEVICE)
EP, BS = 15, 256; spe = len(x20)//BS
opt = torch.optim.AdamW(list(enc_r20.parameters())+list(head.parameters()), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP*spe)
for ep in range(EP):
    enc_r20.train(); perm = torch.randperm(len(x20)); tot = cor = n = 0
    for s in range(spe):
        xb = light_aug(to_float(x20[perm[s*BS:(s+1)*BS]])); k = torch.randint(0, 4, (len(xb),), device=DEVICE)
        for r in range(1, 4):
            m = k == r
            if m.any(): xb[m] = torch.rot90(xb[m], r, dims=(2, 3))
        lg = head(enc_r20(norm(xb))); loss = F.cross_entropy(lg, k)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()*len(xb); cor += (lg.argmax(1)==k).sum().item(); n += len(xb)
    stamp(f"rotation-20k pretext epoch {ep+1}/{EP} loss {tot/n:.4f} rot acc {cor/n*100:.2f}%")
torch.save(enc_r20.state_dict(), "checkpoints/encoder_rotation_20k.pt")

# ---- (1) five seeds / subsets
encs = {"B rotation (100k)": load_enc("checkpoints/encoder_rotation.pt"), "B' rotation (20k, data-matched)": enc_r20,
        "C SimCLR (20k, demo aug)": load_enc("checkpoints/encoder_simclr.pt"),
        "C' SimCLR (20k, strong crop, no hue; ablation)": load_enc("ablation_strongcrop/encoder_simclr_strongcrop.pt")}
R = {"seeds": [], "A": [], "random": []}; R.update({k: [] for k in encs})
for k in range(5):
    seed = SEED + k; R["seeds"].append(seed)
    g["x_lab"], g["y_lab"] = subset(seed)
    R["A"].append(train_supervised(g["x_lab"], g["y_lab"], seed)); stamp(f"seed {seed}: A supervised = {R['A'][-1]*100:.2f}%")
    for name, e in encs.items():
        R[name].append(g["linear_probe"](e, f"seed {seed} {name}", epochs=20)[0])
    set_seed(seed); R["random"].append(g["linear_probe"](make_encoder(), f"seed {seed} random", epochs=20)[0])
print("\nSUMMARY (test acc %, mean +- std over 5 seeds/subsets; seed 3215 is the one reported in the notebook)")
R["summary"] = {}
for k, v in R.items():
    if k in ("seeds", "summary"): continue
    a = np.array(v)*100; R["summary"][k] = {"mean": a.mean(), "std": a.std(ddof=1), "min": a.min(), "max": a.max()}
    print(f"{k:34s} {a.mean():6.2f} +- {a.std(ddof=1):4.2f}   (min {a.min():.2f}, max {a.max():.2f})   per-seed {[round(x,2) for x in a]}")
json.dump(R, open("outputs/robustness.json", "w"), indent=1)
