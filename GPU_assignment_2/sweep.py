"""Part A/B sweeps. Each config runs in its own process so an OOM cannot poison the next run."""
import subprocess, sys, json

PY = sys.executable
CONFIGS = {  # name -> extra flags
    "baseline": [], "amp": ["--amp"], "ckpt": ["--ckpt"], "opt8": ["--opt8"],
    "combo": ["--amp", "--ckpt", "--opt8"],
}


def run(args):
    r = subprocess.run([PY, "hw45.py"] + args, capture_output=True, text=True)
    open("RUN_LOG.txt", "a").write(f"$ hw45.py {' '.join(args)}\n{r.stdout[-1500:]}\n{r.stderr[-300:] if r.returncode else ''}\n")
    for line in reversed(r.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"status": "crash"}


if __name__ == "__main__":
    for name, flags in CONFIGS.items():
        bs = 8
        while bs <= 4096:
            res = run(["--mode", "bench", "--bs", str(bs), "--tag", name, "--out", "scale.jsonl"] + flags)
            print(name, bs, res.get("status"), res.get("samples_per_s"), res.get("peak_gb"), flush=True)
            if res.get("status") != "ok":
                break
            bs *= 2
    # full runs at the HW4 settings (bs 32) for accuracy / wall time
    full = {"amp": ["--amp"], "ckpt": ["--ckpt"], "opt8": ["--opt8"], "combo": ["--amp", "--opt8"],
            "accum": ["--bs", "8", "--accum", "4"]}
    for name, flags in full.items():
        res = run(["--mode", "full", "--tag", name, "--out", "full.jsonl"] + flags)
        print(name, res.get("test_acc"), res.get("wall_s"), res.get("peak_gb"), flush=True)
