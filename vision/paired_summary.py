"""Merge fins_cifar.py result files of the same seeds and compare arms seed by seed.

Runs with --paired reseed before every arm, so arms from separate runs with the same seeds, training-set
size and base augmentation are paired as if they had run together. Prints mean +/- std per arm and
corruption, and the reference arm minus every other arm on the average corruption accuracy, per seed.

Usage:
    python paired_summary.py cifar_n50000_nulls.json cifar_n50000_baselines.json [--ref fins] [--detail null_init]
"""
import argparse
import json
import os

import numpy as np

CORR = ["gaussian_noise", "shot_noise", "impulse_noise", "gaussian_blur", "pixelate", "jpeg", "contrast", "brightness"]
KEYS = ("seed_start", "cifar_train_size", "sym_gen_subset", "paired", "base_aug", "epochs_resnet", "n_aug")


def load(path):
    with open(path) as f:
        runs = json.load(f)
    cfg_path = os.path.splitext(path)[0] + "_config.json"
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    return runs, cfg


def avg(r):
    return float(np.mean([r[c] for c in CORR]))


def main():
    ap = argparse.ArgumentParser(description="Paired comparison of fins_cifar.py result files")
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--ref", default="fins", help="arm compared against every other")
    ap.add_argument("--detail", default=None, help="arm for a per-corruption paired comparison with --ref")
    args = ap.parse_args()

    loaded = [load(p) for p in args.jsons]
    cfgs = [c for _, c in loaded if c is not None]
    for k in KEYS:
        vals = {json.dumps(c.get(k)) for c in cfgs}
        if len(vals) > 1:
            print(f"WARNING: runs differ in {k}: {sorted(vals)}")
    if len(cfgs) < len(loaded):
        print("WARNING: config sidecar missing for some runs; seeds and settings not checked")
    if not all(c.get("paired") for c in cfgs):
        print("WARNING: not every run used --paired; arms from different runs are not paired")
    n = min(len(r) for r, _ in loaded)
    if any(len(r) != n for r, _ in loaded):
        print(f"WARNING: runs have different numbers of seeds; using the first {n}")
    merged = [{} for _ in range(n)]
    for runs, _ in loaded:
        for i in range(n):
            for arm, res in runs[i].items():
                if arm in merged[i] and merged[i][arm] != res:
                    print(f"WARNING: arm {arm} present in several runs with different results; keeping the first")
                merged[i].setdefault(arm, res)
    arms = [a for a in merged[0] if all(a in m for m in merged)]
    start = cfgs[0].get("seed_start") if cfgs else None
    print(f"{n} seeds" + (f" from {start}" if start is not None else "") + f"; arms: {', '.join(arms)}")

    cols = ["clean"] + CORR + ["avg_corrupt"]
    print(f"\n{'corruption':<16s}" + "".join(f"{a:>16s}" for a in arms))
    for c in cols:
        vals = {a: [avg(m[a]) if c == "avg_corrupt" else m[a][c] for m in merged] for a in arms}
        best = max(np.mean(v) for v in vals.values())
        print(f"{c:<16s}" + "".join(f"{np.mean(v):>9.3f}+/-{np.std(v):.3f}{'*' if np.mean(v) >= best - 0.005 else ' '}"
                                    for v in vals.values()))

    if args.ref in arms:
        print(f"\n{args.ref} minus arm, average corruption accuracy per seed; mean +/- std; {args.ref} ahead on")
        for a in arms:
            if a == args.ref:
                continue
            d = [avg(m[args.ref]) - avg(m[a]) for m in merged]
            print(f"  {a:<12s} " + " ".join(f"{x:+.3f}" for x in d)
                  + f";  {np.mean(d):+.3f} +/- {np.std(d):.3f};  {sum(x > 0 for x in d)}/{n}")
    if args.detail in arms and args.ref in arms:
        print(f"\n{args.ref} minus {args.detail} per corruption: mean +/- std; {args.ref} ahead on")
        for c in ["clean"] + CORR:
            d = [m[args.ref][c] - m[args.detail][c] for m in merged]
            print(f"  {c:<16s} {np.mean(d):+.3f} +/- {np.std(d):.3f};  {sum(x > 0 for x in d)}/{n}")


if __name__ == "__main__":
    main()
