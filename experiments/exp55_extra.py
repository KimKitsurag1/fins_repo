"""Experiment 55 -- the two-stage baseline on Burgers and with one library for every equation; FINS weak through the
learned heat operators (README_EXPERIMENTS.md, pre-registration of 07.10, evening).

  two-stage-o4, two-stage-h05-o4   exp50 as in exp51 (H 0.2 / 0.05, 200 subdomains, threshold 0.1) with the library to
                                   u_xxxx (--order 4), Kuramoto-Sivashinsky's, for every equation: heat and KdV at
                                   exp51's conditions (runs in noise_weak_runs), Benjamin-Ono at exp52's (bo_runs),
                                   Burgers clean (burgers_runs); KS's own rows already use it;
  two-stage, two-stage-h05         exp50 with exp51's library (to u_xxx) on Burgers, clean (burgers_runs);
  weak-learned                     FINS weak (exp51's frozen recipe) through the protocol's heat CNOs
                                   (S_theta_heat_s{s}_g64x128_cno_smooth_L1.pt in FINS_RESULTS_DIR), seeds 42-44, clean
                                   (noise_weak_runs).
Scores: exp51's (finite algebra, non-trivial generators, heat's class); `exp51_noise_weak.py summary --exp48` and
`exp52_bo.py summary` pick up the new rows, `exp55_extra.py summary` scores Burgers.

Run:  python exp55_extra.py run [--parts ...] [--seeds ...]
      python exp55_extra.py summary
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp52_bo as e52                                     # noqa: E402

RUNS = os.environ.get("FINS_RESULTS_DIR", "results")      # holds noise_weak_runs, bo_runs, burgers_runs
PARTS = "o4,burgers,heat-learned"
TS = (("two-stage", "0.2", "3"), ("two-stage-h05", "0.05", "3"), ("two-stage-o4", "0.2", "4"),
      ("two-stage-h05-o4", "0.05", "4"))


def two_stage(d, pde, seed, noise, tag, H, order):
    pre = os.path.join(d, f"{tag}_{pde}_s{seed}")
    return (pre + "_soft.npz", ["exp50_two_stage.py", "--pde", pde, "--seed", str(seed), "--data-seed",
                                str(1000 + seed), "--noise", f"{noise:g}", "--H", H, "--K", "200", "--rel", "0.1",
                                "--order", order, "--out", pre])


def jobs(parts, seeds, runs):
    out = []
    if "o4" in parts:
        for pde, conds, sub in (("heat", e51.CONDS, "noise_weak_runs"), ("kdv", e51.CONDS, "noise_weak_runs"),
                                ("bo", e52.CONDS, "bo_runs")):
            for c, sigma in conds.items():
                for seed in seeds:
                    for tag, H, order in TS[2:]:
                        out.append(two_stage(os.path.join(runs, sub, c), pde, seed, sigma, tag, H, order))
    if "burgers" in parts:
        for seed in seeds:
            for tag, H, order in TS:
                out.append(two_stage(os.path.join(runs, "burgers_runs", "clean"), "burgers", seed, 0.0, tag, H, order))
    if "heat-learned" in parts:
        d = os.path.join(runs, "noise_weak_runs", "clean")
        for seed in seeds:
            if seed not in e51.FIT_SEEDS:
                continue
            f = os.path.join(d, f"weak-learned_heat_s{seed}.npz")
            out.append((f, ["exp49_weak.py", "--pde", "heat", "--op", "learned", "--seed", str(seed), "--data-seed",
                            str(1000 + seed), "--tag", "_smooth_L1", "--noise", "0", "--K", "16", "--stride", "1",
                            "--wt", "2", "--wp", "2", "--wx", "auto", "--wx-factor", f"{e51.WX_FACTOR:g}",
                            "--save", f]))
    return out


def summary(args):
    cfg = e31.ExperimentConfig()
    seeds = [int(x) for x in args.seeds.split(",")]
    res = e51.score_cond(os.path.join(args.runs, "burgers_runs", "clean"), ["burgers"], seeds, cfg)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"clean": res}, fh, indent=1)
    lines = ["## burgers, clean: missing dimensions (mean +/- sd over seeds, n seeds; per seed)\n",
             "| row | finite (5) | non-trivial (3) |", "|---|---|---|"]
    for lab, r in res.get("burgers", {}).items():
        if lab.startswith("_"):
            continue
        cells = []
        for cv in ("finite", "non-trivial"):
            v = np.array(list(r[cv]["missing"].values()))
            cells.append(f"{v.mean():.3f} +/- {v.std():.3f} ({len(v)}; max {v.max():.3f})")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=os.path.join("results", "burgers"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.what == "run":
        e51.run(args, jobs(args.parts.split(","), [int(x) for x in args.seeds.split(",")], args.runs))
    else:
        summary(args)


if __name__ == "__main__":
    main()
