"""Experiment 56's registered runs (README_EXPERIMENTS.md, pre-registration of 08.10): reference weak SINDy then Lie
(exp56_wsindy.py) on the evaluation trajectories (seeds 42-51, every condition of exp51 / exp52 / exp53, Burgers clean)
and on the training trajectories of the learned operators (seeds 42-44, clean and 1e-2); outputs beside the other
methods' runs (noise_weak_runs, bo_runs, ks_runs, burgers_runs), tags wsindy and wsindy-train, so the existing
summaries score them.

Run:  python exp56_jobs.py run [--parts eval,train] [--threads 1]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402
import exp52_bo as e52                                     # noqa: E402
import exp53_ks as e53                                     # noqa: E402

RUNS = os.environ.get("FINS_RESULTS_DIR", "results")
SETS = (("heat", e51.CONDS, "noise_weak_runs"), ("kdv", e51.CONDS, "noise_weak_runs"), ("ks", e53.CONDS, "ks_runs"),
        ("bo", e52.CONDS, "bo_runs"), ("burgers", {"clean": 0.0}, "burgers_runs"))


def jobs(parts, runs):
    out = []
    for pde, conds, sub in SETS:
        for c, sigma in conds.items():
            d = os.path.join(runs, sub, c)
            for data, tag, seeds in (("eval", "wsindy", range(42, 52)), ("train", "wsindy-train", (42, 43, 44))):
                if data not in parts or (data == "train" and c not in ("clean", "n1e-2")):
                    continue
                for seed in seeds:
                    pre = os.path.join(d, f"{tag}_{pde}_s{seed}")
                    out.append((pre + "_soft.npz", ["exp56_wsindy.py", "--pde", pde, "--seed", str(seed), "--data",
                                                    data, "--noise", f"{sigma:g}", "--out", pre]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run"])
    ap.add_argument("--parts", default="eval,train")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    e51.run(args, jobs(args.parts.split(","), args.runs))


if __name__ == "__main__":
    main()
