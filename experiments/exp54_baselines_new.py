"""Experiment 54 -- the learned baselines on the two new equations (README_EXPERIMENTS.md, pre-registration of 07.10).

LieGAN, LaLiGAN and Augerino (equation-free) on Benjamin-Ono and Kuramoto-Sivashinsky, and LIG (given the equation's
residual) on Kuramoto-Sivashinsky, as in exp41: trained on the 1024 'mixed' training trajectories of seed s (the FNO
steppers' data), fields on the ten evaluation trajectories of data seed 1000 + s; seeds 42-44, clean data. LIG on
Benjamin-Ono is not run: its residual is evaluated on the transformed, non-uniform grid by finite differences, where
the Hilbert transform is not defined without resampling. LIG as in the protocol: 10 epochs, batch 1, torchdiffeq's
adjoint. Parts and outputs (clean folders of exp52's and exp53's runs, scored by their summaries with the other rows):
  gan        LieGAN, LaLiGAN, Augerino            baselines_<pde>_s<seed>.npz          GPU, ~30 min per run on a T4
  lig        LIG (KS)                             baselines-lig_ks_s<seed>.npz         GPU, ~2 h per run on a T4
  null       the three, untrained (own-class null) baselines-null_<pde>_s<seed>.npz     CPU, minutes
  lig-null   LIG untrained (KS)                   baselines-lig-null_ks_s<seed>.npz    GPU, minutes
exp41 saves after every method, so a rerun resumes at the first missing one.

Run:  python exp54_baselines_new.py run [--pdes bo,ks] [--seeds 42,43,44] [--parts gan,lig,null,lig-null]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402

GAN = "LieGAN,LaLiGAN,Augerino"
LIG = ["--lig-epochs", "10", "--batch-size", "1"]       # the protocol's compute cap and KdV's memory setting
PARTS = {"gan": ("baselines", GAN, []),
         "lig": ("baselines-lig", "LIG", LIG),
         "null": ("baselines-null", GAN, ["--untrained"]),
         "lig-null": ("baselines-lig-null", "LIG", LIG + ["--untrained"])}
RUNS = {"bo": "bo_runs", "ks": "ks_runs"}


def jobs(pdes, seeds, parts, root):
    out = []
    for part in parts:
        name, methods, more = PARTS[part]
        for pde in pdes:
            if part.startswith("lig") and pde != "ks":
                continue
            for seed in seeds:
                f = os.path.join(root, RUNS[pde], "clean", f"{name}_{pde}_s{seed}.npz")
                out.append((f, ["exp41_baselines.py", "--pde", pde, "--seed", str(seed), "--methods", methods, *more,
                                "--out", f]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run"])
    ap.add_argument("--pdes", default="bo,ks")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--parts", default="gan,lig")
    ap.add_argument("--root", default=os.environ.get("FINS_RESULTS_DIR", "results"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    args.root = args.root if os.path.isabs(args.root) else os.path.join(e51.HERE, args.root)
    e51.run(args, jobs(args.pdes.split(","), [int(x) for x in args.seeds.split(",")], args.parts.split(","),
                       args.root))


if __name__ == "__main__":
    main()
