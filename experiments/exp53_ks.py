"""Experiment 53 -- Kuramoto-Sivashinsky: chaotic dynamics (README_EXPERIMENTS.md, pre-registration of 07.10).

u_t + u u_x + u_xx + u_xxxx = 0 on the torus [0, 22) (ks.py): ten evaluation trajectories on the attractor (random
fields of Fourier modes 1..4 run through a transient of 200), 128 x 128 snapshots over T = 20 (about one Lyapunov
time), the band [0.2, 0.8] L scored; algebra d_x, d_t, Galilean t d_x + d_u. The equation lies in a polynomial jet
library (to u_xxxx), so the two-stage pipeline can identify it: the test is chaotic, broadband data. Noise as exp51;
seeds 42-51, evaluation data 1000 + s; scores as exp51 (finite algebra, non-trivial = Galilean, own counts):
  weak, pointwise      FINS through the exact solver (exp49 with exp51's frozen recipe; exp39 as exp51's pointwise);
  two-stage            exp50 as in exp51 with the library to u_xxxx (--order 4), H 0.2 and 0.05;
  weak-learned,        FINS through FNO steppers trained on 1024 clean 'mixed' trajectories (attractor states, half on a
  pointwise-learned    constant background; exp39's recipe: 400 epochs, 64 modes; S_theta_ks_s{s}_mixed_stepper.pt in
                       FINS_RESULTS_DIR), seeds 42-44, clean;
  ...-attractor        the same through steppers trained on zero-mean attractor data (S_theta_ks_s{s}_attractor_...),
                       an ablation: the Galilean boost moves the mean, which no such training trajectory does.
  lienlsd              LieNLSD (exp41; fourth-order prolongation): the exact right-hand side with the
  lienlsd-f            system on the evaluation data (every seed and level), and as published, F fitted on jets to
                       u_xxxx of the 1024 'mixed' training trajectories (seeds 42-44, clean and 1e-2).
The learned arms and LieNLSD are registered separately (README_EXPERIMENTS.md, 07.10).

Run:  python exp53_ks.py run [--conds clean,n1e-2] [--seeds ...] [--parts ...]
      python exp53_ks.py summary
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402

CONDS = {"clean": 0.0, "n1e-2": 1e-2, "n3e-2": 3e-2, "n5e-2": 5e-2, "n1e-1": 1e-1}
PARTS = ("weak,pointwise,two-stage,two-stage-h05,lienlsd,lienlsd-f,weak-learned,pointwise-learned,"
         "weak-learned-attractor,pointwise-learned-attractor")
FIT_CONDS = ("clean", "n1e-2")


def jobs(conds, seeds, parts, runs):
    out = []
    for c in conds:
        d = os.path.join(runs, c)
        nz = ["--noise", f"{CONDS[c]:g}"]
        for seed in seeds:
            ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
            weak = ["--K", "16", "--stride", "1", "--wt", "2", "--wp", "2", "--wx", "auto",
                    "--wx-factor", f"{e51.WX_FACTOR:g}"]
            torus = ["--ic", "periodic", "--n", "10", "--K", "16", "--every", "16", "--thin", "1", "--interior",
                     "--ut", "fd"]
            if "weak" in parts:
                f = os.path.join(d, f"weak_ks_s{seed}.npz")
                out.append((f, ["exp49_weak.py", "--pde", "ks", "--op", "exact", *ds, *nz, *weak, "--save", f]))
            if "pointwise" in parts:
                f = os.path.join(d, f"pointwise_ks_s{seed}.npz")
                out.append((f, ["exp39_localised.py", "--pde", "ks", "--op", "exact", *ds, *torus, *nz, "--save", f]))
            for part, H in (("two-stage", "0.2"), ("two-stage-h05", "0.05")):
                if part in parts:
                    pre = os.path.join(d, f"{part}_ks_s{seed}")
                    out.append((pre + "_soft.npz", ["exp50_two_stage.py", "--pde", "ks", *ds, *nz, "--H", H,
                                                    "--K", "200", "--rel", "0.1", "--order", "4", "--out", pre]))
            if "lienlsd" in parts:
                f = os.path.join(d, f"lienlsd_ks_s{seed}.npz")
                out.append((f, ["exp41_baselines.py", "--pde", "ks", "--seed", str(seed), "--methods",
                                "LieNLSD_RHS@eval", "--lienlsd-fit-points", str(10 * 128 * 128),
                                "--lienlsd-val-points", str(2 * 128 * 128), *nz, "--out", f]))
            if seed not in e51.FIT_SEEDS:
                continue
            if "lienlsd-f" in parts and c in FIT_CONDS:
                f = os.path.join(d, f"lienlsd-f_ks_s{seed}.npz")
                out.append((f, ["exp41_baselines.py", "--pde", "ks", "--seed", str(seed), "--methods",
                                "LieNLSD_F,LieNLSD_F@eval", "--lienlsd-fit-points", str(10 * 128 * 128),
                                "--lienlsd-val-points", str(2 * 128 * 128), *nz, "--out", f]))
            if c != "clean":
                continue
            for data, sfx in (("mixed", ""), ("attractor", "-attractor")):
                if "weak-learned" + sfx in parts:
                    f = os.path.join(d, f"weak-learned{sfx}_ks_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", "--pde", "ks", "--op", "learned", "--train-ic", data, *ds,
                                    "--tag", "", *nz, *weak, "--save", f]))
                if "pointwise-learned" + sfx in parts:
                    f = os.path.join(d, f"pointwise-learned{sfx}_ks_s{seed}.npz")
                    out.append((f, ["exp39_localised.py", "--pde", "ks", "--op", "stepper", "--train-ic", data, *ds,
                                    *torus, *nz, "--save", f]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "ks_runs"))
    ap.add_argument("--out", default=os.path.join("results", "ks"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(e51.HERE, p) for p in (args.runs, args.out))
    args.pdes, args.exp48 = "ks", False
    if args.what == "run":
        e51.run(args, jobs(args.conds.split(","), [int(x) for x in args.seeds.split(",")], args.parts.split(","),
                           args.runs))
    else:
        e51.summary(args, {c: CONDS[c] for c in args.conds.split(",")})


if __name__ == "__main__":
    main()
