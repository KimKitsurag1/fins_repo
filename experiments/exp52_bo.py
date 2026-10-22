"""Experiment 52 -- Benjamin-Ono: a nonlocal equation outside every polynomial jet library (README_EXPERIMENTS.md,
pre-registration of 07.10).

u_t + u u_x = H u_xx on the torus (bo.py): periodised two-soliton evaluation trajectories (c in [3, 5]), 128 x 128 over
T = 0.2, the band [0.2, 0.8] L scored; algebra d_x, d_t, Galilean t d_x + d_u, scaling x d_x + 2 t d_t - u d_u. The
right-hand side has no jet expression, so Lie's algorithm cannot be run on the equation, and the equation-based
pipelines can only use a local surrogate of it. Gaussian noise of sigma times each trajectory's standard deviation
(corrupt.py), sigma in CONDS; seeds 42-51, evaluation data 1000 + s; scores as exp51 (finite algebra, non-trivial
generators, own counts):
  weak, pointwise      FINS through the exact solver (exp49 with exp51's frozen recipe; exp39 as exp51's pointwise);
  two-stage            exp50 as in exp51 (H 0.2 and 0.05, threshold 0.1, library to u_xxx): weak SINDy identifies a
                       local equation, its algebra is read out hard (own count) and soft (k smallest);
  lienlsd-f            LieNLSD as published: F fitted on jets (u, u_x, u_xx, u_xxx) of the training trajectories,
                       determining system on the training (LieNLSD_F) and the evaluation trajectories (LieNLSD_F@eval);
                       seeds 42-44, clean and 1e-2 (about 12 minutes per fit);
  weak-learned,        FINS through FNO steppers trained on 1024 clean 'mixed' BO trajectories (exp39's KdV recipe:
  pointwise-learned    400 epochs, 64 modes; S_theta_bo_s{s}_mixed_stepper.pt in FINS_RESULTS_DIR), seeds 42-44, clean.

Run:  python exp52_bo.py run [--conds clean,n1e-2] [--seeds ...] [--parts ...]
      python exp52_bo.py summary
      python exp52_bo.py fifth      (the fifth weak eigenvalue per seed, registered as reported)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp42_tables as e42                                 # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402

CONDS = {"clean": 0.0, "n1e-2": 1e-2, "n3e-2": 3e-2}
PARTS = "weak,pointwise,two-stage,two-stage-h05,lienlsd-f,weak-learned,pointwise-learned"
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
                f = os.path.join(d, f"weak_bo_s{seed}.npz")
                out.append((f, ["exp49_weak.py", "--pde", "bo", "--op", "exact", *ds, *nz, *weak, "--save", f]))
            if "pointwise" in parts:
                f = os.path.join(d, f"pointwise_bo_s{seed}.npz")
                out.append((f, ["exp39_localised.py", "--pde", "bo", "--op", "exact", *ds, *torus, *nz, "--save", f]))
            for part, H in (("two-stage", "0.2"), ("two-stage-h05", "0.05")):
                if part in parts:
                    pre = os.path.join(d, f"{part}_bo_s{seed}")
                    out.append((pre + "_soft.npz", ["exp50_two_stage.py", "--pde", "bo", *ds, *nz, "--H", H,
                                                    "--K", "200", "--rel", "0.1", "--out", pre]))
            if seed not in e51.FIT_SEEDS:
                continue
            if "lienlsd-f" in parts and c in FIT_CONDS:
                f = os.path.join(d, f"lienlsd-f_bo_s{seed}.npz")
                out.append((f, ["exp41_baselines.py", "--pde", "bo", "--seed", str(seed), "--methods",
                                "LieNLSD_F,LieNLSD_F@eval", "--lienlsd-fit-points", str(10 * 128 * 128),
                                "--lienlsd-val-points", str(2 * 128 * 128), *nz, "--out", f]))
            if c != "clean":
                continue
            if "weak-learned" in parts:
                f = os.path.join(d, f"weak-learned_bo_s{seed}.npz")
                out.append((f, ["exp49_weak.py", "--pde", "bo", "--op", "learned", *ds, "--tag", "", *nz, *weak,
                                "--save", f]))
            if "pointwise-learned" in parts:
                f = os.path.join(d, f"pointwise-learned_bo_s{seed}.npz")
                out.append((f, ["exp39_localised.py", "--pde", "bo", "--op", "stepper", "--train-ic", "mixed", *ds,
                                *torus, *nz, "--save", f]))
    return out


def fifth(args):
    """FINS weak's fifth eigenvalue per level and seed: lambda5 / lambda4, lambda6 / lambda5, and the share of the
    inviscid-Burgers projective field x t d_x + t^2 d_t + (x - t u) d_u, reduced to its part orthogonal to the algebra,
    inside the span of the five lowest eigen-generators (scored region and scaling as the summary)."""
    cfg = e31.ExperimentConfig()
    out = {}
    for c in args.conds.split(","):
        for seed in args.seeds.split(","):
            p = os.path.join(args.runs, c, f"weak_bo_s{seed}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            lam, u, xs, ts = z["lam"], z["u"], z["xs"], z["ts"]
            Nx = u.shape[1]
            keep = (np.arange(Nx) / Nx >= 0.2) & (np.arange(Nx) / Nx <= 0.8)
            w = np.array([1 / ((xs[0, 1] - xs[0, 0]) * Nx), 1 / ts[:, -1].max(), 1 / u.std()])
            G = e42.vectors(e42.ground_truth("bo", u, xs, ts, cfg)[1], keep, w, 8)
            D = e42.vectors(e42.fins_fields({**{k: z[k] for k in z.files}, "k": 5}), keep, w, 8)
            X, T = np.broadcast_to(xs[:, :, None], u.shape), np.broadcast_to(ts[:, None, :], u.shape)
            pv = e42.vectors(np.stack([X * T, T ** 2, X - T * u])[None], keep, w, 8)[0]
            Qg = e42.orth(G)
            pn = pv - Qg @ (Qg.T @ pv)
            share = float(np.linalg.norm(e42.orth(D).T @ pn) / np.linalg.norm(pn))
            out.setdefault(c, {})[seed] = dict(lam4=float(lam[3]), lam5=float(lam[4]), lam6=float(lam[5]),
                                               r54=float(lam[4] / lam[3]), r65=float(lam[5] / lam[4]),
                                               projective=share, nown=int(z["nown"]))
        rows = out.get(c, {})
        if rows:
            q = {key: [r[key] for r in rows.values()] for key in ("r54", "r65", "projective", "nown")}
            print(f"{c}: {len(rows)} seeds; lambda5/lambda4 {min(q['r54']):.3g}-{max(q['r54']):.3g}, "
                  f"lambda6/lambda5 {min(q['r65']):.3g}-{max(q['r65']):.3g}, projective share "
                  f"{min(q['projective']):.3f}-{max(q['projective']):.3f}, own counts {sorted(q['nown'])}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fifth.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary", "fifth"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "bo_runs"))
    ap.add_argument("--out", default=os.path.join("results", "bo"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(e51.HERE, p) for p in (args.runs, args.out))
    args.pdes, args.exp48 = "bo", False
    if args.what == "run":
        e51.run(args, jobs(args.conds.split(","), [int(x) for x in args.seeds.split(",")], args.parts.split(","),
                           args.runs))
    elif args.what == "fifth":
        fifth(args)
    else:
        e51.summary(args, {c: CONDS[c] for c in args.conds.split(",")})


if __name__ == "__main__":
    main()
