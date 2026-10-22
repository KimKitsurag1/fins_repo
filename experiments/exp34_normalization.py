"""Experiment 34 -- an absolute term in the normalisation.

The eigenproblem's denominator ||a||^2 + ||b||^2 (increments) makes a slow
time modulation alpha(t) v of a symmetry v cheap when v's characteristic
evolves fast, and is degenerate for time-independent characteristics. Here
the denominator gains mu_rel * (trace B'B / trace K Q0'Q0) * K Q0'Q0, Q0 the
characteristics at the window starts. Printed per mu_rel: the lowest
eigenvalues, the loss of each algebra member and of chosen non-symmetries,
and the algebra's share inside the lowest eigenvectors (ground truth scores
only).

Run:  python exp34_normalization.py --pde kdv --op stepper --seed 43 --stepper S.pt
"""

import argparse
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp32_blind_readout as e32                          # noqa: E402

A2PI = 2 * np.pi
EXTRA = {
    "kdv": {"t x Galilean": {(0, (0, 2, 0)): 1.0, (2, (0, 1, 0)): 1 / 6},
            "t x d_x": {(0, (0, 1, 0)): 1.0}},
    "burgers": {"t x Galilean": {(0, (0, 2, 0)): A2PI, (2, (0, 1, 0)): 1.0},
                "S2 = t d_t - u d_u": {(1, (0, 1, 0)): 1.0, (2, (0, 0, 1)): -1.0}},
    "heat": {"t x d_x": {(0, (0, 1, 0)): 1.0},
             "phi = x^2 (no 2 alpha t)": {(2, (2, 0, 0)): 1.0}},
}


def solve(Nr, Dr):
    s = np.sqrt(np.clip(np.diag(Dr), 1e-60, None))
    Nn, Dn = Nr / np.outer(s, s), Dr / np.outer(s, s) + 1e-10 * np.eye(len(s))
    lam, V = eigh(Nn, Dn)
    return lam, V / s[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="kdv", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="stepper", choices=["exact", "stepper"])
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--ut", default="central", choices=["central", "central4"])
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=64)
    ap.add_argument("--reliability-filter", type=float, default=2.0)
    ap.add_argument("--mu", default="0,0.01,0.1,1", help="comma-separated mu_rel values")
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)

    xs, ts, u, ut, F, Qb, mons, A, B, keep, scale = e32.prepare(args, d)
    Nt = u.shape[2]
    starts = list(range(0, Nt - args.K, args.every))
    Q0 = np.stack([np.concatenate([Qb[p][:, keep, j].ravel() for j in starts])
                   for p in range(Qb.shape[0])], 1)
    Nr = (A - B).T @ (A - B)
    Dinc = A.T @ A + B.T @ B
    G0 = args.K * Q0.T @ Q0
    unit = np.trace(B.T @ B) / np.trace(G0)

    names, gt = e31.algebra(args.pde, xs, ts, u, cfg.alpha_heat, d)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T,
                             np.stack([g.ravel() for g in gt]).T, rcond=None)
    extra = {}
    for nm, terms in EXTRA[args.pde].items():
        c = np.zeros(3 * len(mons))
        for (comp, pw), raw in terms.items():
            c[comp * len(mons) + mons.index(pw)] = raw * scale[0] ** pw[0] * scale[1] ** pw[1] * scale[2] ** pw[2]
        extra[nm] = c
    reg = (slice(None), slice(None), keep, slice(8, -8))
    gtv = np.stack([g[reg].ravel() for g in gt])
    Fv = F[(slice(None),) + reg].reshape(F.shape[0], -1)
    k = len(names)

    print(f"[{args.pde}, {args.op}] seed {args.seed}, {u.shape[0]} trajectories, degree {d}, "
          f"K = {args.K}, starts every {args.every}; algebra: {k}")
    for mu in [float(v) for v in args.mu.split(",")]:
        Dr = Dinc + mu * unit * G0
        lam, C = solve(Nr, Dr)
        rq = lambda c: float(c @ Nr @ c / (c @ Dr @ c))
        Qe, _ = np.linalg.qr((C[:, :k].T @ Fv).T)
        cont = [np.linalg.norm(Qe.T @ g) / np.linalg.norm(g) for g in gtv]
        print(f"\n  mu_rel = {mu}: lowest eigenvalues " + " ".join(f"{v:.1e}" for v in lam[:k + 3]))
        print("    algebra losses: " + "  ".join(f"{n} {rq(c):.1e}" for n, c in zip(names, cg.T))
              + f"   (lambda{k + 1} = {lam[k]:.1e})")
        print("    non-symmetries: " + "  ".join(f"{n} {rq(c):.1e}" for n, c in extra.items()))
        print(f"    algebra inside the {k} lowest: " + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))


if __name__ == "__main__":
    main()
