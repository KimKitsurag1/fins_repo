"""Experiment 35 -- the periodic part of heat's infinite superposition algebra.

Superpositions b(x, t) d_u with b any heat solution form an infinite-
dimensional subalgebra. On a periodic domain the compatible ones are the
Fourier modes e^{-alpha k^2 t} cos kx, sin kx -- not polynomials. Dictionary:
phi = trig(2 pi n x / L) (t / T)^j, n <= nmax, j <= jmax; the characteristic
is phi itself, and for a linear equation the criterion asks whether phi is
carried by the flow, i.e. whether it solves the equation. The time profile
of each recovered mode is compared with the Taylor series of
e^{-alpha k^2 t} (ground truth, reading only).

Run:  python exp35_periodic_superposition.py --op exact
"""

import argparse
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp34_normalization as e34                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="exact", choices=["exact", "stepper"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--jmax", type=int, default=3)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--every", type=int, default=32)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=64)
    ap.add_argument("--reliability-filter", type=float, default=None)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()

    _, sols, xs_t, ts_t = e31.generate_pde_data("heat", cfg, args.n, seed=args.seed)
    xs, ts = xs_t[:, 0].numpy().astype(np.float64), ts_t[:, 0].numpy().astype(np.float64)
    u = sols[:, 0].numpy().astype(np.float64)
    stp = None
    if args.op == "stepper":
        ns = argparse.Namespace(stepper=args.stepper, pde="heat", seed=args.seed,
                                stepper_modes=args.stepper_modes, stepper_tag="",
                                train_if_missing=False, null_seed=0)
        stp = e31.load_stepper(ns, cfg)
        if args.reliability_filter:
            errs = []
            with torch.no_grad():
                for r in range(u.shape[0]):
                    cur = torch.tensor(u[r, :, :-1].T, dtype=torch.float32)[:, None, :]
                    pred = stp(cur)[:, 0].double().numpy()
                    nxt = u[r, :, 1:].T
                    errs.append(np.linalg.norm(pred - nxt) / np.linalg.norm(nxt - u[r, :, :-1].T))
            errs = np.array(errs)
            keep_r = errs <= args.reliability_filter * np.median(errs)
            xs, ts, u = xs[keep_r], ts[keep_r], u[keep_r]
            print(f"reliability filter: kept {keep_r.sum()} of {len(errs)}")
    N, Nx, Nt = u.shape
    L, T = e31.lengths(xs), ts[:, -1]
    X = xs[:, :, None] / L[:, None, None]
    Tn = np.broadcast_to(ts[:, None, :] / T[:, None, None], u.shape)

    groups, fields = [], []
    for n in range(args.nmax + 1):
        for trig in (("1",) if n == 0 else ("cos", "sin")):
            base = (np.ones_like(X) if trig == "1" else
                    (np.cos if trig == "cos" else np.sin)(2 * np.pi * n * X))
            for j in range(args.jmax + 1):
                fields.append(np.broadcast_to(base, u.shape) * Tn ** j)
                groups.append((n, trig, j))
    Qb = np.stack(fields)
    keep = np.ones(Nx, dtype=bool)
    starts = list(range(0, Nt - args.K, args.every))
    if args.op == "exact":
        A, B = e31.blocks_exact_heat(Qb, xs, ts, starts, args.K, cfg.alpha_heat, keep)
    else:
        A, B = e31.blocks_stepper(Qb, u, starts, args.K, stp, keep)
    Q0 = np.stack([np.concatenate([Qb[p][:, :, j].ravel() for j in starts])
                   for p in range(Qb.shape[0])], 1)
    Nr = (A - B).T @ (A - B)
    G0 = args.K * Q0.T @ Q0
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    lam, C = e34.solve(Nr, Dr)
    rms = np.sqrt((Qb.reshape(Qb.shape[0], -1) ** 2).mean(1))
    W = C * rms[:, None]

    nlow = 2 * args.nmax + 1
    print(f"[heat, {args.op}] seed {args.seed}, {N} trajectories, harmonics <= {args.nmax}, "
          f"t-degree <= {args.jmax} ({len(groups)} fields), K = {args.K}, mu_rel = {args.mu}")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:nlow + 4]))
    akT = cfg.alpha_heat * (2 * np.pi / L[0]) ** 2 * T[0]
    keys = sorted({(n, tr) for n, tr, _ in groups})
    for i in range(nlow + 2):
        w = W[:, i]
        share = {kk: np.linalg.norm([w[p] for p, g in enumerate(groups) if g[:2] == kk]) for kk in keys}
        top = max(share, key=share.get)
        frac = share[top] / np.linalg.norm(w)
        c = [C[p, i] for p, g in enumerate(groups) if g[:2] == top]
        n = top[0]
        prof = f"c1/c0 {c[1] / c[0]:+.3f} (heat {-akT * n * n:+.3f})" if n > 0 and abs(c[0]) > 1e-30 else ""
        if n > 0 and args.jmax >= 2 and abs(c[0]) > 1e-30:
            prof += f", c2/c0 {c[2] / c[0]:+.3f} ({(akT * n * n) ** 2 / 2:+.3f})"
        print(f"  v{i + 1}  loss {lam[i]:.1e}  mode n = {n} {top[1]:<3s} share {frac:.2f}  {prof}")


if __name__ == "__main__":
    main()
