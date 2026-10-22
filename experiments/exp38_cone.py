"""Experiment 38 -- how far an operator carries information in K steps.

An impulse at the torus seam is pushed through the operator's Jacobian for
k = 1..K steps at data states; the response, maximised over states and
steps and normalised by its peak, is read against the distance from the
seam. The distance where it falls below tol is the band the criterion must
exclude. Ground-truth free; works for exact and learned operators alike.

Run:  python exp38_cone.py --pde heat --op stepper --stepper S.pt --K 16
"""

import argparse
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402


def profile(R, tols):
    """Normalised response vs circular distance from the seam; band per tol."""
    Nx = len(R)
    dist = np.minimum(np.arange(Nx), Nx - np.arange(Nx)) / Nx
    R = R / R.max()
    bands = {tol: float(dist[R >= tol].max()) for tol in tols}
    return dist, R, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="stepper", choices=["exact", "stepper"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--tols", default="1e-2,1e-3,1e-4")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=64)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    tols = [float(v) for v in args.tols.split(",")]

    if args.pde == "kdv":
        xs, ts, u = e31.kdv_coarse(cfg, args.n, args.seed)
    elif args.pde == "burgers":
        xs, ts, u = e31.burgers_coarse(cfg, args.n, args.seed)
    else:
        _, sols, xs_t, ts_t = e31.generate_pde_data("heat", cfg, args.n, seed=args.seed)
        xs, ts = xs_t[:, 0].numpy().astype(np.float64), ts_t[:, 0].numpy().astype(np.float64)
        u = sols[:, 0].numpy().astype(np.float64)
    N, Nx, Nt = u.shape
    starts = list(range(0, Nt - args.K, args.every))
    e = np.zeros(Nx)
    e[0] = 1.0

    if args.op == "exact" and args.pde == "heat":
        k2 = (2 * np.pi * np.stack([np.fft.fftfreq(Nx, d=L / Nx) for L in e31.lengths(xs)])) ** 2
        dt = ts[:, 1] - ts[:, 0]
        ek = np.fft.fft(e)
        R = np.max(np.stack([np.abs(np.real(np.fft.ifft(ek[None] * np.exp(-cfg.alpha_heat * k2 * k * dt[:, None]),
                                                         axis=-1))).max(0)
                             for k in range(1, args.K + 1)]), 0)
    elif args.op == "stepper":
        ns = argparse.Namespace(stepper=args.stepper, pde=args.pde, seed=args.seed,
                                stepper_modes=args.stepper_modes, stepper_tag="",
                                train_if_missing=False, null_seed=0)
        stp = e31.load_stepper(ns, cfg)
        u0 = torch.tensor(np.concatenate([u[:, :, j] for j in starts]), dtype=torch.float32)[:, None]
        q = torch.tensor(np.tile(e, (u0.shape[0], 1)), dtype=torch.float32)[:, None]
        R = np.zeros(Nx)
        Rrms = np.zeros(Nx)
        s, t = u0, q
        for _ in range(args.K):
            s, t = jvp(stp, (s,), (t,))
            tt = t[:, 0].detach().double().numpy()
            R = np.maximum(R, np.abs(tt).max(0))
            Rrms = np.maximum(Rrms, np.sqrt((tt ** 2).mean(0)))
        _, Rr, br = profile(Rrms, tols)
        dist_r = np.minimum(np.arange(Nx), Nx - np.arange(Nx)) / Nx
        print("  (rms over states) response: "
              + "  ".join(f"{m:.2f}: {Rr[np.argmin(np.abs(dist_r - m))]:.1e}" for m in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)))
        print("  (rms over states) band per tolerance: " + "  ".join(f"tol {t_:.0e} -> {b:.3f}" for t_, b in br.items()))
    else:
        raise SystemExit("exact cone implemented for heat; use --op stepper otherwise")

    dist, Rn, bands = profile(R, tols)
    print(f"[{args.pde}, {args.op}] seed {args.seed}, {N} trajectories x {len(starts)} starts, K = {args.K}")
    order = np.argsort(dist)
    marks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    print("  response vs distance from the seam: "
          + "  ".join(f"{m:.2f}: {Rn[np.argmin(np.abs(dist - m))]:.1e}" for m in marks))
    print("  band per tolerance: " + "  ".join(f"tol {t:.0e} -> {b:.3f}" for t, b in bands.items()))


if __name__ == "__main__":
    main()
