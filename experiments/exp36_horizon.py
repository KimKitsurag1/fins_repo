"""Experiment 36 -- how each direction's loss grows with the horizon.

A slow modulation alpha(t) v of a symmetry v leaves a residual
-(alpha(t + K dt) - alpha(t)) Q_v, so its loss grows roughly as K^2; a true
symmetry's loss is set by the operator's error and grows slowly. The lowest
eigenvectors found at the first horizon are re-scored, unchanged, at the
others; the growth ratio is ground-truth free. Ground truth is printed for
reading only.

Run:  python exp36_horizon.py --pde kdv --op stepper --seed 43 --Ks 16,32 --stepper S.pt
"""

import argparse
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp32_blind_readout as e32                          # noqa: E402
import exp34_normalization as e34                          # noqa: E402


def blocks(args, cfg, Qb, xs, ts, u, keep, K, stp):
    starts = list(range(0, u.shape[2] - K, args.every))
    if args.op == "stepper":
        A, B = e31.blocks_stepper(Qb, u, starts, K, stp, keep)
    elif args.pde == "kdv":
        A, B = e31.blocks_exact_kdv(Qb, u, ts, starts, K, keep)
    elif args.pde == "burgers":
        A, B = e31.blocks_exact_burgers_aligned(Qb, u, ts, starts, K, cfg.nu_burgers, keep)
    else:
        A, B = e31.blocks_exact_heat(Qb, xs, ts, starts, K, cfg.alpha_heat, keep)
    Q0 = np.stack([np.concatenate([Qb[p][:, keep, j].ravel() for j in starts])
                   for p in range(Qb.shape[0])], 1)
    G0 = K * Q0.T @ Q0
    Nr = (A - B).T @ (A - B)
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    return Nr, Dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="kdv", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="stepper", choices=["exact", "stepper"])
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--Ks", default="16,32")
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--ut", default="central", choices=["central", "central4"])
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=64)
    ap.add_argument("--reliability-filter", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    Ks = [int(k) for k in args.Ks.split(",")]
    args.K = Ks[0]

    xs, ts, u, ut, F, Qb, mons, _, _, keep, scale = e32.prepare(args, d)
    stp = None
    if args.op == "stepper":
        ns = argparse.Namespace(stepper=args.stepper, pde=args.pde, seed=args.seed,
                                stepper_modes=args.stepper_modes, stepper_tag="",
                                train_if_missing=False, null_seed=0)
        stp = e31.load_stepper(ns, cfg)
    pencils = [blocks(args, cfg, Qb, xs, ts, u, keep, K, stp) for K in Ks]
    lam, C = e34.solve(*pencils[0])
    rq = lambda c, P: float(c @ P[0] @ c / (c @ P[1] @ c))

    names, gt = e31.algebra(args.pde, xs, ts, u, cfg.alpha_heat, d)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T,
                             np.stack([g.ravel() for g in gt]).T, rcond=None)
    rms = np.sqrt((F.reshape(F.shape[0], -1) ** 2).mean(1))
    Qg, _ = np.linalg.qr(cg * rms[:, None])
    extra = {}
    for nm, terms in e34.EXTRA[args.pde].items():
        c = np.zeros(3 * len(mons))
        for (comp, pw), raw in terms.items():
            c[comp * len(mons) + mons.index(pw)] = raw * scale[0] ** pw[0] * scale[1] ** pw[1] * scale[2] ** pw[2]
        extra[nm] = c

    hdr = " -> ".join(f"K={K}" for K in Ks)
    print(f"[{args.pde}, {args.op}] seed {args.seed}, {u.shape[0]} trajectories, degree {d}, "
          f"mu_rel = {args.mu}; losses {hdr}, growth = last / first")
    print("  lowest eigenvectors at the first horizon (ground-truth free), [share outside the algebra, GT]:")
    for i in range(args.show):
        c = C[:, i]
        ls = [rq(c, P) for P in pencils]
        w = c * rms
        out = np.linalg.norm(w - Qg @ (Qg.T @ w)) / np.linalg.norm(w)
        print(f"    v{i + 1}  " + "  ".join(f"{v:.1e}" for v in ls)
              + f"   growth {ls[-1] / ls[0]:5.1f}   [outside {out:.2f}]")
    print("  algebra (GT):")
    for nm, c in zip(names, cg.T):
        ls = [rq(c, P) for P in pencils]
        print(f"    {nm:<12} " + "  ".join(f"{v:.1e}" for v in ls) + f"   growth {ls[-1] / ls[0]:5.1f}")
    print("  non-symmetries:")
    for nm, c in extra.items():
        ls = [rq(c, P) for P in pencils]
        print(f"    {nm:<24} " + "  ".join(f"{v:.1e}" for v in ls) + f"   growth {ls[-1] / ls[0]:5.1f}")


if __name__ == "__main__":
    main()
