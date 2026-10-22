"""Experiment 32 -- what does the eigenproblem actually output, read blind?

exp31's lowest eigenvectors are mixtures. Here the subspace they span is
rotated to its sparsest form (varimax on the coefficients, each weighted by
its basis field's RMS so a coefficient measures a contribution to the
field) and the generators are printed as formulas in x, t, u. No ground
truth enters the discovery, the rotation or the printout; it is used only
for the closing comparison line.

The rank is not known blind, so the subspace size m is a parameter: run it
below, at and above the true dimension and see where readable formulas end.

Run:  python exp32_blind_readout.py --pde burgers --op stepper --m 5 \
          --stepper S_theta_burgers_s42_g64x128_stepper.pt
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

COMP = ("xi", "eta", "phi")


def varimax(Phi, iters=500, tol=1e-10):
    p, k = Phi.shape
    R, d = np.eye(k), 0.0
    for _ in range(iters):
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / p))
        R = u @ vh
        d_new = s.sum()
        if d and d_new / d < 1 + tol:
            break
        d = d_new
    return Phi @ R


def monomial_label(a, b, c):
    parts = [("x", a), ("t", b), ("u", c)]
    s = "".join(v if n == 1 else f"{v}^{n}" for v, n in parts if n)
    return s or "1"


def print_readout(C, F, Nr, Dr, mons, scale, m, show=0.05, note=""):
    """Varimax readout of the m lowest eigenvectors C[:, :m] (mu-normalised eigenproblem Nr, Dr),
    printed as formulas in the unscaled coordinates, sorted by loss."""
    rms = np.sqrt((F.reshape(F.shape[0], -1) ** 2).mean(1))
    Rot = varimax(np.linalg.qr(C[:, :m] * rms[:, None])[0])
    gens = []
    for i in range(m):
        w = Rot[:, i]
        c = w / rms
        gens.append((float(c @ Nr @ c / (c @ Dr @ c)), w, c))
    gens.sort(key=lambda g: g[0])
    for i, (loss, w, c) in enumerate(gens):
        big = np.abs(w).max()
        terms = []
        for j in np.argsort(-np.abs(w)):
            if abs(w[j]) < show * big:
                break
            comp, (a, b, cc) = j // len(mons), mons[j % len(mons)]
            terms.append((comp, c[j] / (scale[0] ** a * scale[1] ** b * scale[2] ** cc), monomial_label(a, b, cc)))
        lead = terms[0][1]
        rest = float(np.sqrt(sum(w[q] ** 2 for q in range(len(w)) if abs(w[q]) < show * big)) / np.linalg.norm(w))
        form = ";  ".join(f"{COMP[q]} = " + " ".join(f"{r / lead:+.3g} {lab}" for cmp, r, lab in terms if cmp == q)
                          for q in range(3) if any(t_[0] == q for t_ in terms))
        print(f"  v{i + 1}  loss {loss:.1e}  {len(terms)} terms (rest {rest:.2f}):  {form}{note}")


def prepare(args, d):
    """Data, basis and increment blocks for one run."""
    cfg = e31.ExperimentConfig()

    if args.pde == "kdv":
        xs, ts, u, ut = e31.kdv_stencil(cfg, args.n, args.seed, 2e-5)
    elif args.pde == "burgers" and args.aligned:
        xs, ts, u = e31.burgers_coarse(cfg, args.n, args.seed)
        ut = e31.time_derivative(u, ts, args.ut)
    else:
        _, sols, xs_t, ts_t = e31.generate_pde_data(args.pde, cfg, args.n, seed=args.seed)
        xs, ts = xs_t[:, 0].numpy().astype(np.float64), ts_t[:, 0].numpy().astype(np.float64)
        u = sols[:, 0].numpy().astype(np.float64)
        ut = e31.time_derivative(u, ts, args.ut)
    stp = None
    if args.op == "stepper":
        ns = argparse.Namespace(stepper=args.stepper, pde=args.pde, seed=args.seed,
                                stepper_modes=args.stepper_modes, stepper_tag="",
                                train_if_missing=False, null_seed=0)
        stp = e31.load_stepper(ns, cfg)
        if args.reliability_filter:
            # the same equation-free rule as exp31
            errs = []
            with torch.no_grad():
                for r in range(u.shape[0]):
                    cur = torch.tensor(u[r, :, :-1].T, dtype=torch.float32)[:, None, :]
                    nxt = u[r, :, 1:].T
                    pred = stp(cur)[:, 0].double().numpy()
                    errs.append(np.linalg.norm(pred - nxt) / np.linalg.norm(nxt - u[r, :, :-1].T))
            errs = np.array(errs)
            keep_r = errs <= args.reliability_filter * np.median(errs)
            xs, ts, u, ut = xs[keep_r], ts[keep_r], u[keep_r], ut[keep_r]
            print(f"reliability filter: kept {keep_r.sum()} of {len(errs)}")
    Nx, Nt = u.shape[1:]
    frac = np.arange(Nx) / Nx
    keep = (frac >= args.seam) & (frac <= 1 - args.seam)
    scale = (e31.lengths(xs).max(), ts[:, -1].max(), u.std())
    F, Qb, mons = e31.basis(u, xs, ts, d, ut, scale)
    starts = list(range(0, Nt - args.K, args.every))
    if args.op == "stepper":
        A, B = e31.blocks_stepper(Qb, u, starts, args.K, stp, keep)
    elif args.pde == "burgers" and args.aligned:
        A, B = e31.blocks_exact_burgers_aligned(Qb, u, ts, starts, args.K, cfg.nu_burgers, keep)
    elif args.pde == "burgers":
        A, B = e31.blocks_exact(Qb, u, ts[0], starts, args.K, cfg.nu_burgers, keep)
    elif args.pde == "kdv":
        A, B = e31.blocks_exact_kdv(Qb, u, ts, starts, args.K, keep)
    else:
        A, B = e31.blocks_exact_heat(Qb, xs, ts, starts, args.K, cfg.alpha_heat, keep)
    return xs, ts, u, ut, F, Qb, mons, A, B, keep, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="stepper", choices=["exact", "stepper"])
    ap.add_argument("--m", type=int, default=5, help="subspace size (the rank is not known blind)")
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=24)
    ap.add_argument("--reliability-filter", type=float, default=None)
    ap.add_argument("--ut", default="central", choices=["central", "central4"])
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--show", type=float, default=0.05,
                    help="print terms carrying at least this share of a generator's largest term")
    args = ap.parse_args()
    d = args.degree or (3 if args.pde == "heat" else 2)
    xs, ts, u, ut, F, Qb, mons, A, B, keep, scale = prepare(args, d)
    lam, C, Nm, Dm, sc = e31.discover(A, B)

    # coefficients weighted by each basis field's RMS: a weighted coefficient
    # measures that term's contribution to the field
    rms = np.sqrt((F.reshape(F.shape[0], -1) ** 2).mean(1))
    W = C[:, :args.m] * rms[:, None]
    Q, _ = np.linalg.qr(W)
    Rot = varimax(Q)
    rq = lambda c: float((c * sc) @ Nm @ (c * sc) / ((c * sc) @ Dm @ (c * sc)))
    gens = []
    for j in range(Rot.shape[1]):
        w = Rot[:, j]
        c = w / rms
        gens.append((rq(c), w, c))
    gens.sort(key=lambda g: g[0])

    print(f"[{args.pde}, {args.op}] seed {args.seed}, {u.shape[0]} trajectories, degree {d}, "
          f"subspace of the {args.m} lowest eigenvectors, rotated to its sparsest form")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:args.m + 2]))
    for i, (loss, w, c) in enumerate(gens):
        big = np.abs(w).max()
        terms = []
        lead = None
        for idx in np.argsort(-np.abs(w)):
            if abs(w[idx]) < args.show * big:
                break
            comp, (a, b, cc) = divmod(idx, len(mons))[0], mons[idx % len(mons)]
            raw = c[idx] / (scale[0] ** a * scale[1] ** b * scale[2] ** cc)
            if lead is None:
                lead = raw
            terms.append((comp, raw, monomial_label(a, b, cc)))
        rest = np.sqrt(sum(w[k] ** 2 for k in range(len(w)) if abs(w[k]) < args.show * big)) / np.linalg.norm(w)
        form = []
        for comp in range(3):
            tt = [f"{r / lead:+.3g} {lab}" for cmp, r, lab in terms if cmp == comp]
            if tt:
                form.append(f"{COMP[comp]} = " + " ".join(tt))
        print(f"  v{i + 1}  loss {loss:.1e}  {len(terms)} terms (rest {rest:.2f}):  " + ";  ".join(form))
    if args.pde == "burgers":
        a = 2 * np.pi
        print(f"  (for comparison only -- the Burgers algebra on this data: d_x; d_t; Galilean xi = "
              f"{a:.3g} t, phi = 1; scaling xi = x, eta = 2t, phi = -u; projective xi = xt, eta = t^2, "
              f"phi = {1 / a:.3g} x - tu)")


if __name__ == "__main__":
    main()
