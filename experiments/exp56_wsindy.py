"""Experiment 56 -- the two-stage baseline like for like: reference weak SINDy (Messenger and Bortz 2021, the authors'
algorithm ported in pde/method_wsindy.py: spectral test functions, MSTLS) then Lie's algorithm, on the evaluation
trajectories (as exp50) and on the training trajectories of the learned operators (the data FINS's learned operators
see), with one library for every equation: d_x^k (u^j), j <= 4, k <= 6, and a constant.

Stage 2 as exp50: the determining system of the identified u_t = F in FINS's normalised variables, exact rationals,
read hard (null space below --tol: the identified equation's algebra, its own count) and soft (the k smallest right
singular vectors). Scored on the clean evaluation trajectories (exp51's scorer), so the rows sit beside FINS's.

Oracle (exp57, fractional Burgers): --oracle 0.5,0.75,... adds -Lambda^beta (u^j) columns, j in --oracle-powers
(default 1: a linear nonlocal term; Fourier multipliers on the periodic data, method_wsindy.py), to the library; the
identified equation and the time-scaling exponent at which its terms balance (x -> l x, t -> l^a t, u -> l^c u) go to
<out>_oracle.json, without a Lie stage (a nonlocal term has no jet expression).

Run:  python exp56_wsindy.py --pde kdv --seed 42 --data train --noise 0.01 --out wsindy-train_kdv_s42
"""

import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp40_window as e40                                 # noqa: E402
import exp50_two_stage as e50                              # noqa: E402
import corrupt as cr                                       # noqa: E402
import method_wsindy as mw                                 # noqa: E402

POLYS, MAX_DX = (0, 1, 2, 3, 4), 6
N_TRAIN = 1024
ROWS_TRAIN = 50000          # about the evaluation data's rows (10 trajectories x the authors' K_max = 5000)


def training(args, cfg):
    """[N_TRAIN, Nx, Nt] corrupted training trajectories of the learned operator of seed args.seed, and x, t."""
    if args.pde in e39.TORUS:
        T = e39.horizon(args.pde)
        _, x, t = e39.grid(args.pde, cfg, 128, 128, T)
        tr = e39.training_data(args.pde, N_TRAIN, args.seed, "mixed", x, T, 128, 0.002)
    else:
        smooth, only_L = (True, 1.0) if args.pde == "heat" else (False, None)
        xs, ts, tr, _ = e40.trajectories(args.pde, cfg, N_TRAIN, args.seed, smooth, only_L)
        x, t = xs[0], ts[0]
    return cr.corrupt(tr, args.noise, 1, args.seed, cr.TRAIN)[1], x, t


def minus_lambda(beta):
    """Symbol of -Lambda^beta, Lambda = (-d_xx)^(1/2)."""
    return lambda kap: -np.abs(kap) ** beta


def balance(orders):
    """Time-scaling exponent a at which every term (j, order) of u_t = sum c d^order (u^j) balances under
    x -> l x, t -> l^a t, u -> l^c u: c (j - 1) - order + a = 0 for every term. None if no (a, c) does, or if a is not
    fixed (one power j != 1 only)."""
    A = np.array([[1.0, j - 1.0] for j, _ in orders])
    b = np.array([float(o) for _, o in orders])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.abs(A @ sol - b).max() > 1e-9:
        return None
    if np.linalg.matrix_rank(A) < 2 and any(j != 1 for j, _ in orders):
        return None
    return float(sol[0])


def F_from_terms(terms, coords, order):
    """u_t = sum c d_x^k (u^j) (physical) -> U_T as a polynomial in (U, U_X, ..., U_{order X}) (normalised)."""
    U = sp.Symbol("U")
    nj = 2 * order + 2
    J = sp.symbols(f"U1:{nj + 1}")
    sig, Lw, Tm = (sp.Rational(float(coords[key])) for key in ("sigma", "Lw", "T_max"))

    def DX(g):
        out = sp.diff(g, U) * J[0]
        for i in range(nj - 1):
            out += sp.diff(g, J[i]) * J[i + 1]
        return out

    F = 0
    for j, k, c in terms:
        c = sp.Rational(float(c))
        g = (sig * U) ** j
        for _ in range(k):
            g = DX(g)
        F += c * Tm / sig * g / Lw ** k
    return sp.expand(F)


def determining_F(F, d, order):
    """exp50.determining with a given F (a polynomial in U and its X-jets to `order`)."""
    X, T, U = sp.symbols("X T U")
    nj = 2 * order + 2
    J = sp.symbols(f"U1:{nj + 1}")

    def Dx(g):
        out = sp.diff(g, X) + sp.diff(g, U) * J[0]
        for i in range(nj - 1):
            out += sp.diff(g, J[i]) * J[i + 1]
        return out

    DxF = [F]
    for _ in range(nj - 1):
        DxF.append(sp.expand(Dx(DxF[-1])))

    def Dt(g):
        out = sp.diff(g, T) + sp.diff(g, U) * F
        for i in range(nj - 1):
            out += sp.diff(g, J[i]) * DxF[i + 1]
        return out

    Fu = [sp.diff(F, U)] + [sp.diff(F, J[i]) for i in range(order)]
    mons = [mm for mm in itertools.product(range(d + 1), repeat=3) if sum(mm) <= d]
    cols = []
    gens = (X, T, U) + J
    for comp in range(3):
        for a, b, c in mons:
            f = X ** a * T ** b * U ** c
            Q = f if comp == 2 else (-f * J[0] if comp == 0 else -f * F)
            cond = Dt(Q)
            DQ = Q
            for jj in range(order + 1):
                if jj:
                    DQ = Dx(DQ)
                cond -= Fu[jj] * DQ
            cols.append(sp.Poly(sp.expand(cond), *gens).as_dict())
    keys = sorted({kk for col in cols for kk in col})
    index = {kk: i for i, kk in enumerate(keys)}
    M = np.zeros((len(keys), len(cols)))
    for jcol, col in enumerate(cols):
        for kk, v in col.items():
            M[index[kk], jcol] = float(v)
    return M, mons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pde", required=True, choices=["heat", "burgers", "kdv", "bo", "ks", *e39.fb.EQUATIONS,
                                                       *e39.ccf.EQUATIONS])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--data", default="eval", choices=["eval", "train"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-8, help="hard null space: singular values below tol x largest")
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--out", default=None, help="prefix: <out>_hard.npz, <out>_soft.npz")
    ap.add_argument("--oracle", default=None, help="comma list of beta: -Lambda^beta (u^j) columns; <out>_oracle.json")
    ap.add_argument("--oracle-powers", default="1", help="comma list of j for the -Lambda^beta (u^j) columns")
    args = ap.parse_args()
    args.dseed = 1000 + args.seed if args.data_seed is None else args.data_seed
    args.thin = 1
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    u, u_clean, xs, ts, coords = e50.data(args, cfg)          # evaluation trajectories: scoring and coordinates
    t0 = time.time()
    if args.data == "eval":
        Ufit, x, t, rows = u, xs[0], ts[0], None
    else:
        Ufit, x, t = training(args, cfg)
        rows = ROWS_TRAIN
    betas = [] if args.oracle is None else [float(b) for b in args.oracle.split(",")]
    w = mw.WSINDy(polys=POLYS, max_dx=MAX_DX, multipliers=tuple((minus_lambda(b), b) for b in betas),
                  mult_polys=tuple(int(j) for j in args.oracle_powers.split(",")))
    W, lam, resid = w.fit(Ufit, x, t, rows_total=rows)
    if betas:
        sel = [(j, k, mi, float(c)) for (j, k, mi), c in zip(w.terms, W) if c != 0]
        names = " ".join(f"{c:+.5g} " + (f"d{k}(u^{j})" if mi < 0 else f"(-L^{betas[mi]:g})(u^{j})")
                         for j, k, mi, c in sel)
        a = balance([(j, k if mi < 0 else betas[mi]) for j, k, mi, _ in sel])
        print(f"[{args.pde}] oracle, {args.data} data ({len(Ufit)} trajectories), data seed {args.dseed}, noise "
              f"{args.noise:g}: u_t = {names}   (lambda {lam:.2e}, residual {resid:.2e}); exponent at which the terms "
              f"balance: {'none' if a is None else f'{a:.4g}'}; {time.time() - t0:.0f} s", flush=True)
        if args.out:
            with open(args.out + "_oracle.json", "w") as fh:
                json.dump(dict(pde=args.pde, seed=args.seed, data_seed=args.dseed, data=args.data, noise=args.noise,
                               betas=betas, powers=args.oracle_powers, lam=float(lam), resid=resid, balance=a,
                               names=names,
                               terms=[[j, k, None if mi < 0 else betas[mi], c] for j, k, mi, c in sel],
                               wsindy={kk: float(v) for kk, v in w.info.items()}), fh, indent=1)
            print(f"  saved {args.out}_oracle.json")
        return
    terms = [(j, k, float(c)) for (j, k, _), c in zip(w.terms, W) if c != 0]
    t1 = time.time()
    names = " ".join(f"{c:+.5g} d{k}(u^{j})" for j, k, c in terms)
    print(f"[{args.pde}] {args.data} data ({len(Ufit)} trajectories), data seed {args.dseed}, noise {args.noise:g}: "
          f"u_t = {names}   (lambda {lam:.2e}, residual {resid:.2e}; m {w.info['mx']},{w.info['mt']}, "
          f"p {w.info['px']},{w.info['pt']}, {w.info['rows']} rows; {t1 - t0:.0f} s)", flush=True)
    order = max([k for _, k, _ in terms] + [1])
    F = F_from_terms(terms, coords, order)
    M, mons = determining_F(F, d, order)
    M = e50.normalise_rows(M)
    _, s, Vh = np.linalg.svd(M, full_matrices=True)
    sv = np.zeros(Vh.shape[0])
    sv[:len(s)] = s
    o = np.argsort(sv)
    sv, V = sv[o], Vh[o].T
    nm = len(mons)
    V = V * np.repeat([coords["Lw"], coords["T_max"], coords["sigma"]], nm)[:, None]
    k = len(e31.algebra(args.pde, xs, ts, u_clean, cfg.alpha_heat, d)[0])
    hard = int((sv < args.tol * sv.max()).sum())
    print(f"  determining system {M.shape[0]} x {M.shape[1]}: smallest singular values / largest "
          + " ".join(f"{v / sv.max():.1e}" for v in sv[:k + 3]) + f"; hard algebra dimension {hard} (k = {k}); "
          f"{time.time() - t1:.0f} s", flush=True)
    if args.out:
        common = dict(pde=args.pde, seed=args.seed, data_seed=args.dseed, mons=np.array(mons), degree=d,
                      u=u_clean, xs=xs, ts=ts, noise=args.noise, thin=1, sv=sv[:k + 6] / sv.max(),
                      terms=np.array([f"{c:+.6g} d{kk}(u^{j})" for j, kk, c in terms]), resid=resid, lam=lam,
                      data=args.data, wsindy=np.array([f"{kk}={v}" for kk, v in w.info.items()]), **coords)
        np.savez(args.out + "_soft.npz", method="wsindy, soft", C=V[:, :k], k=k, **common)
        np.savez(args.out + "_hard.npz", method="wsindy, hard", C=V[:, :max(hard, 1)] if hard else
                 np.zeros((V.shape[0], 1)), k=max(hard, 1), hard=hard, **common)
        print(f"  saved {args.out}_soft.npz, {args.out}_hard.npz")


if __name__ == "__main__":
    main()
