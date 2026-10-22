"""Experiment 50 -- two-stage baseline: weak SINDy identifies the equation, Lie's algorithm gives its algebra.

Stage 1 (PySINDy's WeakPDELibrary, Messenger and Bortz's weak SINDy): library u^a d_x^m u, a <= 2, m <= --order,
plus a constant, integrated against compact test functions on --K random subdomains of every evaluation trajectory
(the trajectories FINS uses); sequential thresholding of the column-normalised coefficients relative to the largest
(--rel), least squares refit on the kept terms.
Stage 2: the determining system of the identified u_t = F, written in FINS's normalised variables X = (x - x_off) /
Lw, T = t / T_max, U = u / sigma, for generators polynomial of degree d in (X, T, U) (FINS's full class): with
Q = phi - xi U_X - tau F, D_T Q - sum_j F_{U_j} D_X^j Q = 0 identically in X, T, U and the jets, one row per
monomial (rows normalised). Hard algebra: the null space below --tol of the largest singular value (the exact
algebra of the identified equation); soft: the m = k smallest right singular vectors (as FINS reads its pencil).
Saves exp42-compatible npz files (hard, soft) with the clean evaluation data, in the coordinates of FINS's runs.

Run:  python exp50_two_stage.py --pde kdv --data-seed 1007 --noise 0.01 --out two_stage_kdv
"""

import argparse
import itertools
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pysindy as ps
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp40_window as e40                                 # noqa: E402
import corrupt as cr                                       # noqa: E402



def data(args, cfg):
    """Noisy and clean evaluation trajectories [N, Nx, Nt], their grids, and FINS's coordinates for the class."""
    if args.pde in e39.TORUS:
        L, x, t = e39.grid(args.pde, cfg, 128, 128, e39.horizon(args.pde))
        u, _, ts = e39.evaluation_data(args.pde, "periodic", args.n, args.dseed, x, t, 0.002)
        xs = np.tile(x, (args.n, 1))
        coords = dict(Lw=L, x_offset=np.zeros(args.n))
    else:
        smooth, only_L = (True, 1.0) if args.pde == "heat" else (False, None)
        xs, ts, u, _ = e40.trajectories(args.pde, cfg, args.n, args.dseed, smooth, only_L)
        Nx = u.shape[1]
        W = 4 * Nx
        dx = e31.lengths(xs) / Nx
        coords = dict(Lw=(W * dx).max(), x_offset=dx * ((W // 2) % Nx))
    u_clean, u = cr.corrupt(u, args.noise, args.thin, args.dseed, cr.EVAL)
    ts = ts[:, ::args.thin]
    coords.update(T_max=ts[:, -1].max(), sigma=u.std())
    return u, u_clean, xs, ts, coords


def identify(u, xs, ts, args):
    """Weak SINDy over all trajectories: (names, powers a, orders m, coefficients) of the kept terms."""
    np.random.seed(args.sindy_seed)
    lib_terms = []
    rows, rhs = [], []
    for r in range(len(u)):
        X, T = np.meshgrid(xs[r], ts[r], indexing="ij")
        grid = np.asarray([X, T]).transpose([1, 2, 0])
        Lxt = np.array([xs[r][-1] - xs[r][0], ts[r][-1] - ts[r][0]])
        lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
                                derivative_order=args.order, spatiotemporal_grid=grid, K=args.K, p=4,
                                H_xt=list(args.H * Lxt), include_bias=True, periodic=True, is_uniform=True)
        ur = ps.AxesArray(u[r][:, :, None], {"ax_spatial": 0, "ax_time": 1, "ax_coord": 2})
        rows.append(np.asarray(lib.fit_transform([ur])[0]))
        rhs.append(np.asarray(lib.convert_u_dot_integral(ur))[:, 0])
        if not lib_terms:
            lib_terms = lib.get_feature_names()
    Th, b = np.concatenate(rows), np.concatenate(rhs)
    norms = np.linalg.norm(Th, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    A = Th / norms
    active = np.ones(A.shape[1], bool)
    for _ in range(50):
        c = np.zeros(A.shape[1])
        c[active] = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
        keep = np.abs(c) >= args.rel * np.abs(c).max()
        if (keep == active).all():
            break
        active = keep
    coef = c / norms
    terms = []
    for name, cj in zip(lib_terms, coef):
        if cj == 0:
            continue
        terms.append((name,) + parse_term(name, args.order) + (float(cj),))
    resid = float(np.linalg.norm(A @ c - b) / np.linalg.norm(b))
    return terms, resid


def parse_term(name, order):
    """PySINDy names ('1', 'x0', 'x0^2', 'x0_11', 'x0x0_1', 'x0^2x0_111') -> (power of u, derivative order)."""
    if name == "1":
        return 0, -1
    m = 0
    for k in range(order, 0, -1):
        tag = "x0_" + "1" * k
        if name.endswith(tag):
            m, name = k, name[: -len(tag)]
            break
    if m == 0:
        a = 1 if name == "x0" else int(name.split("^")[1])
        return a, 0
    a = 0 if name == "" else (1 if name == "x0" else int(name.split("^")[1]))
    return a, m


def determining(terms, coords, d, order):
    """Matrix M [rows, 3 nm] of the determining system of the identified equation in FINS's class."""
    X, T, U = sp.symbols("X T U")
    nj = 2 * order + 2
    J = sp.symbols(f"U1:{nj + 1}")
    # exact rationals (the floats' binary values): the expansion then cancels what cancels in exact arithmetic, so no
    # rounding residue survives as a spurious row (rows are normalised below)
    sig, Lw, Tm = (sp.Rational(float(coords[key])) for key in ("sigma", "Lw", "T_max"))
    # u_t = sum c u^a d_x^m u  ->  U_T = (T_max / sigma) sum c (sigma U)^a (sigma / Lw^m) d_X^m U
    F = 0
    for _, a, m, c in terms:
        c = sp.Rational(float(c))
        if m < 0:
            F += c * Tm / sig
        elif m == 0:
            F += c * Tm / sig * (sig * U) ** a
        else:
            F += c * Tm / sig * (sig * U) ** a * sig / Lw ** m * J[m - 1]
    F = sp.expand(F)

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
            for j in range(order + 1):
                if j:
                    DQ = Dx(DQ)
                cond -= Fu[j] * DQ
            cols.append(sp.Poly(sp.expand(cond), *gens).as_dict())
    keys = sorted({k for col in cols for k in col})
    index = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(cols)))
    for jcol, col in enumerate(cols):
        for k, v in col.items():
            M[index[k], jcol] = float(v)
    return M, mons


def normalise_rows(M):
    """Unit rows (each monomial equation weighs the same); exact zeros dropped."""
    nrm = np.linalg.norm(M, axis=1)
    keep = nrm > 0
    return M[keep] / nrm[keep, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["heat", "burgers", "kdv", "bo", "ks", *e39.fb.EQUATIONS,
                                                       *e39.ccf.EQUATIONS])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--order", type=int, default=3, help="highest x-derivative in the library")
    ap.add_argument("--K", type=int, default=200, help="weak-form subdomains per trajectory")
    ap.add_argument("--H", type=float, default=0.05, help="subdomain half-width, fraction of the x and t extents "
                    "(PySINDy's default 1/20)")
    ap.add_argument("--rel", type=float, default=0.05, help="threshold relative to the largest normalised coefficient")
    ap.add_argument("--tol", type=float, default=1e-8, help="hard null space: singular values below tol x largest")
    ap.add_argument("--sindy-seed", type=int, default=0)
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--out", default=None, help="prefix: <out>_hard.npz, <out>_soft.npz")
    args = ap.parse_args()
    args.dseed = 1000 + args.seed if args.data_seed is None else args.data_seed
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    u, u_clean, xs, ts, coords = data(args, cfg)
    t0 = time.time()
    terms, resid = identify(u, xs, ts, args)
    t1 = time.time()
    print(f"[{args.pde}] data seed {args.dseed}, noise {args.noise:g}, thin {args.thin}: u_t = "
          + " ".join(f"{c:+.5g} {n}" for n, _, _, c in terms) + f"   (weak residual {resid:.2e}; {t1 - t0:.0f} s)")
    M, mons = determining(terms, coords, d, args.order)
    M = normalise_rows(M)
    _, s, Vh = np.linalg.svd(M, full_matrices=True)
    sv = np.zeros(Vh.shape[0])
    sv[:len(s)] = s
    order_ = np.argsort(sv)
    sv, V = sv[order_], Vh[order_].T
    # generators in (X, T, U) -> physical components xi = Lw xi_X, tau = T_max tau_T, phi = sigma phi_U (FINS's C)
    nm = len(mons)
    V = V * np.repeat([coords["Lw"], coords["T_max"], coords["sigma"]], nm)[:, None]
    k = len(e31.algebra(args.pde, xs, ts, u_clean, cfg.alpha_heat, d)[0])
    hard = int((sv < args.tol * sv.max()).sum())
    print(f"  determining system {M.shape[0]} x {M.shape[1]}: smallest singular values / largest "
          + " ".join(f"{v / sv.max():.1e}" for v in sv[:k + 3]) + f"; hard algebra dimension {hard} (k = {k}); "
          f"{time.time() - t1:.0f} s")
    if args.out:
        common = dict(pde=args.pde, seed=args.seed, data_seed=args.dseed, mons=np.array(mons), degree=d,
                      u=u_clean, xs=xs, ts=ts, noise=args.noise, thin=args.thin, sv=sv[:k + 6] / sv.max(),
                      terms=np.array([f"{c:+.6g} {n}" for n, _, _, c in terms]), resid=resid, rel=args.rel,
                      **coords)
        np.savez(args.out + "_soft.npz", method="two-stage, soft", C=V[:, :k], k=k, **common)
        np.savez(args.out + "_hard.npz", method="two-stage, hard", C=V[:, :max(hard, 1)] if hard else
                 np.zeros((V.shape[0], 1)), k=max(hard, 1), hard=hard, **common)
        print(f"  saved {args.out}_soft.npz, {args.out}_hard.npz")


if __name__ == "__main__":
    main()
