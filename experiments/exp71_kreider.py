"""Kreider, Harlim and Huang (arXiv 2511.09779) as a baseline on the protocol's evaluation data (pde/method_kreider.py).

Two readings of the same estimator:
  sheet    the authors' setting: per evaluation trajectory, frames of the sampled solution's graph and its
           prolongations to the equation's order, rows S . pr v from every normal of the prolonged graph, accumulated
           over the trajectories; its null space holds the symmetries shared by the sampled solutions;
  surface  the equation surface u_t = F(u, u_x, ..., u_{x^p}) (autonomous, the input LieNLSD's network is given)
           sampled by the jets of all trajectories: codimension-1 normals of the pooled jets by the same SVD + GMLS
           estimator, rows n . pr v; its null space holds the symmetries of the estimated equation.
Jets in both readings are the authors' (graph frames, level by level). Coordinates are FINS's normalised (X, T, U);
the x-period is closed by periodic copies of the cloud; rows come from snapshots tm .. Nt - tm. Columns of A are
scaled by their rms as in LieNLSD's readout. Readouts saved in FINS's C format as the two-stage pipeline's: soft (the
k smallest right singular vectors) and hard (the own count: the largest ratio of consecutive singular values of A
among the lowest, e49.own_count).
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import corrupt as cr                                       # noqa: E402
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp41_baselines as e41                              # noqa: E402
import exp49_weak as e49                                   # noqa: E402
import exp50_two_stage as e50                              # noqa: E402
import method_kreider as mk                                # noqa: E402

ORDER = {"burgers": 2, "heat": 2, "kdv": 3, "bo": 3, "ks": 4, "fb125": 2, "fb150": 2, "fb175": 2, "fb200": 2}
WRAP = 16          # periodic copies of this many columns on each side of the x-period
COMP_TO_FINS = {"xi_x": 0, "xi_t": 1, "phi": 2}


def cloud(u_r, x_r, t_r, coords, r):
    """Normalised grid points (T, X, U) of one trajectory with periodic copies; the original points' indices."""
    Nx, Nt = u_r.shape
    X = (x_r - coords["x_offset"][r]) / coords["Lw"]
    P = Nx * (x_r[1] - x_r[0]) / coords["Lw"]
    T = t_r / coords["T_max"]
    U = u_r / coords["sigma"]
    cols = np.concatenate([np.arange(Nx - WRAP, Nx), np.arange(Nx), np.arange(WRAP)])
    shift = np.concatenate([np.full(WRAP, -P), np.zeros(Nx), np.full(WRAP, P)])
    XX = (X[cols] + shift)[:, None] * np.ones(Nt)[None, :]
    TT = np.ones(len(cols))[:, None] * T[None, :]
    UU = U[cols]
    orig = np.zeros(XX.shape, bool)
    orig[WRAP:WRAP + Nx] = True
    return TT.ravel(), XX.ravel(), UU.ravel(), orig.ravel()


def interior(Nx, Nt, tm):
    keep = np.zeros((Nx, Nt), bool)
    keep[:, tm:Nt - tm] = True
    return keep.ravel()


def grid_neighbours(ncols, Nt, K):
    """The K nearest grid points of every point of a [ncols, Nt] grid in index units (the point itself first)."""
    g = np.stack(np.meshgrid(np.arange(ncols), np.arange(Nt), indexing="ij"), -1).reshape(-1, 2).astype(float)
    return cKDTree(g).query(g, k=K, workers=-1)[1]


def jet_frames_std(t, x, u, p, K, ell, J, idx):
    """mk.jet_frames on a sampled grid: neighbours taken on the grid of (t, x) (a search in (t, x, u, ...) picks points
    along the flat direction of steep data), every level's coordinates scaled by their rms increment to the four
    nearest grid neighbours before the SVD and GMLS (a locally isotropic cloud; the derivative columns are orders of
    magnitude larger than (t, x, u)); tangents mapped back as vectors, normals as covectors."""
    Z = np.stack([t, x, u], 1).astype(float)
    for k in range(p + 1):
        sd = np.sqrt(((Z[idx[:, 1:5]] - Z[:, None, :]) ** 2).mean((0, 1)))
        sd[sd == 0] = 1.0
        Ts, Ss = mk.local_frames(Z / sd, 2, K, ell, J, idx=idx)
        T = Ts * sd[None, :, None]
        S = Ss / sd[None, :, None]
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        if k < p:
            Z = np.concatenate([Z, mk.next_jet_level(Z, T, k)], 1)
    return Z, T, S


def jets_of(u, xs, ts, coords, p, K, ell, J, tm, keep_normals=True):
    """Per trajectory: the authors' jets Z and prolonged-graph normals S (None if not kept) at the original interior
    points."""
    out = []
    for r in range(len(u)):
        T, X, U, orig = cloud(u[r], xs[r], ts[r], coords, r)
        if r == 0:
            idx = grid_neighbours(u.shape[1] + 2 * WRAP, u.shape[2], K)
        Z, _, S = jet_frames_std(T, X, U, p, K, ell, J, idx)
        sel = np.flatnonzero(orig)[interior(u.shape[1], u.shape[2], tm)]
        out.append((Z[sel], S[sel] if keep_normals else None))
    return out


def scaled_svd(A):
    """Right singular vectors of A with its columns scaled by their rms, mapped back; singular values ascending."""
    dA = np.diag(A)
    sc = np.sqrt(np.where(dA > 1e-12 * dA.max(), dA, np.median(dA[dA > 1e-12 * dA.max()])))   # exact null columns
    # (translations on the autonomous surface) keep a typical scale instead of a vanishing one
    s, V = mk.null_space_svd(A / np.outer(sc, sc))
    o = np.argsort(s)
    V = V[:, o] / sc[:, None]
    return s[o], V / np.linalg.norm(V, axis=0)


def to_fins(V, gens, mons, coords):
    """Coefficient columns of the Kreider class (components xi_t, xi_x, phi in (T, X, U)) -> FINS's C rows q * nm + j,
    physical components xi = Lw xi_X, tau = T_max xi_T, phi = sigma phi_U."""
    nm = len(mons)
    scale = {"xi_x": coords["Lw"], "xi_t": coords["T_max"], "phi": coords["sigma"]}
    C = np.zeros((3 * nm, V.shape[1]))
    for comp_i, comp in enumerate(mk.COMPONENTS):
        for j, (et, ex, eu) in enumerate(gens.monomials):
            row = COMP_TO_FINS[comp] * nm + mons.index((ex, et, eu))
            C[row] = scale[comp] * V[comp_i * gens.n_mono + j]
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pde", required=True, choices=sorted(ORDER))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--variant", choices=("sheet", "surface"), required=True)
    ap.add_argument("--K", type=int, default=40, help="neighbours of the graph frames (jets)")
    ap.add_argument("--ell", type=int, default=4, help="GMLS degree of the graph frames")
    ap.add_argument("--J", type=int, default=2, help="GMLS iterations")
    ap.add_argument("--K-surface", type=int, default=80, help="neighbours of the pooled surface normals")
    ap.add_argument("--ell-surface", type=int, default=2)
    ap.add_argument("--order", type=int, default=None, help="jet order (default the equation's)")
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--tm", type=int, default=8, help="snapshots left out at each end for the rows")
    ap.add_argument("--train-n", type=int, default=0, help="surface: pool the first N of the learned operators' 1024 "
                    "training trajectories (the data LieNLSD's network is fitted on) instead of the evaluation data")
    ap.add_argument("--out", required=True, help="prefix: <out>_soft.npz, <out>_hard.npz")
    args = ap.parse_args()
    args.dseed = 1000 + args.seed if args.data_seed is None else args.data_seed
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    p = args.order or ORDER[args.pde]
    u, u_clean, xs, ts, coords = e50.data(args, cfg)
    t0 = time.time()
    gens = mk.PolynomialGenerators(p, degree=d)
    mons = [m for m in itertools.product(range(d + 1), repeat=3) if sum(m) <= d]
    if args.train_n:
        if args.variant != "surface":
            raise SystemExit("--train-n: surface reading only")
        tr, x_tr, t_tr, _, _, _ = e41.protocol_data(args.pde, cfg, args.seed, args.dseed, 1024, 1)
        tr = cr.corrupt(tr[:args.train_n, 0].numpy().astype(np.float64), args.noise, args.thin, args.seed, cr.TRAIN)[1]
        t_tr = t_tr[::args.thin]
        n_tr = len(tr)
        per = jets_of(tr, np.tile(x_tr, (n_tr, 1)), np.tile(t_tr, (n_tr, 1)),
                      dict(coords, x_offset=np.full(n_tr, coords["x_offset"][0])), p, args.K, args.ell, args.J,
                      args.tm, keep_normals=False)
    else:
        per = jets_of(u, xs, ts, coords, p, args.K, args.ell, args.J, args.tm, args.variant == "sheet")
    t1 = time.time()
    labels = mk.jet_labels(p)
    if args.variant == "sheet":
        A = sum(gens.gram(Z, S) for Z, S in per)
        info = {}
    else:
        red = [labels.index(n) for n in ["u"] + ["u_" + "x" * j for j in range(1, p + 1)] + ["u_t"]]
        Z = np.concatenate([Z for Z, _ in per])
        Y = Z[:, red]
        sd = Y.std(0)
        _, N = mk.pooled_normals(Y / sd, 1, args.K_surface, args.ell_surface, args.J)
        n = N[:, :, 0] / sd
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        S = np.zeros((len(Z), Z.shape[1], 1))
        S[:, red, 0] = n
        A = gens.gram(Z, S)
        # the estimated surface's slope du_t/d(jet) against the equation, where it is known (diagnostic only)
        info = dict(normal_ut_share=float(np.median(np.abs(n[:, -1]))))
    s, V = scaled_svd(A)
    k = len(e31.algebra(args.pde, xs, ts, u_clean, cfg.alpha_heat, d)[0])
    lam = s / s.max()
    hard = e49.own_count(lam)
    C = to_fins(V, gens, mons, coords)
    t2 = time.time()
    print(f"[{args.pde}] Kreider {args.variant}: data seed {args.dseed}, noise {args.noise:g}, thin {args.thin}, "
          f"order {p}, degree {d}, K {args.K}, ell {args.ell}, J {args.J}"
          + (f", surface K {args.K_surface}, ell {args.ell_surface}" if args.variant == "surface" else "")
          + f"; jets {t1 - t0:.0f} s, rows and SVD {t2 - t1:.0f} s")
    print("  smallest singular values / largest: " + " ".join(f"{v:.1e}" for v in lam[:k + 4])
          + f"; own count {hard} (k = {k})" + (f"; {json.dumps(info)}" if info else ""))
    common = dict(pde=args.pde, seed=args.seed, data_seed=args.dseed, mons=np.array(mons), degree=d,
                  u=u_clean, xs=xs, ts=ts, noise=args.noise, thin=args.thin, sv=lam[:k + 6], variant=args.variant,
                  order=p, K=args.K, ell=args.ell, J=args.J, K_surface=args.K_surface, ell_surface=args.ell_surface,
                  tm=args.tm, train_n=args.train_n, seconds=t2 - t0, info=json.dumps(info), **coords)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out + "_soft.npz", method=f"Kreider {args.variant}, soft", C=C[:, :k], k=k, **common)
    np.savez(args.out + "_hard.npz", method=f"Kreider {args.variant}, hard", C=C[:, :hard], k=hard, hard=hard,
             **common)
    print(f"  saved {args.out}_soft.npz, {args.out}_hard.npz")


if __name__ == "__main__":
    main()
