"""Experiment 37 -- non-periodic data: heat between Dirichlet walls.

Boundary conditions break some symmetries of the equation (walls break
translation, a fixed length breaks scaling). The criterion targets the
equation, so it is scored only in an interior window the walls cannot reach
within the horizon -- the seam band is the torus case of the same rule.
Data: u(0) = u(L) = 0, exact sine-series evolution; u_x from the sine
series; u_t by fourth-order differences (equation-free); the exact
propagator acts on a perturbation through its sine projection, so a
perturbation that violates the walls is contaminated there, as a real
operator would be. Polynomial class, mu-normalised eigenproblem (exp34).

Run:  python exp37_dirichlet_heat.py --K 16 --wall 0.2
"""

import argparse
import itertools
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp34_normalization as e34                          # noqa: E402


def cone_band(steps, frac, tol):
    """Distance (domain fraction) beyond which the operator's response to an impulse
    at the first grid point stays below tol times its maximum over all K steps.
    steps: callables q -> k-step response. Ground-truth free."""
    e = np.zeros(len(frac))
    e[0] = 1.0
    R = np.max(np.abs(np.stack([f(e) for f in steps])), axis=0)
    R = R / R.max()
    return float(frac[np.where(R >= tol)[0].max()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--wall", type=float, default=0.2,
                    help="excluded fraction next to each wall; negative: from the influence cone")
    ap.add_argument("--tol", type=float, default=1e-3, help="influence-cone amplitude tolerance")
    ap.add_argument("--mu", type=float, default=1.0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L, Nx, Nt, a = 1.0, args.nx, args.nt, args.alpha

    x = np.arange(1, Nx + 1) * L / (Nx + 1)
    t = np.linspace(0, args.T, Nt)
    modes = np.arange(1, Nx + 1)
    kn = modes * np.pi / L
    S = np.sin(np.outer(x, kn))                               # u = S c
    Sinv = np.linalg.inv(S)
    Dx = np.cos(np.outer(x, kn)) * kn[None, :] @ Sinv          # u_x = Dx u

    u = np.empty((args.n, Nx, Nt))
    for r in range(args.n):
        c = rng.standard_normal(Nx) / modes ** 1.5 * (modes <= Nx // 2)
        c *= rng.uniform(0.3, 4.0) / np.sqrt((S @ c) ** 2).mean()
        u[r] = S @ (c[:, None] * np.exp(-a * kn[:, None] ** 2 * t[None, :]))
    xs = np.tile(x, (args.n, 1))
    ts = np.tile(t, (args.n, 1))
    ux = np.einsum("ij,rjt->rit", Dx, u)
    ut = e31.time_derivative(u, ts, "central4")

    scale = (L, args.T, u.std())
    X = np.broadcast_to(xs[:, :, None] / scale[0], u.shape)
    Tn = np.broadcast_to(ts[:, None, :] / scale[1], u.shape)
    Un = u / scale[2]
    mons = [m for m in itertools.product(range(args.degree + 1), repeat=3) if sum(m) <= args.degree]
    F, Qb = [], []
    for comp in range(3):
        for p, q, s in mons:
            f = X ** p * Tn ** q * Un ** s
            fld = np.zeros((3,) + u.shape)
            fld[comp] = f
            F.append(fld)
            Qb.append(-f * ux if comp == 0 else (-f * ut if comp == 1 else f))
    F, Qb = np.stack(F), np.stack(Qb)

    frac = x / L
    starts = list(range(0, Nt - args.K, args.every))
    dt = t[1] - t[0]
    prop = [S @ np.diag(np.exp(-a * kn ** 2 * k * dt)) @ Sinv for k in range(1, args.K + 1)]
    wall = args.wall if args.wall >= 0 else cone_band([lambda q, P=P: P @ q for P in prop], frac, args.tol)
    keep = (frac >= wall) & (frac <= 1 - wall)
    A, B, Q0 = [], [], []
    for p in range(Qb.shape[0]):
        ap_, bp_, qp_ = [], [], []
        for j in starts:
            q0 = Qb[p][:, :, j]                                 # [N, Nx]
            ev = np.stack([q0 @ P.T for P in prop], -1)         # [N, Nx, K]
            ap_.append((ev - q0[:, :, None])[:, keep])
            bp_.append((Qb[p][:, :, j + 1:j + args.K + 1] - Qb[p][:, :, j:j + 1])[:, keep])
            qp_.append(q0[:, keep])
        A.append(np.stack(ap_).ravel())
        B.append(np.stack(bp_).ravel())
        Q0.append(np.stack(qp_).ravel())
    A, B, Q0 = np.stack(A, 1), np.stack(B, 1), np.stack(Q0, 1)
    Nr = (A - B).T @ (A - B)
    G0 = args.K * Q0.T @ Q0
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    lam, C = e34.solve(Nr, Dr)

    names, gt = e31.algebra("heat", xs, ts, u, a, args.degree)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T,
                             np.stack([g.ravel() for g in gt]).T, rcond=None)
    rq = [float(c @ Nr @ c / (c @ Dr @ c)) for c in cg.T]
    reg = (slice(None), slice(None), keep, slice(8, -8))
    gtv = np.stack([g[reg].ravel() for g in gt])
    Fv = F[(slice(None),) + reg].reshape(F.shape[0], -1)
    k = len(names)
    Qe, _ = np.linalg.qr((C[:, :k].T @ Fv).T)
    cont = [np.linalg.norm(Qe.T @ g) / np.linalg.norm(g) for g in gtv]
    print(f"[heat, Dirichlet walls, exact] seed {args.seed}, {args.n} trajectories, degree {args.degree}, "
          f"K = {args.K}, wall band {wall:.3f}" + (f" (influence cone, tol {args.tol})" if args.wall < 0 else "")
          + f", mu_rel = {args.mu}; algebra in the class: {k}")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:k + 3]))
    print("  algebra losses: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq))
          + f"   (lambda{k + 1} = {lam[k]:.1e})")
    print(f"  algebra inside the {k} lowest: " + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))


if __name__ == "__main__":
    main()
