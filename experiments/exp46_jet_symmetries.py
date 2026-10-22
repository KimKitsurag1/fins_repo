"""Experiment 46 -- generalised (Lie-Baecklund) symmetries by FINS in a jet-polynomial class.

A generalised symmetry in evolutionary form is a characteristic Q[u] -- a function of u and finitely many
x-derivatives of u -- that solves the linearised equation on solutions (Olver 1993, Ch. 5): exactly FINS's transport
condition. Replacing the point-field class by monomials in the jets (u, u_x, ..., u_5x), the same generalised
eigenproblem looks for them. The class here: the 15 monomials of KdV weight <= 7 (u weight 2, d/dx weight 1).

Equations on the periodic interval [0, 2 pi), pseudo-spectral with the 2/3 mask; integrating-factor RK4 whose
tangent goes through the same stages (the exact Jacobian of the discrete map):
  kdv   u_t = -u_xxx - 6 u u_x  integrable: u_x, K3 = u_xxx + 6 u u_x, K5 = u_5x + 10 u u_xxx + 20 u_x u_xx + 30 u^2 u_x
  pme   u_t = (u u_x)_x          not integrable: u_x and u u_xx + u_x^2
  heat  u_t = alpha u_xx         linear: 1, u, u_x, ..., u_5x
Expected ranks in the class: 3, 2, 7.

  python exp46_jet_symmetries.py --pde kdv --seed 0 --save jets/kdv_s0
"""

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
from scipy.linalg import eigh

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
NJ = 6
MONS = sorted([e for e in itertools.product(range(4), repeat=NJ) if sum(ek * (2 + k) for k, ek in enumerate(e)) <= 7],
              key=lambda e: (sum(ek * (2 + k) for k, ek in enumerate(e)), e[::-1]))
DEFAULTS = {"kdv": dict(n=256, dt=0.002, nsub=8, nt=33),
            "pme": dict(n=128, dt=0.004, nsub=6, nt=33),
            "heat": dict(n=128, dt=0.005, nsub=1, nt=33)}
TOL = 1e-6


def mon_name(e):
    names = ["u", "u1", "u2", "u3", "u4", "u5"]
    parts = [names[k] + (f"^{ek}" if ek > 1 else "") for k, ek in enumerate(e) if ek]
    return "*".join(parts) if parts else "1"


class Solver:
    """u_t = L u + N(u) on [0, 2 pi), rfft state, integrating-factor RK4 (Lawson) and its exact tangent."""

    def __init__(self, pde, n, alpha=0.5):
        self.pde, self.n = pde, n
        k = torch.fft.rfftfreq(n, d=2 * math.pi / n, dtype=torch.float64) * 2 * math.pi
        self.k = k
        self.ik = 1j * k
        self.mask = (k.abs() < (2.0 / 3.0) * k.abs().max()).to(torch.float64)
        if pde == "kdv":
            self.L = (1j * k ** 3).to(torch.complex128)
        elif pde == "heat":
            self.L = (-alpha * k ** 2).to(torch.complex128)
        else:
            self.L = torch.zeros_like(k, dtype=torch.complex128)

    def phys(self, v):
        return torch.fft.irfft(v, n=self.n)

    def N(self, v):
        u = self.phys(v)
        if self.pde == "kdv":
            return self.mask * (-3 * self.ik * torch.fft.rfft(u * u))
        if self.pde == "pme":
            return self.mask * (-(self.k ** 2) * torch.fft.rfft(0.5 * u * u))
        return torch.zeros_like(v)

    def dN(self, v, dv):
        u, du = self.phys(v), self.phys(dv)
        if self.pde == "kdv":
            return self.mask * (-6 * self.ik * torch.fft.rfft(u * du))
        if self.pde == "pme":
            return self.mask * (-(self.k ** 2) * torch.fft.rfft(u * du))
        return torch.zeros_like(dv)

    def step(self, v, h, dv=None):
        E, E2 = torch.exp(self.L * h), torch.exp(self.L * h / 2)
        k1 = self.N(v)
        a2 = E2 * (v + h / 2 * k1)
        k2 = self.N(a2)
        a3 = E2 * v + h / 2 * k2
        k3 = self.N(a3)
        a4 = E * v + h * E2 * k3
        k4 = self.N(a4)
        vn = E * v + h / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
        if dv is None:
            return vn
        l1 = self.dN(v, dv)
        d2 = E2 * (dv + h / 2 * l1)
        l2 = self.dN(a2, d2)
        d3 = E2 * dv + h / 2 * l2
        l3 = self.dN(a3, d3)
        d4 = E * dv + h * E2 * l3
        l4 = self.dN(a4, d4)
        return vn, E * dv + h / 6 * (E * l1 + 2 * E2 * (l2 + l3) + l4)

    def solve(self, u0, nt, dt, nsub):
        v = self.mask * torch.fft.rfft(u0)
        out = [self.phys(v)]
        for _ in range(nt - 1):
            for _ in range(nsub):
                v = self.step(v, dt / nsub)
            out.append(self.phys(v))
        return torch.stack(out, 1)

    def transport(self, u0, d0, K, dt, nsub):
        """J_k d0, k = 1..K: u0 [P, n], d0 [P, m, n] -> [P, K, m, n]."""
        v = (self.mask * torch.fft.rfft(u0))[:, None]
        dv = self.mask * torch.fft.rfft(d0)
        out = []
        for _ in range(K):
            for _ in range(nsub):
                v, dv = self.step(v, dt / nsub, dv)
            out.append(self.phys(dv))
        return torch.stack(out, 1)

    def jets(self, u):
        v = torch.fft.rfft(u)
        return torch.stack([torch.fft.irfft((self.ik ** j) * v, n=self.n) for j in range(NJ)], 0)


def initial(pde, n, n_traj, rng, ic="default"):
    x = np.arange(n) * 2 * np.pi / n
    u0 = np.zeros((n_traj, n))
    for r in range(n_traj):
        if pde == "kdv" and ic == "fourier":
            for k in range(1, 9):
                a, b = rng.standard_normal(2) / k
                u0[r] += 0.6 * (a * np.cos(k * x) + b * np.sin(k * x))
        elif pde == "kdv":
            u0[r] = rng.uniform(-0.3, 0.3)
            for _ in range(rng.integers(1, 4)):
                a, w, c = rng.uniform(0.5, 2.0), rng.uniform(0.4, 0.8), rng.uniform(0, 2 * np.pi)
                for m in (-1, 0, 1):
                    u0[r] += a / np.cosh((x - c - 2 * np.pi * m) / w) ** 2
        elif pde == "pme":
            u0[r] = rng.uniform(0.25, 0.45)
            for _ in range(rng.integers(2, 4)):
                a, w, c = rng.uniform(0.4, 0.8), rng.uniform(0.3, 0.6), rng.uniform(0, 2 * np.pi)
                for m in (-1, 0, 1):
                    u0[r] += a * np.exp(-((x - c - 2 * np.pi * m) / w) ** 2)
        else:
            u0[r] = rng.standard_normal()
            for k in range(1, 11):
                a, b = rng.standard_normal(2) / k
                u0[r] += a * np.cos(k * x) + b * np.sin(k * x)
    return torch.tensor(u0)


def true_chars(pde, J):
    """Ground-truth characteristics from physical jets J [6, ...]."""
    u, u1, u2, u3, u4, u5 = J
    if pde == "kdv":
        return ["u_x", "K3", "K5"], [u1, u3 + 6 * u * u1, u5 + 10 * u * u3 + 20 * u1 * u2 + 30 * u * u * u1]
    if pde == "pme":
        return ["u_x", "u u_xx + u_x^2"], [u1, u * u2 + u1 * u1]
    return ["1", "u", "u_x", "u_xx", "u_xxx", "u_4x", "u_5x"], [torch.ones_like(u), u, u1, u2, u3, u4, u5]


class Basis:
    """Scaled monomials: J_k / (sigma k0^k), sigma and k0 from the training set."""

    def __init__(self, solver, u_train):
        J = solver.jets(u_train)
        self.sigma = float(u_train.std())
        self.k0 = float(((J[1] ** 2).mean() / ((u_train - u_train.mean()) ** 2).mean()).sqrt())
        self.solver = solver

    def values(self, u):
        J = self.solver.jets(u)
        S = torch.stack([J[k] / (self.sigma * self.k0 ** k) for k in range(NJ)], 0)
        return torch.stack([torch.prod(torch.stack([S[k] ** ek for k, ek in enumerate(e)], 0), 0) for e in MONS], 0)


def gsolve(N, D):
    s = np.sqrt(np.clip(np.diag(D), 1e-60, None))
    lam, V = eigh(N / np.outer(s, s), D / np.outer(s, s) + 1e-10 * np.eye(len(s)))
    return lam, V / s[:, None]


def orth(V):
    U, s, _ = np.linalg.svd(V.T, full_matrices=False)
    return U[:, s > TOL * s[0]] if s.size and s[0] > 0 else U[:, :0]


def missing(Fa, Fb):
    Qa, Qb = orth(Fa), orth(Fb)
    return float(Fb.shape[0] - ((Qa.T @ Qb) ** 2).sum())


def largest_ratio_rank(lam):
    r = np.where(lam[:-1] > 0, lam[1:] / np.where(lam[:-1] > 0, lam[:-1], 1.0), -np.inf)
    return int(np.argmax(r)) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pde", required=True, choices=["kdv", "pme", "heat"])
    ap.add_argument("--seed", type=int, default=0, help="data seeds 4600 + s (training) and 5600 + s (held-out)")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--nsub", type=int, default=None)
    ap.add_argument("--nt", type=int, default=None)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--ic", default="default", choices=["default", "fourier"], help="kdv: sech^2 humps or random modes")
    ap.add_argument("--save", default=None, help="stem under FINS_RESULTS_DIR (json, npz)")
    args = ap.parse_args()
    cfg = dict(DEFAULTS[args.pde])
    for key in ("n", "dt", "nsub", "nt"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    t0 = time.time()
    sol = Solver(args.pde, cfg["n"], args.alpha)
    ut = sol.solve(initial(args.pde, cfg["n"], args.n_train, np.random.default_rng(4600 + args.seed), args.ic),
                   cfg["nt"], cfg["dt"], cfg["nsub"])
    ue = sol.solve(initial(args.pde, cfg["n"], args.n_eval, np.random.default_rng(5600 + args.seed), args.ic),
                   cfg["nt"], cfg["dt"], cfg["nsub"])
    basis = Basis(sol, ut)
    nb = len(MONS)
    # pencil on (trajectory, start) pairs, all points (the class is x-free, the data periodic)
    N = np.zeros((nb, nb)); DA = np.zeros((nb, nb)); G0 = np.zeros((nb, nb)); trBB = 0.0
    starts = list(range(0, cfg["nt"] - args.K, args.every))
    for r in range(args.n_train):
        Qall = basis.values(ut[r])                                                     # [nb, nt, n]
        for j in starts:
            dl = sol.transport(ut[r, j][None], Qall[:, j][None], args.K, cfg["dt"], cfg["nsub"])[0]   # [K, nb, n]
            A = (dl - Qall[:, j][None]).permute(1, 0, 2).reshape(nb, -1).numpy()
            B = (Qall[:, j + 1:j + 1 + args.K] - Qall[:, j][:, None]).reshape(nb, -1).numpy()
            Q0 = Qall[:, j].numpy()
            R = A - B
            N += R @ R.T
            DA += A @ A.T + B @ B.T
            G0 += args.K * Q0 @ Q0.T
            trBB += float((B ** 2).sum())
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = gsolve(N, D)
    names, gt_train = true_chars(args.pde, sol.jets(ut))
    M = basis.values(ut).reshape(nb, -1).numpy()
    Gt = np.stack([g.reshape(-1).numpy() for g in gt_train], 1)
    Ct, *_ = np.linalg.lstsq(M.T, Gt, rcond=None)
    fit = float(np.linalg.norm(M.T @ Ct - Gt) / np.linalg.norm(Gt))
    rq = [float(c @ N @ c / (c @ D @ c)) for c in Ct.T]
    r_true = len(names)
    # held-out scores: characteristics at all points of the held-out snapshots
    Me = basis.values(ue).reshape(nb, -1).numpy()
    _, gt_eval = true_chars(args.pde, sol.jets(ue))
    Qb = np.stack([g.reshape(-1).numpy() for g in gt_eval])
    miss = missing((C[:, :r_true].T @ Me), Qb)
    rng = np.random.default_rng(args.seed)
    nulls = [missing(rng.standard_normal((r_true, nb)) @ Me, Qb) for _ in range(args.null_draws)]
    rel = float(np.sqrt(cmu))
    out = dict(pde=args.pde, seed=args.seed, ic=args.ic, config=cfg, K=args.K, mu=args.mu, monomials=[mon_name(e) for e in MONS],
               lam=lam.tolist(), true_names=names, true_rq=rq, fit_residual=fit, r_true=r_true,
               gap=float(lam[r_true] / lam[r_true - 1]) if lam[r_true - 1] > 0 else float("inf"),
               rank_largest_ratio=largest_ratio_rank(lam), missing=miss, null_missing=[float(np.mean(nulls)),
               float(np.std(nulls))], rel_change=rel, u_range=[float(ut.min()), float(ut.max())],
               time=time.time() - t0)
    print(f"[{args.pde}] seed {args.seed}: n {cfg['n']}, dt {cfg['dt']}, nsub {cfg['nsub']}, nt {cfg['nt']}, K {args.K}; "
          f"u in [{out['u_range'][0]:.2f}, {out['u_range'][1]:.2f}]; relative change over K {rel:.2f}; "
          f"{len(starts) * args.n_train} pairs, {time.time() - t0:.0f} s")
    print("  spectrum: " + " ".join(f"{v:.1e}" for v in lam))
    print("  true characteristics (fit residual %.1e): " % fit + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq)))
    print(f"  expected rank {r_true}: lambda{r_true + 1} / lambda{r_true} {out['gap']:.3g}; largest-ratio rank "
          f"{out['rank_largest_ratio']}; held-out missing of the {r_true} lowest {miss:.4f} (random {np.mean(nulls):.3f} "
          f"+- {np.std(nulls):.3f})")
    for j in range(min(r_true + 2, nb)):
        c = C[:, j] / np.abs(C[:, j]).max()
        top = np.argsort(-np.abs(c))[:4]
        print(f"    eigenvector {j + 1} ({lam[j]:.1e}): " + ", ".join(f"{c[i]:+.2f} {mon_name(MONS[i])}" for i in top))
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", N=N, D=D, C=C, lam=lam, Ct=Ct, sigma=basis.sigma, k0=basis.k0)
        print(f"  saved {stem}.json/.npz")


if __name__ == "__main__":
    main()
