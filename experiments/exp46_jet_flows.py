"""Finite flows of jet characteristics (exp46): do the generalised symmetries found in the jet class exponentiate to
flows that commute with the evolution, and do the first non-symmetries break down?

A characteristic Q[u] = sigma sum_i c_i m_i(S), S_k = u_kx / (sigma k0^k) (exp46's scaled jets), generates the
evolutionary flow dw/ds = P Q[w]. For a generalised symmetry the flow commutes with the evolution (for KdV, the flows
of K3 and K5 are the KdV and fifth-order KdV equations, which commute: integrability). Integration: Lawson RK4 with the
constant-coefficient linear part of Q exact in Fourier space and the rest explicit; the step is at most
1 / (safety x rate) with the explicit part's largest rate re-evaluated every 16 steps, and at most |s| / 8. P is the
solver's 2/3 band, narrowed for KdV to |k| < 32: there the explicit stage of the fifth-order flow is stable and
accurate (its stiffness grows like 10 u k^3), and the KdV data hold at most 1e-9 of their norm above k = 32 (with
|k| < 42 the K5 flow drifted to E ~ 1 at eps = 1; with |k| < 64 it diverged). s is signed so that the leading
even-order part of the linearised flow (its real symbol at the band edge, averaged over the data) is dissipative;
with none, s > 0. A flow whose leading even-order coefficient changes sign along the data is ill-posed in both
directions.
Defect after K snapshots E = ||S_K(T_s u0) - T_s u_K|| / ||T_s u_K - u_K||, size eps = ||T_s u0 - u0|| /
||u0 - mean u0||, r1 = ||J_K Q0 - Q_K|| / ||Q_K|| with the exact tangent (the s -> 0 limit, except for flows whose
linear part rotates the phases of the data's modes by more than a radian at the smallest size, such as K5's).

Generators: the true characteristics (fitted in the class) and the r + 2 lowest eigenvectors of exp46's pencil
(results/jets/{pde}_s{seed}.npz); flows on exp46's held-out trajectories (data 5600 + s), starts 0 and 16.

  python exp46_jet_flows.py --pde kdv --seed 42 --n-eval 2 --out results/jets/flows/kdv_s42.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import exp46_jet_symmetries as j46

NB = len(j46.MONS)


class JetFlow:
    def __init__(self, sol, sigma, k0, c, band=None):
        self.sol, self.sigma, self.k0 = sol, sigma, k0
        kb = float(sol.k.max()) * (2.0 / 3.0) if band is None else band
        self.mask = (sol.k.abs() < kb).to(torch.float64) * sol.mask
        self.c = np.asarray(c, dtype=np.float64)
        unit = {tuple(int(i == k) for i in range(j46.NJ)): k for k in range(j46.NJ)}
        L = torch.zeros_like(sol.k, dtype=torch.complex128)
        self.const, self.nl = 0.0, []
        for i, e in enumerate(j46.MONS):
            if self.c[i] == 0.0:
                continue
            if e in unit:
                L = L + self.c[i] * (sol.ik / k0) ** unit[e]
            elif sum(e) == 0:
                self.const = sigma * self.c[i]
            else:
                self.nl.append((self.c[i], e))
        self.L = L * self.mask

    def scaled(self, w):
        J = self.sol.jets(w)
        return [J[k] / (self.sigma * self.k0 ** k) for k in range(j46.NJ)]

    def N(self, v):
        """Masked Fourier transform of the explicit part (nonlinear monomials and the constant)."""
        S = self.scaled(self.sol.phys(v))
        q = torch.full_like(S[0], self.const)
        for c, e in self.nl:
            m = torch.ones_like(S[0])
            for k, ek in enumerate(e):
                if ek:
                    m = m * S[k] ** ek
            q = q + self.sigma * c * m
        return self.mask * torch.fft.rfft(q)

    def Q(self, w):
        v = self.sol.mask * torch.fft.rfft(w)
        return self.sol.phys(self.L * v + self.N(v))

    def coefficients(self, w):
        """a_j(x) = dQ / d(u_jx) of the full characteristic at w (physical units), j = 0..5: [6, ..., n]."""
        S = self.scaled(w)
        a = [torch.zeros_like(S[0]) for _ in range(j46.NJ)]
        for i, e in enumerate(j46.MONS):
            if self.c[i] == 0.0:
                continue
            for jj, ej in enumerate(e):
                if ej == 0:
                    continue
                m = torch.full_like(S[0], float(ej))
                for k, ek in enumerate(e):
                    p = ek - (k == jj)
                    if p:
                        m = m * S[k] ** p
                a[jj] = a[jj] + self.c[i] * m / self.k0 ** jj
        return a

    def rate(self, w):
        """Largest rate of the explicit part: sum over nonlinear monomials of |d m / d u_jx| k_max^j (sup over x)."""
        S = self.scaled(w)
        kmax = float((self.sol.k * self.mask).max())
        r = torch.zeros(w.shape[:-1], dtype=w.dtype)
        for c, e in self.nl:
            for jj, ej in enumerate(e):
                if ej == 0:
                    continue
                m = torch.full_like(S[0], float(ej))
                for k, ek in enumerate(e):
                    p = ek - (k == jj)
                    if p:
                        m = m * S[k] ** p
                r = r + abs(c) * m.abs().amax(-1) * (kmax / self.k0) ** jj
        return r

    def sign(self, w):
        """-sign of the mean real symbol of the linearised flow at the band edge (even orders); +1 without one."""
        kmax = float((self.sol.k * self.mask).max())
        a = self.coefficients(w)
        re = sum(float(a[jj].mean()) * (-1) ** (jj // 2) * kmax ** jj for jj in (2, 4))
        im = sum(abs(float(a[jj].mean())) * kmax ** jj for jj in (1, 3, 5))
        if abs(re) <= 1e-9 * (abs(re) + im) or re == 0.0:
            return 1.0
        return -1.0 if re > 0 else 1.0

    def apply(self, w, s, safety=2.0, every=16, max_steps=200_000):
        """Lawson RK4 in s for a batch w [B, n] with per-element s [B]. The step is at most 1 / (safety x rate),
        re-evaluated every `every` steps at the current state (a steepening flow takes smaller steps), and at most
        |s| / 8; an element whose rate becomes non-finite, or that has not reached s after max_steps (a flow
        approaching a blow-up), is marked failed (NaN)."""
        v = self.sol.mask * torch.fft.rfft(w)
        left, cap = s.clone(), s.abs() / 8
        steps = torch.zeros_like(s, dtype=torch.long)
        n = 0
        while bool((left != 0).any()) and n < max_steps:
            if n % every == 0:
                hmax = 1.0 / (safety * self.rate(self.sol.phys(v)))
                bad = torch.isnan(hmax) | (hmax <= 0)
                if bool(bad.any()):
                    v = torch.where(bad[:, None], torch.full_like(v, float("nan")), v)
                    left = torch.where(bad, torch.zeros_like(left), left)
                    hmax = torch.where(bad, torch.ones_like(hmax), hmax)
            hb = torch.sign(left) * torch.minimum(left.abs(), torch.minimum(hmax, cap))
            h = hb[:, None]
            E, E2 = torch.exp(self.L * h), torch.exp(self.L * h / 2)
            k1 = self.N(v)
            a2 = E2 * (v + h / 2 * k1)
            k2 = self.N(a2)
            a3 = E2 * v + h / 2 * k2
            k3 = self.N(a3)
            a4 = E * v + h * E2 * k3
            k4 = self.N(a4)
            v = E * v + h / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
            steps = steps + (hb != 0).long()
            left = left - hb
            left = torch.where(left.abs() <= 1e-12 * s.abs(), torch.zeros_like(left), left)
            n += 1
        v = torch.where((left != 0)[:, None], torch.full_like(v, float("nan")), v)
        return self.sol.phys(v), steps


def run_generator(flow, sol, cfg, ue, pairs, K, eps_list, safety):
    r, jj = pairs[:, 0], pairs[:, 1]
    u0, uK = ue[r, jj], ue[r, jj + K]
    P = len(pairs)
    Q0, QK = flow.Q(u0), flow.Q(uK)
    JQ = sol.transport(u0, Q0[:, None], K, cfg["dt"], cfg["nsub"])[:, -1, 0]
    r1 = math.sqrt(float(((JQ - QK) ** 2).sum()) / float((QK ** 2).sum()))
    sign = flow.sign(torch.cat([u0, uK]))
    base = ((u0 - u0.mean(-1, keepdim=True)) ** 2).sum(-1).sqrt()
    q0 = (Q0 ** 2).sum(-1).sqrt()
    out = {}
    for x in eps_list:
        s = sign * x * base / q0.clamp_min(1e-30)
        ab, ns = flow.apply(torch.cat([u0, uK]), torch.cat([s, s]), safety)
        a, b = ab[:P], ab[P:]
        flow_ok = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
        sk = sol.solve(torch.where(flow_ok[:, None], a, u0), K + 1, cfg["dt"], cfg["nsub"])[:, -1]
        G, Dl = sk - b, b - uK
        ok = flow_ok & torch.isfinite(G).all(-1)
        n_flow, n_solve = int((~flow_ok).sum()), int((flow_ok & ~ok).sum())
        num, den = float((G[ok] ** 2).sum()), float((Dl[ok] ** 2).sum())
        real = (((a - u0) ** 2).sum(-1).sqrt() / base)[ok]
        out[str(x)] = dict(E=math.sqrt(num / den) if (n_flow + n_solve == 0 and den > 0) else float("inf"),
                           eps=float(real.mean()) if len(real) else float("nan"), steps=int(ns.max()),
                           flow_fail=n_flow, solve_fail=n_solve)
    return r1, sign, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pde", required=True, choices=["kdv", "pme", "heat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stem", default=None, help="exp46 output stem (default results/jets/{pde}_s{seed})")
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--starts", default="0,16")
    ap.add_argument("--eps", default="0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--extra", type=int, default=2, help="eigenvectors beyond the expected rank")
    ap.add_argument("--safety", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=None, help="flow band |k| < band (default: the solver's 2/3 band; kdv: 32)")
    ap.add_argument("--nsub", type=int, default=None, help="as used for the exp46 run (kdv: 32)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    stem = args.stem or os.path.join(j46.RESULTS, "jets", f"{args.pde}_s{args.seed}")
    with open(stem + ".json") as f:
        meta = json.load(f)
    z = np.load(stem + ".npz")
    cfg = dict(meta["config"])
    if args.nsub is not None:
        cfg["nsub"] = args.nsub
    t0 = time.time()
    sol = j46.Solver(args.pde, cfg["n"])
    ue = sol.solve(j46.initial(args.pde, cfg["n"], args.n_eval, np.random.default_rng(5600 + args.seed), meta.get("ic", "default")),
                   cfg["nt"], cfg["dt"], cfg["nsub"])
    sigma, k0 = float(z["sigma"]), float(z["k0"])
    lam, C, Ct = z["lam"], z["C"], z["Ct"]
    r_true = meta["r_true"]
    band = args.band if args.band is not None else (32.0 if args.pde == "kdv" else None)
    eps_list = [float(x) for x in args.eps.split(",")]
    pairs = torch.tensor([(r, j) for r in range(args.n_eval) for j in (int(v) for v in args.starts.split(","))])
    gens = [(f"true {n}", "true", Ct[:, i]) for i, n in enumerate(meta["true_names"])]
    gens += [(f"eigenvector {i + 1} ({lam[i]:.1e})", "discovered" if i < r_true else "next", C[:, i] / np.abs(C[:, i]).max())
             for i in range(r_true + args.extra)]
    report = dict(pde=args.pde, seed=args.seed, config=cfg, K=args.K, safety=args.safety, band=band,
                  n_eval=args.n_eval, lam=lam[:r_true + 4].tolist(),
                  r_true=r_true, gens={})
    print(f"[{args.pde} flows] seed {args.seed}: n {cfg['n']}, dt {cfg['dt']}, nsub {cfg['nsub']}, K {args.K}, "
          f"{len(pairs)} pairs, eps {eps_list} ({time.time() - t0:.0f} s)", flush=True)
    for name, kind, c in gens:
        t1 = time.time()
        flow = JetFlow(sol, sigma, k0, c, band=band)
        r1, sign, res = run_generator(flow, sol, cfg, ue, pairs, args.K, eps_list, args.safety)
        report["gens"][name] = dict(kind=kind, sign=sign, r1=r1, E=res)
        print(f"  {name:<26s} s {'+' if sign > 0 else '-'}, r1 {r1:.2e} | E(eps): "
              + "  ".join(f"{res[str(x)]['E']:.2e} ({res[str(x)]['eps']:.3f})" for x in eps_list)
              + f"  [{time.time() - t1:.0f} s, max steps {max(v['steps'] for v in res.values())}, failed "
              f"{sum(v['flow_fail'] for v in res.values())}/{sum(v['solve_fail'] for v in res.values())}]", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
