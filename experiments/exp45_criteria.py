"""Criteria that separate the symmetry algebra from near-symmetries in exp45 (2D), without ground truth in the
criterion (ground truth only scores it).

  sensitivity -- Rayleigh quotients of the exact operator's eigenvectors through the learned FNO, same data, pairs and
                 basis: operator-limited losses (exact symmetries) rise with the operator's error, intrinsic losses
                 (near-symmetries) do not;
  affine      -- the matrix class aff(4) on (x, y, t, u): the 20 fields of degree <= 1 (5 x 5 matrices in homogeneous
                 coordinates), as the sub-pencil of the saved degree-2 pencils;
  mu          -- spectrum against the absolute normalisation term (exact operator, pencils recomputed);
  closure     -- escaped share of the brackets of the k lowest eigenvectors (exp44's measure), k = 1..10, and the
                 closure-gated rank: the largest k whose span leaves at most --tau of its brackets.

  python exp45_criteria.py --dir results/neural2d --seeds 42,43,44 --out results/neural2d/criteria.json
"""

import argparse
import json
import os

import numpy as np
import torch

import exp45_neural_2d as e

NM = len(e.MONS)
AFFINE = [q * NM + i for q in range(4) for i, m in enumerate(e.MONS) if sum(m) <= 1]


def fields(C):
    """Coefficient matrix [60, k] -> k fields, each four polynomial dicts {(a, b, c, d): coef} in X, Y, T, U
    (scaled components)."""
    return [[{e.MONS[i]: float(C[q * NM + i, j]) for i in range(NM) if C[q * NM + i, j] != 0.0} for q in range(4)]
            for j in range(C.shape[1])]


def pmul(p, r):
    o = {}
    for m, v in p.items():
        for n, w in r.items():
            key = tuple(a + b for a, b in zip(m, n))
            o[key] = o.get(key, 0.0) + v * w
    return o


def pdiff(p, axis):
    o = {}
    for m, v in p.items():
        if m[axis] > 0:
            key = tuple(a - (i == axis) for i, a in enumerate(m))
            o[key] = o.get(key, 0.0) + v * m[axis]
    return o


def padd(p, r, s=1.0):
    o = dict(p)
    for m, v in r.items():
        o[m] = o.get(m, 0.0) + s * v
    return o


def bracket(v, w):
    """[v, w] in the scaled coordinates (X, Y, T, U); components scale like the coordinates, so this is the
    physical bracket in scaled units."""
    out = []
    for q in range(4):
        acc = {}
        for r in range(4):
            acc = padd(acc, pmul(v[r], pdiff(w[q], r)))
            acc = padd(acc, pmul(w[r], pdiff(v[q], r)), -1.0)
        out.append(acc)
    return out


class Evaluator:
    """Polynomial fields at the probe points of a held-out set (exp45's Probe), scaled components, flattened."""

    def __init__(self, d, every_t=4, every_x=2):
        g = d.g
        sl = slice(g.sl.start, g.sl.stop, every_x)
        js = torch.arange(2, d.u.shape[1] - 2, every_t)
        R, J = torch.meshgrid(torch.arange(d.u.shape[0]), js, indexing="ij")
        u, _, _, _, X, Y, T = d.at(R, J, sl)
        self.V = [a.reshape(-1).numpy() for a in (X, Y, T, u / d.s["sigma"])]
        self.cache = {}

    def mono(self, m):
        if m not in self.cache:
            self.cache[m] = np.prod([v ** k for v, k in zip(self.V, m)], axis=0)
        return self.cache[m]

    def __call__(self, f):
        comps = []
        for q in range(4):
            acc = np.zeros_like(self.V[0])
            for m, c in f[q].items():
                acc = acc + c * self.mono(m)
            comps.append(acc)
        return np.concatenate(comps)


def closure(C, ev):
    """escaped share and bracket size of the span of C [60, k] (exp44's definition); escaped is nan for an abelian
    span."""
    F = fields(C)
    Dv = np.stack([ev(f) for f in F])
    _, s, Vt = np.linalg.svd(Dv.T, full_matrices=False)
    r = int((s > e.TOL * s[0]).sum())
    A = Vt[:r].T / s[:r]
    Qv = A.T @ Dv
    B = {(i, j): ev(bracket(F[i], F[j])) for i in range(len(F)) for j in range(i + 1, len(F))}
    tot = out = 0.0
    for a in range(r):
        for b in range(a + 1, r):
            g = sum((A[i, a] * A[j, b] - A[j, a] * A[i, b]) * B[i, j] for (i, j) in B)
            res = g - Qv.T @ (Qv @ g)
            tot += g @ g
            out += res @ res
    return (float(np.sqrt(out / tot)) if tot > 1e-24 else float("nan")), float(np.sqrt(2 * tot))


def gated_rank(esc, tau):
    """Largest k whose span leaves at most tau of its brackets (abelian spans count as closed)."""
    ok = [k + 1 for k, v in enumerate(esc) if not (v > tau)]
    return max(ok) if ok else 0


def rq(N, D, c):
    return float(c @ N @ c / (c @ D @ c))


def largest_ratio_rank(lam, n=10):
    first = lam[:n]
    r = np.where(first[:-1] > 0, first[1:] / np.where(first[:-1] > 0, first[:-1], 1.0), -np.inf)
    return int(np.argmax(r)) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(e.RESULTS, "neural2d"))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--mus", default="1,0.1,0.01")
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    g = e.Grid()
    report = {}
    for s in (int(x) for x in args.seeds.split(",")):
        dtr = e.Data(g, e.trajectories(g, 16, 4000 + s, 51, 0.02, 3), 0.02)
        dev = e.Data(g, e.trajectories(g, 8, 5000 + s, 51, 0.02, 3), 0.02, dtr.s)
        probe, ev = e.Probe(dev), Evaluator(dev)
        z = {op: np.load(os.path.join(args.dir, f"{op}_m6_s{s}.npz")) for op in ("exact", "fno")}
        P = {op: (z[op]["poly_N"], z[op]["poly_D"]) for op in z}
        rep = {}

        # V1: operator sensitivity
        lam_e, C_e = e.gsolve(*P["exact"])
        ratios = [rq(*P["fno"], C_e[:, i]) / lam_e[i] for i in range(12)]
        rep["sensitivity"] = dict(lam_exact=lam_e[:12].tolist(), rq_fno=[rq(*P["fno"], C_e[:, i]) for i in range(12)],
                                  ratio=ratios)
        print(f"[s{s}] V1 operator sensitivity, Rayleigh quotient through the FNO / exact eigenvalue, eigenvectors 1-12: "
              + " ".join(f"{v:.3g}" for v in ratios), flush=True)

        # V2: affine (matrix) class
        rep["affine"] = {}
        for op in ("exact", "fno"):
            N, D = P[op]
            la, Ca = e.gsolve(N[np.ix_(AFFINE, AFFINE)], D[np.ix_(AFFINE, AFFINE)])
            C60 = np.zeros((4 * NM, Ca.shape[1]))
            C60[AFFINE] = Ca
            sc = probe.scores(*probe.poly(C60[:, :6]))
            lam2 = e.gsolve(N, D)[0]
            rep["affine"][op] = dict(lam=la[:12].tolist(), gap=float(la[6] / la[5]), gap_degree2=float(lam2[6] / lam2[5]),
                                     rank_largest_ratio=largest_ratio_rank(la), missing_q=sc["missing_q"],
                                     missing_f=sc["missing_f"])
            print(f"[s{s}] V2 affine class, {op}: lowest " + " ".join(f"{v:.1e}" for v in la[:9])
                  + f"; lambda7/lambda6 {la[6] / la[5]:.0f} (degree 2: {lam2[6] / lam2[5]:.0f}); largest-ratio rank "
                  f"{largest_ratio_rank(la)}; held-out missing {sc['missing_q']:.4f} (fields {sc['missing_f']:.4f})", flush=True)

        # V3: normalisation sensitivity (exact operator, pencil recomputed)
        op64 = e.ExactOp(g, 0.02, 3)
        N, DA, G0, trBB = e.pencil(op64, dtr, e.chars_poly, e.pairs_of(16, 51, 8, 4), 8, 2)
        cmu = trBB / np.trace(G0)
        lam1, C1 = e.gsolve(N, DA + cmu * G0)
        rep["mu"] = {}
        for mu in (float(x) for x in args.mus.split(",")):
            Dm = DA + mu * cmu * G0
            lm, Cm = e.gsolve(N, Dm)
            sc = probe.scores(*probe.poly(Cm[:, :6]))
            r78 = [rq(N, Dm, C1[:, i]) / lam1[i] for i in (6, 7)]
            rep["mu"][str(mu)] = dict(lam=lm[:10].tolist(), rq78_ratio=r78, missing_q=sc["missing_q"])
            print(f"[s{s}] V3 mu = {mu:g}: lowest " + " ".join(f"{v:.1e}" for v in lm[:9])
                  + f"; mu = 1 eigenvectors 7, 8: quotient / mu = 1 eigenvalue {r78[0]:.3g}, {r78[1]:.3g}; six lowest "
                  f"held-out missing {sc['missing_q']:.4f}", flush=True)

        # V4: closure-gated rank (degree-2 class)
        rep["closure"] = {}
        rng = np.random.default_rng(s)
        for op in ("exact", "fno"):
            lam_o, C_o = e.gsolve(*P[op])
            esc = [closure(C_o[:, :k], ev)[0] for k in range(1, args.kmax + 1)]
            rep["closure"][op] = dict(escaped=esc, rank=gated_rank(esc, args.tau))
            print(f"[s{s}] V4 closure, {op}: escaped share, k = 1..{args.kmax}: "
                  + " ".join("abel" if np.isnan(v) else f"{v:.3f}" for v in esc)
                  + f"; closure-gated rank {gated_rank(esc, args.tau)}", flush=True)
        nulls = {k: [closure(rng.standard_normal((4 * NM, k)), ev)[0] for _ in range(args.null_draws)] for k in (6, 7)}
        rep["closure"]["null"] = {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in nulls.items()}
        Ct = e.true_coefficients()
        rep["closure"]["ground_truth"] = closure(Ct, ev)[0]
        print(f"[s{s}] V4 null (Gaussian coefficients): k = 6 {np.mean(nulls[6]):.3f} +- {np.std(nulls[6]):.3f}, "
              f"k = 7 {np.mean(nulls[7]):.3f} +- {np.std(nulls[7]):.3f}; ground-truth algebra {rep['closure']['ground_truth']:.2e}",
              flush=True)
        report[s] = rep
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
