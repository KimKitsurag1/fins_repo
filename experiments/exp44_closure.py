"""Experiment 44 -- closure of a discovered span under the Lie bracket.

FINS's generators are polynomials in X = (x - x0) / Lw, T = t / T_max, U = u / sigma, so their brackets are
exact polynomial computations. With e_1..e_r an orthonormal basis of the span V on the scored region
(exp42's region and component scaling),
    escaped   = ( sum_ab ||(I - P_V) [e_a, e_b]||^2 / sum_ab ||[e_a, e_b]||^2 )^1/2
    structure = ( sum_ab ||[e_a, e_b]||^2 )^1/2,
both independent of the basis chosen in V. escaped is 0 for a closed span; structure ~ 0 flags an abelian
or degenerate span, for which escaped is undefined. References per seed: the ground-truth algebra of the
class (fitted to the class off-shell, on random u) and random k-subspaces of the class (Gaussian
coefficients, as exp42's class null).

Run:  python exp44_closure.py --dir results [--pdes burgers,kdv,heat] [--seeds 42,43,44]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
from exp42_tables import DEGREE, FINITE, TOL, ground_truth  # noqa: E402


# ---- polynomial vector fields: component q -> {(a, b, c): coefficient} in X, T, U ----
def fields_from_coef(C, mons):
    """Coefficient matrix [3 nm, k] -> k fields, each a list of three polynomial dicts."""
    nm = len(mons)
    out = []
    for j in range(C.shape[1]):
        out.append([{tuple(int(e) for e in m): float(C[q * nm + i, j]) for i, m in enumerate(mons)
                     if C[q * nm + i, j] != 0.0} for q in range(3)])
    return out


def pmul(p, r):
    o = {}
    for (a, b, c), v in p.items():
        for (d, e, f), w in r.items():
            key = (a + d, b + e, c + f)
            o[key] = o.get(key, 0.0) + v * w
    return o


def pdiff(p, axis):
    o = {}
    for m, v in p.items():
        if m[axis] > 0:
            key = tuple(e - (i == axis) for i, e in enumerate(m))
            o[key] = o.get(key, 0.0) + v * m[axis]
    return o


def padd(p, r, s=1.0):
    o = dict(p)
    for m, v in r.items():
        o[m] = o.get(m, 0.0) + s * v
    return o


def bracket(v, w, S):
    """[v, w] of physical fields (components xi, eta, phi as polynomials in the scaled variables);
    S = (Lw, T_max, sigma) converts between physical and scaled components."""
    vn = [{m: c / S[q] for m, c in v[q].items()} for q in range(3)]
    wn = [{m: c / S[q] for m, c in w[q].items()} for q in range(3)]
    out = []
    for q in range(3):
        acc = {}
        for r in range(3):
            acc = padd(acc, pmul(vn[r], pdiff(wn[q], r)))
            acc = padd(acc, pmul(wn[r], pdiff(vn[q], r)), -1.0)
        out.append({m: c * S[q] for m, c in acc.items()})
    return out


class Evaluator:
    """Fields evaluated on exp42's scored region with its component scaling, flattened."""

    def __init__(self, z, band):
        u, xs, ts = z["u"], z["xs"], z["ts"]
        Nx = u.shape[1]
        frac = np.arange(Nx) / Nx
        keep = (frac >= band) & (frac <= 1 - band)
        L = (xs[0, 1] - xs[0, 0]) * Nx
        self.w = np.array([1 / L, 1 / ts[:, -1].max(), 1 / u.std()])
        X = ((xs - z["x_offset"][:, None]) / z["Lw"])[:, :, None] * np.ones_like(u)
        T = (ts / z["T_max"])[:, None, :] * np.ones_like(u)
        U = u / z["sigma"]
        crop = lambda a: a[:, keep][..., 8:-8]
        self.V = [crop(X), crop(T), crop(U)]
        self.cache = {}

    def mono(self, m):
        if m not in self.cache:
            self.cache[m] = self.V[0] ** m[0] * self.V[1] ** m[1] * self.V[2] ** m[2]
        return self.cache[m]

    def __call__(self, f):
        comps = []
        for q in range(3):
            acc = np.zeros_like(self.V[0])
            for m, c in f[q].items():
                acc = acc + c * self.mono(m)
            comps.append(acc.ravel() * self.w[q])
        return np.concatenate(comps)


def closure(C, mons, S, ev):
    """escaped, structure and the rank of the span of coefficient vectors C [3 nm, k]."""
    F = fields_from_coef(C, mons)
    Dv = np.stack([ev(f) for f in F])                               # [k, dim]
    _, s, Vt = np.linalg.svd(Dv.T, full_matrices=False)
    r = int((s > TOL * s[0]).sum())
    A = Vt[:r].T / s[:r]                                            # e_a = sum_i A[i, a] v_i, orthonormal
    Qv = A.T @ Dv                                                   # [r, dim]
    k = len(F)
    B = {}
    for i in range(k):
        for j in range(i + 1, k):
            B[i, j] = ev(bracket(F[i], F[j], S))
    tot, out = 0.0, 0.0
    for a in range(r):
        for b in range(a + 1, r):
            g = sum(A[i, a] * A[j, b] * B[i, j] - A[j, a] * A[i, b] * B[i, j] for (i, j) in B)
            res = g - Qv.T @ (Qv @ g)
            tot += g @ g
            out += res @ res
    return (np.sqrt(out / tot) if tot > 0 else float("nan")), np.sqrt(2 * tot), r


def fit_ground_truth(z, pde, cfg, rng):
    """Ground-truth algebra as coefficients of the class, fitted off-shell (random u) where the class
    spans it exactly; returns [3 nm, kB], names and the relative fit residual."""
    u = z["u"]
    ur = u.mean() + u.std() * rng.standard_normal(u.shape)
    names, gt = ground_truth(pde, ur, z["xs"], z["ts"], cfg)
    zr = dict(z, u=ur)
    X = ((zr["xs"] - zr["x_offset"][:, None]) / zr["Lw"])[:, :, None] * np.ones_like(ur)
    T = (zr["ts"] / zr["T_max"])[:, None, :] * np.ones_like(ur)
    U = ur / zr["sigma"]
    mono = np.stack([(X ** a * T ** b * U ** c).ravel() for a, b, c in z["mons"]], 1)   # [pts, nm]
    nm = mono.shape[1]
    M = np.zeros((3 * mono.shape[0], 3 * nm))
    for q in range(3):
        M[q * mono.shape[0]:(q + 1) * mono.shape[0], q * nm:(q + 1) * nm] = mono
    G = np.stack([g.reshape(3, -1).ravel() for g in gt], 1)
    Cg, *_ = np.linalg.lstsq(M, G, rcond=None)
    resid = float(np.linalg.norm(M @ Cg - G) / np.linalg.norm(G))
    return Cg, list(names), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("FINS_RESULTS_DIR", "."))
    ap.add_argument("--pdes", default="burgers,kdv,heat")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--band", type=float, default=0.2)
    ap.add_argument("--null-draws", type=int, default=30)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    print("closure of the span: escaped share of the brackets (0 closed); structure = bracket size of an "
          "orthonormal basis; null = random k-subspaces of the class, mean +/- sd")
    for pde in args.pdes.split(","):
        for s in [int(x) for x in args.seeds.split(",")]:
            runs = {}
            for tag, label in (("fins", "FINS"), ("exact", "FINS, exact operator")):
                p = os.path.join(args.dir, f"{tag}_{pde}_s{s}.npz")
                if os.path.exists(p):
                    runs[label] = dict(np.load(p, allow_pickle=True))
            if not runs:
                continue
            z0 = next(iter(runs.values()))
            ev = Evaluator(z0, args.band)
            mons = [tuple(int(e) for e in m) for m in z0["mons"]]
            S = (float(z0["Lw"]), float(z0["T_max"]), float(z0["sigma"]))
            rng = np.random.default_rng(s)
            Cg, names, resid = fit_ground_truth(z0, pde, cfg, rng)
            k = int(z0["k"])
            lines = [f"[{pde} s{s}] data seed {int(z0['data_seed'])}, k = {k}, ground truth fitted off-shell "
                     f"(relative residual {resid:.1e})"]
            refs = {f"ground truth, finite ({FINITE[pde]})": Cg[:, :FINITE[pde]]}
            if Cg.shape[1] > FINITE[pde]:
                refs[f"ground truth, full in the class ({Cg.shape[1]})"] = Cg
            for label, z in runs.items():
                refs[label] = z["C"][:, :int(z["k"])]
            for label, C in refs.items():
                e, st, r = closure(C, mons, S, ev)
                lines.append(f"  {label:<36s} escaped {e:.3f}  structure {st:.2e}  rank {r}")
            nulls = [closure(rng.standard_normal((3 * len(mons), k)), mons, S, ev) for _ in range(args.null_draws)]
            ne = np.array([n[0] for n in nulls]); ns = np.array([n[1] for n in nulls])
            lines.append(f"  {'null, Gaussian coefficients':<36s} escaped {ne.mean():.3f} +/- {ne.std():.3f}  "
                         f"structure {ns.mean():.2e}")
            print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
