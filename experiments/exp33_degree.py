"""Experiment 33 -- the spectrum as a function of the class degree d.

Blocks are computed once at the largest degree; the problem at a smaller
degree is the same problem restricted to the columns of monomials of degree
<= d. Printed per degree, ground-truth free: the lowest eigenvalues, the
share of each low eigenvector carried by monomials of exactly degree d
("new at this degree"), and how much of the degree-d low subspace survives
at degree d + 1. The share outside the algebra is ground truth, printed for
reading only.

Run:  python exp33_degree.py --pde burgers --op exact --aligned --ut central4 --dmax 4
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


def null_cosines(P, q, rng, draws=500, pct=99):
    """Per-rank percentile of principal-angle cosines between random q-dim subspaces of R^P."""
    cs = [np.linalg.svd(np.linalg.qr(rng.standard_normal((P, q)))[0].T
                        @ np.linalg.qr(rng.standard_normal((P, q)))[0], compute_uv=False)
          for _ in range(draws)]
    return np.percentile(np.array(cs), pct, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="exact", choices=["exact", "stepper"])
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--ut", default="central", choices=["central", "central4"])
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--stepper-modes", type=int, default=64)
    ap.add_argument("--reliability-filter", type=float, default=None)
    ap.add_argument("--top", type=int, default=12, help="low eigen-directions shown per degree")
    ap.add_argument("--q", type=int, default=6, help="low directions compared across degrees")
    ap.add_argument("--cv", action="store_true", help="two-fold cross-validation over trajectories")
    ap.add_argument("--qcv", type=int, default=8, help="low directions compared between folds")
    ap.add_argument("--stable", type=float, default=0.95, help="cosine counted as agreement")
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()

    xs, ts, u, ut, F, Qb, mons, A, B, keep, scale = e32.prepare(args, args.dmax)
    nm = len(mons)
    deg = np.array([sum(mons[p % nm]) for p in range(3 * nm)])
    rms = np.sqrt((F.reshape(F.shape[0], -1) ** 2).mean(1))

    # ground truth, for reading only: algebra fields and their degree-dmax coefficients
    names, gt = e31.algebra(args.pde, xs, ts, u, cfg.alpha_heat, args.dmax)
    Fm = F.reshape(F.shape[0], -1).T
    cg, *_ = np.linalg.lstsq(Fm, np.stack([g.ravel() for g in gt]).T, rcond=None)
    gdeg = [int(deg[np.abs(c * rms) > 1e-6 * np.abs(c * rms).max()].max()) for c in cg.T]

    print(f"[{args.pde}, {args.op}] seed {args.seed}, {u.shape[0]} trajectories, u_t {args.ut}, "
          f"K = {args.K}, starts every {args.every}; algebra: "
          + ", ".join(f"{n} (degree {g})" for n, g in zip(names, gdeg)))
    prev = None
    counts = {"v1": {}, "v2": {}}
    rng = np.random.default_rng(0)
    for d in range(1, args.dmax + 1):
        idx = np.where(deg <= d)[0]
        lam, C, Nm, Dm, sc = e31.discover(A[:, idx], B[:, idx])
        W = C * rms[idx, None]                                   # RMS-weighted coefficients
        top = min(args.top, W.shape[1])
        new = [float(np.linalg.norm(W[deg[idx] == d, i]) / np.linalg.norm(W[:, i])) for i in range(top)]
        # ground truth (reading only): share outside the algebra members representable at d
        inside = [j for j, g in enumerate(gdeg) if g <= d]
        G = cg[idx][:, inside] * rms[idx, None]
        Qg, _ = np.linalg.qr(G) if inside else (np.zeros((len(idx), 0)), None)
        out = [float(np.linalg.norm(W[:, i] - Qg @ (Qg.T @ W[:, i])) / np.linalg.norm(W[:, i]))
               for i in range(top)]
        print(f"\n  degree {d}: {len(idx)} coefficients; algebra members in the class: {len(inside)}")
        print("    i   lambda    new-at-d  outside(GT)")
        for i in range(top):
            print(f"    {i + 1:<3d} {lam[i]:.1e}   {new[i]:.2f}      {out[i]:.2f}")
        if prev is not None:
            # survival of the previous degree's low subspace (first q directions) at this degree
            pidx, pW, q = prev
            emb = np.zeros((len(idx), q))
            pos = {p: k for k, p in enumerate(idx)}
            for k, p in enumerate(pidx):
                emb[pos[p]] = pW[k, :q]
            Qp, _ = np.linalg.qr(emb)
            Qn, _ = np.linalg.qr(W[:, :q + 3])
            s = np.linalg.svd(Qn.T @ Qp, compute_uv=False)
            print(f"    survival of degree {d - 1}'s {q} lowest directions inside degree {d}'s "
                  f"{q + 3} lowest (cosines): " + " ".join(f"{v:.2f}" for v in s))
        prev = (idx, W, min(args.q, W.shape[1] - 3))
        if args.cv:
            # two folds over trajectories: fit on one, score the fitted
            # eigenvectors on the other (ground-truth free)
            J = len(range(0, u.shape[2] - args.K, args.every))
            Ar = A[:, idx].reshape(J, u.shape[0], -1, len(idx))
            Br = B[:, idx].reshape(J, u.shape[0], -1, len(idx))
            subs = []
            for fold in (0, 1):
                fit = np.arange(u.shape[0]) % 2 == fold
                Af, Bf = Ar[:, fit].reshape(-1, len(idx)), Br[:, fit].reshape(-1, len(idx))
                Av, Bv = Ar[:, ~fit].reshape(-1, len(idx)), Br[:, ~fit].reshape(-1, len(idx))
                lf, Cf, _, _, _ = e31.discover(Af, Bf)
                Nv = (Av - Bv).T @ (Av - Bv)
                Dv = Av.T @ Av + Bv.T @ Bv
                val = [float(c @ Nv @ c / (c @ Dv @ c)) for c in Cf[:, :top].T]
                Wf = Cf[:, :top] * rms[idx, None]
                outf = [float(np.linalg.norm(w - Qg @ (Qg.T @ w)) / np.linalg.norm(w)) for w in Wf.T]
                print(f"    fold {fold} (fit on {fit.sum()}, score on {(~fit).sum()} trajectories): "
                      "own / held-out loss [outside(GT)]")
                print("      " + "  ".join(f"{a:.0e}/{b:.0e}[{o:.1f}]" for a, b, o in zip(lf[:top], val, outf)))
                subs.append(Cf * rms[idx, None])
            for rule, q in (("v1", 8), ("v2", min(16, len(idx) // 2 - 1))):
                S0 = np.linalg.qr(subs[0][:, :q])[0]
                S1 = np.linalg.qr(subs[1][:, :q])[0]
                U, cos, _ = np.linalg.svd(S0.T @ S1)
                thr = np.maximum(args.stable, null_cosines(len(idx), q, rng))
                ok = cos >= thr
                counts[rule][d] = int(ok.sum())
                print(f"    [{rule}] fold agreement, {q} lowest directions: "
                      + " ".join(f"{v:.2f}" for v in cos) + f"   stable beyond chance: {int(ok.sum())}")
                if rule == "v2" and ok.any():
                    # ground truth, reading only: what the stable subspace holds
                    P_st = np.linalg.qr(S0 @ U[:, ok])[0]
                    cont = [float(np.linalg.norm(P_st.T @ g) / np.linalg.norm(g)) for g in G.T]
                    outs = [float(np.linalg.norm(p - Qg @ (Qg.T @ p))) for p in P_st.T]
                    print("         (GT) algebra members inside the stable subspace: "
                          + "  ".join(f"{names[j]} {c:.2f}" for j, c in zip(inside, cont)))
                    print("         (GT) stable directions outside the algebra: "
                          + " ".join(f"{v:.2f}" for v in outs))


    if counts["v1"]:
        c1 = counts["v1"]
        best = max(c1, key=lambda k: (c1[k], -k))
        print(f"\n  rule v1 (q = 8, most stable, ties -> lower): degree {best}, rank {c1[best]}  {c1}")
        c2 = counts["v2"]
        ds = sorted(c2)
        best2 = ds[-1]
        for a, b in zip(ds, ds[1:]):
            if c2[b] < c2[a]:
                best2 = a
                break
        print(f"  rule v2 (q adaptive, raise d while stable count does not fall): degree {best2}, "
              f"rank {c2[best2]}  {c2}")


if __name__ == "__main__":
    main()
