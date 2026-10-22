"""exp23_stats — aggregate discovery cells into reportable statistics.

Reads the npz files written by exp23/exp24 and produces the reported
statistics. Three choices:

* WITHIN-SEED PAIRED CONTRASTS, not cross-cell means. The absolute level
  of novel containment moves with slot count and with the evaluation set,
  so only differences measured at identical settings are portable. The
  sign test over seeds is the headline; the mean difference is secondary.
* BOOTSTRAP CI over seeds, reported alongside n. With three to five seeds
  a CI is wide by construction, and showing it is the point.
* TWO METRICS side by side: per-slot best-match cosine (what LieSym and
  FINS report) and novel-component containment relative to the essential
  trio (what survives the correlation inflation of exp19). They answer
  different questions and disagree, so both go in the table.

Run:  python exp23_stats.py --dir results
      python exp23_stats.py --dir results --baseline on_spectral
"""

import argparse
import glob
import itertools
import os
import re

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pde"))
from closure_metrics import closure_report, describe, fit_affine  # noqa: E402

AL, L, T = 0.01, 2 * np.pi, 1.0
NAMES = ["d_x", "d_t", "u d_u", "Galilean", "scaling", "projective"]
ESS = ("d_x", "d_t", "u d_u")


def gt_fields(x, t, u):
    X, Tm = np.meshgrid(x, t, indexing="ij")
    Xb, Tb = np.broadcast_to(X, u.shape), np.broadcast_to(Tm, u.shape)
    Z, O = np.zeros_like(u), np.ones_like(u)
    return {
        "d_x": (O, Z, Z), "d_t": (Z, O, Z), "u d_u": (Z, Z, u),
        "Galilean": (2 * AL * Tb, Z, -Xb * u),
        "scaling": (Xb, 2 * Tb, Z),
        "projective": (4 * AL * Tb * Xb, 4 * AL * Tb ** 2,
                       -(Xb ** 2 + 2 * AL * Tb) * u),
    }


def rows(triples):
    M = np.stack([np.concatenate([c.ravel() for c in tr]) for tr in triples])
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)


def gt_affine(x, t):
    """The heat algebra in the (xi, eta, a, b) representation."""
    X, Tm = np.meshgrid(x, t, indexing="ij")
    Z, O = np.zeros_like(X), np.ones_like(X)
    return [(O, Z, Z, Z), (Z, O, Z, Z), (Z, Z, O, Z),
            (2 * AL * Tm, Z, -X, Z), (X, 2 * Tm, Z, Z),
            (4 * AL * Tm * X, 4 * AL * Tm ** 2, -(X ** 2 + 2 * AL * Tm), Z)]


def analyse(path):
    z = np.load(path)
    F, u, x, t = z["fields"], z["u"], z["x"], z["t"]
    gt = gt_fields(x, t, u)
    A = rows([tuple(F[k]) for k in range(F.shape[0])])
    G = rows([gt[n] for n in NAMES])
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[sv / sv[0] > 1e-6]
    ess = [NAMES.index(n) for n in ESS]
    Qe, _ = np.linalg.qr(G[ess].T)
    out = {"n_slots": F.shape[0], "dim": V.shape[0],
           "random_ref": float(np.sqrt(V.shape[0] / A.shape[1]))}
    sim = np.abs(A @ G.T)                                  # [slots, gt]
    for j, n in enumerate(NAMES):
        out[f"cos:{n}"] = float(sim[:, j].max())           # best-match
        g = G[j] - Qe @ (Qe.T @ G[j])
        nf = float(np.linalg.norm(g))
        out[f"novel:{n}"] = (float(np.linalg.norm(V @ (g / nf)))
                             if nf > 1e-8 else np.nan)

    # closure: recover (xi, eta, a, b) per slot
    # from the multi-sample fields, then bracket in that representation
    dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
    if u.ndim >= 3 and u.shape[0] >= 2:
        gens, spreads = [], []
        for k in range(F.shape[0]):
            g, sp = fit_affine(F[k, 0], F[k, 1], F[k, 2], u)
            gens.append(g)
            spreads.append(sp)
        rep = closure_report(gens, dx, dt, basis=(x, t))
        out["escaped"] = rep["escaped"]
        out["structure"] = rep["structure"]
        out["abelian_pairs"] = rep["abelian_pairs"]
        out["n_pairs"] = rep["n_pairs"]
        out["xi_spread"] = float(np.mean(spreads))
        out["verdict"] = describe(rep)
    return out


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) < 2:
        return (np.nan, np.nan)
    s = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--pattern", default="exp2*_*_s*.npz")
    ap.add_argument("--baseline", default=None,
                    help="cell name used as the paired baseline, "
                         "e.g. on_spectral")
    ap.add_argument("--metric", default="novel",
                    choices=["novel", "cos", "both"])
    args = ap.parse_args()

    cells = {}
    for p in sorted(glob.glob(os.path.join(args.dir, args.pattern))):
        m = re.match(r".*?exp\d+_(.+)_s(\d+)\.npz$", os.path.basename(p))
        if not m:
            continue
        cells.setdefault(m.group(1), {})[int(m.group(2))] = analyse(p)
    if not cells:
        print(f"no npz matching {args.pattern} in {args.dir}")
        return

    print("=" * 92)
    print("PER-CELL SUMMARY   (mean over seeds; bootstrap 95% CI; n = seeds)")
    print("=" * 92)
    keys = ([f"novel:{n}" for n in NAMES if n not in ESS]
            if args.metric != "cos" else [])
    keys += [f"cos:{n}" for n in NAMES] if args.metric != "novel" else []
    for cell, per in sorted(cells.items()):
        seeds = sorted(per)
        ref = np.mean([per[s]["random_ref"] for s in seeds])
        print(f"\n  {cell}   seeds {seeds}   slots "
              f"{per[seeds[0]]['n_slots']}   random ref {ref:.3f}")
        for k in keys:
            v = [per[s][k] for s in seeds]
            lo, hi = boot_ci(v)
            print(f"    {k:<22} {np.mean(v):.3f}   "
                  f"[{lo:.3f}, {hi:.3f}]   " +
                  " ".join(f"s{s}={per[s][k]:.3f}" for s in seeds))
        if "escaped" in per[seeds[0]]:
            e = [per[s]["escaped"] for s in seeds]
            st = [per[s]["structure"] for s in seeds]
            sp = [per[s]["xi_spread"] for s in seeds]
            ab = [f"{per[s]['abelian_pairs']}/{per[s]['n_pairs']}"
                  for s in seeds]
            print(f"    {'closure: escaped':<22} {np.nanmean(e):.3f}   "
                  + " ".join(f"s{s}={per[s]['escaped']:.3f}" for s in seeds))
            print(f"    {'closure: structure':<22} {np.mean(st):.2e}   "
                  f"abelian pairs {' '.join(ab)}   "
                  f"xi/eta spread {np.mean(sp):.2f}")
            print(f"    -> {per[seeds[0]]['verdict']}")

    if args.baseline is None:
        pairs = list(itertools.combinations(sorted(cells), 2))
    else:
        if args.baseline not in cells:
            print(f"\nbaseline {args.baseline} not found")
            return
        pairs = [(args.baseline, c) for c in sorted(cells)
                 if c != args.baseline]
    print("\n" + "=" * 92)
    print("WITHIN-SEED PAIRED CONTRASTS   (the portable statistic)")
    print("=" * 92)
    for a, b in pairs:
        common = sorted(set(cells[a]) & set(cells[b]))
        if not common:
            continue
        print(f"\n  {b}  minus  {a}   (seeds {common})")
        for k in keys:
            d = [cells[b][s][k] - cells[a][s][k] for s in common]
            lo, hi = boot_ci(d)
            pos = sum(1 for v in d if v > 0)
            print(f"    {k:<22} {np.mean(d):+.3f}   [{lo:+.3f}, {hi:+.3f}]"
                  f"   sign {pos}/{len(d)}   " +
                  " ".join(f"{v:+.3f}" for v in d))
    print("\nRead the sign test first: with 3-5 seeds the CI is wide by "
          "construction.\nAbsolute levels are NOT comparable across "
          "different slot counts.")


if __name__ == "__main__":
    main()
