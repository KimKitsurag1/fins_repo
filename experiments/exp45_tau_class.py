"""The tau = tau(t) class (V5): point symmetries of a second-order evolution equation have tau depending on t alone,
so the degree-2 class of exp45 can drop every tau monomial in x, y or u (60 -> 48 fields) without losing a symmetry.
Sub-pencil of the saved 60 x 60 pencils (exact_m6 and fno_m6, training pairs), scored on held-out probe points; the
degree-2 and affine classes for comparison.

  python exp45_tau_class.py --dir results/neural2d --seeds 42,43,44 --out results/neural2d/tau_class.json
"""

import argparse
import json
import os

import numpy as np

import exp45_criteria as cr
import exp45_neural_2d as e

NM = len(e.MONS)
TAU_T = [q * NM + i for q in range(4) for i, m in enumerate(e.MONS) if q != 2 or (m[0] == 0 and m[1] == 0 and m[3] == 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(e.RESULTS, "neural2d"))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    g = e.Grid()
    report = {}
    print(f"tau = tau(t) class: {len(TAU_T)} of {4 * NM} fields; affine class {len(cr.AFFINE)}", flush=True)
    for s in (int(x) for x in args.seeds.split(",")):
        dtr = e.Data(g, e.trajectories(g, 16, 4000 + s, 51, 0.02, 3), 0.02)
        dev = e.Data(g, e.trajectories(g, 8, 5000 + s, 51, 0.02, 3), 0.02, dtr.s)
        probe = e.Probe(dev)
        rep = {}
        for op in ("exact", "fno"):
            z = np.load(os.path.join(args.dir, f"{op}_m6_s{s}.npz"))
            N, D = z["poly_N"], z["poly_D"]
            out = {}
            for name, idx in (("degree2", list(range(4 * NM))), ("tau_t", TAU_T), ("affine", cr.AFFINE)):
                lam, C = e.gsolve(N[np.ix_(idx, idx)], D[np.ix_(idx, idx)])
                C60 = np.zeros((4 * NM, C.shape[1]))
                C60[idx] = C
                sc = probe.scores(*probe.poly(C60[:, :6]))
                out[name] = dict(lam=lam[:10].tolist(), gap=float(lam[6] / lam[5]), rank=cr.largest_ratio_rank(lam),
                                 missing_q=sc["missing_q"], missing_f=sc["missing_f"])
            rep[op] = out
            print(f"[s{s}] {op}: " + "; ".join(
                f"{name} lambda6 {v['lam'][5]:.1e} lambda7 {v['lam'][6]:.1e} gap {v['gap']:.0f} rank {v['rank']} missing "
                f"{v['missing_q']:.4f}" for name, v in out.items()), flush=True)
        report[s] = rep
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
