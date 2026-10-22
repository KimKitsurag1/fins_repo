"""Closure-gated rank for saved 1D FINS runs (exp39 / exp40 npz): escaped share of the brackets of the k lowest
eigenvectors (exp44's measure) for k = 1..(saved columns), and the largest k whose span leaves at most --tau.

  python exp44_closure_rank.py --files results/exact_kdv_s42.npz results/exact_heat_s42.npz
"""

import argparse

import numpy as np

import exp44_closure as e44


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--band", type=float, default=0.2)
    args = ap.parse_args()
    for path in args.files:
        z = dict(np.load(path, allow_pickle=True))
        ev = e44.Evaluator(z, args.band)
        mons = [tuple(int(v) for v in m) for m in z["mons"]]
        S = (float(z["Lw"]), float(z["T_max"]), float(z["sigma"]))
        C = z["C"]
        esc = [e44.closure(C[:, :k], mons, S, ev)[0] for k in range(1, C.shape[1] + 1)]
        ok = [k + 1 for k, v in enumerate(esc) if not (v > args.tau)]
        lam = z["lam"]
        r = lam[1:len(esc)] / lam[:len(esc) - 1]
        print(f"{path}: k_true {int(z['k'])}; escaped k = 1..{len(esc)}: "
              + " ".join("abel" if np.isnan(v) else f"{v:.3f}" for v in esc)
              + f"; closure-gated rank {max(ok) if ok else 0}; largest-ratio rank {int(np.argmax(r)) + 1}", flush=True)


if __name__ == "__main__":
    main()
