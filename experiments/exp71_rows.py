"""Rows of Kreider et al. (exp71, both readings, soft readout) in the paper's tables tab:main (clean, means over seeds
42-44), tab:noise (means over seeds 42-51, finite / non-trivial) and tab:new-learned (clean, means over seeds 42-44,
finite / non-trivial), from the studies' summaries.

Run:  python exp71_rows.py [--results results]
"""

import argparse
import json
import os

import numpy as np

LEVELS = ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1")
STUDY = {"heat": "noise_weak", "kdv": "noise_weak", "bo": "bo", "ks": "ks", "burgers": "burgers"}
ROWS = (("Kreider et al., solutions", "Kreider et al., solutions, soft"),
        ("Kreider et al., equation surface", "Kreider et al., equation surface, soft"))
SEEDS_MAIN = ("42", "43", "44")


def missing(summ, cond, pde, label, cv):
    return summ.get(cond, {}).get(pde, {}).get(label, {}).get(cv, {}).get("missing", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    summ = {pde: json.load(open(os.path.join(args.results, d, "summary.json"))) for pde, d in STUDY.items()}
    print("% tab:main, means over seeds 42-44: Burgers (5), KdV (4), heat (6), heat class (10)")
    for name, label in ROWS:
        cells = []
        for pde, cv in (("burgers", "finite"), ("kdv", "finite"), ("heat", "finite"), ("heat", "full")):
            m = missing(summ[pde], "clean", pde, label, cv)
            cells.append(f"{np.mean([m[s] for s in SEEDS_MAIN]):.2f}")
        print(f"{name} & " + " & ".join(cells) + " \\\\")
    print("% tab:noise, means over seeds 42-51: finite / non-trivial, levels " + ", ".join(LEVELS))
    for pde in ("heat", "kdv", "bo", "ks"):
        print(f"% {pde}")
        for name, label in ROWS:
            cells = []
            for c in LEVELS:
                f = missing(summ[pde], c, pde, label, "finite")
                n = missing(summ[pde], c, pde, label, "non-trivial")
                cells.append(f"{np.mean(list(f.values())):.2f} / {np.mean(list(n.values())):.2f}" if f else "--")
            print(f"{name} & " + " & ".join(cells) + " \\\\")
    print("% tab:new-learned, means over seeds 42-44: finite / non-trivial; Benjamin-Ono, Kuramoto-Sivashinsky")
    for name, label in ROWS:
        cells = []
        for pde in ("bo", "ks"):
            f, n = (missing(summ[pde], "clean", pde, label, cv) for cv in ("finite", "non-trivial"))
            cells.append(f"{np.mean([f[s] for s in SEEDS_MAIN]):.2f} / {np.mean([n[s] for s in SEEDS_MAIN]):.2f}")
        print(f"{name} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
