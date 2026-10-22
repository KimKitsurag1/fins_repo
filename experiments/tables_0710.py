"""LaTeX rows of the PDE paper's noise table (Table tab:noise) from the summaries of exp51 (heat, KdV; --exp48 rows
included), exp52 (Benjamin-Ono) and exp53 (Kuramoto-Sivashinsky): missing dimensions, finite algebra and non-trivial
generators, means over the seeds each row has (printed as a superscript when not ten).

Run:  python tables_0710.py [--results results] [--out tab_noise.tex]
"""

import argparse
import json
import os

import numpy as np

CONDS = ("clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1")
EQS = (("heat", "noise_weak", "heat (6)"), ("kdv", "noise_weak", "KdV (4)"), ("bo", "bo", "Benjamin--Ono (4)"),
       ("ks", "ks", "Kuramoto--Sivashinsky (3)"))
ROWS = (("FINS, weak", ("FINS weak",)),
        ("FINS, pointwise", ("FINS pointwise",)),
        ("two-stage, $H=0.2$, hard", ("two-stage, H 0.2, hard (own count)",)),
        ("two-stage, $H=0.05$, hard", ("two-stage, H 0.05, hard (own count)",)),
        ("LieNLSD, exact RHS$^\\dagger$", ("LieNLSD_RHS@eval",)),
        ("LieNLSD as published", ("LieNLSD_F", "LieNLSD_F (exp48)")))


def cell(d, c, pde, labels, cv):
    for lab in labels:
        v = d.get(c, {}).get(pde, {}).get(lab, {}).get(cv, {}).get("missing", {})
        if v:
            m = float(np.mean(list(v.values())))
            m = 0.0 if abs(m) < 5e-4 else m
            return f"{m:.2f}" + ("" if len(v) == 10 else f"$^{{{len(v)}}}$")
    return "--"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = {}
    for _, d, _ in EQS:
        p = os.path.join(args.results, d, "summary.json")
        data[d] = json.load(open(p)) if os.path.exists(p) else {}
    lines = []
    for pde, d, title in EQS:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{title}}} \\\\")
        for name, labels in ROWS:
            cells = []
            for c in CONDS:
                f, n = cell(data[d], c, pde, labels, "finite"), cell(data[d], c, pde, labels, "non-trivial")
                cells.append(f if n == "--" else f"{f} / {n}")
            if all(x == "--" for x in cells):
                continue
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    text = "\n".join(lines[:-1]) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
