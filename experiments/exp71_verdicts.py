"""exp71's verdicts KR1-KR6 from the noise studies' summaries (missing dimensions of the finite algebra, soft readout).

Inputs: results/noise_weak/summary.json (heat, KdV), results/ks/summary.json, results/bo/summary.json,
results/burgers/summary.json, each from its study's summary command after the Kreider runs.
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIES = {"heat": "noise_weak", "kdv": "noise_weak", "ks": "ks", "bo": "bo", "burgers": "burgers"}
SHEET, SURFACE = "Kreider et al., solutions, soft", "Kreider et al., equation surface, soft"
WEAK, LEARNED = "FINS weak", "FINS weak, learned operator"
# LieNLSD as published, clean, the paper's values: the noise table's row (heat finite, KdV, BO, KS) and the main
# table's (Burgers); the main table's heat (6) and KdV entries in brackets
PAPER = {"heat": (0.55, 0.32), "burgers": (0.55,), "kdv": (0.91, 0.90), "bo": (1.354,), "ks": (0.0,)}
SEEDS, LEARNED_SEEDS = [str(s) for s in range(42, 52)], ["42", "43", "44"]
CONDS = {"heat": ["clean", "n1e-3", "n1e-2", "n3e-2", "n5e-2", "n1e-1"], "kdv": ["clean", "n1e-3", "n1e-2", "n3e-2",
         "n5e-2", "n1e-1"], "ks": ["clean", "n1e-2", "n3e-2", "n5e-2", "n1e-1"], "bo": ["clean", "n1e-2", "n3e-2"],
         "burgers": ["clean"]}                     # the registered levels


def missing(summ, pde, cond, label):
    """{seed: missing dimensions of the finite algebra} of a row, {} if absent."""
    return summ.get(cond, {}).get(pde, {}).get(label, {}).get("finite", {}).get("missing", {})


def line(name, ok, detail):
    print(f"({name}) {'MET' if ok else 'NOT MET'}: {detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    summ = {pde: json.load(open(os.path.join(args.results, d, "summary.json"))) for pde, d in SUMMARIES.items()}
    eqs = ("heat", "burgers", "kdv", "bo", "ks")
    out = {}
    # KR1, KR2: clean, every equation, >= 9/10 seeds
    for name, label, bound in (("KR1", SHEET, 1.5), ("KR2", SURFACE, 0.5)):
        parts, ok = [], True
        for pde in eqs:
            m = missing(summ[pde], pde, "clean", label)
            n_ok = sum(m[s] >= bound for s in SEEDS if s in m)
            ok &= sorted(m) == sorted(SEEDS) and n_ok >= 9
            parts.append(f"{pde} {n_ok}/{len(m)} (min {min(m.values()):.3f}, mean {np.mean(list(m.values())):.3f})")
            out.setdefault(name, {})[pde] = m
        line(name, ok, f">= {bound} on " + "; ".join(parts))
    # KR3, KR4: FINS weak (exact; learned 42-44) misses fewer than the surface reading, clean, every seed
    for name, label, seeds in (("KR3", WEAK, SEEDS), ("KR4", LEARNED, LEARNED_SEEDS)):
        parts, ok = [], True
        for pde in ("heat", "kdv", "bo", "ks"):
            f = missing(summ[pde], pde, "clean", label)
            s = missing(summ[pde], pde, "clean", SURFACE)
            both = [k for k in seeds if k in f and k in s]
            n_ok = sum(f[k] < s[k] for k in both)
            ok &= len(both) == len(seeds) and n_ok == len(seeds)
            parts.append(f"{pde} {n_ok}/{len(both)} (FINS max {max(f[k] for k in both):.3f}, surface min "
                         f"{min(s[k] for k in both):.3f})")
        line(name, ok, "; ".join(parts))
    # KR5: surface median >= 0.5 at every noise level on every equation (Burgers: clean only)
    parts, ok = [], True
    for pde in eqs:
        meds = []
        for cond in CONDS[pde]:
            m = missing(summ[pde], pde, cond, SURFACE)
            med = float(np.median(list(m.values()))) if m else float("nan")
            meds.append(f"{cond} {med:.3f} ({len(m)})")
            ok &= med >= 0.5 and sorted(m) == sorted(SEEDS)
        parts.append(f"{pde}: " + ", ".join(meds))
    line("KR5", ok, "medians " + "; ".join(parts))
    # KR6: surface mean over 42-44 more than LieNLSD as published
    parts, ok = [], True
    for pde in eqs:
        m = missing(summ[pde], pde, "clean", SURFACE)
        mean = float(np.mean([m[k] for k in LEARNED_SEEDS]))
        ok &= mean > PAPER[pde][0]
        extra = f" [main table {PAPER[pde][1]:.2f}: {'more' if mean > PAPER[pde][1] else 'not more'}]" \
            if len(PAPER[pde]) > 1 else ""
        parts.append(f"{pde} {mean:.3f} against {PAPER[pde][0]:g}{extra}")
    line("KR6", ok, "; ".join(parts))
    # without predictions: hard readouts and own counts
    print("\nwithout predictions (clean, mean over seeds 42-51):")
    for pde in eqs:
        cells = []
        for label in (SHEET, SURFACE, SHEET.replace("soft", "hard (own count)"),
                      SURFACE.replace("soft", "hard (own count)"), WEAK, LEARNED):
            m = missing(summ[pde], pde, "clean", label)
            if m:
                cells.append(f"{label} {np.mean(list(m.values())):.3f} ({len(m)})")
        print(f"  {pde}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
