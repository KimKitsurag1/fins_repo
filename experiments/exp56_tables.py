"""Experiment 56's tables (README_EXPERIMENTS.md, 08.10), reported without predictions:

readouts  the time-scaling exponent a-hat and its containment (exponent_readout.py) of every row exp51's scorer
          reads, on Benjamin-Ono (bo_runs), KdV (noise_weak_runs) and Burgers (burgers_runs), every condition and seed:
          results/exponent/summary.{json,md}.
paired    the paired readouts from the summaries' JSON: m = k beside the soft two-stage readouts, the own count beside
          the hard ones, missing and spurious dimensions and the dimension read: results/paired/summary.md.

Run (after the summaries):  python exp56_tables.py readouts|paired
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp52_bo as e52                                     # noqa: E402
import exp53_ks as e53                                     # noqa: E402
import exponent_readout as er                              # noqa: E402

RUNS = os.environ.get("FINS_RESULTS_DIR", "results")
EXPONENT = (("bo", e52.CONDS, "bo_runs", 2.0), ("kdv", e51.CONDS, "noise_weak_runs", 3.0),
            ("burgers", {"clean": 0.0}, "burgers_runs", 2.0))
SUMMARIES = (("heat", "noise_weak", e51.CONDS, ("finite", "full")),
             ("kdv", "noise_weak", e51.CONDS, ("finite", "non-trivial")),
             ("burgers", "burgers", {"clean": 0.0}, ("finite", "non-trivial")),
             ("bo", "bo", e52.CONDS, ("finite", "non-trivial")), ("ks", "ks", e53.CONDS, ("finite", "non-trivial")))
SOFT = ("FINS weak", "FINS weak, learned operator", "WSINDy, evaluation data, soft", "WSINDy, training data, soft",
        "two-stage, H 0.2, soft", "two-stage, H 0.05, soft")
HARD = ("FINS weak, own count", "FINS weak, learned operator, own count", "WSINDy, evaluation data, hard (own count)",
        "WSINDy, training data, hard (own count)", "two-stage, H 0.2, hard (own count)",
        "two-stage, H 0.05, hard (own count)")


def readouts(args, sets=EXPONENT, write=True):
    """{condition: {equation: {row: {seed: [a-hat, containment]}}}} over sets [(equation, conditions, runs
    subdirectory, true a)]; written to args.out_exponent if write."""
    cfg = e31.ExperimentConfig()
    res = {}
    for pde, conds, sub, _ in sets:
        for c in conds:
            d = os.path.join(args.runs, sub, c)
            for seed in range(42, 52):
                fields, ref = e51.rows_of(d, pde, seed)
                if ref is None:
                    continue
                geo = er.geometry(pde, ref["u"], ref["xs"], ref["ts"], cfg)
                for label, f in fields.items():
                    a, cont = er.read_fields(f, geo)
                    res.setdefault(c, {}).setdefault(pde, {}).setdefault(label, {})[str(seed)] = [a, cont]
            print(f"{pde} {c}: {len(res.get(c, {}).get(pde, {}))} rows", flush=True)
    if write:
        lines = exponent_lines(res, sets)
        os.makedirs(args.out_exponent, exist_ok=True)
        with open(os.path.join(args.out_exponent, "summary.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        with open(os.path.join(args.out_exponent, "summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return res


def exponent_lines(res, sets):
    lines = ["# Time-scaling exponent a-hat of each row's span (exponent_readout.py): mean [min, max] over seeds, "
             "mean containment, n seeds\n"]
    for pde, conds, _, truth in sets:
        cs = [c for c in conds if pde in res.get(c, {})]
        if not cs:
            continue
        labels = []
        for c in cs:
            labels += [lab for lab in res[c][pde] if lab not in labels]
        lines.append(f"## {pde} (a = {truth:g})\n")
        lines.append("| row | " + " | ".join(cs) + " |")
        lines.append("|---" * (len(cs) + 1) + "|")
        for lab in labels:
            cells = []
            for c in cs:
                v = np.array(list(res[c][pde].get(lab, {}).values()))
                cells.append(f"{v[:, 0].mean():.3f} [{v[:, 0].min():.2f}, {v[:, 0].max():.2f}] c {v[:, 1].mean():.3f} "
                             f"({len(v)})" if len(v) else "")
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def paired(args):
    lines = ["# Paired readouts: m = k beside the soft two-stage readouts, the own count beside the hard ones. Cells: "
             "missing / spurious dimensions [dimension read], means over the seeds the row has (n)\n"]
    for pde, sub, conds, cvs in SUMMARIES:
        p = os.path.join(args.results, sub, "summary.json")
        if not os.path.exists(p):
            continue
        js = json.load(open(p))
        cs = [c for c in conds if pde in js.get(c, {})]
        for cv in cvs:
            lines.append(f"## {pde}, {cv} algebra\n")
            lines.append("| readout | row | " + " | ".join(cs) + " |")
            lines.append("|---" * (len(cs) + 2) + "|")
            for kind, rows in (("m = k / soft", SOFT), ("own count / hard", HARD)):
                for lab in rows:
                    cells, any_ = [], False
                    for c in cs:
                        r = js[c][pde].get(lab, {}).get(cv, {})
                        if not r.get("missing"):
                            cells.append("")
                            continue
                        any_ = True
                        seeds = sorted(r["missing"])
                        m, s, k = (np.mean([r[key][q] for q in seeds]) for key in ("missing", "spurious", "k"))
                        cells.append(f"{m:.3f} / {s:.3f} [{k:.1f}] ({len(seeds)})")
                    if any_:
                        lines.append(f"| {kind} | {lab} | " + " | ".join(cells) + " |")
            lines.append("")
    os.makedirs(args.out_paired, exist_ok=True)
    with open(os.path.join(args.out_paired, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["readouts", "paired"])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-exponent", default=os.path.join("results", "exponent"))
    ap.add_argument("--out-paired", default=os.path.join("results", "paired"))
    args = ap.parse_args()
    {"readouts": readouts, "paired": paired}[args.what](args)


if __name__ == "__main__":
    main()
