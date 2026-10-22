"""Summaries of fins_image.py results without recomputing pencils.

planted: per seed, share of the planted generator outside the 3 lowest (trained, control, margin), the ratio of
its Rayleigh quotient in the control to that in the planted network (how much training lowered its loss), and
FINS against LieGG in GL(2).
encoders: missing dimensions of every expected subspace (EXPECTED in fins_image.py) in the k lowest eigenvectors
of every encoder, k the subspace dimension, against a random k-subspace.
"""
import argparse
import json

import numpy as np

from fins_image import EXPECTED, field_name, make_class, named, score


def planted(path):
    res = json.load(open(path))["results"]
    keys = sorted({(r["arch"], r["output"], r.get("sigma", 0.0), r["group"]) for r in res})
    print(f"{path}: share outside the 3 lowest, trained / control (margin); loss ratio control / trained")
    for arch, out, sg, grp in keys:
        rs = sorted([r for r in res if (r["arch"], r["output"], r.get("sigma", 0.0), r["group"]) == (arch, out, sg, grp)],
                    key=lambda r: r["seed"])
        cells = []
        for r in rs:
            t, c = r["trained"], r["none"]
            cells.append(f"{t['share_out_3']:.2f}/{c['share_out_3']:.2f} ({c['share_out_3'] - t['share_out_3']:+.2f}) "
                         f"x{c['loss'] / t['loss']:.2f}")
        line = f"  {arch:<4s} {out:<4s} sigma {sg:<4g} {grp:<11s} " + " | ".join(cells)
        if "fins_gl2" in rs[0]:
            line += " || GL(2) FINS " + " ".join(f"{r['fins_gl2']['trained']:.2f}" for r in rs) + \
                    ", LieGG " + " ".join(f"{r['liegg']['share_out']:.2f}" for r in rs)
        print(line)


def encoders(path):
    d = json.load(open(path))
    fields = make_class(3, 2)
    assert d["fields"] == [field_name(f) for f in fields]
    gts = named(fields, 3)
    sets = {k: v for k, v in EXPECTED.items() if v}
    print(f"{path}: missing dimensions of each expected subspace in the k lowest (random k-subspace)")
    for r in d["results"]:
        N, D = (np.array(a) for a in r["pencil"])
        cells = []
        for name, gens in sets.items():
            sc = score(N, D, {g: gts[g] for g in gens})
            cells.append(f"{name} {sc['missing']:.2f} of {sc['k_gt']} ({sc['chance_missing']:.2f})")
        print(f"  {r['encoder']:<14s} sigma {r.get('sigma', 0.0):<4g} " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planted", nargs="*", default=[])
    ap.add_argument("--encoders", nargs="*", default=[])
    a = ap.parse_args()
    for p in a.planted:
        planted(p)
    for p in a.encoders:
        encoders(p)


if __name__ == "__main__":
    main()
