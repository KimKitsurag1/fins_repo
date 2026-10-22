"""exp64, post hoc (not registered): the obs library holds exact identities of incompressible flow,
d_x(u q) = d_xx(uv) + d_xy(v^2 - u^2) / 2 and d_y(v q) = -d_yy(uv) + d_xy(v^2 - u^2) / 2 for q = curl u (with rd only
their sum holds, by u . grad psi = 0), so the regime's equation has more than one sparse form in it, and the registered
scorer counts only the transport form. This script rescores the obs rows' equations by function: the identified
right-hand side, with u = -psi_y, v = psi_x, q = Lap psi - kd2 psi, expanded as a differential polynomial in psi; each
monomial of the regime's within exp61's 2 % of its coefficient, any other monomial below 1e-3 of the smallest of the
regime's (the registered rule admits no extra term); the regressed kd2 within 0.05, as registered. The algebra is not
rescored.

Run:  python exp64_posthoc_identity.py --runs <dir with lib_{regime}_s{seed}.json> [--out results/pyqg]
"""

import argparse
import glob
import json
import os

import sympy as sp

TRUE_KD2 = {"f": 0.0, "beta": 0.0, "rd": 1.0, "drag": 0.0}
EXTRA = {"f": [], "beta": [("obs", "v", 0, 0, -0.5)], "rd": [], "drag": [("loc", 1, 0, 0, -0.3)]}
X, Y = sp.symbols("x y")
PSI = sp.Function("psi")(X, Y)


def fields(kd2):
    u, v = -sp.diff(PSI, Y), sp.diff(PSI, X)
    q = sp.diff(PSI, X, 2) + sp.diff(PSI, Y, 2) - kd2 * PSI
    return dict(u=u, v=v, uq=u * q, vq=v * q, uu=u * u, vv=v * v, uv=u * v, q=q)


def term(F, kind, a, p, r):
    return sp.diff(F["q"] ** a if kind == "loc" else F[a], X, p, Y, r)


def poly(expr):
    expr = sp.expand(expr)
    ders = sorted(expr.atoms(sp.Derivative), key=sp.default_sort_key) + [PSI]
    syms = sp.symbols(f"d0:{len(ders)}")
    return sp.Poly(expr.subs(dict(zip(ders, syms)), simultaneous=True), *syms).as_dict(), dict(zip(syms, ders))


def rhs(F, terms):
    return sum(c * term(F, k, a, p, r) for k, a, p, r, c in terms)


def functional(row, regime):
    F = fields(TRUE_KD2[regime])
    got = [(t["kind"], *t["key"][1:], t["c"]) for t in row["terms"]]
    want = [("obs", "uq", 1, 0, -1.0), ("obs", "vq", 0, 1, -1.0)] + EXTRA[regime]
    diff, _ = poly(rhs(F, got) - rhs(F, want))
    ref, _ = poly(rhs(F, want))
    small = min(abs(float(c)) for c in ref.values())
    err = max([abs(float(c)) / abs(float(ref[m])) for m, c in diff.items() if m in ref] + [0.0])
    extra = max([abs(float(c)) / small for m, c in diff.items() if m not in ref] + [0.0])
    kd = row.get("kd2_regressed")
    return max(err, extra), err <= 0.02 and extra <= 1e-3 and kd is not None and abs(kd - TRUE_KD2[regime]) <= 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res, lines = {}, ["# exp64 obs rows rescored by function (post hoc)\n",
                      "| regime | n | true (registered) | true (by function) | changed runs |", "|---|---|---|---|---|"]
    for regime in ("f", "beta", "rd", "drag"):
        cell = {}
        for p in sorted(glob.glob(os.path.join(args.runs, f"lib_{regime}_s*.json"))):
            row = json.load(open(p))["rows"]["obs"]
            err, ok = functional(row, regime)
            cell[os.path.basename(p)[:-5]] = dict(registered=bool(row["true_eq"]), functional=bool(ok), rel_err=err,
                                                  names=row["names"])
        res[regime] = cell
        changed = [k.split("_s")[-1] for k, v in cell.items() if v["registered"] != v["functional"]]
        lines.append(f"| {regime} | {len(cell)} | {sum(v['registered'] for v in cell.values())} | "
                     f"{sum(v['functional'] for v in cell.values())} | {', '.join(changed) or '-'} |")
    print("\n".join(lines))
    for regime, cell in res.items():
        for k, v in cell.items():
            if not v["functional"] or v["registered"] != v["functional"]:
                print(f"  {k}: registered {v['registered']}, by function {v['functional']} (rel. {v['rel_err']:.1e}): "
                      f"{v['names']}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "libraries_obs_posthoc.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        with open(os.path.join(args.out, "libraries_obs_posthoc.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
