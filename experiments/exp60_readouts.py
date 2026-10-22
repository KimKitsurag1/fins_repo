"""exp60 readouts without the scans' priors, from the saved pencils and identified models (README_EXPERIMENTS.md,
entry of 09.10):
  sl2    -- the pencil restricted to the traceless linear fields (x d_x - y d_y, y d_x, x d_y): the deepest field's
            invariant ellipse gives eps-hat and the angle of its axes (no aligned-axes family);
  shift  -- the scaling family with and without the shift d_u: (x d_x + y d_y, t d_t, u d_u) against + d_u;
  soft   -- weak SINDy's identified models read like FINS's scans: eps from the nonlinear part and the argmin over e of
            the linear symbol's variation on the ellipses kx^2 + e^2 ky^2 = r^2.

Run:  python exp60_readouts.py all --runs <aniso2d_runs> --seeds 42,...,51 [--out results/aniso2d]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402
import qg2d                                                # noqa: E402

AS = ("1.25", "1.5", "1.75")
ROWS = ("iso-free", "dir-free", "harm-free", "ell-free", "nested")


def restricted(N, D, V):
    return eigh(V.T @ N @ V, V.T @ D @ V)


def sl2(N, D):
    """Deepest traceless linear field M = [[a, b], [c, -a]] (xi = a x + b y, eta = c x - a y); its invariant form
    S = [[1, -a / c], [-a / c, -b / c]]: eps-hat = sqrt(min / max eigenvalue), angle of the minor-eigenvalue axis."""
    V = qg2d.coefficients([[(0, (1, 0, 0, 0), 1.0), (1, (0, 1, 0, 0), -1.0)], [(0, (0, 1, 0, 0), 1.0)],
                           [(1, (1, 0, 0, 0), 1.0)]])
    lam, W = restricted(N, D, V)
    a, b, c = W[:, 0]
    det = -a * a - b * c
    out = dict(lam=lam.tolist(), det_sign=float(np.sign(det)), field=[float(a), float(b), float(c)])
    if det <= 0 or c == 0:
        return dict(out, eps=None, angle=None)
    S = np.array([[1.0, -a / c], [-a / c, -b / c]])
    if S[1, 1] < 0:
        S = -S
    ls, vs = np.linalg.eigh(S)
    ang = float(np.degrees(np.arctan2(vs[1, 0], vs[0, 0])))
    ang = (ang + 90.0) % 180.0 - 90.0
    return dict(out, eps=float(np.sqrt(ls[0] / ls[1])), angle=ang)


def shift_family(N, D, sigma):
    """(x d_x + y d_y, t d_t, u d_u) and the same plus d_u: eigenvalues, ratio of the two lowest, the deepest
    direction's a = w_t / w_x, c = w_u / w_x and shift (physical, w_s sigma / w_x)."""
    base = [[(0, (1, 0, 0, 0), 1.0), (1, (0, 1, 0, 0), 1.0)], [(2, (0, 0, 1, 0), 1.0)], [(3, (0, 0, 0, 1), 1.0)]]
    out = {}
    for name, ents in (("B3", base), ("B4", base + [[(3, (0, 0, 0, 0), 1.0)]])):
        lam, W = restricted(N, D, qg2d.coefficients(ents))
        w = W[:, 0] / W[0, 0]
        out[name] = dict(lam=lam.tolist(), ratio=float(lam[1] / lam[0]), a=float(w[1]), c=float(w[2]),
                         shift=float(w[3] * sigma) if len(w) > 3 else None)
    return out


def sym(kind, prm, KX, KY):
    K2 = KX ** 2 + KY ** 2
    if kind == "iso":
        return -K2 ** (prm[0] / 2)
    if kind == "dir_x":
        return -np.abs(KX) ** prm[0]
    if kind == "dir_y":
        return -np.abs(KY) ** prm[0]
    if kind == "ell":
        return -(KX ** 2 + prm[0] ** 2 * KY ** 2) ** (prm[1] / 2)
    c2 = (KX ** 2 - KY ** 2) / K2
    m = int(kind[4:])
    return -K2 ** (prm[0] / 2) * (1.0 if m == 0 else (c2 if m == 2 else 2 * c2 ** 2 - 1))


def lin_symbol(terms, KX, KY):
    s = np.zeros(KX.shape, dtype=complex)
    for t in terms:
        if t["kind"] == "loc" and t["j"] == 1:
            s += t["c"] * (1j * KX) ** t["p"] * (1j * KY) ** t["q"]
        elif t["kind"] != "loc":
            s += t["c"] * sym(t["kind"], t["prm"], KX, KY)
    return s


def defect(terms, e, radii=(0.5, 1.0, 2.0, 4.0), n=96):
    """Largest relative variation of the linear symbol along the ellipses kx^2 + e^2 ky^2 = r^2."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = 0.0
    for r in radii:
        v = lin_symbol(terms, r * np.cos(th), r * np.sin(th) / e)
        if np.abs(v).max() > 0:
            out = max(out, np.abs(v - v.mean()).max() / np.abs(v).max())
    return out


def soft(terms):
    cxx = sum(t["c"] for t in terms if t["kind"] == "loc" and t["j"] == 2 and (t["p"], t["q"]) == (2, 0))
    cyy = sum(t["c"] for t in terms if t["kind"] == "loc" and t["j"] == 2 and (t["p"], t["q"]) == (0, 2))
    es = np.arange(0.2, 2.0 + 1e-12, 0.001)
    d = np.array([defect(terms, e) for e in es])
    i = int(np.argmin(d))
    return dict(e_n=float(np.sqrt(cyy / cxx)) if cxx > 0 and cyy > 0 else None, e_soft=float(es[i]),
                defect=float(d[i]))


def run(args):
    res = {}
    for a in AS:
        for s in args.seeds.split(","):
            p = os.path.join(args.runs, f"fins_a{a}_s{s}.npz")
            if os.path.exists(p):
                z = np.load(p)
                sigma = float(np.asarray(z["u_train"]).std())
                res.setdefault(a, {})[s] = dict(sl2=sl2(z["N"], z["D"]), shift=shift_family(z["N"], z["D"], sigma))
            q = os.path.join(args.runs, f"regress_a{a}_s{s}.json")
            if args.soft and os.path.exists(q):
                j = json.load(open(q))
                res.setdefault(a, {}).setdefault(s, {})["soft"] = {r: dict(soft(j["rows"][r]["terms"]),
                                                                           a_hat=j["rows"][r]["a_hat"])
                                                                   for r in ROWS if r in j["rows"]}
    lines = ["# exp60 readouts without the scans' priors\n",
             "| a | n | sl(2): eps-hat | angle (deg) | lam2 / lam1 | scaling B3: ratio, a-hat | B4 (+ d_u): ratio, a-hat |",
             "|---|---|---|---|---|---|---|"]
    rng = (lambda v, f="{:.3f}": "-" if not v else f"{f.format(min(v))} to {f.format(max(v))}")
    for a, cell in res.items():
        S = [c["sl2"] for c in cell.values() if "sl2" in c]
        B = [c["shift"] for c in cell.values() if "shift" in c]
        lines.append(f"| {a} | {len(S)} | {rng([x['eps'] for x in S if x['eps'] is not None], '{:.4f}')} | "
                     f"{rng([x['angle'] for x in S if x['angle'] is not None], '{:.2f}')} | "
                     f"{rng([x['lam'][1] / x['lam'][0] for x in S], '{:.0f}')} | "
                     f"{rng([b['B3']['ratio'] for b in B], '{:.0f}')}, {rng([b['B3']['a'] for b in B])} | "
                     f"{rng([b['B4']['ratio'] for b in B], '{:.0f}')}, {rng([b['B4']['a'] for b in B])} |")
    if args.soft:
        lines += ["", "| a | row | e_n | soft e | defect at min | a-hat |", "|---|---|---|---|---|---|"]
        for a, cell in res.items():
            for r in ROWS:
                v = [c["soft"][r] for c in cell.values() if "soft" in c and r in c["soft"]]
                if v:
                    lines.append(f"| {a} | {r} | {rng([x['e_n'] for x in v if x['e_n'] is not None], '{:.4f}')} | "
                                 f"{rng([x['e_soft'] for x in v])} | {rng([x['defect'] for x in v], '{:.1e}')} | "
                                 f"{sorted(set(str(x['a_hat']) for x in v))} |")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "readouts_summary.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        with open(os.path.join(args.out, "readouts_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
