"""Experiment 61, weak SINDy arm -- weak SINDy in 2D on pyqg's data (README_EXPERIMENTS.md; development 09.10).

The data and regimes of exp61_pyqg.py (pyqg's barotropic model; q, 128^2). Rows: the reference library
d_x^p d_y^q (q^j), j <= 3, p + q <= 4, and a constant (ref); with the quasi-geostrophic completion d_x(u q), d_y(v q),
psi, psi_x, psi_y, psi = Lap^-1 q, u = -psi_y, v = psi_x (qg-free); the same with psi = (Lap - kd2)^-1 q and kd2
profiled by the residual (qg-rd); the nested rule (the next row only when the residual drops more than 1.25 times).
Each identified model's algebra by its structure (qg_algebra) against the regime's.

Run:  python exp61_wsindy.py regress --regime beta --seed 7 --data-seed 100 --save <stem>
      python exp61_wsindy.py run --runs <dir> --threads 1;  python exp61_wsindy.py summary --runs <dir>
"""

import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp51_noise_weak as e51                             # noqa: E402
import exp58_symmetry_library as e58                       # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import method_wsindy2d as mw2                              # noqa: E402
import qg2d                                                # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# weak SINDy on pyqg's data: the local library and its quasi-geostrophic completion

POLYS, MAX_D = (0, 1, 2, 3), 4
KD2S = np.round(np.arange(0.0, 4.0 + 1e-9, 0.25), 4)
NEST = 1.25
NONLOCAL = (("adv_x", 1, 0), ("adv_y", 0, 1), ("psi", 0, 0), ("psi_x", 1, 0), ("psi_y", 0, 1))
TRUE_TERMS = {"f": {("adv_x",): -1.0, ("adv_y",): -1.0},
              "beta": {("adv_x",): -1.0, ("adv_y",): -1.0, ("psi_x",): -0.5},
              "rd": {("adv_x",): -1.0, ("adv_y",): -1.0},
              "drag": {("adv_x",): -1.0, ("adv_y",): -1.0, ("loc", 1, 0, 0): -0.3}}
COEF_TOL = 0.02


class QGLib:
    """Weak SINDy in 2D (pde/method_wsindy2d.py) on trajectories U [N, Nx, Ny, Nt]: the local library once; the
    completion d_x(u q), d_y(v q), psi, psi_x, psi_y with psi = (Lap - kd2)^-1 q (the mean of q dropped at kd2 = 0),
    u = -psi_y, v = psi_x, per kd2."""

    def __init__(self, U, x, t):
        self.w = mw2.WSINDy2D(POLYS, MAX_D)
        self.U = U
        self.b, self.Mlhs = self.w.prepare(U, x, x, t)
        self.local = list(self.w.terms)
        self.G, self.M = self.w.columns(U, [(j, p, q, -1) for j, p, q in self.local])
        self.A, self.c, self.bb = self.G.T @ self.G, self.G.T @ self.b, float(self.b @ self.b)
        k = 2 * np.pi * np.fft.fftfreq(U.shape[1], d=x[1] - x[0])
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.cache = {}

    def fields(self, u, kd2):
        return qg_fields(self.KX, self.KY, u, kd2)

    def nonlocal_cols(self, kd2):
        key = float(np.round(kd2, 6))
        if key not in self.cache:
            p = self.w.prep
            su = p["su"]
            cols = [[] for _ in NONLOCAL]
            for u in self.U:
                psi, uq, vq = self.fields(u, kd2)
                for i, (name, px, qy) in enumerate(NONLOCAL):
                    f = {"adv_x": uq / su ** 2, "adv_y": vq / su ** 2}.get(name, psi / su)
                    cols[i].append(mw2.conv_nd(f, (p["Cx"][px], p["Cy"][qy], p["Ct"][0]), p["s"]).ravel())
            G = np.stack([np.concatenate(c) for c in cols], 1)
            M = np.array([(su ** -2 if name.startswith("adv") else su ** -1) * p["sx"] ** -px * p["sy"] ** -qy
                          for name, px, qy in NONLOCAL])
            self.cache[key] = dict(G=G, M=M, X=self.G.T @ G, A=G.T @ G, c=G.T @ self.b)
        return self.cache[key]

    def resid(self, kd2):
        m = self.nonlocal_cols(kd2)
        A = np.block([[self.A, m["X"]], [m["X"].T, m["A"]]])
        c = np.concatenate([self.c, m["c"]])
        M = np.concatenate([self.M, m["M"]]) / self.Mlhs
        return e58.mstls_gram(A, c, self.bb, M, nterms=len(c))

    def fit(self, kd2=None):
        """The reference MSTLS on the local library (kd2 None) or with the completion at kd2: (terms, resid, lam)."""
        G, M = self.G, self.M
        names = [("loc", j, p, q) for j, p, q in self.local]
        if kd2 is not None:
            m = self.nonlocal_cols(kd2)
            G, M = np.concatenate([G, m["G"]], 1), np.concatenate([M, m["M"]])
            names = names + [(name,) for name, _, _ in NONLOCAL]
        W, lam, resid = self.w.mstls(G, self.b, M / self.Mlhs)
        terms = [dict(kind=n[0], key=list(n), c=float(cc), kd2=None if n[0] == "loc" else float(kd2))
                 for n, cc in zip(names, W) if cc != 0]
        return terms, float(resid), float(lam)


def qg_fields(KX, KY, u, kd2):
    """psi, u q, v q of one trajectory u [Nx, Ny, Nt] (psi_hat = -q_hat / (|k|^2 + kd2), zero at k = 0 when kd2 = 0)."""
    K2 = KX ** 2 + KY ** 2
    a = -1.0 / np.where(K2 + kd2 > 0, K2 + kd2, 1.0) * (K2 + kd2 > 0)
    ph = a[:, :, None] * np.fft.fft2(u, axes=(0, 1))
    inv = (lambda h: np.real(np.fft.ifft2(h, axes=(0, 1))))
    return inv(ph), inv(-1j * KY[:, :, None] * ph) * u, inv(1j * KX[:, :, None] * ph) * u


def qg_term_name(t):
    if t["kind"] == "loc":
        _, j, p, q = t["key"]
        d = "".join(f"d{ax}^{n}" if n > 1 else f"d{ax}" for ax, n in (("x", p), ("y", q)) if n)
        return f"{t['c']:+.4g} {d}(q^{j})" if j else f"{t['c']:+.4g}"
    return f"{t['c']:+.4g} {t['kind']}" + (f"[kd2 {t['kd2']:g}]" if t["kd2"] else "")


WEIGHTS = {"adv_x": (0.0, 0.0, 2.0), "adv_y": (0.0, 0.0, 2.0), "psi": (2.0, 0.0, 1.0), "psi_x": (1.0, 0.0, 1.0),
           "psi_y": (1.0, 0.0, 1.0)}                         # (wx, wt, wu) weights of the completion's terms


def qg_algebra(terms):
    """The identified model's algebra by its structure: rotation (equal advection coefficients, no psi_x or psi_y,
    isotropic local symbols per power), shift q -> q + c (no q^j beyond linear derivatives, divergence-free advection,
    no screened psi), and the scalings x -> l^wx x, t -> l^wt t, q -> l^wu q balancing every term against q_t (the
    null space of the weight equations; a screened inversion fixes wx = 0)."""
    cx = sum(t["c"] for t in terms if t["kind"] == "adv_x")
    cy = sum(t["c"] for t in terms if t["kind"] == "adv_y")
    eq_adv = abs(cx - cy) <= 1e-3 * max(abs(cx), abs(cy), 1e-300)
    screened = any(t["kind"] != "loc" and t["kd2"] for t in terms)
    rot = eq_adv and not any(t["kind"] in ("psi_x", "psi_y") for t in terms)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for j in sorted({t["key"][1] for t in terms if t["kind"] == "loc"}):
        for r in (0.5, 1.0, 2.0, 4.0):
            KX, KY = r * np.cos(th), r * np.sin(th)
            v = sum(t["c"] * (1j * KX) ** t["key"][2] * (1j * KY) ** t["key"][3] for t in terms
                    if t["kind"] == "loc" and t["key"][1] == j)
            if np.abs(v).max() > 0 and np.abs(v - v.mean()).max() > 1e-6 * np.abs(v).max():
                rot = False
    shift = eq_adv and not any(t["kind"] == "loc" and t["key"][1] >= 1 and (t["key"][1] >= 2 or
                                                                             t["key"][2] + t["key"][3] == 0)
                               for t in terms) and not any(t["kind"] == "psi" and t["kd2"] for t in terms)
    rows = []
    for t in terms:
        if t["kind"] == "loc":
            _, j, p, q = t["key"]
            wv = (-(p + q), 0.0, float(j))
        else:
            wv = WEIGHTS[t["kind"]]
        rows.append(np.array(wv) - np.array([0.0, -1.0, 1.0]))
    if screened:
        rows.append(np.array([1.0, 0.0, 0.0]))
    R = np.array(rows) if rows else np.zeros((0, 3))
    _, sv, Vt = np.linalg.svd(np.vstack([R, np.zeros((1, 3))]))
    rank = int((sv > 1e-9 * max(sv.max(), 1.0)).sum()) if sv.size else 0
    basis = Vt[rank:].tolist()
    return dict(rotation=bool(rot), shift=bool(shift), n_scalings=3 - rank, scalings=basis)


def qg_true_algebra(regime):
    sc = e61.SCALINGS[regime]
    return dict(rotation=regime != "beta", shift=True, n_scalings=len(sc), scalings=sc)


def same_span(A, B, tol=1e-6):
    if len(A) != len(B):
        return False
    if not len(A):
        return True
    Qa = np.linalg.qr(np.array(A, dtype=float).T)[0]
    Bn = np.array(B, dtype=float)
    Bn = Bn / np.linalg.norm(Bn, axis=1, keepdims=True)
    return float(len(B) - ((Qa.T @ Bn.T) ** 2).sum()) <= tol


def qg_true_eq(terms, regime):
    want = TRUE_TERMS[regime]
    got = {tuple(t["key"]) if t["kind"] == "loc" else (t["kind"],): t for t in terms}
    if set(got) != set(want):
        return False
    if regime == "rd" and any(abs(t["kd2"] - 1.0) > 1e-9 for t in terms if t["kind"] != "loc"):
        return False
    if regime != "rd" and any(t["kd2"] for t in terms if t["kind"] != "loc"):
        return False
    return all(abs(got[k]["c"] / v - 1) <= COEF_TOL for k, v in want.items())


def qg_rhs_error(terms, g, U):
    """Relative error of the identified right-hand side against pyqg's (the filter excluded) on U [N, Nx, Ny, Nt]."""
    num = den = 0.0
    k = 2 * np.pi * np.fft.fftfreq(U.shape[1], d=g.dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    for u in U:
        q = torch.tensor(np.moveaxis(u, -1, 0), dtype=torch.float64)
        F0 = np.moveaxis(qg2d.rhs(g, q).numpy(), 0, -1)
        F = np.zeros_like(u)
        flds = {kd: qg_fields(KX, KY, u, kd) for kd in {t["kd2"] for t in terms if t["kind"] != "loc"}}
        for t in terms:
            if t["kind"] == "loc":
                _, j, p, q_ = t["key"]
                f = np.fft.fft2(u ** j, axes=(0, 1)) if j else np.fft.fft2(np.ones_like(u), axes=(0, 1))
                F += t["c"] * np.real(np.fft.ifft2((1j * KX[..., None]) ** p * (1j * KY[..., None]) ** q_ * f,
                                                   axes=(0, 1)))
            else:
                psi, uq, vq = flds[t["kd2"]]
                f, px, qy = {"adv_x": (uq, 1, 0), "adv_y": (vq, 0, 1), "psi": (psi, 0, 0), "psi_x": (psi, 1, 0),
                             "psi_y": (psi, 0, 1)}[t["kind"]]
                F += t["c"] * np.real(np.fft.ifft2((1j * KX[..., None]) ** px * (1j * KY[..., None]) ** qy
                                                   * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))
        num += float(((F - F0) ** 2).sum())
        den += float((F0 ** 2).sum())
    return float(np.sqrt(num / den))


def regress(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    t0 = time.time()
    U = e61.make_data(args, g, args.n_train, dseed).numpy().transpose(0, 2, 3, 1)
    x = g.x.numpy()
    t = args.dt * np.arange(args.nt)
    lib = QGLib(U, x, t)
    print(f"[{args.regime} (beta {g.beta}, rd {g.rd}, rek {g.rek}), {args.data} data] seed {s} (data {dseed}): "
          f"{len(U)} trajectories, {lib.w.info['rows']} rows, test supports m {lib.w.info['mx']},{lib.w.info['my']},"
          f"{lib.w.info['mt']}; {time.time() - t0:.0f} s", flush=True)
    truth = qg_true_algebra(args.regime)
    rows = {}

    def put(name, terms, resid, lam, prm=None):
        alg = qg_algebra(terms)
        right = (alg["rotation"] == truth["rotation"] and alg["shift"] == truth["shift"]
                 and same_span(alg["scalings"], truth["scalings"], 1e-4))
        rows[name] = dict(terms=terms, names=" ".join(qg_term_name(tt) for tt in terms), resid=resid, lam=lam,
                          prm=prm, true_eq=qg_true_eq(terms, args.regime), right_algebra=bool(right), algebra=alg,
                          eF=qg_rhs_error(terms, g, U))
        r = rows[name]
        print(f"  {name:8s} {'TRUE ' if r['true_eq'] else 'wrong'} algebra {'right' if right else 'WRONG'} (rotation "
              f"{alg['rotation']}, shift {alg['shift']}, scalings {alg['n_scalings']}) eF {r['eF']:.3g} resid "
              f"{resid:.3g}" + (f" kd2 {prm}" if prm is not None else "") + f"  q_t = {r['names']}", flush=True)

    t1 = time.time()
    put("ref", *lib.fit())
    print(f"  ({time.time() - t1:.0f} s)", flush=True)
    t1 = time.time()
    put("qg-free", *lib.fit(0.0), 0.0)
    print(f"  ({time.time() - t1:.0f} s)", flush=True)
    t1 = time.time()
    best = min(KD2S, key=lib.resid)
    put("qg-rd", *lib.fit(float(best)), float(best))
    print(f"  ({time.time() - t1:.0f} s)", flush=True)
    pick = "ref"
    for nxt in ("qg-free", "qg-rd"):
        if rows[pick]["resid"] > NEST * rows[nxt]["resid"]:
            pick = nxt
    rows["nested"] = dict(rows[pick], picked=pick)
    print(f"  nested -> {pick}; total {time.time() - t0:.0f} s", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(e61.RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(dict(config=vars(args), data_seed=dseed, rows=rows,
                           wsindy={k: float(v) for k, v in lib.w.info.items()}), fh, indent=1)
        print(f"  saved {stem}.json")


def jobs(args):
    """(output, command) per regime and seed: regress on the seed's training data (pyqg's, 128^2)."""
    out = []
    for regime in qg2d.REGIMES:
        for s in args.seeds.split(","):
            stem = os.path.join(args.runs, f"regress_{regime}_s{s}")
            out.append((stem + ".json", ["exp61_wsindy.py", "regress", "--regime", regime, "--seed", s, "--threads",
                                         "1", "--save", stem]))
    return out


def summary(args):
    """results/pyqg/wsindy_summary.{json,md}: per regime and row, per seed: true equation, algebra, model."""
    res, lines = {}, ["# Weak SINDy on pyqg's data (exp61): counts over seeds\n",
                      "| regime | row | n | true equation | right algebra | rotation / shift / scalings (modal) | "
                      "eF (median) |", "|---|---|---|---|---|---|---|"]
    for regime in qg2d.REGIMES:
        cell = {}
        for s in args.seeds.split(","):
            p = os.path.join(args.runs, f"regress_{regime}_s{s}.json")
            if os.path.exists(p):
                j = json.load(open(p))
                for r, v in j["rows"].items():
                    cell.setdefault(r, {})[s] = {k: v.get(k) for k in ("true_eq", "right_algebra", "algebra", "names",
                                                                       "eF", "resid", "prm", "picked")}
        if not cell:
            continue
        res[regime] = cell
        for r, v in cell.items():
            alg = collections.Counter((a["algebra"]["rotation"], a["algebra"]["shift"], a["algebra"]["n_scalings"])
                                      for a in v.values()).most_common(1)[0][0]
            lines.append(f"| {regime} | {r} | {len(v)} | {sum(a['true_eq'] for a in v.values())} | "
                         f"{sum(a['right_algebra'] for a in v.values())} | {alg} | "
                         f"{np.median([a['eF'] for a in v.values()]):.2g} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pyqg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "wsindy_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "wsindy_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["regress", "run", "summary"])
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(e61.RESULTS, "pyqg_ws_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--regime", default="f", choices=list(qg2d.REGIMES))
    ap.add_argument("--var", default="q", choices=["q"])
    ap.add_argument("--data", default="pyqg", choices=["port", "pyqg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--nsub", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"regress": regress, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
