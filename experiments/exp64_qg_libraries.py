"""Experiment 64 -- two-stage rows on pyqg's data that need no quasi-geostrophic knowledge (README_EXPERIMENTS.md,
entry of 09.10):
  obs   -- weak SINDy with the reference library in q plus a local library in pyqg's observed velocity: u, v and
           d_x^p d_y^r (f), f in {u q, v q, u^2, v^2, u v}, p + r <= 2 (derivatives of u and v alone are left out: the
           identities div u = 0 and curl u = q + kd2 psi make them collinear with q); kd2 from q = Lap psi - kd2 psi,
           a pointwise regression on pyqg's psi;
  riesz -- the reference library plus the generalized SQG (alpha-turbulence) completion: transport by
           u_s = grad^perp psi_s, psi_s = -(|k|^2 + kappa)^(-s/2) q, with psi_s, psi_s,x, psi_s,y (QG: s = 2); s and kappa
           profiled by the residual (a coarse grid in both, then golden section); riesz-free fixes kappa = 0.
Each identified model's algebra by its structure (exp61_wsindy.qg_algebra with the terms' scaling weights at s), scored
on the zero-mean states: rotation and scalings (the shift d_q leaves that space; reported, not scored).
Data and settings of exp61_wsindy.py (pyqg's data, 128^2, 16 training trajectories of data seed 4000 + s).

Run:  python exp64_qg_libraries.py regress --regime rd --seed 7 --data-seed 100 --save <stem>
"""

import argparse
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
import exp61_wsindy as e61w                                # noqa: E402
import method_wsindy2d as mw2                              # noqa: E402
import qg2d                                                # noqa: E402

OBS = ("u", "v", "uq", "vq", "uu", "vv", "uv")
OBS_W = {"u": (1.0, 0.0, 1.0), "v": (1.0, 0.0, 1.0), "uq": (1.0, 0.0, 2.0), "vq": (1.0, 0.0, 2.0),
         "uu": (2.0, 0.0, 2.0), "vv": (2.0, 0.0, 2.0), "uv": (2.0, 0.0, 2.0)}
OBS_MAP = {("uq", 1, 0): "adv_x", ("vq", 0, 1): "adv_y", ("v", 0, 0): "psi_x", ("u", 0, 0): "psi_y"}
S_GRID = np.arange(0.5, 3.0 + 1e-9, 0.5)
K_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
TRUE_KD2 = {"f": 0.0, "beta": 0.0, "rd": 1.0, "drag": 0.0}


def riesz_fields(KX, KY, q, s, kappa):
    """psi_s, u_s q, v_s q of one trajectory q [Nx, Ny, Nt]: psi_s_hat = -(|k|^2 + kappa)^(-s/2) q_hat (0 where
    |k|^2 + kappa = 0)."""
    K2 = KX ** 2 + KY ** 2 + kappa
    a = -np.where(K2 > 0, np.where(K2 > 0, K2, 1.0) ** (-s / 2), 0.0)
    ph = a[:, :, None] * np.fft.fft2(q, axes=(0, 1))
    inv = (lambda h: np.real(np.fft.ifft2(h, axes=(0, 1))))
    return inv(ph), inv(-1j * KY[:, :, None] * ph) * q, inv(1j * KX[:, :, None] * ph) * q


class Lib(e61w.QGLib):
    """exp61's weak SINDy library with the generalized transport keyed by (s, kappa) and the observed-velocity columns."""

    def riesz_cols(self, s, kappa):
        key = (float(np.round(s, 6)), float(np.round(kappa, 6)))
        if key not in self.cache:
            p = self.w.prep
            su = p["su"]
            cols = [[] for _ in e61w.NONLOCAL]
            for u in self.U:
                psi, uq, vq = riesz_fields(self.KX, self.KY, u, s, kappa)
                for i, (name, px, qy) in enumerate(e61w.NONLOCAL):
                    f = {"adv_x": uq / su ** 2, "adv_y": vq / su ** 2}.get(name, psi / su)
                    cols[i].append(mw2.conv_nd(f, (p["Cx"][px], p["Cy"][qy], p["Ct"][0]), p["s"]).ravel())
            G = np.stack([np.concatenate(c) for c in cols], 1)
            M = np.array([(su ** -2 if name.startswith("adv") else su ** -1) * p["sx"] ** -px * p["sy"] ** -qy
                          for name, px, qy in e61w.NONLOCAL])
            self.cache[key] = dict(G=G, M=M, X=self.G.T @ G, A=G.T @ G, c=G.T @ self.b)
        return self.cache[key]

    def riesz_resid(self, s, kappa):
        m = self.riesz_cols(s, kappa)
        A = np.block([[self.A, m["X"]], [m["X"].T, m["A"]]])
        c = np.concatenate([self.c, m["c"]])
        M = np.concatenate([self.M, m["M"]]) / self.Mlhs
        return e58.mstls_gram(A, c, self.bb, M, nterms=len(c))

    def riesz_fit(self, s, kappa):
        m = self.riesz_cols(s, kappa)
        G, M = np.concatenate([self.G, m["G"]], 1), np.concatenate([self.M, m["M"]])
        names = [("loc", j, p, q) for j, p, q in self.local] + [(name,) for name, _, _ in e61w.NONLOCAL]
        W, lam, resid = self.w.mstls(G, self.b, M / self.Mlhs)
        terms = [dict(kind=n[0], key=list(n), c=float(cc), kd2=None if n[0] == "loc" else float(kappa))
                 for n, cc in zip(names, W) if cc != 0]
        return terms, float(resid), float(lam)

    def obs_fit(self, V):
        """The reference library plus d_x^p d_y^r (f) of the observed velocity V [N, 2, Nx, Ny, Nt]."""
        p = self.w.prep
        su = p["su"]
        cols, names, M = [], [], []
        for f in OBS:
            for px in range(1 if f in ("u", "v") else 3):
                for qy in range(1 if f in ("u", "v") else 3 - px):
                    col = []
                    for q, (u, v) in zip(self.U, V):
                        fld = {"u": u, "v": v, "uq": u * q, "vq": v * q, "uu": u * u, "vv": v * v, "uv": u * v}[f]
                        lin = f in ("u", "v")
                        col.append(mw2.conv_nd(fld / (su if lin else su ** 2), (p["Cx"][px], p["Cy"][qy], p["Ct"][0]),
                                               p["s"]).ravel())
                    cols.append(np.concatenate(col))
                    names.append(("obs", f, px, qy))
                    M.append((su ** -1 if f in ("u", "v") else su ** -2) * p["sx"] ** -px * p["sy"] ** -qy)
        G = np.concatenate([self.G, np.stack(cols, 1)], 1)
        Mv = np.concatenate([self.M, np.array(M)])
        allnames = [("loc", j, a, b) for j, a, b in self.local] + names
        W, lam, resid = self.w.mstls(G, self.b, Mv / self.Mlhs)
        terms = [dict(kind=n[0], key=list(n), c=float(cc), kd2=None) for n, cc in zip(allnames, W) if cc != 0]
        return terms, float(resid), float(lam)


def golden(f, a, b, n=8):
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (c, fc) if fc < fd else (d, fd)


def profile(lib, free):
    """(s, kappa) by the residual: the grid S_GRID x K_GRID (kappa 0 only if free), then golden section on s within
    +-0.25 and on kappa between the neighbouring grid values (kept at 0 if that is no worse)."""
    grid = [(v, k) for v in S_GRID for k in ((0.0,) if free else K_GRID)]
    s, kappa = min(grid, key=lambda p: lib.riesz_resid(*p))
    s, _ = golden(lambda v: lib.riesz_resid(v, kappa), max(0.25, s - 0.25), s + 0.25)
    if not free:
        i = K_GRID.index(kappa)
        lo, hi = (K_GRID[i - 1] if i else 0.0), (K_GRID[i + 1] if i + 1 < len(K_GRID) else K_GRID[i] + 1.0)
        kappa, _ = golden(lambda k: lib.riesz_resid(s, k), lo, hi)
        kappa = 0.0 if lib.riesz_resid(s, 0.0) <= lib.riesz_resid(s, kappa) else kappa
    return float(s), float(kappa)


def weights_at(s):
    """Scaling weights (wx, wt, wu) of the completion's terms at transport exponent s (QG: s = 2, as
    exp61_wsindy.WEIGHTS)."""
    return {"adv_x": (s - 2.0, 0.0, 2.0), "adv_y": (s - 2.0, 0.0, 2.0), "psi": (s, 0.0, 1.0),
            "psi_x": (s - 1.0, 0.0, 1.0), "psi_y": (s - 1.0, 0.0, 1.0)}


def algebra(terms, s=2.0, screened=False, other=()):
    """exp61_wsindy.qg_algebra generalized to transport exponent s; obs terms without a quasi-geostrophic equivalent
    (other) break the rotation and the shift and add their weight rows."""
    saved = dict(e61w.WEIGHTS)
    try:
        e61w.WEIGHTS.update(weights_at(s))
        tt = [dict(t, kd2=(1.0 if screened else None) if t["kind"] != "loc" else None) for t in terms]
        alg = e61w.qg_algebra(tt)
    finally:
        e61w.WEIGHTS.clear()
        e61w.WEIGHTS.update(saved)
    if other:
        rows = [np.array(OBS_W[t["key"][1]]) - np.array([t["key"][2] + t["key"][3], 0.0, 0.0])
                - np.array([0.0, -1.0, 1.0]) for t in other]
        for t in terms:
            if t["kind"] == "loc":
                _, j, p, q = t["key"]
                rows.append(np.array([-(p + q), 0.0, float(j)]) - np.array([0.0, -1.0, 1.0]))
            else:
                rows.append(np.array(weights_at(s)[t["kind"]]) - np.array([0.0, -1.0, 1.0]))
        if screened:
            rows.append(np.array([1.0, 0.0, 0.0]))
        R = np.vstack(rows + [np.zeros(3)])
        _, sv, Vt = np.linalg.svd(R)
        rank = int((sv > 1e-9 * max(sv.max(), 1.0)).sum())
        alg = dict(rotation=False, shift=False, n_scalings=3 - rank, scalings=Vt[rank:].tolist())
    return alg


def to_qg(terms):
    out, other = [], []
    for t in terms:
        if t["kind"] != "obs":
            out.append(t)
            continue
        _, f, p, r = t["key"]
        m = OBS_MAP.get((f, p, r))
        if m:
            out.append(dict(t, kind=m, key=[m], c=-t["c"] if m == "psi_y" else t["c"]))
        else:
            other.append(t)
    return out, other


def true_eq(terms, regime, s=2.0, kappa=None, kd2=None):
    """The identified model is the regime's equation: the quasi-geostrophic terms (mapped) with coefficients within
    exp61's 2 %, s within 0.02 of 2, the screening (kappa, or the regressed kd2) within 0.05 of the truth."""
    want = e61w.TRUE_TERMS[regime]
    got = {tuple(t["key"]) if t["kind"] == "loc" else (t["kind"],): t for t in terms}
    if set(got) != set(want) or abs(s - 2.0) > 0.02:
        return False
    k = kappa if kappa is not None else kd2
    if k is None or abs(k - TRUE_KD2[regime]) > 0.05:
        return False
    return all(abs(got[key]["c"] / v - 1) <= e61w.COEF_TOL for key, v in want.items())


def regress(args):
    s0 = args.seed
    dseed = 4000 + s0 if args.data_seed is None else args.data_seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    t0 = time.time()
    Q = e61.make_data(args, g, args.n_train, dseed)
    psi = qg2d.to_psi(g, Q)
    uv = torch.stack([qg2d.irfft(g, -g.iky * torch.fft.rfft2(psi)), qg2d.irfft(g, g.ikx * torch.fft.rfft2(psi))], 2)
    U = Q.numpy().transpose(0, 2, 3, 1)
    V = uv.numpy().transpose(0, 2, 3, 4, 1)
    lib = Lib(U, g.x.numpy(), args.dt * np.arange(args.nt))
    truth = e61w.qg_true_algebra(args.regime)
    lap = qg2d.irfft(g, -g.K2 * torch.fft.rfft2(psi))
    A = torch.stack([lap.flatten()[::7], psi.flatten()[::7]], 1).numpy()
    c = np.linalg.lstsq(A, Q.flatten()[::7].numpy(), rcond=None)[0]
    kd2 = float(-c[1] / c[0])
    print(f"[{args.regime} (beta {g.beta}, rd {g.rd}, rek {g.rek})] seed {s0} (data {dseed}): {len(U)} trajectories, "
          f"{lib.w.info['rows']} rows; q = {c[0]:.6f} Lap psi + {c[1]:+.6f} psi (kd2 {kd2:.6f}); "
          f"{time.time() - t0:.0f} s", flush=True)
    rows = {}

    def put(name, terms, resid, lam, s=2.0, kappa=None, kdreg=None, prm=None):
        qgt, other = to_qg(terms)
        screened = (kappa or 0.0) > 0 or (kdreg is not None and abs(kdreg) > 0.05)
        alg = algebra(qgt, s, screened, other)
        right = alg["rotation"] == truth["rotation"] and e61w.same_span(alg["scalings"], truth["scalings"], 1e-4)
        rows[name] = dict(terms=terms, names=" ".join(e61w.qg_term_name(dict(t, kind=t["kind"] if t["kind"] != "obs"
                                                                               else "/".join(map(str, t["key"]))))
                                                       for t in terms),
                          resid=resid, lam=lam, prm=prm, s=s, kappa=kappa, kd2_regressed=kdreg,
                          true_eq=true_eq(qgt + other, args.regime, s, kappa, kdreg), right_algebra=bool(right),
                          algebra=alg)
        r = rows[name]
        print(f"  {name:10s} {'TRUE ' if r['true_eq'] else 'wrong'} algebra {'right' if right else 'WRONG'} (rotation "
              f"{alg['rotation']}, shift {alg['shift']}, scalings {alg['n_scalings']}) resid {resid:.3g}"
              + (f" s {s:.4f} kappa {kappa:.4f}" if kappa is not None else "") + f"  q_t = {r['names']}", flush=True)

    t1 = time.time()
    put("ref", *lib.fit(), kdreg=kd2)
    put("obs", *lib.obs_fit(V), kdreg=kd2)
    print(f"  ({time.time() - t1:.0f} s)", flush=True)
    if args.obs_only:
        return
    for name, free in (("riesz-free", True), ("riesz", False)):
        t1 = time.time()
        s, kappa = profile(lib, free)
        put(name, *lib.riesz_fit(s, kappa), s=s, kappa=kappa, prm=[s, kappa])
        print(f"  ({time.time() - t1:.0f} s, {len(lib.cache)} evaluations cached)", flush=True)
    pick = "ref"
    for nxt in ("riesz-free", "riesz"):
        if rows[pick]["resid"] > e61w.NEST * rows[nxt]["resid"]:
            pick = nxt
    rows["nested"] = dict(rows[pick], picked=pick)
    print(f"  nested -> {pick}; total {time.time() - t0:.0f} s", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(e61.RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(dict(config=vars(args), data_seed=dseed, kd2_regressed=kd2, rows=rows,
                           wsindy={k: float(v) for k, v in lib.w.info.items()}), fh, indent=1)
        print(f"  saved {stem}.json")


def jobs(args):
    out = []
    for regime in qg2d.REGIMES:
        for s in args.seeds.split(","):
            stem = os.path.join(args.runs, f"lib_{regime}_s{s}")
            out.append((stem + ".json", ["exp64_qg_libraries.py", "regress", "--regime", regime, "--seed", s,
                                         "--threads", "1", "--save", stem]))
    return out


def summary(args):
    """results/pyqg/libraries_summary.{json,md}."""
    res, lines = {}, ["# Two-stage rows without quasi-geostrophic knowledge on pyqg's data (exp64)\n",
                      "| regime | row | n | true equation | right algebra | s (range) | kappa (range) |",
                      "|---|---|---|---|---|---|---|"]
    for regime in qg2d.REGIMES:
        cell = {}
        for s in args.seeds.split(","):
            p = os.path.join(args.runs, f"lib_{regime}_s{s}.json")
            if os.path.exists(p):
                j = json.load(open(p))
                for r, v in j["rows"].items():
                    cell.setdefault(r, {})[s] = {k: v.get(k) for k in ("true_eq", "right_algebra", "algebra", "names",
                                                                       "resid", "s", "kappa", "kd2_regressed",
                                                                       "picked")}
        if not cell:
            continue
        res[regime] = cell
        for r, v in cell.items():
            ss = [a["s"] for a in v.values() if a.get("kappa") is not None]
            ks = [a["kappa"] for a in v.values() if a.get("kappa") is not None]
            lines.append(f"| {regime} | {r} | {len(v)} | {sum(a['true_eq'] for a in v.values())} | "
                         f"{sum(a['right_algebra'] for a in v.values())} | "
                         + (f"{min(ss):.4f} to {max(ss):.4f} | {min(ks):.4f} to {max(ks):.4f} |" if ss else "- | - |"))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pyqg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "libraries_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "libraries_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["regress", "run", "summary"])
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(e61.RESULTS, "pyqg_lib_runs"))
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
    ap.add_argument("--obs-only", action="store_true")
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
