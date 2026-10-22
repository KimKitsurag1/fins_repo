"""Experiment 62 -- the drag's deformed time scaling, read by FINS in a class with tau(t) and the q-coefficient to
degree 6 in t (README_EXPERIMENTS.md, entry of 09.10).

pyqg's drag q_t + J(psi, q) = -rek Lap psi (= -rek (q - <q>)) is point-equivalent to the inviscid flow by
q = e^(-r t) w, s = (1 - e^(-r t)) / r: its time scaling is V_r = ((e^(r t) - 1) / r) d_t - e^(r t) q d_q, outside
exp45's degree-2 class. Class here: the 60 fields plus tau T^3..T^6 and phi T^2 U..T^6 U (69 fields; scaled
components, as exp45). A 70th field, the exact V_r characteristic, enters the pencil for its quotient only.
Readouts: own count (exp49's floor 1e-6); the pencil restricted to the time fields (tau T^0..T^6, phi U T^0..T^6):
the direction of the two deepest with tau(0) = 0 and phi = -q at t = 0 gives rho = r T_max from its phi T U
coefficient (rho-hat) and its tau T^2 / T coefficients; missing of the regime's algebra (exact V_r) in spans.

Run:  python exp62_drag_tau.py fins --regime drag --rek 0.3 --seed 7 --data-seed 100 --eval-seed 101 --save <stem>
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
NM = len(e45.MONS)
EXTRA = [(2, k) for k in range(3, 7)] + [(3, k) for k in range(2, 7)]      # (component, power of T); 3: times U
NF = 4 * NM + len(EXTRA)
DEG = 6
RATES = (0.1, 0.3, 0.6)


def index(c, k=0, u=0, a=0, b=0):
    """Position of component c's monomial X^a Y^b T^k U^u in the 69-field basis."""
    mon = (a, b, k, u)
    if sum(mon) <= 2:
        return c * NM + e45.MONS.index(mon)
    return 4 * NM + EXTRA.index((c, k))


def time_fields():
    """tau T^0..T^6 and phi U T^0..T^6 (no shift): indices into the class."""
    return [index(2, k) for k in range(DEG + 1)] + [index(3, k, 1) for k in range(DEG + 1)]


def vec(entries):
    v = np.zeros(NF)
    for c, k, u, a, b, val in entries:
        v[index(c, k, u, a, b)] += val
    return v


def vr_taylor(rho):
    """V_r truncated at degree 6 in scaled components: tau / T = sum rho^(k-1) T^k / k!, phi / sigma =
    -sum rho^k T^k U / k!."""
    ents = [(2, k, 0, 0, 0, rho ** (k - 1) / math.factorial(k)) for k in range(1, DEG + 1)]
    ents += [(3, k, 1, 0, 0, -rho ** k / math.factorial(k)) for k in range(DEG + 1)]
    return vec(ents)


def candidates(rho):
    return {"d_x": vec([(0, 0, 0, 0, 0, 1.0)]), "d_y": vec([(1, 0, 0, 0, 0, 1.0)]),
            "d_t": vec([(2, 0, 0, 0, 0, 1.0)]), "shift": vec([(3, 0, 0, 0, 0, 1.0)]),
            "rotation": vec([(0, 0, 0, 0, 1, -1.0), (1, 0, 0, 1, 0, 1.0)]),
            "D_x": vec([(0, 0, 0, 1, 0, 1.0), (1, 0, 0, 0, 1, 1.0)]),
            "D_t": vec([(2, 1, 0, 0, 0, 1.0), (3, 0, 1, 0, 0, -1.0)]),
            "V_r taylor": vr_taylor(rho)}


def chars_ext(d, r, j, reg):
    u, ux, uy, ut, X, Y, T = d.at(r, j, reg)
    fac, U = d.factors(u, ux, uy, ut), u / d.s["sigma"]
    ext = [fac[..., c, :, :] * T ** k * (U if c == 3 else 1.0) for c, k in EXTRA]
    return torch.cat([e45.chars_poly(d, r, j, reg), torch.stack(ext, -3)], -3)


def chars_vr(rate):
    """The exact characteristic of V_rate (Q / sigma), rate 0: t d_t - q d_q."""
    def f(d, r, j, reg):
        u, ux, uy, ut, X, Y, T = d.at(r, j, reg)
        fac, t = d.factors(u, ux, uy, ut), T * d.s["T"]
        tau = t if rate == 0 else torch.expm1(rate * t) / rate
        return (tau / d.s["T"]) * fac[..., 2, :, :] - torch.exp(rate * t) * (u / d.s["sigma"]) * fac[..., 3, :, :]
    return f


def chars_run(rate):
    vr = chars_vr(rate)

    def f(d, r, j, reg):
        return torch.cat([chars_ext(d, r, j, reg), vr(d, r, j, reg).unsqueeze(-3)], -3)
    return f


def pencil_terms(op, d, charfn, r, j0, K):
    """exp45.pencil_terms with B'B returned as a matrix."""
    sl = d.g.sl
    Q0 = charfn(d, r, j0, slice(None))
    dl = op.transport(d.u[r, j0], Q0, K)[..., sl, sl]
    jk = j0[:, None] + torch.arange(1, K + 1)
    Qk = charfn(d, r[:, None].expand_as(jk), jk, sl)
    Q0s = Q0[..., sl, sl][:, None]
    m = Q0.shape[1]
    A = (dl - Q0s).movedim(2, 0).reshape(m, -1).double()
    B = (Qk - Q0s).movedim(2, 0).reshape(m, -1).double()
    Q = Q0s[:, 0].movedim(1, 0).reshape(m, -1).double()
    R = A - B
    BB = B @ B.T
    return R @ R.T, A @ A.T + BB, K * (Q @ Q.T), BB


def pencil(op, d, charfn, pairs, K, chunk=2):
    tot = None
    with torch.no_grad():
        for c in range(0, len(pairs), chunk):
            p = pairs[c:c + chunk]
            terms = pencil_terms(op, d, charfn, p[:, 0], p[:, 1], K)
            tot = list(terms) if tot is None else [a + b for a, b in zip(tot, terms)]
    return [t.numpy() for t in tot]


def quotient(N, D, c):
    return float(c @ N @ c / (c @ D @ c))


def rho_of(w):
    """rho-hat of a time-field direction normalised to phi U = -1: (from phi T U, from tau T^2 / tau T)."""
    t, p = w[:DEG + 1], w[DEG + 1:]
    t, p = t / -p[0], p / -p[0]
    return float(-p[1]), float(2 * t[2] / t[1])


def time_readout(N, D):
    """Pencil restricted to the time fields: eigenvalues, and the deepest two combined to tau(0) = 0."""
    idx = time_fields()
    lt, Wt = eigh(N[np.ix_(idx, idx)], D[np.ix_(idx, idx)])
    w1, w2 = Wt[:, 0], Wt[:, 1]
    w = w1[0] * w2 - w2[0] * w1
    rp, rt = rho_of(w)
    return dict(lam=lt[:4].tolist(), rho_phi=rp, rho_tau=rt, vec=(w / -w[DEG + 1]).tolist())


def span_readout(C):
    """Inside a span of class vectors C [69, m]: the 2D subspace lying in the time fields (smallest singular values
    of its components outside them), combined to tau(0) = 0."""
    idx = time_fields()
    out = [i for i in range(NF) if i not in idx]
    _, s, Vh = np.linalg.svd(C[out], full_matrices=True)
    s = np.concatenate([s, np.zeros(C.shape[1] - len(s))]) if len(s) < C.shape[1] else s
    a1, a2 = Vh[-1], Vh[-2]
    w1, w2 = C[idx] @ a1, C[idx] @ a2
    w = w1[0] * w2 - w2[0] * w1
    rp, rt = rho_of(w)
    return dict(sing=sorted(s.tolist())[:3], rho_phi=rp, rho_tau=rt)


class Points:
    """Held-out sample points (exp45.Probe's: snapshots every 4 in [2, nt - 3], every second point of the scored
    block): class characteristics [69, M] and exact V_rate characteristics."""

    def __init__(self, d, every_t=4, every_x=2):
        g = d.g
        self.reg = slice(g.sl.start, g.sl.stop, every_x)
        js = torch.arange(2, d.u.shape[1] - 2, every_t)
        self.r, self.j = torch.meshgrid(torch.arange(d.u.shape[0]), js, indexing="ij")
        self.d = d
        with torch.no_grad():
            self.Q = chars_ext(d, self.r, self.j, self.reg).movedim(2, 0).reshape(NF, -1).numpy()

    def exact(self, rate):
        with torch.no_grad():
            return chars_vr(rate)(self.d, self.r, self.j, self.reg).reshape(1, -1).numpy()


def fins(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    rek = args.rek if args.regime == "drag" else 0.0
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score, rek=rek)
    rate = rek if rek > 0 else args.vr_rate
    t0 = time.time()
    dtr = e45.Data(g, e61.make_data(args, g, args.n_train, dseed), args.dt)
    dev = e45.Data(g, e61.make_data(args, g, args.n_eval, eseed), args.dt, dtr.s)
    Ts = dtr.s["T"]
    print(f"[{args.regime} (rek {g.rek}), q, {args.data} data] seed {s}: {args.n_train} training trajectories (data "
          f"seed {dseed}), {args.n_eval} held-out ({eseed}); data {time.time() - t0:.0f} s; T {Ts:g}", flush=True)
    op = qg2d.ExactOp(g, args.dt, args.nsub, "q")
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, BB = pencil(op, dtr, chars_run(rate), ptr, args.K)
    cmu = float(np.trace(BB[:NF, :NF]) / np.trace(G0[:NF, :NF]))
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N[:NF, :NF], D[:NF, :NF])
    lf6 = np.maximum(lam[:17], 1e-6)
    own = int(np.argmax(lf6[1:] / lf6[:-1])) + 1
    cand = candidates(rek * Ts)
    rq = {n: quotient(N[:NF, :NF], D[:NF, :NF], c) for n, c in cand.items()}
    e70 = np.zeros(NF + 1)
    e70[NF] = 1.0
    rq[f"V_{rate:g} exact"] = quotient(N, D, e70)
    tr = time_readout(N[:NF, :NF], D[:NF, :NF])
    print(f"  class of {NF} fields ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change over K "
          f"{np.sqrt(cmu):.2e}")
    print("    lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:12]))
    print(f"    own count (floor 1e-6) {own}")
    print("    quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in rq.items()))
    print(f"    time fields: eigenvalues " + " ".join(f"{v:.1e}" for v in tr["lam"])
          + f"; r-hat {tr['rho_phi'] / Ts:.4f} (phi), {tr['rho_tau'] / Ts:.4f} (tau); true {rek:g}", flush=True)
    pts = Points(dev)
    names = ["d_x", "d_y", "d_t", "shift", "rotation", "D_x"]
    true_q = np.concatenate([np.stack([cand[n] for n in names]) @ pts.Q, pts.exact(rek)])
    names.append("V_r" if rek > 0 else "D_t")
    spans = {}
    for lab, m in (("span_k", len(names)), ("span_own", own)):
        Qc = C[:, :m].T @ pts.Q
        spans[lab] = dict(m=int(m), missing=e45.missing(Qc, true_q), novel=e45.novel(Qc, true_q).tolist(),
                          **{f"time_{k}": v for k, v in span_readout(C[:, :m]).items()})
        sp = spans[lab]
        print(f"    span of the {m} lowest ({lab}): missing {sp['missing']:.3f}; novel "
              + " ".join(f"{n} {v:.2f}" for n, v in zip(names, sp["novel"]))
              + f"; time sub-span r-hat {sp['time_rho_phi'] / Ts:.4f} (phi), {sp['time_rho_tau'] / Ts:.4f} (tau)",
              flush=True)
    out = dict(config=vars(args), data_seed=dseed, eval_seed=eseed, rek=rek, rate=rate, T=Ts, lam=lam[:20].tolist(),
               own=own, cmu=cmu, quotients=rq, names=names, time=tr,
               r_hat=dict(phi=tr["rho_phi"] / Ts, tau=tr["rho_tau"] / Ts), **spans)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", C=C[:, :20], lam=lam, N=N, D=D)
        print(f"  saved {stem}.json/.npz")


ROWS = [("drag", r) for r in RATES] + [("f", 0.0)]


def stem_of(runs, regime, rek, s):
    return os.path.join(runs, f"tau_{regime}" + (f"_r{rek:g}" if regime == "drag" else "") + f"_s{s}")


def jobs(args):
    out = []
    for regime, rek in ROWS:
        for s in args.seeds.split(","):
            stem = stem_of(args.runs, regime, rek, s)
            out.append((stem + ".json", ["exp62_drag_tau.py", "fins", "--regime", regime, "--rek", str(rek),
                                         "--seed", s, "--threads", "1", "--save", stem]))
    return out


def summary(args):
    """results/drag_tau/summary.{json,md}: per row and seed, own count, missing, r-hat, the quotients."""
    res, lines = {}, ["# The drag's time scaling in a tau(t) class (exp62)\n",
                      "| row | n | own = 7 | missing (span 7) <= 0.05 | r-hat (phi) | V_r exact (median) | D_t (median) |",
                      "|---|---|---|---|---|---|---|"]
    for regime, rek in ROWS:
        cell = {}
        for s in args.seeds.split(","):
            p = stem_of(args.runs, regime, rek, s) + ".json"
            if os.path.exists(p):
                j = json.load(open(p))
                cell[s] = {k: j[k] for k in ("own", "span_k", "span_own", "r_hat", "quotients", "lam", "time")}
        if not cell:
            continue
        lab = f"{regime} rek {rek:g}"
        res[lab] = cell
        rh = [c["r_hat"]["phi"] for c in cell.values()]
        vx = [v for c in cell.values() for k, v in c["quotients"].items() if k.endswith("exact")]
        dt = [c["quotients"]["D_t"] for c in cell.values()]
        lines.append(f"| {lab} | {len(cell)} | {sum(c['own'] == 7 for c in cell.values())} | "
                     f"{sum(c['span_k']['missing'] <= 0.05 for c in cell.values())} | {min(rh):.4f} to {max(rh):.4f} | "
                     f"{np.median(vx):.1e} | {np.median(dt):.1e} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "drag_tau")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["fins", "run", "summary"])
    ap.add_argument("--regime", default="drag", choices=["drag", "f"])
    ap.add_argument("--rek", type=float, default=0.3)
    ap.add_argument("--vr-rate", type=float, default=0.3)
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "drag_tau_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--var", default="q")
    ap.add_argument("--data", default="pyqg", choices=["port", "pyqg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--nsub", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"fins": fins, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
