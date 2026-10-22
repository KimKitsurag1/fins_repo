"""Experiment 65 -- symmetry audit of PG8's FNO2d emulators of pyqg (notebook 9; README_EXPERIMENTS.md).

  fins    -- FINS through an emulator on its own rollouts: the emulator rolled out from the initial states of PG8's
             FINS data (16 training trajectories of data seed 4000 + s, 8 held-out of 5000 + s), so the pencil's data
             term follows the emulator and the quotients measure its symmetry defects rather than its error against
             pyqg (PG8 used pyqg's data, where every quotient carries the error). exp61's readouts.
  consequence -- the emulator's one-step error against the port (qg2d, two RK4 steps with pyqg's filter) on held-out
             pyqg states transformed by finite flows of the regime's generators, over its error on the untransformed
             states: shifts by half a cell (d_x, d_y), rotation by 30 degrees, x -> 1.25 x (D_x), q -> 1.25 q (D_t),
             both with q -> 1.25 q (D_beta); spline interpolation (order 3) about the box centre, pyqg's filter after.

Run:  python exp65_emulator_audit.py fins --regime f --seed 42 --save <stem>
      python exp65_emulator_audit.py consequence --regime f --seed 42 --save <stem>
"""

import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from scipy.ndimage import affine_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
GENS = {"f": ("d_x", "d_y", "rotation", "D_x", "D_t"), "beta": ("d_x", "d_y", "D_beta")}


def rollout(model, q0, nt):
    """The emulator's own trajectories from q0 [B, n, n]: [B, nt, n, n]."""
    out, q = [q0], q0
    with torch.no_grad():
        for _ in range(nt - 1):
            q = model(q)
            out.append(q)
    return torch.stack(out, 1)


def device_of(args):
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def fins(args):
    s = args.seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    model = e61.load_fno(args, g.cast(torch.float32))
    dev = device_of(args)
    dtype = torch.float32 if dev != "cpu" else torch.float64
    m = copy.deepcopy(model).to(dtype).to(dev)
    t0 = time.time()
    qtr = e61.make_data(args, g, args.n_train, 4000 + s)
    qev = e61.make_data(args, g, args.n_eval, 5000 + s)
    rtr = rollout(m, qtr[:, 0].to(dtype).to(dev), args.nt).double().cpu()
    rev = rollout(m, qev[:, 0].to(dtype).to(dev), args.nt).double().cpu()
    drift = float((rtr - qtr).norm() / qtr.norm())
    print(f"[{args.regime}, emulator s{s}] own rollouts: {args.n_train} + {args.n_eval} from pyqg's initial states; "
          f"relative distance to pyqg over the window {drift:.3f}; |q| max {float(rtr.abs().max()):.3f} (pyqg "
          f"{float(qtr.abs().max()):.3f}); {time.time() - t0:.0f} s", flush=True)
    dtr = e45.Data(g, rtr, args.dt)
    dvl = e45.Data(g, rev, args.dt, dtr.s)
    op = e45.FNOOp(m)
    d_ev = dtr if dev == "cpu" else dtr.cast(dtype, dev)
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = e45.pencil(op, d_ev, e45.chars_poly, ptr, args.K, args.chunk)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N, D)
    names, ents = qg2d.algebra(args.regime, "q")
    Ct = qg2d.coefficients(ents)
    rq = {n: e61.quotient(N, D, c) for n, c in e61.candidates("q").items()}
    lf6 = np.maximum(lam[:17], 1e-6)
    own = int(np.argmax(lf6[1:] / lf6[:-1])) + 1
    pr = e61.plane_readouts(N, D, "q")
    probe = e45.Probe(dvl)
    true_q = probe.poly(Ct)[1]
    spans = {}
    for lab, k in (("span_k", Ct.shape[1]), ("span_own", own)):
        Qc = probe.poly(C[:, :k])[1]
        spans[lab] = dict(m=int(k), missing=e45.missing(np.where(np.isfinite(Qc), Qc, 0.0), true_q))
    print(f"  pencil on the emulator's rollouts ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change "
          f"{np.sqrt(cmu):.2e}; lowest " + " ".join(f"{v:.1e}" for v in lam[:10]) + f"; own count {own}")
    print("    quotients: " + "  ".join(f"{n} {v:.1e}" + ("*" if n in names else "") for n, v in rq.items()))
    print(f"    rotation plane {pr['rot_lam'][0]:.1e} (stretch {pr['rot_e']}); scaling space " + "; ".join(
        f"{l:.1e} (" + ", ".join(f"{x:+.3f}" for x in v) + ")" for l, v in zip(pr["scal_lam"], pr["scal_vec"])))
    print("    " + "; ".join(f"{k} missing {v['missing']:.3f}" for k, v in spans.items()), flush=True)
    out = dict(config=vars(args), drift=drift, lam=lam[:20].tolist(), names=names, quotients=rq, own=own, cmu=cmu,
               planes=pr, **spans)
    save(args, out, dict(C=C[:, :20], lam=lam, N=N, D=D))


def transform(q, kind, g, lam=1.25, theta=30.0, shift=0.5):
    """A finite flow of a generator applied to q [n, n] (numpy; the box centre is the origin)."""
    n = q.shape[0]
    if kind == "identity":
        return q.copy()
    if kind in ("d_x", "d_y"):
        k = 2 * np.pi * np.fft.fftfreq(n, d=g.dx)
        ph = np.exp(-1j * k * shift * g.dx)
        h = np.fft.fft2(q) * (ph[:, None] if kind == "d_x" else ph[None, :])
        return np.real(np.fft.ifft2(h))
    if kind == "D_t":
        return lam * q
    if kind == "rotation":
        t = np.radians(theta)
        M = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    elif kind in ("D_x", "D_beta"):
        M = np.eye(2) / lam
    else:
        raise ValueError(kind)
    c = np.array([n / 2.0, n / 2.0])
    out = affine_transform(q, M, offset=c - M @ c, order=3, mode="grid-wrap")
    return lam * out if kind == "D_beta" else out


def consequence(args):
    s = args.seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    model = e61.load_fno(args, g.cast(torch.float32)).double()
    qev = e61.make_data(args, g, args.n_eval, 5000 + s)
    states = qev[:, list(range(args.snap_first, args.nt, args.snap_every))].reshape(-1, g.n, g.n)
    kinds = ("identity",) + GENS[args.regime]
    res = {}
    for kind in kinds:
        X = torch.stack([torch.tensor(transform(q.numpy(), kind, g)) for q in states])
        X = qg2d.filt(g, X)
        with torch.no_grad():
            F = model(X)
        P = X
        for _ in range(args.nsub):
            P = qg2d.step(g, P, args.dt / args.nsub)
        num = ((F - P) ** 2).sum((-2, -1)).sqrt()
        den = ((P - X) ** 2).sum((-2, -1)).sqrt()
        e = (num / den).numpy()
        res[kind] = dict(err_mean=float(e.mean()), err_median=float(np.median(e)), err=e.tolist())
        print(f"  {kind:9s} one-step error relative to the increment: mean {e.mean():.4f}, median {np.median(e):.4f}",
              flush=True)
    base = res["identity"]["err_mean"]
    ratio = {k: v["err_mean"] / base for k, v in res.items() if k != "identity"}
    print("  ratio to identity: " + "  ".join(f"{k} {v:.3f}" for k, v in ratio.items()), flush=True)
    save(args, dict(config=vars(args), n_states=int(states.shape[0]), errors=res, ratio=ratio))


def save(args, out, arrays=None):
    if not args.save:
        return
    stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=1)
    if arrays is not None:
        np.savez(stem + ".npz", **arrays)
    print(f"  saved {stem}.json")


def stem_of(runs, what, regime, s):
    return os.path.join(runs, f"audit_{what}_{regime}_s{s}")


def jobs(args):
    out = []
    for what in ("fins", "consequence"):
        for regime in ("f", "beta"):
            for s in args.seeds.split(","):
                stem = stem_of(args.runs, what, regime, s)
                out.append((stem + ".json", ["exp65_emulator_audit.py", what, "--regime", regime, "--seed", s,
                                             "--save", stem]))
    return out


def summary(args):
    """results/pyqg/audit_summary.{json,md}: per regime and seed, the self-consistent readouts and the error ratios."""
    res = {"fins": {}, "consequence": {}}
    for what in res:
        for regime in ("f", "beta"):
            for s in args.seeds.split(","):
                p = stem_of(args.runs, what, regime, s) + ".json"
                if os.path.exists(p):
                    j = json.load(open(p))
                    keys = ("quotients", "planes", "own", "lam", "drift", "span_k") if what == "fins" else ("ratio",)
                    res[what].setdefault(regime, {})[s] = {k: j[k] for k in keys}
    lines = ["# Audit of PG8's emulators (exp65)\n", "| regime | seed | rotation plane | quotients (self-consistent) | "
             "error ratio (transformed / untransformed) |", "|---|---|---|---|---|"]
    for regime, cell in res["fins"].items():
        for s, r in cell.items():
            ratio = res["consequence"].get(regime, {}).get(s, {}).get("ratio", {})
            lines.append(f"| {regime} | {s} | {r['planes']['rot_lam'][0]:.1e} | "
                         + " ".join(f"{g} {r['quotients'][g]:.1e}" for g in GENS[regime]) + " | "
                         + " ".join(f"{g} {v:.3f}" for g, v in ratio.items()) + " |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pyqg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "audit_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "audit_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["fins", "consequence", "run", "summary"])
    ap.add_argument("--regime", default="f", choices=["f", "beta"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(RESULTS, "audit_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fno-tag", default="")
    ap.add_argument("--train-fno-if-missing", action="store_true")
    ap.add_argument("--fno-n", type=int, default=256)
    ap.add_argument("--fno-width", type=int, default=32)
    ap.add_argument("--fno-modes", type=int, default=24)
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-epochs", type=int, default=40)
    ap.add_argument("--fno-batch", type=int, default=32)
    ap.add_argument("--fno-lr", type=float, default=1e-3)
    ap.add_argument("--var", default="q")
    ap.add_argument("--data", default="pyqg", choices=["port", "pyqg"])
    ap.add_argument("--seed", type=int, default=42)
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
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--snap-first", type=int, default=10)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"fins": fins, "consequence": consequence, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
