"""Experiment 61 -- point symmetries of quasi-geostrophic flow from pyqg's solution operator, and which physical
parameters break them (README_EXPERIMENTS.md; development 09.10).

qg2d.py: pyqg's barotropic model (BTModel) in torch with its exact tangent. Regimes (qg2d.REGIMES): inviscid f-plane,
beta-plane, a deformation radius rd, a bottom drag rek. FINS: exp45's polynomial class (60 fields, degree 2 in
x, y, t, u) through the exact tangent, u = q (potential vorticity) or u = psi (pressure); relative pencil on the
training trajectories; readouts on the pencil restricted to the plane (y d_x, x d_y) (rotation) and to the scaling space
(x d_x + y d_y, t d_t, u d_u): their generalized eigenpairs, so the count of deep directions and their exponents.

Run:  python exp61_pyqg.py check --regime f
      python exp61_pyqg.py fins --regime beta --seed 7 --data-seed 100 --eval-seed 101 --save <stem>
"""

import argparse
import collections
import copy
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pyqg
import torch
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp60_aniso2d as e60                                # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
CANDIDATES = ("d_x", "d_y", "d_t", "shift", "rotation", "D_x", "D_t", "D_beta")


def pyqg_run(g, q0, T, h):
    """pyqg's BTModel (AB3) from q0 [n, n] (our layout), round(T / h) steps of h: q(T) in our layout."""
    m = pyqg.BTModel(nx=g.n, L=g.L, beta=g.beta, rd=g.rd, rek=g.rek, U=g.U, filterfac=g.filterfac, dt=h, tmax=T,
                     twrite=10 ** 9, tavestart=10 * T, log_level=0)
    m.set_q(q0.numpy().T[None].copy())
    for _ in range(int(round(T / h))):
        m._step_forward()
    return torch.tensor(np.asarray(m.q)[0].T.copy()), m.t


def pyqg_trajectories(g, n_traj, seed, nt, dt, nsub, var="q"):
    """qg2d's initial conditions integrated by pyqg itself (AB3, step dt / nsub): [B, nt, n, n] in our layout."""
    q0 = qg2d.initial(g, n_traj, np.random.default_rng(seed))
    out = np.zeros((n_traj, nt, g.n, g.n))
    for r in range(n_traj):
        m = pyqg.BTModel(nx=g.n, L=g.L, beta=g.beta, rd=g.rd, rek=g.rek, U=g.U, filterfac=g.filterfac, dt=dt / nsub,
                         tmax=dt * nt, twrite=10 ** 9, tavestart=10 * dt * nt, log_level=0)
        m.set_q(q0[r].double().numpy().T[None].copy())
        out[r, 0] = np.asarray(m.q)[0].T
        for j in range(1, nt):
            for _ in range(nsub):
                m._step_forward()
            out[r, j] = np.asarray(m.q)[0].T
    return qg2d.observe(g, torch.tensor(out, dtype=g.dtype), var)


def make_data(args, g, n_traj, seed):
    if args.data == "pyqg":
        return pyqg_trajectories(g, n_traj, seed, args.nt, args.dt, args.nsub, args.var)
    return qg2d.trajectories(g, n_traj, seed, args.nt, args.dt, args.nsub, args.var)


def check(args):
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    t0 = time.time()
    q = qg2d.trajectories(g, 4, 100, args.nt, args.dt, args.nsub)
    T = args.dt * (args.nt - 1)
    print(f"[{args.regime}: beta {g.beta}, rd {g.rd}, rek {g.rek}] 4 trajectories, T {T:g}, step "
          f"{args.dt / args.nsub:g}: {time.time() - t0:.1f} s; q in [{float(q.min()):.3f}, {float(q.max()):.3f}]")
    ref = q[:, -1]
    h = args.dt / args.nsub
    out, tt = zip(*[pyqg_run(g, q[r, 0], T, h) for r in range(4)])
    out = torch.stack(out)
    print(f"  pyqg AB3, same step (t {tt[0]:.4f}): |q - pyqg| / |q| {float((out - ref).norm() / ref.norm()):.2e}")
    dq = q[:, 1:] - q[:, :-1]
    print(f"  (relative to the change over one snapshot: {float((out - ref).norm() / dq[:, -1].norm()):.2e})")
    half = qg2d.solve(g, q[:, 0], args.nt, args.dt, 2 * args.nsub)[:, -1]
    g0 = qg2d.Grid(g.n, g.L, g.score, beta=g.beta, rd=g.rd, rek=g.rek, filterfac=0.0)
    nof = qg2d.solve(g0, q[:, 0], args.nt, args.dt, args.nsub)[:, -1]
    print(f"  RK4 half step (the filter acts per step): {float((half - ref).norm() / ref.norm()):.2e}; without the "
          f"filter (exp(0) = 1): {float((nof - ref).norm() / ref.norm()):.2e}")
    seam = (q[..., :4, :] ** 2).mean().sqrt() / (q ** 2).mean().sqrt()
    print(f"  rms of q within 1 of the seam / overall: {float(seam):.1e} (max over time of the seam rms "
          f"{float((q[..., :4, :] ** 2).mean((0, 2, 3)).sqrt().max()):.1e})")
    d = e45.Data(g, q, args.dt)
    rel = float(((q[:, args.K:] - q[:, :-args.K]) ** 2).mean().sqrt() / (q ** 2).mean().sqrt())
    print(f"  relative change over K = {args.K} snapshots {rel:.3f}")
    op = qg2d.ExactOp(g, args.dt, args.nsub)
    r, j = torch.tensor([0, 1]), torch.tensor([5, 20])
    qc = e45.chars_poly(d, r, j, slice(None))[:, [1, 17, 40]]
    jv = op.transport(d.u[r, j], qc, 3)[:, -1]
    eps = 1e-5 * float(d.u.abs().max()) / float(qc.abs().max())
    up = qg2d.solve(g, (d.u[r, j][:, None] + eps * qc).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    um = qg2d.solve(g, (d.u[r, j][:, None] - eps * qc).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    fd = ((up - um) / (2 * eps)).reshape(jv.shape)
    print(f"  tangent vs central differences: rel err {float((jv - fd).norm() / fd.norm()):.1e}")


def candidates(var):
    """Every generator any regime keeps, in q or psi weights: {name: coefficient vector [60]}."""
    ents = dict(qg2d.GENERATORS)
    ents["D_x"] = qg2d.scaling(1.0, 0.0, 0.0 if var == "q" else 2.0)
    ents["D_t"] = qg2d.scaling(0.0, 1.0, -1.0)
    ents["D_beta"] = qg2d.scaling(1.0, -1.0, 1.0 if var == "q" else 3.0)
    return {k: qg2d.coefficients([ents[k]])[:, 0] for k in CANDIDATES}


def quotient(N, D, c):
    return float(c @ N @ c / (c @ D @ c))


def sub_eig(N, D, V):
    """Generalized eigenpairs of the pencil restricted to the columns of V (ascending)."""
    return eigh(V.T @ N @ V, V.T @ D @ V)


def plane_readouts(N, D, var):
    """Rotation plane (y d_x, x d_y): eigenvalues, stretch e = sqrt(-c2 / c1) of the deepest direction (1: the
    rotation). Scaling space (x d_x + y d_y, t d_t, u d_u): eigenvalues and eigenvectors scaled to unit max entry."""
    V = qg2d.coefficients([[(0, (0, 1, 0, 0), 1.0)], [(1, (1, 0, 0, 0), 1.0)]])
    lr, Wr = sub_eig(N, D, V)
    c1, c2 = Wr[0, 0], Wr[1, 0]
    S = qg2d.coefficients([qg2d.scaling(1.0, 0.0, 0.0), qg2d.scaling(0.0, 1.0, 0.0), qg2d.scaling(0.0, 0.0, 1.0)])
    ls, Ws = sub_eig(N, D, S)
    vecs = [(w / w[np.argmax(np.abs(w))]).tolist() for w in Ws.T]
    return dict(rot_lam=lr.tolist(), rot_e=float(np.sqrt(-c2 / c1)) if c1 * c2 < 0 else None,
                scal_lam=ls.tolist(), scal_vec=vecs)


KEPT = {"f": CANDIDATES, "beta": ("d_x", "d_y", "d_t", "shift", "D_beta"),
        "rd": ("d_x", "d_y", "d_t", "shift", "rotation", "D_t"),
        "drag": ("d_x", "d_y", "d_t", "shift", "rotation", "D_x")}
SCALINGS = {"f": [(1.0, 0.0, 0.0), (0.0, 1.0, -1.0)], "beta": [(1.0, -1.0, 1.0)], "rd": [(0.0, 1.0, -1.0)],
            "drag": [(1.0, 0.0, 0.0)]}


def fno_path(args):
    path = os.path.join(RESULTS, "pyqg", f"fno2d_{args.regime}_{args.var}_n{args.n}_s{args.seed}{args.fno_tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def train_fno(args, g):
    """exp45's recipe on pyqg's trajectories (args.data) of data seed 6000 + seed: one-step training with a two-step
    rollout term, Adam, cosine."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    u = make_data(args, g, args.fno_n, 6000 + args.seed).float()
    model = e45.FNO2d(args.fno_width, args.fno_modes, args.fno_layers)
    inc = float((u[:, 1:] - u[:, :-1]).std())
    model.norm.copy_(torch.tensor([float(u.mean()), float(u.std()), inc]))
    model = model.to(dev)
    x0, x1, x2 = (u[:, i:u.shape[1] - 2 + i].reshape(-1, g.n, g.n) for i in range(3))
    n = x0.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.fno_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.fno_epochs * (n // args.fno_batch))
    print(f"training FNO2d on {args.fno_n} trajectories ({n} pairs), width {args.fno_width}, modes {args.fno_modes}, "
          f"{args.fno_layers} layers, {args.fno_epochs} epochs, {dev}", flush=True)
    t0 = time.time()
    for ep in range(args.fno_epochs):
        perm, tot = torch.randperm(n), 0.0
        for b in range(n // args.fno_batch):
            i = perm[b * args.fno_batch:(b + 1) * args.fno_batch]
            a, y1, y2 = x0[i].to(dev), x1[i].to(dev), x2[i].to(dev)
            p1 = model(a)
            p2 = model(p1)
            loss = (((p1 - y1) ** 2).mean() + ((p2 - y2) ** 2).mean()) / inc ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % max(1, args.fno_epochs // 20) == 0 or ep == args.fno_epochs - 1:
            print(f"  epoch {ep}: loss {tot / max(1, n // args.fno_batch):.3e} ({time.time() - t0:.0f} s)", flush=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=sd, cfg=dict(width=args.fno_width, modes=args.fno_modes, layers=args.fno_layers)),
               fno_path(args))
    print(f"saved {fno_path(args)}", flush=True)


def load_fno(args, g):
    path = fno_path(args)
    if not os.path.exists(path):
        if not args.train_fno_if_missing:
            raise SystemExit(f"no FNO at {path}; pass --train-fno-if-missing")
        train_fno(args, g)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = e45.FNO2d(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


def fins(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    t0 = time.time()
    dtr = e45.Data(g, make_data(args, g, args.n_train, dseed), args.dt)
    dev = e45.Data(g, make_data(args, g, args.n_eval, eseed), args.dt, dtr.s)
    print(f"[{args.regime} (beta {g.beta}, rd {g.rd}, rek {g.rek}), {args.var}, {args.data} data] seed {s}: "
          f"{args.n_train} training trajectories (data seed {dseed}), {args.n_eval} held-out ({eseed}); data "
          f"{time.time() - t0:.0f} s", flush=True)
    out = dict(config=vars(args), data_seed=dseed, eval_seed=eseed)
    if args.op == "exact":
        op, d_ev = qg2d.ExactOp(g, args.dt, args.nsub, args.var), dtr
    else:
        model = load_fno(args, g.cast(torch.float32))
        op64 = e45.FNOOp(copy.deepcopy(model).double())
        for name, dd in (("training", dtr), ("held-out", dev)):
            e1, eK, per = e45.operator_errors(op64.model, dd, args.K)
            out[f"fno_error_{name}"] = [e1, eK]
            out[f"fno_error_{name}_per_trajectory"] = per.tolist()
            print(f"  FNO rollout error relative to the data increment ({name}): 1 step {e1:.4f} (per trajectory "
                  f"{per.min():.4f}-{per.max():.4f}), K steps {eK:.4f}", flush=True)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        op, d_ev = (op64, dtr) if device == "cpu" else (op64.cast(torch.float32, device),
                                                        dtr.cast(torch.float32, device))
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = e45.pencil(op, d_ev, e45.chars_poly, ptr, args.K, 2)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N, D)
    names, ents = qg2d.algebra(args.regime, args.var)
    Ct = qg2d.coefficients(ents)
    k = Ct.shape[1]
    cand = candidates(args.var)
    rq = {n: quotient(N, D, c) for n, c in cand.items()}
    lf6 = np.maximum(lam[:17], 1e-6)
    own = dict(floor12=e60.own_count(lam, k), floor6=int(np.argmax(lf6[1:] / lf6[:-1])) + 1)
    pr = plane_readouts(N, D, args.var)
    print(f"  polynomial class ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change over K {np.sqrt(cmu):.2e}")
    print("    lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:12]))
    print(f"    algebra ({k}: {', '.join(names)}); own count {own}")
    print("    quotients: " + "  ".join(f"{n} {v:.1e}" + ("*" if n in names else "") for n, v in rq.items()))
    print(f"    rotation plane: eigenvalues {pr['rot_lam'][0]:.1e}, {pr['rot_lam'][1]:.1e}; stretch {pr['rot_e']}")
    print("    scaling space: " + "; ".join(f"{l:.1e} (" + ", ".join(f"{x:+.3f}" for x in v) + ")"
                                         for l, v in zip(pr["scal_lam"], pr["scal_vec"])), flush=True)
    probe = e45.Probe(dev)
    true_q = probe.poly(Ct)[1]
    spans = {}
    for name, m in (("span_k", k), ("span_own6", own["floor6"])):
        Qc = probe.poly(C[:, :m])[1]
        Qc = np.where(np.isfinite(Qc), Qc, 0.0)
        spans[name] = dict(m=int(m), missing=e45.missing(Qc, true_q), novel=e45.novel(Qc, true_q).tolist())
        print(f"    span of the {m} lowest ({name}): missing {spans[name]['missing']:.3f}; novel "
              + " ".join(f"{n} {v:.2f}" for n, v in zip(names, spans[name]["novel"])), flush=True)
    out.update(lam=lam[:20].tolist(), names=names, quotients=rq, own=own, cmu=cmu, planes=pr, **spans)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", C=C[:, :20], lam=lam, N=N, D=D)
        print(f"  saved {stem}.json/.npz")


ROWS = [("q", r, 128) for r in ("f", "beta", "rd", "drag")] + [("psi", "f", 128), ("psi", "rd", 128), ("q", "f", 64)]


def stem_of(runs, var, regime, n, s, op="exact"):
    tag = "fins" if op == "exact" else "fins-fno"
    return os.path.join(runs, f"{tag}_{regime}_{var}" + ("" if n == 128 else f"_n{n}") + f"_s{s}")


def jobs(args):
    """(output, command) of the registered CPU runs: q in every regime, psi (pyqg's gauge) on the f-plane and with
    rd, q on the f-plane at 64^2; pyqg's data, n = 128 unless stated."""
    out = []
    for var, regime, n in ROWS:
        for s in args.seeds.split(","):
            stem = stem_of(args.runs, var, regime, n, s)
            out.append((stem + ".json", ["exp61_pyqg.py", "fins", "--regime", regime, "--var", var, "--data", "pyqg",
                                         "--n", str(n), "--every", "8", "--seed", s, "--threads", "1",
                                         "--save", stem]))
    return out


def summary(args):
    """results/pyqg/summary.{json,md}: per row (variable, regime, grid, operator) and seed, the readouts."""
    res, lines = {}, ["# Quasi-geostrophic flow, pyqg (exp61): counts over seeds\n"]
    keys = ("names", "quotients", "own", "planes", "span_k", "cmu", "lam", "fno_error_training")
    rows = [r + ("exact",) for r in ROWS] + [("q", "f", 128, "fno"), ("q", "beta", 128, "fno")]
    lines.append("| row | n | own count = k | missing <= 0.05 | max kept quotient (median) | min broken quotient "
                 "(median) | rotation plane (median) | scaling eigenvalues <= 1e-4 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for var, regime, n, op in rows:
        lab = f"{op} {var} {regime}" + ("" if n == 128 else f" n{n}")
        cell = {}
        for s in args.seeds.split(","):
            p = stem_of(args.runs, var, regime, n, s, op) + ".json"
            if os.path.exists(p):
                j = json.load(open(p))
                cell[s] = {k: j.get(k) for k in keys}
        if not cell:
            continue
        res[lab] = cell
        kept = KEPT[regime] if var == "q" else None
        k_ok = sum(len(r["names"]) == r["own"]["floor6"] for r in cell.values())
        m_ok = sum(r["span_k"]["missing"] <= 0.05 for r in cell.values())
        mk = [max(r["quotients"][c] for c in kept) for r in cell.values()] if kept else []
        mb = [min(r["quotients"][c] for c in CANDIDATES if c not in kept) for r in cell.values()] \
            if kept and len(kept) < len(CANDIDATES) else []
        rot = [r["planes"]["rot_lam"][0] for r in cell.values()]
        sc = collections.Counter(sum(v <= 1e-4 for v in r["planes"]["scal_lam"]) for r in cell.values())
        f = (lambda v: f"{np.median(v):.1e}" if v else "-")
        lines.append(f"| {lab} | {len(cell)} | {k_ok} | {m_ok} | {f(mk)} | {f(mb)} | {f(rot)} | {dict(sc)} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pyqg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "fins", "run", "summary"])
    ap.add_argument("--op", default="exact", choices=["exact", "fno"])
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
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "pyqg_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--regime", default="f", choices=list(qg2d.REGIMES))
    ap.add_argument("--var", default="q", choices=["q", "psi", "psi_far"])
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
        {"check": check, "fins": fins, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
