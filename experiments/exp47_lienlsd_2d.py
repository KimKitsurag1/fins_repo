"""Experiment 47 -- LieNLSD (Hu et al., ICML 2025) on the 2D equation of exp45, u_t = div(u grad u).

LieNLSD as its repository runs its 2D tasks: an MLP F fitted to u_t as a function of (u, u_x, u_y, u_xx, u_yy, u_xy)
on jets of the training trajectories, F's Jacobian in the determining system. Point fields of FINS's class (monomials
of degree <= 2 in X = x / 8, Y = y / 8, T = t / T, U = u / sigma in each of xi, eta, tau, phi: 60 fields, exp45's
layout); the m = 6 smallest right singular vectors (and the own count, below 0.01 of the largest) scored on exp45's
held-out probe like FINS's polynomial class. Jets: central differences (as published) or FINS's derivatives (spectral
in x, y; fourth order in t). F: fitted, at initialisation, or the exact right-hand side.

  python exp47_lienlsd_2d.py --seeds 42,43,44 --out results/neural2d/lienlsd_2d.json
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp45_neural_2d as e                                            # noqa: E402
from method_lienlsd import Adan, RHSNet                                # noqa: E402

COLS = ("U", "UX", "UY", "UXX", "UYY", "UXY")
_T, _X, _Y, _U, _UX, _UY, _UXX, _UYY, _UXY, _UT = sp.symbols("T X Y U UX UY UXX UYY UXY UT")


def library():
    """Callables for theta, D_X, D_Y, D_X^2, D_Y^2, D_X D_Y, D_T of exp45's monomials (MONS: powers of X, Y, T, U)."""
    th = [_X ** a * _Y ** b * _T ** c * _U ** d for a, b, c, d in e.MONS]

    def dx(f):
        return sp.diff(f, _X) + sp.diff(f, _U) * _UX

    def dy(f):
        return sp.diff(f, _Y) + sp.diff(f, _U) * _UY

    ops = [th, [dx(f) for f in th], [dy(f) for f in th],
           [sp.diff(f, _X, 2) + 2 * _UX * sp.diff(f, _X, _U) + _UX ** 2 * sp.diff(f, _U, 2) + _UXX * sp.diff(f, _U) for f in th],
           [sp.diff(f, _Y, 2) + 2 * _UY * sp.diff(f, _Y, _U) + _UY ** 2 * sp.diff(f, _U, 2) + _UYY * sp.diff(f, _U) for f in th],
           [sp.diff(f, _X, _Y) + _UY * sp.diff(f, _X, _U) + _UX * sp.diff(f, _Y, _U) + _UX * _UY * sp.diff(f, _U, 2)
            + _UXY * sp.diff(f, _U) for f in th],
           [sp.diff(f, _T) + _UT * sp.diff(f, _U) for f in th]]
    args = (_T, _X, _Y, _U, _UX, _UY, _UXX, _UYY, _UXY, _UT)
    return [sp.lambdify(args, sp.Matrix(v), "numpy") for v in ops]


def jets(u, dx, dt, kind):
    """Scaled-free jets of u [B, nt, n, n] (x along axis -2, y along -1) on snapshots 2 .. nt - 3.
    kind 'fd': central differences, second order in t (LieNLSD); 'spectral': FINS's derivatives."""
    if kind == "fd":
        def d1(f, ax):
            return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * dx)

        def d2(f, ax):
            return (np.roll(f, -1, ax) - 2 * f + np.roll(f, 1, ax)) / dx ** 2

        def dxy(f):
            return d1(d1(f, -2), -1)
        ut = (u[:, 3:-1] - u[:, 1:-3]) / (2 * dt)
        dX, dY, dXX, dYY, dXY = (lambda f: d1(f, -2)), (lambda f: d1(f, -1)), (lambda f: d2(f, -2)), (lambda f: d2(f, -1)), dxy
    else:
        n = u.shape[-1]
        k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        KX, KY = k[:, None], k[None, :]

        def sd(f, px, py):
            return np.real(np.fft.ifft2((1j * KX) ** px * (1j * KY) ** py * np.fft.fft2(f, axes=(-2, -1)), axes=(-2, -1)))
        ut = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)
        dX, dY, dXX, dYY, dXY = (lambda f: sd(f, 1, 0)), (lambda f: sd(f, 0, 1)), (lambda f: sd(f, 2, 0)), \
            (lambda f: sd(f, 0, 2)), (lambda f: sd(f, 1, 1))
    v = u[:, 2:-2]
    return dict(u=v, ux=dX(v), uy=dY(v), uxx=dXX(v), uyy=dYY(v), uxy=dXY(v), ut=ut, utx=dX(ut), uty=dY(ut))


def scaled(J, s):
    """Jets in X = x / Lx, T = t / T, U = u / sigma."""
    L, T, sg = s["Lx"], s["T"], s["sigma"]
    return dict(U=J["u"] / sg, UX=J["ux"] * L / sg, UY=J["uy"] * L / sg, UXX=J["uxx"] * L ** 2 / sg,
                UYY=J["uyy"] * L ** 2 / sg, UXY=J["uxy"] * L ** 2 / sg, UT=J["ut"] * T / sg,
                UTX=J["utx"] * T * L / sg, UTY=J["uty"] * T * L / sg)


def exact_jacobian(P, kappa):
    U, UX, UY, UXX, UYY = (P[:, i] for i in range(5))
    return np.stack([kappa * (UXX + UYY), 2 * kappa * UX, 2 * kappa * UY, kappa * U, kappa * U, 0 * U], -1)


def exact_rhs(P, kappa):
    U, UX, UY, UXX, UYY = (P[:, i] for i in range(5))
    return kappa * (U * (UXX + UYY) + UX ** 2 + UY ** 2)


def block_points(S, sl, rng, n_points, trajs):
    """(trajectory, snapshot, i, j) drawn uniformly over the given trajectories, all jet snapshots and the scored block."""
    nt = S["U"].shape[1]
    r = rng.choice(trajs, n_points)
    return r, rng.integers(0, nt, n_points), rng.integers(sl.start, sl.stop, n_points), rng.integers(sl.start, sl.stop, n_points)


def gather(S, idx, keys):
    r, j, a, b = idx
    return np.stack([S[k][r, j, a, b] for k in keys], -1)


def fit(S, sl, epochs, seed, n_fit, n_val, batch=256, lr=1e-3, every=10):
    """F: scaled jets -> U_T, official configuration (MLP 200 x 3 sigmoid, Adan, MSE, batch 256); checkpoint every 10
    epochs chosen on the last two training trajectories. Returns (net, log)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    B = S["U"].shape[0]
    tr, va = block_points(S, sl, rng, n_fit, np.arange(B - 2)), block_points(S, sl, rng, n_val, np.arange(B - 2, B))
    Xtr, ytr, Xva, yva = gather(S, tr, COLS), S["UT"][tr], gather(S, va, COLS), S["UT"][va]
    net = RHSNet(Xtr.mean(0), Xtr.std(0) + 1e-12, ytr.mean(), ytr.std() + 1e-12)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)[:, None]
    opt = Adan(net.parameters(), lr) if Adan is not None else torch.optim.Adam(net.parameters(), lr)

    def val():
        with torch.no_grad():
            return float(((net(Xv) - yv) ** 2).mean() / (yv ** 2).mean())

    log = dict(optimizer=type(opt).__name__, n_train=n_fit, n_val=n_val, epochs=epochs, val=[(0, val())])
    best, state = log["val"][0][1], copy.deepcopy(net.state_dict())
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), batch):
            i = perm[b:b + batch]
            loss = tf.mse_loss(net(Xt[i]), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % every == 0 or ep + 1 == epochs:
            v = val()
            log["val"].append((ep + 1, v))
            if v < best:
                best, state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.eval()
    log["best_epoch"] = min(log["val"], key=lambda x: x[1])[0]
    log["val_rel_mse"] = best
    return net, log


def net_jacobian(net, P):
    z = torch.tensor(P, dtype=torch.float32, requires_grad=True)
    g, = torch.autograd.grad(net(z).sum(), z)
    return g.detach().numpy().astype(np.float64)


def determining(S, pts, J, fns, tgrid):
    """Rows of the linearised determining system over the 60 fields, blocks (xi, eta, tau, phi) in exp45's order."""
    r, j, a, b = pts
    nm = len(e.MONS)
    rows = []
    for q in range(len(r)):
        v = {k: float(S[k][r[q], j[q], a[q], b[q]]) for k in S if k not in ("Xg", "Yg")}
        T, X, Y = tgrid[j[q]], S["Xg"][a[q], b[q]], S["Yg"][a[q], b[q]]
        th, dX, dY, dXX, dYY, dXY, dT = (np.asarray(f(T, X, Y, v["U"], v["UX"], v["UY"], v["UXX"], v["UYY"], v["UXY"], v["UT"]),
                                                    dtype=np.float64).ravel() * np.ones(nm) for f in fns)
        z = np.zeros(nm)
        blocks = []
        for c, w in ((0, "UX"), (1, "UY"), (2, "UT")):
            wx, wy = {"UX": ("UXX", "UXY"), "UY": ("UXY", "UYY"), "UT": ("UTX", "UTY")}[w]
            blocks.append(dict(P0=z, PX=-v[w] * dX, PY=-v[w] * dY, PXX=-(v[w] * dXX + 2 * v[wx] * dX),
                               PYY=-(v[w] * dYY + 2 * v[wy] * dY), PXY=-(v[w] * dXY + v[wx] * dY + v[wy] * dX),
                               PT=-v[w] * dT))
        blocks.append(dict(P0=th, PX=dX, PY=dY, PXX=dXX, PYY=dYY, PXY=dXY, PT=dT))
        P = {k: np.concatenate([bl[k] for bl in blocks]) for k in blocks[0]}
        jr = J[q]
        rows.append(P["PT"] - jr[0] * P["P0"] - jr[1] * P["PX"] - jr[2] * P["PY"] - jr[3] * P["PXX"]
                    - jr[4] * P["PYY"] - jr[5] * P["PXY"])
    return np.array(rows)


def readout(W, m, threshold=0.01):
    sc = np.sqrt((W ** 2).mean(0))
    sc = np.where(sc > 1e-12, sc, 1.0)
    _, S, Vh = np.linalg.svd(W / sc, full_matrices=False)
    C = np.stack([Vh[-(i + 1)] / sc for i in range(m)], 1)
    n_own = max(1, min(int(np.sum(S / S[0] < threshold)), m))
    return C, n_own, (S[::-1] / S[0])[:m + 4].tolist()


def run_seed(s, args, fns):
    g = e.Grid()
    dtr = e.Data(g, e.trajectories(g, 16, 4000 + s, 51, 0.02, 3), 0.02)
    dev = e.Data(g, e.trajectories(g, 8, 5000 + s, 51, 0.02, 3), 0.02, dtr.s)
    probe = e.Probe(dev)
    sc = dtr.s
    kappa = sc["sigma"] * sc["T"] / sc["Lx"] ** 2
    tgrid = dtr.t.numpy()[2:-2] / sc["T"]
    Xg, Yg = (g.X / sc["Lx"]).numpy(), (g.Y / sc["Lx"]).numpy()
    out = {}
    for kind in ("fd", "spectral"):
        S = scaled(jets(dtr.u.numpy(), g.dx, 0.02, kind), sc)
        Se = scaled(jets(dev.u.numpy(), g.dx, 0.02, kind), sc)
        S["Xg"], S["Yg"] = Xg, Yg
        rng = np.random.default_rng(0)
        pts = block_points(S, g.sl, rng, args.n_points * args.n_traj, np.arange(args.n_traj))
        P = gather(S, pts, COLS)
        Jex = exact_jacobian(P, kappa)
        Pe = np.stack([Se[k][:, :, g.sl, g.sl].ravel() for k in COLS], -1)
        ute = Se["UT"][:, :, g.sl, g.sl].ravel()
        floor = float(np.linalg.norm(exact_rhs(Pe, kappa) - ute) / np.linalg.norm(ute))
        for variant in ("exact", "fitted", "untrained"):
            t0 = time.time()
            diag = dict(rhs_rel_exact=floor)
            if variant == "exact":
                J = Jex
            else:
                net, diag["fit"] = fit(S, g.sl, args.epochs if variant == "fitted" else 0, s, args.n_fit, args.n_val)
                J = net_jacobian(net, P)
                with torch.no_grad():
                    fe = net(torch.tensor(Pe, dtype=torch.float32))[:, 0].numpy()
                diag["rhs_rel_fit"] = float(np.linalg.norm(fe - ute) / np.linalg.norm(ute))
            diag["jac_rel"] = float(np.linalg.norm(J - Jex) / np.linalg.norm(Jex))
            W = determining(S, pts, J, fns, tgrid)
            C, n_own, sv = readout(W, 6)
            res = dict(m6=probe.scores(*probe.poly(C)), own=probe.scores(*probe.poly(C[:, :n_own])), n_own=n_own,
                       sv_rel=sv, diag=diag, time=time.time() - t0)
            out[f"{kind}/{variant}"] = res
            print(f"[s{s}] {kind:8s} {variant:9s} m=6 missing_q {res['m6']['missing_q']:.4f} missing_f "
                  f"{res['m6']['missing_f']:.4f}; own count {n_own}: missing_q {res['own']['missing_q']:.4f}; "
                  f"novel {' '.join(f'{x:.2f}' for x in res['m6']['novel_q'])}; jac {diag['jac_rel']:.3f}; "
                  f"rhs fit {diag.get('rhs_rel_fit', float('nan')):.4f} exact {floor:.4f}; "
                  + (f"F val {diag['fit']['val_rel_mse']:.2e} at epoch {diag['fit']['best_epoch']}; " if "fit" in diag else "")
                  + f"sv {' '.join(f'{x:.1e}' for x in sv[:8])} ({res['time']:.0f} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-fit", type=int, default=2 ** 17)
    ap.add_argument("--n-val", type=int, default=2 ** 15)
    ap.add_argument("--n-traj", type=int, default=10)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    fns = library()
    report = dict(config=vars(args))
    for s in (int(x) for x in args.seeds.split(",")):
        report[s] = run_seed(s, args, fns)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    if args.out:
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
