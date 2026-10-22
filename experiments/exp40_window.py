"""Experiment 40 -- discovery through a boundary-free window operator.

Periodic trajectories are read as solutions on the line: a window spans
several periods with a continuous coordinate, so no characteristic jumps
anywhere (projective's u-independent x / 2 pi included). u_x is a
fourth-order difference inside the window; the window stepper's K-step
output lives on the central W - 2rK points, where every step is scored.
Same mu-normalised eigenproblem as exp34; ground truth scores only.

--noise / --thin (exp48): every --thin-th snapshot and Gaussian noise of --noise times each trajectory's standard
deviation (corrupt.py), on the evaluation data and on a trained operator's training data; u_t by fourth-order
differences of the corrupted snapshots; K and --every count kept snapshots; the heat kernel is composed --thin times
(still exact). Saved u: the clean evaluation data.

Run:  python exp40_window.py --pde burgers --train-if-missing --epochs 400
"""

import argparse
import itertools
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp32_blind_readout as e32                          # noqa: E402
import exp34_normalization as e34                          # noqa: E402
import corrupt as cr                                       # noqa: E402
from window_stepper import (GlobalStepper, WindowStepper, to_float64,  # noqa: E402
                            train_global_stepper, train_window_stepper)


def moment_kernel(sigma2, r):
    """Symmetric (2r+1)-tap kernel whose even moments 0..2r match the heat step
    exp(-sigma2 theta^2) (Gaussian of variance 2 sigma2): exact to O(theta^(2r+2))."""
    n = np.arange(r + 1)
    V = np.array([[(1.0 if nn == 0 else 0.0) if i == 0 else 2.0 * i ** (2 * nn) for i in range(r + 1)]
                  for nn in n])
    dfact = np.array([np.prod(np.arange(2 * nn - 1, 0, -2)) if nn else 1.0 for nn in n])
    w = np.linalg.solve(V, dfact * (2 * sigma2) ** n)
    return np.concatenate([w[:0:-1], w])


def truncated_kernel(sigma2, r, Nx):
    m = np.fft.fftfreq(Nx, d=1.0 / Nx)
    h = np.real(np.fft.ifft(np.exp(-sigma2 * (2 * np.pi * m / Nx) ** 2)))
    return np.concatenate([h[-r:], h[:r + 1]])


class HeatKernelWindow(torch.nn.Module):
    """Exact-by-construction heat step on the line as a valid convolution."""

    def __init__(self, ker):
        super().__init__()
        self.register_buffer("w", torch.tensor(np.asarray(ker)[::-1].copy())[None, None])
        self.r = (len(ker) - 1) // 2

    def forward(self, u):
        return torch.nn.functional.conv1d(u, self.w)


def trajectories(pde, cfg, n, seed, smooth=False, only_L=None):
    """xs [n, Nx], ts [n, Nt], u [n, Nx, Nt], u_t. Heat: smooth = benchmark initial data low-passed
    and evolved exactly; only_L = keep one domain length (the benchmark cycles five, so 5 n are drawn)."""
    if pde == "heat" and (smooth or only_L):
        n_gen = 5 * n if only_L else n
        ics, sols, xs_t, ts_t = e31.generate_pde_data("heat", cfg, n_gen, seed=seed)
        xs, ts = xs_t[:, 0].numpy().astype(np.float64), ts_t[:, 0].numpy().astype(np.float64)
        if smooth:
            # benchmark initial data low-passed (no energy near Nyquist), evolved exactly
            Nx = xs.shape[1]
            m = np.fft.fftfreq(Nx, d=1.0 / Nx)
            c0 = np.fft.fft(ics[:, 0].numpy().astype(np.float64), axis=-1) * np.exp(-(m / (Nx / 6)) ** 8)
            k2 = (2 * np.pi * m[None] / e31.lengths(xs)[:, None]) ** 2
            u = np.real(np.fft.ifft(c0[:, :, None] * np.exp(-cfg.alpha_heat * k2[:, :, None] * ts[:, None, :]),
                                    axis=1))
        else:
            u = sols[:, 0].numpy().astype(np.float64)
        if only_L:
            sel = np.flatnonzero(np.isclose(e31.lengths(xs), only_L))[:n]
            if len(sel) < n:
                raise SystemExit(f"only {len(sel)} heat samples with L = {only_L}")
            xs, ts, u = xs[sel], ts[sel], u[sel]
        return xs, ts, u, e31.time_derivative(u, ts, "central4")
    if pde == "kdv":
        return e31.kdv_stencil(cfg, n, seed, 2e-5)
    if pde == "burgers":
        xs, ts, u = e31.burgers_coarse(cfg, n, seed)
    else:
        _, sols, xs_t, ts_t = e31.generate_pde_data(pde, cfg, n, seed=seed)
        xs, ts = xs_t[:, 0].numpy().astype(np.float64), ts_t[:, 0].numpy().astype(np.float64)
        u = sols[:, 0].numpy().astype(np.float64)
    return xs, ts, u, e31.time_derivative(u, ts, "central4")


def dx4(u, dx):
    """Fourth-order central difference along axis 1; second order at the two edge points."""
    o = np.empty_like(u)
    o[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / 12
    o[:, 1] = (u[:, 2] - u[:, 0]) / 2
    o[:, -2] = (u[:, -1] - u[:, -3]) / 2
    o[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / 2
    o[:, -1] = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / 2
    return o / dx[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers", choices=["burgers", "heat", "kdv"])
    ap.add_argument("--op", default="window", choices=["window", "cno", "fnopad", "heat-kernel"],
                    help="window: local valid-convolution stepper; cno / fnopad: same-size non-periodic "
                         "global steppers; heat-kernel: the exact heat map as a window operator")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--smooth", action="store_true",
                    help="heat: low-passed initial data, evolved exactly (evaluation, and training when a "
                         "same-size stepper is trained here)")
    ap.add_argument("--kernel-fit", default="moments", choices=["moments", "truncated"])
    ap.add_argument("--ux", default="fd", choices=["fd", "spectral"],
                    help="spectral: exact derivative of the periodic data (isolates the operator)")
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=0, help="blind readout of the m lowest (0: off)")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-if-missing", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--unroll", type=int, default=4, help="training rollout length")
    ap.add_argument("--score-margin", type=int, default=None,
                    help="same-size operators: points excluded at each window edge (default 6 K, the "
                         "window stepper's r = 6 cone, so all operators are scored on the same region)")
    ap.add_argument("--train-margin", type=int, default=6, help="same-size operators: training crop per step")
    ap.add_argument("--cno-layers", type=int, default=3)
    ap.add_argument("--cno-res", type=int, default=2)
    ap.add_argument("--cno-mult", type=int, default=16)
    ap.add_argument("--fno-modes", type=int, default=32)
    ap.add_argument("--fno-width", type=int, default=48)
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-pad", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=None, help="training learning rate (default per operator)")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="seed of the evaluation trajectories (default --seed: inside the training set)")
    ap.add_argument("--only-L", type=float, default=None,
                    help="heat: keep one domain length (training and evaluation data)")
    ap.add_argument("--save", default=None, help="npz with the eigen-generators and the evaluation data")
    ap.add_argument("--reliability-filter", type=float, default=None,
                    help="drop trajectories whose one-step error exceeds this multiple of the median")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise, times each trajectory's std (corrupt.py)")
    ap.add_argument("--thin", type=int, default=1, help="keep every thin-th snapshot (K, --every count kept ones)")
    ap.add_argument("--interior", action="store_true",
                    help="windows start at snapshot 2 and end by Nt - 3 (no one-sided u_t; exp48)")
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    arch = dict(width=args.width, layers=args.layers, kernel=args.kernel)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.op == "heat-kernel":
        xs0, ts0, _, _ = trajectories("heat", cfg, 1, args.seed)
        dx0, dt0 = e31.lengths(xs0)[0] / cfg.grid_size, ts0[0, 1] - ts0[0, 0]
        sig2 = cfg.alpha_heat * dt0 / dx0 ** 2
        ker = moment_kernel(sig2, 6) if args.kernel_fit == "moments" else truncated_kernel(sig2, 6, cfg.grid_size)
        th = np.linspace(0, np.pi, 2001)
        M = ker[6] + 2 * sum(ker[6 + i] * np.cos(i * th) for i in range(1, 7))
        band = th <= 2 * np.pi / 6
        tgt = np.exp(-sig2 * th ** 2)
        print(f"kernel ({args.kernel_fit}): sum {ker.sum():.12f}; multiplier error relative to the "
              f"increment over the data band {np.max(np.abs(M - tgt)[band][1:] / (1 - tgt[band][1:])):.1e}; "
              f"max |multiplier| on [0, pi] {np.abs(M).max():.3f}")
        base = ker
        for _ in range(args.thin - 1):           # one step between kept snapshots = thin original steps
            ker = np.convolve(ker, base)
        stp = HeatKernelWindow(ker)
    elif args.op in ("cno", "fnopad"):
        Wn = args.periods * cfg.grid_size
        garch = dict(cno_layers=args.cno_layers, cno_res=args.cno_res, cno_mult=args.cno_mult,
                     fno_modes=args.fno_modes, fno_width=args.fno_width, fno_layers=args.fno_layers,
                     fno_pad=args.fno_pad)
        name = args.stepper or (f"S_theta_{args.pde}_s{args.seed}_g{cfg.grid_size}x{cfg.time_steps}"
                                f"_{args.op}{args.tag}.pt")
        path = name if os.path.isabs(name) else os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), name)
        stp = GlobalStepper(args.op, Wn, **garch)
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            stp.load_state_dict({k: v for k, v in sd.items() if torch.is_tensor(v)})
        elif args.train_if_missing:
            n_tr = args.n_train or cfg.n_train
            tr = trajectories(args.pde, cfg, n_tr, args.seed, args.smooth, args.only_L)[2]
            tr = torch.tensor(cr.corrupt(tr, args.noise, args.thin, args.seed, cr.TRAIN)[1], dtype=torch.float32)[:, None]
            print(f"training {args.pde} {args.op} stepper on {n_tr} trajectories" + (" (smooth)" if args.smooth else "")
                  + f", seed {args.seed}, "
                  f"{args.epochs} epochs, window {Wn}, unroll {args.unroll}, {dev}", flush=True)
            # partial state every 10 epochs next to the checkpoint: an interrupted run resumes exactly
            trained = train_global_stepper(tr, dev, kind=args.op, size=Wn, epochs=args.epochs,
                                           unroll=args.unroll, margin=args.train_margin, seed=args.seed,
                                           lr=args.lr or 1e-3, ckpt=path + ".partial", **garch)
            sd = {k: v.cpu() for k, v in trained.state_dict().items() if torch.is_tensor(v)}
            stp.load_state_dict(sd)
            torch.save(sd, path)
            if os.path.exists(path + ".partial"):
                os.remove(path + ".partial")
            print(f"saved {path}", flush=True)
        else:
            raise SystemExit(f"no {args.op} stepper at {path}; pass --train-if-missing")
    else:
        name = args.stepper or (f"S_theta_{args.pde}_s{args.seed}_g{cfg.grid_size}x{cfg.time_steps}"
                                f"_window{args.tag}.pt")
        path = name if os.path.isabs(name) else os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), name)
        stp = WindowStepper(**arch)
        if os.path.exists(path):
            stp.load_state_dict(torch.load(path, map_location="cpu"))
        elif args.train_if_missing:
            n_tr = args.n_train or cfg.n_train
            tr = trajectories(args.pde, cfg, n_tr, args.seed)[2]
            tr = torch.tensor(cr.corrupt(tr, args.noise, args.thin, args.seed, cr.TRAIN)[1], dtype=torch.float32)[:, None]
            print(f"training {args.pde} window stepper on {n_tr} trajectories, seed {args.seed}, "
                  f"{args.epochs} epochs, radius {stp.r}, unroll {args.unroll}, {dev}", flush=True)
            trained = train_window_stepper(tr, dev, epochs=args.epochs, seed=args.seed,
                                           unroll=args.unroll, **arch)
            stp.load_state_dict({k: v.cpu() for k, v in trained.state_dict().items()})
            torch.save(stp.state_dict(), path)
            print(f"saved {path}", flush=True)
        else:
            raise SystemExit(f"no window stepper at {path}; pass --train-if-missing")
    stp = to_float64(stp).eval()
    shrink = stp.r * args.K if stp.r > 0 else (args.score_margin or 6 * args.K)
    for p in stp.parameters():
        p.requires_grad_(False)

    dseed = args.seed if args.data_seed is None else args.data_seed
    xs, ts, u, ut = trajectories(args.pde, cfg, args.n, dseed, args.smooth, args.only_L)
    u_clean, u = cr.corrupt(u, args.noise, args.thin, dseed, cr.EVAL)
    ts = ts[:, ::args.thin]
    if cr.corrupted(args):
        ut = e31.time_derivative(u, ts, "central4")
    tm = max(1, 8 // args.thin)                 # scored times: the same physical margin as 8 of the original
    su, sut = e31.band_shares(u, ut)
    print(f"  band check, energy above k = Nx/4: u {su:.1e}, u_t {sut:.1e}"
          + ("   FLAG: u_t lives where the operator has seen little" if sut > 1e-2 else ""))
    # equation-free reliability: one-step error on the scored region of each trajectory
    Nx0 = u.shape[1]
    W0 = args.periods * Nx0
    idx0 = np.arange(W0) % Nx0
    with torch.no_grad():
        errs = []
        for r in range(u.shape[0]):
            cur = torch.tensor(u[r][idx0].T[:-1])[:, None]
            pred = stp(cur)[:, 0].numpy()[:, shrink - stp.r:W0 - shrink - stp.r]
            tgt = u[r][idx0].T[1:, shrink:W0 - shrink]
            inc = tgt - u[r][idx0].T[:-1, shrink:W0 - shrink]
            errs.append(float(np.linalg.norm(pred - tgt) / np.linalg.norm(inc)))
    errs = np.array(errs)
    print("  per-trajectory one-step error: " + " ".join(f"{e:.3f}" for e in errs))
    if args.reliability_filter:
        keep_r = errs <= args.reliability_filter * np.median(errs)
        xs, ts, u, ut, u_clean = xs[keep_r], ts[keep_r], u[keep_r], ut[keep_r], u_clean[keep_r]
        print(f"  reliability filter: kept {int(keep_r.sum())} of {len(errs)} (<= {args.reliability_filter} x median)")
    N, Nx, Nt = u.shape
    W = args.periods * Nx
    if W - 2 * shrink < Nx // 2:
        raise SystemExit(f"window {W} too narrow for radius {stp.r} x K {args.K}")
    idx = np.arange(W) % Nx
    dx = e31.lengths(xs) / Nx
    xw = dx[:, None] * np.arange(W)[None]
    uw, utw = u[:, idx], ut[:, idx]
    if args.ux == "spectral":
        k = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)[None, :, None] / dx[:, None, None]
        uxw = np.real(np.fft.ifft(1j * k * np.fft.fft(u, axis=1), axis=1))[:, idx]
    else:
        uxw = dx4(uw, dx)

    Lw, T = (W * dx).max(), ts[:, -1].max()
    X = (xw[:, :, None] - xw[:, W // 2:W // 2 + 1, None]) / Lw
    Tn = ts[:, None, :] / T
    Un = uw / uw.std()
    mons = [mm for mm in itertools.product(range(d + 1), repeat=3) if sum(mm) <= d]
    F, Qb = [], []
    for comp in range(3):
        for a, b, c in mons:
            f = np.broadcast_to(X ** a * Tn ** b, uw.shape) * Un ** c
            fld = np.zeros((3,) + uw.shape)
            fld[comp] = f
            F.append(fld)
            Qb.append(-f * uxw if comp == 0 else (-f * utw if comp == 1 else f))
    F, Qb = np.stack(F), np.stack(Qb)

    # --interior: every characteristic of a window on times 2 .. Nt - 3, where u_t is the central five-point difference
    starts = list(range(2, Nt - args.K - 2, args.every) if args.interior else range(0, Nt - args.K, args.every))
    reg = slice(shrink, W - shrink)
    u0 = torch.tensor(np.concatenate([uw[:, :, j] for j in starts]))[:, None]
    # ground-truth free: the operator's rollout error on the scored region, 1 and K steps,
    # for u and for its window derivative (what d_x's residual differentiates)
    with torch.no_grad():
        s_ = u0
        for kk in range(1, args.K + 1):
            s_ = stp(s_)
            if kk in (1, args.K):
                off = shrink - kk * stp.r
                pred = s_[:, 0, off:off + (W - 2 * shrink)].numpy()
                tgt = np.concatenate([uw[:, reg, j + kk] for j in starts])
                cur = np.concatenate([uw[:, reg, j] for j in starts])
                dxr = np.tile(dx, len(starts))
                e_u = np.linalg.norm(pred - tgt) / np.linalg.norm(tgt - cur)
                e_d = (np.linalg.norm(dx4(pred[:, :, None], dxr)[:, 2:-2] - dx4(tgt[:, :, None], dxr)[:, 2:-2])
                       / np.linalg.norm(dx4(tgt[:, :, None], dxr)[:, 2:-2] - dx4(cur[:, :, None], dxr)[:, 2:-2]))
                print(f"  rollout error relative to the data increment after {kk} step(s): u {e_u:.3f}, u_x {e_d:.3f}")
    # ground-truth free: response in the scored region to a unit impulse at the window's first point,
    # pushed through the operator's Jacobian (zero for an operator whose influence cone excludes it)
    imp = torch.zeros_like(u0)
    imp[:, 0, 0] = 1.0
    s_, t_ = u0, imp
    leak = 0.0
    for kk in range(1, args.K + 1):
        s_, t_ = jvp(stp, (s_,), (t_,))
        off = shrink - kk * stp.r
        leak = max(leak, float(t_[:, 0, off:off + (W - 2 * shrink)].pow(2).mean(1).sqrt().mean()))
    print(f"  edge leak: rms response in the scored region to a unit impulse at the window edge, "
          f"max over {args.K} steps: {leak:.1e}")
    A, B, Q0 = [], [], []
    for p in range(Qb.shape[0]):
        q0 = np.concatenate([Qb[p][:, :, j] for j in starts])
        s, t = u0, torch.tensor(q0)[:, None]
        a_, b_ = [], []
        for k in range(1, args.K + 1):
            s, t = jvp(stp, (s,), (t,))
            off = shrink - k * stp.r
            tk = t[:, 0, off:off + (W - 2 * shrink)].numpy()
            a_.append(tk - q0[:, reg])
            b_.append(np.concatenate([Qb[p][:, reg, j + k] - Qb[p][:, reg, j] for j in starts]))
        A.append(np.stack(a_, -1).ravel())
        B.append(np.stack(b_, -1).ravel())
        Q0.append(q0[:, reg].ravel())
    A, B, Q0 = np.stack(A, 1), np.stack(B, 1), np.stack(Q0, 1)
    Nr = (A - B).T @ (A - B)
    G0 = args.K * Q0.T @ Q0
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    lam, C = e34.solve(Nr, Dr)
    # ground-truth free: how far the characteristics move over the horizon, relative to their size
    rel_change = float(np.sqrt(np.trace(B.T @ B) / np.trace(G0)))

    # the class measures x from each window's centre; the ground truth must too
    names, gt = e31.algebra(args.pde, xw - xw[:, W // 2:W // 2 + 1], ts, uw, cfg.alpha_heat, d)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T, np.stack([g.ravel() for g in gt]).T, rcond=None)
    rq = [float(c @ Nr @ c / (c @ Dr @ c)) for c in cg.T]
    k = len(names)
    sel = (slice(None), slice(None), reg, slice(tm, -tm))
    gtv = np.stack([g[sel].ravel() for g in gt])
    Fv = F[(slice(None),) + sel].reshape(F.shape[0], -1)
    Qe, _ = np.linalg.qr((C[:, :k].T @ Fv).T)
    cont = [np.linalg.norm(Qe.T @ g) / np.linalg.norm(g) for g in gtv]
    print(f"[{args.pde}, {args.op} operator r = {stp.r}] seed {args.seed}, data seed {dseed}, {N} trajectories, window "
          f"{args.periods} periods ({W} points), scored on the central {W - 2 * shrink}, degree {d}, "
          f"K = {args.K}, mu_rel = {args.mu}, u_x {args.ux}, noise {args.noise:g}, thin {args.thin}")
    print(f"  relative change of the basis characteristics, rms over k = 1..K steps: {rel_change:.1e}")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:k + 3]))
    print("  algebra losses: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq)) + f"   (lambda{k + 1} = {lam[k]:.1e})")
    print(f"  algebra inside the {k} lowest: " + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))
    if args.save:
        # eigen-generator j = sum_i C[i, j] * monomial_i in component i // len(mons), monomials in
        # X = (x - x_offset) / Lw, T = t / T_max, U = u / sigma, x the torus coordinate of xs
        np.savez(args.save, pde=args.pde, method=f"FINS-{args.op}", seed=args.seed, data_seed=dseed,
                 C=C[:, :k + 3], lam=lam[:k + 6], mons=np.array(mons), degree=d, Lw=Lw, T_max=T,
                 sigma=uw.std(), x_offset=dx * ((W // 2) % Nx), u=u_clean, xs=xs, ts=ts, names=np.array(names),
                 losses=np.array(rq), containment=np.array(cont), rel_change=rel_change,
                 band=np.array([su, sut]), K=args.K, k=k, noise=args.noise, thin=args.thin, interior=args.interior)
        print(f"  saved {args.save}")
    if args.m:
        rms = np.sqrt((F.reshape(F.shape[0], -1) ** 2).mean(1))
        Rot = e32.varimax(np.linalg.qr(C[:, :args.m] * rms[:, None])[0])
        scale = (Lw, T, uw.std())
        for i in range(args.m):
            w = Rot[:, i]
            c = w / rms
            big = np.abs(w).max()
            terms = []
            for j in np.argsort(-np.abs(w)):
                if abs(w[j]) < 0.05 * big:
                    break
                comp, (a, b, cc) = j // len(mons), mons[j % len(mons)]
                terms.append((comp, c[j] / (scale[0] ** a * scale[1] ** b * scale[2] ** cc),
                              e32.monomial_label(a, b, cc)))
            lead = terms[0][1]
            loss = float(c @ Nr @ c / (c @ Dr @ c))
            rest = float(np.sqrt(sum(w[q] ** 2 for q in range(len(w)) if abs(w[q]) < 0.05 * big)) / np.linalg.norm(w))
            form = ";  ".join(f"{('xi', 'eta', 'phi')[q]} = " + " ".join(f"{r / lead:+.3g} {lab}" for cmp, r, lab in terms if cmp == q)
                              for q in range(3) if any(t_[0] == q for t_ in terms))
            print(f"  v{i + 1}  loss {loss:.1e}  {len(terms)} terms (rest {rest:.2f}):  {form}   (x from the window centre)")


if __name__ == "__main__":
    main()
