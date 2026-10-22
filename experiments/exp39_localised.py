"""Experiment 39 -- fields localised away from the torus seam.

For transport and dispersion no interior band excludes the seam (exp38).
If the field is constant near the seam, characteristic terms proportional to
u_x or u_t have no jump there and terms proportional to u only a constant
one; u-independent x-dependent terms (Burgers projective's phi = x / 2 pi)
still jump. Localised and broadband initial data are compared under the same
equation parameters, grid and horizon, through the exact operator or a
learned torus stepper.

Burgers: u_t + 2 pi u u_x = 4 pi^2 nu u_xx on the unit torus (the benchmark
solver's scaling). KdV: u_t + 6 u u_x + u_xxx = 0 on [0, 2 pi); localised data
are well-separated sech^2 solitons (no radiation), broadband data random
sech^2 humps.

--op stepper: a torus FNO stepper (no positional embedding, shift-equivariant)
trained on trajectories from its own seed. KdV training data (--train-ic):
one or two solitons (c in [10, 16]) in the central band on a zero or constant
background, or half of those and half broadband humps; evaluation data: the
--ic generator at --data-seed. Blocks by
forward-mode JVP in float64.

--noise / --thin (exp48): every --thin-th snapshot and Gaussian noise of --noise times each trajectory's standard
deviation (corrupt.py), on the evaluation data and on a stepper's training data; K and --every count kept snapshots,
the exact solver integrates on the original grid. --ut fd: u_t by fourth-order differences of the (corrupted)
snapshots instead of the aligned solver stencil (required for corrupted data). Saved u: the clean evaluation data.

Run:  python exp39_localised.py --pde kdv --ic localised --op stepper --train-if-missing --data-seed 1042
"""

import argparse
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
import bo                                                  # noqa: E402
import ccf                                                 # noqa: E402
import fburgers as fb                                      # noqa: E402
import ks                                                  # noqa: E402
from stepper_surrogate import Stepper, train_stepper       # noqa: E402
from window_stepper import to_float64                      # noqa: E402

PER_SAVE = {"burgers": 8, "kdv": 40}
KS_MIXED = "ks-mixed-background-v2"     # KS 'mixed' training data with constant backgrounds (ks.training_initial)
TORUS = ("kdv", "bo", "ks") + fb.EQUATIONS + ccf.EQUATIONS


def horizon(pde):
    """T of an equation's trajectories (Burgers and heat: the windows' 0.05, as KdV's)."""
    return fb.T if pde in fb.EQUATIONS + ccf.EQUATIONS else {"bo": bo.T, "ks": ks.T}.get(pde, 0.05)


def initial(pde, ic, n, x, rng):
    if pde == "bo":
        return bo.initial("periodic" if ic in ("localised", "periodic") else "broad", n, x, rng)
    if pde == "ks":
        return ks.initial(ic, n, x, rng)
    if pde in fb.EQUATIONS + ccf.EQUATIONS:
        return fb.initial("periodic" if ic in ("localised", "periodic") else "broad", n, x, rng)
    L = x[-1] + (x[1] - x[0])
    u0 = np.zeros((n, len(x)))
    for r in range(n):
        if pde == "burgers":
            if ic == "localised":
                for _ in range(rng.integers(1, 3)):
                    c, w, a = rng.uniform(0.4, 0.5) * L, rng.uniform(0.03, 0.05) * L, rng.uniform(0.3, 0.8)
                    u0[r] += a * np.exp(-0.5 * ((x - c) / w) ** 2)
            else:
                for k in range(1, 7):
                    u0[r] += rng.standard_normal() / k * np.cos(2 * np.pi * k * x / L + rng.uniform(0, 2 * np.pi))
                u0[r] *= 0.5 / np.abs(u0[r]).max()
        else:
            if ic in ("localised", "periodic"):
                cs = rng.uniform(12.0, 16.0, 2)
                xs0 = (rng.uniform(0.30, 0.34), rng.uniform(0.48, 0.52))
                # 'periodic': the same solitons summed over periodic images, so the torus
                # field has no jump at the seam (the plain profile leaves one of ~3e-2)
                images = (-2, -1, 0, 1, 2) if ic == "periodic" else (0,)
                for c, x0 in zip(cs, xs0):
                    for m in images:
                        u0[r] += c / 2 / np.cosh(np.sqrt(c) / 2 * (x - x0 * L - m * L)) ** 2
            else:
                for _ in range(rng.integers(1, 4)):
                    a, w, x0 = rng.uniform(0.5, 4.0), rng.uniform(0.2, 1.0), rng.uniform(0, L)
                    d = (x - x0 + L / 2) % L - L / 2
                    u0[r] += a / np.cosh(d / w) ** 2
    return u0


def training_initial(pde, n, x, rng, kind="solitons"):
    """Stepper training data. KdV 'solitons': one or two solitons, c in [10, 16], in the bands
    [0.29, 0.35] L and [0.47, 0.53] L, background 0 (p = 0.5) or uniform in [-0.5, 0.5];
    a soliton on a constant background is an exact travelling wave (speed c + 6 b), so the
    field stays constant near the seam over T (the plain profile leaves a seam jump of ~3e-2
    that radiates high wavenumbers). 'mixed': half of those summed over periodic images (no
    jump), half broadband sech^2 humps anywhere on the torus, shuffled; 'narrow', 'wide' (exp72):
    'mixed' with the backgrounds scaled by ks.SCALE. Burgers: the localised generator."""
    if pde == "burgers":
        return initial(pde, "localised", n, x, rng)
    if pde == "bo":
        return bo.training_initial(n, x, rng, kind)
    if pde == "ks":
        return ks.training_initial(n, x, rng, kind)
    if pde in fb.EQUATIONS + ccf.EQUATIONS:
        return fb.training_initial(n, x, rng, kind)
    L = x[-1] + (x[1] - x[0])
    bands = ((0.29, 0.35), (0.47, 0.53))
    n_sol = n if kind == "solitons" else n - n // 2
    images = (0,) if kind == "solitons" else (-2, -1, 0, 1, 2)
    u0 = np.zeros((n_sol, len(x)))
    for r in range(n_sol):
        u0[r] = 0.0 if rng.random() < 0.5 else rng.uniform(-0.5, 0.5)
        u0[r] *= ks.SCALE.get(kind, 1.0)
        chosen = bands if rng.integers(1, 3) == 2 else (bands[rng.integers(0, 2)],)
        for lo, hi in chosen:
            c, x0 = rng.uniform(10.0, 16.0), rng.uniform(lo, hi)
            for m in images:
                u0[r] += c / 2 / np.cosh(np.sqrt(c) / 2 * (x - x0 * L - m * L)) ** 2
    if kind in ks.SCALE:
        u0 = np.concatenate([u0, initial(pde, "broad", n // 2, x, rng)])[rng.permutation(n)]
    return u0


def solve(pde, u0, T, Nt, nu):
    if pde == "bo":
        return bo.solve(u0, T, Nt)
    if pde == "ks":
        return ks.solve(u0, T, Nt)
    if pde in fb.EQUATIONS:
        return fb.solve(u0, T, Nt, fb.alpha(pde))
    if pde in ccf.EQUATIONS:
        return ccf.solve(u0, T, Nt, ccf.alpha(pde))
    dt = T / ((Nt - 1) * PER_SAVE[pde] + 0.5)
    if pde == "burgers":
        return np.asarray(e31.PDESolver.solve_burgers(u0, nu, T, Nt, dt_internal=dt, verbose=False))
    return np.asarray(e31.PDESolver.solve_kdv(u0, T, Nt, dt_max=dt, verbose=False))


def grid(pde, cfg, nx, nt, T):
    if pde in ("bo", "ks") + fb.EQUATIONS + ccf.EQUATIONS:
        m = bo if pde == "bo" else (ks if pde == "ks" else fb)
        return m.L, np.arange(nx) * m.L / nx, np.linspace(0, T, nt)
    _, _, xb, _ = e31.generate_pde_data(pde, cfg, 1, seed=0)
    L = float(e31.lengths(xb[:, 0].numpy().astype(np.float64))[0])
    return L, np.arange(nx) * L / nx, np.linspace(0, T, nt)


def training_data(pde, n, seed, kind, x, T, Nt, nu):
    """[n, Nx, Nt] stepper training trajectories."""
    return solve(pde, training_initial(pde, n, x, np.random.default_rng(seed), kind), T, Nt, nu)


def evaluation_data(pde, ic, n, dseed, x, t, nu):
    """u [n, Nx, Nt], u_t and ts [n, Nt] of the evaluation trajectories. KdV: the centre of an aligned
    three-point stencil around every snapshot (exp31), so every time shifts by delta."""
    Nx, Nt, T = len(x), len(t), t[-1]
    u = solve(pde, initial(pde, ic, n, x, np.random.default_rng(dseed)), T, Nt, nu)
    ts = np.tile(t, (n, 1))
    if pde == "kdv":
        delta = 2e-5
        flat = np.moveaxis(u, 2, 1).reshape(-1, Nx)
        snap = np.asarray(e31.PDESolver.solve_kdv(flat, 2 * delta, 3, dt_max=2 * delta / 4.5, verbose=False))
        u = np.moveaxis(snap[:, :, 1].reshape(n, Nt, Nx), 1, 2)
        ut = np.moveaxis(((snap[:, :, 2] - snap[:, :, 0]) / (2 * delta)).reshape(n, Nt, Nx), 1, 2)
        return u, ut, ts + delta
    return u, e31.time_derivative(u, ts, "central4"), ts


def stepper_path(args):
    data = "localised" if args.train_ic == "solitons" else args.train_ic
    name = args.stepper or f"S_theta_{args.pde}_s{args.seed}_{data}_stepper{args.tag}.pt"
    return name if os.path.isabs(name) else os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), name)


def load_or_train(args, x, T, Nt):
    stp = Stepper(modes=args.modes, positional_embedding=None)
    path = stepper_path(args)
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if args.pde == "ks" and args.train_ic == "mixed" and sd.get("recipe") != KS_MIXED:
            raise SystemExit(f"{path} was trained on zero-mean data (before 07.10): rename it to "
                             f"{path.replace('_mixed_', '_attractor_')}")
    elif args.train_if_missing:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tr = training_data(args.pde, args.n_train, args.seed, args.train_ic, x, T, Nt, args.nu)
        tr = torch.tensor(cr.corrupt(tr, args.noise, args.thin, args.seed, cr.TRAIN)[1], dtype=torch.float32)[:, None]
        print(f"training {args.pde} torus stepper on {args.n_train} trajectories ({args.train_ic}), seed {args.seed}, "
              f"{args.epochs} epochs, {args.modes} modes, no positional embedding, {dev}", flush=True)
        # partial state every 10 epochs next to the checkpoint: an interrupted run resumes exactly
        trained = train_stepper(tr, dev, epochs=args.epochs, seed=args.seed, loss="abs",
                                modes=args.modes, positional_embedding=None, ckpt=path + ".partial")
        sd = {k: v.detach().cpu() for k, v in trained.state_dict().items() if torch.is_tensor(v)}
        if args.pde == "ks" and args.train_ic == "mixed":
            sd["recipe"] = KS_MIXED                 # tells these from the zero-mean ones trained under this name
        torch.save(sd, path)
        if os.path.exists(path + ".partial"):
            os.remove(path + ".partial")
        print(f"saved {path}", flush=True)
    else:
        raise SystemExit(f"no stepper at {path}; pass --train-if-missing")
    stp.load_state_dict({k: v for k, v in sd.items() if torch.is_tensor(v)})
    stp = to_float64(stp).eval()
    for p in stp.parameters():
        p.requires_grad_(False)
    return stp


def blocks_exact(args, Qb, u, starts, dtK, keep):
    u0s = np.concatenate([u[:, :, j] for j in starts])
    q0 = np.stack([np.concatenate([Qb[i][:, :, j] for j in starts]) for i in range(Qb.shape[0])])
    qn = np.linalg.norm(q0, axis=2, keepdims=True)
    eps = np.where(qn > 1e-12, 1e-4 * np.linalg.norm(u0s, axis=1)[None, :, None] / np.maximum(qn, 1e-30), 0.0)
    batch = np.concatenate([np.concatenate([u0s + eps[i] * q0[i], u0s - eps[i] * q0[i]]) for i in range(Qb.shape[0])])
    # the solver integrates on the original grid (thin x as many saves), so thinning leaves the operator exact
    sol = solve(args.pde, batch, dtK, args.K * args.thin + 1, args.nu)[:, :, ::args.thin]
    NJ = u0s.shape[0]
    A = []
    for i in range(Qb.shape[0]):
        blk = sol[i * 2 * NJ:(i + 1) * 2 * NJ]
        e = np.where(eps[i] > 0, eps[i], 1.0)[:, :, None]
        dS = np.where(eps[i][:, :, None] > 0, (blk[:NJ] - blk[NJ:]) / (2 * e), 0.0)
        A.append((dS[:, :, 1:] - q0[i][:, :, None])[:, keep].ravel())
    return np.stack(A, 1)


def blocks_stepper(args, Qb, u, starts, stp, keep):
    u0 = torch.tensor(np.concatenate([u[:, :, j] for j in starts]))[:, None]
    A = []
    for i in range(Qb.shape[0]):
        q0 = np.concatenate([Qb[i][:, :, j] for j in starts])
        s, t = u0, torch.tensor(q0)[:, None]
        a = []
        for _ in range(args.K):
            s, t = jvp(stp, (s,), (t,))
            a.append(t[:, 0].numpy() - q0)
        A.append(np.stack(a, -1)[:, keep].ravel())
    return np.stack(A, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="burgers", choices=["burgers", "kdv", "bo", "ks", *fb.EQUATIONS, *ccf.EQUATIONS])
    ap.add_argument("--ic", default="localised", choices=["localised", "periodic", "broad"],
                    help="periodic: the localised solitons summed over periodic images (no seam jump)")
    ap.add_argument("--op", default="exact", choices=["exact", "stepper"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42, help="data seed (exact); stepper training seed (stepper)")
    ap.add_argument("--data-seed", type=int, default=None, help="evaluation data seed (default --seed)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--nu", type=float, default=0.002)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=0, help="blind readout of the m lowest (0: off)")
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-if-missing", action="store_true")
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--train-ic", default="solitons", choices=["solitons", "mixed", "attractor"],
                    help="stepper training data: localised solitons, or half solitons and half broadband humps; KS: "
                         "'mixed' on constant backgrounds, 'attractor' zero mean")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--modes", type=int, default=64)
    ap.add_argument("--save", default=None, help="npz with the eigen-generators and the evaluation data")
    ap.add_argument("--reliability-filter", type=float, default=None,
                    help="drop trajectories whose one-step error exceeds this multiple of the median")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise, times each trajectory's std (corrupt.py)")
    ap.add_argument("--thin", type=int, default=1, help="keep every thin-th snapshot (K, --every count kept ones)")
    ap.add_argument("--interior", action="store_true",
                    help="windows start at snapshot 2 and end by Nt - 3 (no one-sided u_t; exp48)")
    ap.add_argument("--ut", default="stencil", choices=["stencil", "fd"],
                    help="KdV u_t: aligned solver stencil of the evaluation data, or fourth-order differences of the "
                         "(corrupted) snapshots")
    args = ap.parse_args()
    if args.train_ic == "attractor" and args.pde != "ks":
        raise SystemExit("--train-ic attractor: Kuramoto-Sivashinsky only")
    if cr.corrupted(args) and args.pde == "kdv" and args.ut == "stencil":
        raise SystemExit("corrupted data carry no solver stencil: pass --ut fd")
    cfg = e31.ExperimentConfig()
    dseed = args.seed if args.data_seed is None else args.data_seed
    T = args.T or horizon(args.pde)
    Nx, Nt = args.nx, args.nt
    L, x, t = grid(args.pde, cfg, Nx, Nt, T)
    stp = load_or_train(args, x, T, Nt) if args.op == "stepper" else None
    u, ut, ts = evaluation_data(args.pde, args.ic, args.n, dseed, x, t, args.nu)
    u_clean, u = cr.corrupt(u, args.noise, args.thin, dseed, cr.EVAL)
    ts = ts[:, ::args.thin]
    if cr.corrupted(args) or args.ut == "fd":
        ut = e31.time_derivative(u, ts, "central4")
    Nt = u.shape[2]
    tm = max(1, 8 // args.thin)                 # scored times: the same physical margin as 8 of the original

    su, sut = e31.band_shares(u, ut)
    print(f"  band check, energy above k = Nx/4: u {su:.1e}, u_t {sut:.1e}"
          + ("   FLAG: u_t lives where the operator has seen little" if sut > 1e-2 else ""))
    frac = np.arange(Nx) / Nx
    keep = (frac >= args.seam) & (frac <= 1 - args.seam)
    if stp is not None:
        # equation-free reliability: one-step error on each trajectory relative to the data increment
        with torch.no_grad():
            errs = np.array([float(np.linalg.norm(stp(torch.tensor(u[r, :, :-1].T)[:, None])[:, 0].numpy()
                                                  - u[r, :, 1:].T) / np.linalg.norm(u[r, :, 1:].T - u[r, :, :-1].T))
                             for r in range(u.shape[0])])
        print("  per-trajectory one-step error: " + " ".join(f"{e:.3f}" for e in errs))
        if args.reliability_filter:
            sel = errs <= args.reliability_filter * np.median(errs)
            u, ut, ts, u_clean = u[sel], ut[sel], ts[sel], u_clean[sel]
            print(f"  reliability filter: kept {int(sel.sum())} of {len(errs)} (<= {args.reliability_filter} x median)")
    N = u.shape[0]
    xs = np.tile(x, (N, 1))
    near = (frac < 0.1) | (frac > 0.9)
    seam_share = float(np.sqrt((u[:, near] ** 2).mean() / (u ** 2).mean()))
    seam_grad = float(np.sqrt((np.gradient(u, axis=1)[:, near] ** 2).mean() / (np.gradient(u, axis=1) ** 2).mean()))

    scale = (L, T, u.std())
    F, Qb, mons = e31.basis(u, xs, ts, 2, ut, scale)
    # --interior: every characteristic of a window on times 2 .. Nt - 3, where u_t is the central five-point difference
    starts = list(range(2, Nt - args.K - 2, args.every) if args.interior else range(0, Nt - args.K, args.every))
    dtK = args.K * (ts[0, 1] - ts[0, 0])
    if stp is not None:
        # ground-truth free: rollout error on the scored band, 1 and K steps, relative to the data increment
        u0t = torch.tensor(np.concatenate([u[:, :, j] for j in starts]))[:, None]
        with torch.no_grad():
            s_ = u0t
            for kk in range(1, args.K + 1):
                s_ = stp(s_)
                if kk in (1, args.K):
                    tgt = np.concatenate([u[:, :, j + kk] for j in starts])[:, keep]
                    cur = np.concatenate([u[:, :, j] for j in starts])[:, keep]
                    e_u = np.linalg.norm(s_[:, 0].numpy()[:, keep] - tgt) / np.linalg.norm(tgt - cur)
                    print(f"  rollout error relative to the data increment after {kk} step(s): {e_u:.3f}")
        A = blocks_stepper(args, Qb, u, starts, stp, keep)
    else:
        A = blocks_exact(args, Qb, u, starts, dtK, keep)
    B, Q0 = [], []
    for i in range(Qb.shape[0]):
        qb = np.concatenate([Qb[i][:, :, j:j + args.K + 1] for j in starts])
        B.append((qb[:, :, 1:] - qb[:, :, :1])[:, keep].ravel())
        Q0.append(qb[:, keep, 0].ravel())
    B, Q0 = np.stack(B, 1), np.stack(Q0, 1)
    Nr = (A - B).T @ (A - B)
    G0 = args.K * Q0.T @ Q0
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    lam, C = e34.solve(Nr, Dr)
    # ground-truth free: how far the characteristics move over the horizon, relative to their size
    rel_change = float(np.sqrt(np.trace(B.T @ B) / np.trace(G0)))

    names, gt = e31.algebra(args.pde, xs, ts, u, cfg.alpha_heat)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T, np.stack([g.ravel() for g in gt]).T, rcond=None)
    rq = [float(c @ Nr @ c / (c @ Dr @ c)) for c in cg.T]
    k = len(names)
    reg = (slice(None), slice(None), keep, slice(tm, -tm))
    gtv = np.stack([g[reg].ravel() for g in gt])
    Fv = F[(slice(None),) + reg].reshape(F.shape[0], -1)
    Qe, _ = np.linalg.qr((C[:, :k].T @ Fv).T)
    cont = [np.linalg.norm(Qe.T @ g) / np.linalg.norm(g) for g in gtv]
    opname = "exact" if stp is None else f"stepper s{args.seed}{args.tag}"
    print(f"[{args.pde}, {opname}, {args.ic}] data seed {dseed}, {N} trajectories, Nx {Nx}, T {T}, "
          + (f"nu {args.nu}, " if args.pde == "burgers" else "") + f"K = {args.K}, seam band {args.seam}; "
          f"L {L:.3f}; within 0.1 of the seam / overall, rms of u: {seam_share:.1e}, of u_x: {seam_grad:.1e}; "
          f"u_t {args.ut}, noise {args.noise:g}, thin {args.thin}")
    print(f"  relative change of the basis characteristics, rms over k = 1..K steps: {rel_change:.1e}")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:k + 3]))
    print("  algebra losses: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq)) + f"   (lambda{k + 1} = {lam[k]:.1e})")
    print(f"  algebra inside the {k} lowest: " + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))
    if args.save:
        # eigen-generator j = sum_i C[i, j] * monomial_i in component i // len(mons), monomials in
        # X = x / L, T = t / T_max, U = u / sigma (e31.basis)
        np.savez(args.save, pde=args.pde, method="FINS-exact" if stp is None else "FINS-stepper",
                 seed=args.seed, data_seed=dseed, C=C[:, :k + 3], lam=lam[:k + 6], mons=np.array(mons), degree=2,
                 Lw=scale[0], T_max=scale[1], sigma=scale[2], x_offset=np.zeros(N), u=u_clean, xs=xs, ts=ts,
                 names=np.array(names), losses=np.array(rq), containment=np.array(cont), rel_change=rel_change,
                 band=np.array([su, sut]), K=args.K, k=k, noise=args.noise, thin=args.thin, ut=args.ut, interior=args.interior)
        print(f"  saved {args.save}")
    if args.m:
        e32.print_readout(C, F, Nr, Dr, mons, scale, args.m)


if __name__ == "__main__":
    main()
