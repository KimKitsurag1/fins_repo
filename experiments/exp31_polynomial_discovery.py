"""Experiment 31 -- FINS discovery in a finite-dimensional generator class.

With xi, eta, phi spanned by polynomials in (x, t, u) of total degree <= d,
the multiple-shooting increment loss is a ratio of quadratic forms in the
coefficients. Discovery is its generalised eigenproblem: the eigenvectors
with the smallest eigenvalues are the discovered generators. No training.
Equation-free: the operator is the exact solver or a learned stepper and
u_t is a finite difference of the data; the ground truth only scores.

Burgers: the 5-dim algebra. Heat (u_t = alpha u_xx): 6 generators plus,
the equation being linear, superpositions u -> u + beta, beta a solution;
within total degree <= 3 these are beta = 1, x, x^2 + 2 alpha t,
x^3 + 6 alpha x t -- 10 directions in all. Heat samples have their own L and
T = L^2, so on the grid every sample obeys one equation: one stepper serves
all of them, while generators live in absolute per-sample coordinates.

Reported: share of each algebra generator inside the k lowest eigen-
generators; the loss of each algebra generator itself through the operator
(low baseline: ~1 through an untrained operator); the share of each
discovered eigen-generator outside the algebra.

Run:  python exp31_polynomial_discovery.py --pde heat --op stepper --n 10
"""

import argparse
import itertools
import math
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pde"))
from config import ExperimentConfig                      # noqa: E402
from data_generation import PDESolver, generate_pde_data  # noqa: E402
from ground_truth import GroundTruthAlgebra, get_gt_names  # noqa: E402
from stepper_surrogate import Stepper, train_stepper       # noqa: E402
from exp30_loss_spectrum import blocks_exact, blocks_stepper  # noqa: E402
import bo                                                  # noqa: E402
import ccf                                                 # noqa: E402
import fburgers as fb                                      # noqa: E402
import ks                                                  # noqa: E402


def lengths(xs):
    return xs[:, -1] + (xs[:, 1] - xs[:, 0])


def x_derivative(u, xs):
    k = 2 * np.pi * np.stack([np.fft.fftfreq(u.shape[1], d=L / u.shape[1])
                              for L in lengths(xs)])
    return np.real(np.fft.ifft(1j * k[:, :, None] * np.fft.fft(u, axis=1), axis=1))


def time_derivative(u, ts, mode, stp=None):
    """u_t for the characteristics, per-sample dt, all equation-free.
    central: central difference of the data (the default and the best);
    forward: forward difference; stepper: (Step(u) - u) / dt."""
    dt = (ts[:, 1] - ts[:, 0])[:, None, None]
    if mode == "central":
        return np.gradient(u, axis=2, edge_order=2) / dt
    if mode == "central4":
        # fourth order: five-point central inside, one-sided fourth-order
        # formulas on the two first and two last time slices
        o = np.empty_like(u)
        o[:, :, 2:-2] = (-u[:, :, 4:] + 8 * u[:, :, 3:-1] - 8 * u[:, :, 1:-3] + u[:, :, :-4]) / 12
        o[:, :, 0] = (-25 * u[:, :, 0] + 48 * u[:, :, 1] - 36 * u[:, :, 2] + 16 * u[:, :, 3] - 3 * u[:, :, 4]) / 12
        o[:, :, 1] = (-3 * u[:, :, 0] - 10 * u[:, :, 1] + 18 * u[:, :, 2] - 6 * u[:, :, 3] + u[:, :, 4]) / 12
        o[:, :, -1] = (25 * u[:, :, -1] - 48 * u[:, :, -2] + 36 * u[:, :, -3] - 16 * u[:, :, -4] + 3 * u[:, :, -5]) / 12
        o[:, :, -2] = (3 * u[:, :, -1] + 10 * u[:, :, -2] - 18 * u[:, :, -3] + 6 * u[:, :, -4] - u[:, :, -5]) / 12
        return o / dt
    out = np.empty_like(u)
    if mode == "forward":
        out[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    else:
        with torch.no_grad():
            flat = torch.tensor(np.moveaxis(u, 2, 1).reshape(-1, u.shape[1]),
                                dtype=torch.float32)[:, None, :]
            nxt = stp(flat)[:, 0].double().numpy()
        nxt = np.moveaxis(nxt.reshape(u.shape[0], u.shape[2], u.shape[1]), 1, 2)
        out[:, :, :-1] = (nxt - u)[:, :, :-1]
    out[:, :, -1] = out[:, :, -2]
    return out / dt


def band_shares(u, ut, frac=4):
    """Energy share above k = Nx / frac (axis 1) of u and of u_t: the protocol's band check. A u_t
    share above 1e-2 flags data whose time derivative lives where the operator has seen little."""
    k = np.abs(np.fft.fftfreq(u.shape[1], d=1.0 / u.shape[1]))
    hi = (k > u.shape[1] // frac)[None, :, None]
    eu, et = np.abs(np.fft.fft(u, axis=1)) ** 2, np.abs(np.fft.fft(ut, axis=1)) ** 2
    return float((eu * hi).sum() / eu.sum()), float((et * hi).sum() / et.sum())


def basis(u, xs, ts, d, ut, scale):
    """Basis generators (one component = one monomial of the scaled absolute
    coordinates) and their characteristics: F [nb, 3, N, Nx, Nt], Q [nb, N,
    Nx, Nt]. One coefficient vector is one vector field for every sample."""
    ux = x_derivative(u, xs)
    X = np.broadcast_to(xs[:, :, None] / scale[0], u.shape)
    T = np.broadcast_to(ts[:, None, :] / scale[1], u.shape)
    Un = u / scale[2]
    mons = [(a, b, c) for a, b, c in itertools.product(range(d + 1), repeat=3)
            if a + b + c <= d]
    F, Q = [], []
    for comp in range(3):
        for a, b, c in mons:
            f = X ** a * T ** b * Un ** c
            fld = np.zeros((3,) + u.shape)
            fld[comp] = f
            F.append(fld)
            Q.append(-f * ux if comp == 0 else (-f * ut if comp == 1 else f))
    return np.stack(F), np.stack(Q), mons


def algebra(pde, xs, ts, u, alpha, dmax=3):
    """Names and fields [3, N, Nx, Nt] of the algebra inside the class."""
    N = u.shape[0]
    if pde == "burgers":
        per = [GroundTruthAlgebra.burgers_generators(xs[r], ts[r], u[r]) for r in range(N)]
        names = get_gt_names("burgers")
    elif pde == "kdv":
        per = [GroundTruthAlgebra.kdv_generators(xs[r], ts[r], u[r]) for r in range(N)]
        names = get_gt_names("kdv")
    elif pde == "bo":
        per = [bo.generators(xs[r], ts[r], u[r]) for r in range(N)]
        names = list(bo.NAMES)
    elif pde == "ks":
        per = [ks.generators(xs[r], ts[r], u[r]) for r in range(N)]
        names = list(ks.NAMES)
    elif pde in fb.EQUATIONS:
        per = [fb.generators(xs[r], ts[r], u[r], pde) for r in range(N)]
        names = fb.names(pde)
    elif pde in ccf.EQUATIONS:
        per = [ccf.generators(xs[r], ts[r], u[r], pde) for r in range(N)]
        names = ccf.names(pde)
    else:
        per = [GroundTruthAlgebra.heat_generators(xs[r], ts[r], u[r], alpha=alpha)
               for r in range(N)]
        # superpositions b(x, t) d_u, b a heat polynomial: one per degree <= dmax
        labels = ["b=1", "b=x", "b=x2+2at", "b=x3+6axt"]
        names = list(get_gt_names("heat")) + [labels[n] if n < 4 else f"b=h{n}"
                                              for n in range(dmax + 1)]
        for r in range(N):
            X, T = np.meshgrid(xs[r], ts[r], indexing="ij")
            Z = np.zeros_like(X)
            for n in range(dmax + 1):
                beta = sum(math.factorial(n) / (math.factorial(k) * math.factorial(n - 2 * k))
                           * (alpha * T) ** k * X ** (n - 2 * k) for k in range(n // 2 + 1))
                per[r] = list(per[r]) + [(Z, Z, beta)]
    return names, [np.stack([np.stack([np.asarray(per[r][j][c]) for r in range(N)])
                             for c in range(3)]) for j in range(len(names))]


def blocks_exact_heat(Qb, xs, ts, starts, K, alpha, keep):
    """Increment blocks through the exact heat propagator (linear, so the
    Frechet derivative is the propagator itself), per-sample L and dt."""
    k2 = (2 * np.pi * np.stack([np.fft.fftfreq(Qb.shape[2], d=L / Qb.shape[2])
                                for L in lengths(xs)])) ** 2           # [N, Nx]
    dt = ts[:, 1] - ts[:, 0]
    decay = np.exp(-alpha * k2[:, :, None] * (dt[:, None, None]
                                              * np.arange(1, K + 1)[None, None, :]))
    A, B = [], []
    for Q in Qb:
        a_, b_ = [], []
        for j in starts:
            q0 = Q[:, :, j]
            prop = np.real(np.fft.ifft(np.fft.fft(q0, axis=1)[:, :, None] * decay, axis=1))
            a_.append((prop - q0[:, :, None])[:, keep])
            b_.append((Q[:, :, j + 1:j + K + 1] - Q[:, :, j:j + 1])[:, keep])
        A.append(np.stack(a_).ravel())
        B.append(np.stack(b_).ravel())
    return np.stack(A, 1), np.stack(B, 1)


BURGERS_PER_SAVE = 79       # internal solver steps per saved snapshot (~1e-4 each)


def burgers_coarse(cfg, n, seed):
    """Burgers trajectories with every snapshot on the solver's internal grid.
    The benchmark's solver rounds save times to that grid (78 or 79 steps
    per save, +-0.6 percent of a save step), which no finite-difference
    order can undo in u_t."""
    ics, _, xs, ts = generate_pde_data("burgers", cfg, n, seed=seed)
    T, Nt = cfg.T_final, cfg.time_steps
    u = np.asarray(PDESolver.solve_burgers(ics[:, 0].numpy(), cfg.nu_burgers, T, Nt,
                                           dt_internal=T / ((Nt - 1) * BURGERS_PER_SAVE + 0.5),
                                           verbose=False))
    return (xs[:, 0].numpy().astype(np.float64), ts[:, 0].numpy().astype(np.float64),
            u.astype(np.float64))


def blocks_exact_burgers_aligned(Qb, u, ts, starts, K, nu, keep):
    """exp30's exact Burgers blocks with the re-solve's saves aligned to the
    internal grid, matching burgers_coarse."""
    dt_save = ts[0, 1] - ts[0, 0]
    nb = Qb.shape[0]
    u0 = np.concatenate([u[:, :, j] for j in starts])
    q0 = np.stack([np.concatenate([Qb[i][:, :, j] for j in starts]) for i in range(nb)])
    qn = np.linalg.norm(q0, axis=2, keepdims=True)
    eps = np.where(qn > 1e-12, 1e-4 * np.linalg.norm(u0, axis=1)[None, :, None]
                   / np.maximum(qn, 1e-30), 0.0)
    batch = np.concatenate([np.concatenate([u0 + eps[i] * q0[i], u0 - eps[i] * q0[i]])
                            for i in range(nb)])
    T = K * dt_save
    sol = np.asarray(PDESolver.solve_burgers(batch, nu, T, K + 1,
                                             dt_internal=T / (K * BURGERS_PER_SAVE + 0.5),
                                             verbose=False))
    NJ = u0.shape[0]
    A, B = [], []
    for i in range(nb):
        blk = sol[i * 2 * NJ:(i + 1) * 2 * NJ]
        e = np.where(eps[i] > 0, eps[i], 1.0)[:, :, None]
        dS = np.where(eps[i][:, :, None] > 0, (blk[:NJ] - blk[NJ:]) / (2 * e), 0.0)
        a = dS[:, :, 1:K + 1] - q0[i][:, :, None]
        qb = np.concatenate([Qb[i][:, :, j:j + K + 1] for j in starts])
        b = qb[:, :, 1:] - qb[:, :, :1]
        A.append(a[:, keep].ravel())
        B.append(b[:, keep].ravel())
    return np.stack(A, 1), np.stack(B, 1)


KDV_PER_SAVE = 400          # internal solver steps per saved snapshot


def kdv_coarse(cfg, n, seed):
    """KdV trajectories with every snapshot exactly on the solver's internal
    grid. The benchmark's own saves round to that grid (393 or 394 steps per
    save), which is harmless for u but corrupts any finite difference in
    time. Returns xs [n, Nx], ts [n, Nt], u [n, Nx, Nt]."""
    ics, _, xs, ts = generate_pde_data("kdv", cfg, n, seed=seed)
    T, Nt = cfg.T_final, cfg.time_steps
    steps = (Nt - 1) * KDV_PER_SAVE
    u = np.asarray(PDESolver.solve_kdv(ics[:, 0].numpy(), T, Nt,
                                       dt_max=T / (steps + 0.5), verbose=False))
    return (xs[:, 0].numpy().astype(np.float64), ts[:, 0].numpy().astype(np.float64),
            u.astype(np.float64))


def kdv_stencil(cfg, n, seed, delta):
    """KdV data with an equation-free u_t: around every saved snapshot a
    three-point stencil at spacing delta, on the solver's internal grid.
    KdV's u_t lives in modes 8-21 (omega = k^3 up to ~9e3) that a save step
    of 1/127 aliases (central-difference error 0.99); an aligned stencil at
    delta = 2e-5 gives 0.003. The stencil centre becomes the snapshot (all
    times shift by delta). Measurement design, not equation knowledge."""
    xs, ts, u = kdv_coarse(cfg, n, seed)
    N, Nx, Nt = u.shape
    flat = np.moveaxis(u, 2, 1).reshape(-1, Nx)
    snap = np.asarray(PDESolver.solve_kdv(flat, 2 * delta, 3,
                                          dt_max=2 * delta / 4.5, verbose=False))
    centre = np.moveaxis(snap[:, :, 1].reshape(N, Nt, Nx), 1, 2)
    ut = np.moveaxis(((snap[:, :, 2] - snap[:, :, 0]) / (2 * delta)).reshape(N, Nt, Nx), 1, 2)
    return xs, ts + delta, centre, ut


def blocks_exact_kdv(Qb, u, ts, starts, K, keep):
    """Increment blocks through the exact KdV solver (central finite
    differences), saves aligned to the internal grid."""
    dt_save = ts[0, 1] - ts[0, 0]
    nb, N = Qb.shape[0], u.shape[0]
    u0 = np.concatenate([u[:, :, j] for j in starts])
    q0 = np.stack([np.concatenate([Qb[i][:, :, j] for j in starts]) for i in range(nb)])
    qn = np.linalg.norm(q0, axis=2, keepdims=True)
    eps = np.where(qn > 1e-12, 1e-4 * np.linalg.norm(u0, axis=1)[None, :, None]
                   / np.maximum(qn, 1e-30), 0.0)
    batch = np.concatenate([np.concatenate([u0 + eps[i] * q0[i], u0 - eps[i] * q0[i]])
                            for i in range(nb)])
    T = K * dt_save
    sol = np.asarray(PDESolver.solve_kdv(batch, T, K + 1,
                                         dt_max=T / (K * KDV_PER_SAVE + 0.5), verbose=False))
    NJ = u0.shape[0]
    A, B = [], []
    for i in range(nb):
        blk = sol[i * 2 * NJ:(i + 1) * 2 * NJ]
        e = np.where(eps[i] > 0, eps[i], 1.0)[:, :, None]
        dS = np.where(eps[i][:, :, None] > 0, (blk[:NJ] - blk[NJ:]) / (2 * e), 0.0)
        a = dS[:, :, 1:K + 1] - q0[i][:, :, None]
        qb = np.concatenate([Qb[i][:, :, j:j + K + 1] for j in starts])
        b = qb[:, :, 1:] - qb[:, :, :1]
        A.append(a[:, keep].ravel())
        B.append(b[:, keep].ravel())
    return np.stack(A, 1), np.stack(B, 1)


def discover(A, B):
    s = np.sqrt((A ** 2).sum(0) + (B ** 2).sum(0))
    s = np.where(s > 1e-30, s, 1.0)
    An, Bn = A / s, B / s
    Nm = (An - Bn).T @ (An - Bn)
    Dm = An.T @ An + Bn.T @ Bn + 1e-10 * np.eye(A.shape[1])
    lam, V = eigh(Nm, Dm)
    return lam, V / s[:, None], Nm, Dm, s


def load_stepper(args, cfg):
    arch = {"modes": args.stepper_modes,
            "positional_embedding": None if getattr(args, "stepper_pos_emb", "grid") == "none" else "grid"}
    stp = Stepper(**arch)
    if args.stepper == "random":
        # own-class null: an untrained stepper of the same architecture
        torch.manual_seed(args.null_seed)
        stp = Stepper(**arch)
        return stp.eval()
    name = args.stepper or (f"S_theta_{args.pde}_s{args.seed}_g{cfg.grid_size}"
                            f"x{cfg.time_steps}_stepper{args.stepper_tag}.pt")
    path = name if os.path.isabs(name) else os.path.join(
        os.environ.get("FINS_RESULTS_DIR", "."), name)
    if not os.path.exists(path):
        if not args.train_if_missing:
            raise SystemExit(f"no stepper at {path} (use --train-if-missing)")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        n_tr = args.stepper_n_train or cfg.n_train
        if args.pde == "kdv":
            tr = torch.tensor(kdv_coarse(cfg, n_tr, args.seed)[2], dtype=torch.float32)[:, None]
        elif args.pde == "burgers" and args.aligned:
            tr = torch.tensor(burgers_coarse(cfg, n_tr, args.seed)[2], dtype=torch.float32)[:, None]
        else:
            _, tr, _, _ = generate_pde_data(args.pde, cfg, n_tr, seed=args.seed)
        print(f"training {args.pde} stepper on {tr.shape[0]} ICs, seed {args.seed}, "
              f"{args.stepper_epochs} epochs, loss {args.stepper_loss}, {dev}", flush=True)
        trained = train_stepper(tr, dev, epochs=args.stepper_epochs, seed=args.seed,
                                loss=args.stepper_loss, **arch)
        sd = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
              for k, v in trained.state_dict().items()}
        torch.save(sd, path)
        print(f"saved {path}", flush=True)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    # a fresh CPU module: neuralop's positional embedding caches its grid as a
    # plain tensor, which .cpu() would leave on the GPU
    stp.load_state_dict({k: v for k, v in sd.items() if k != "_metadata"})
    return stp.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=["burgers", "heat", "kdv"], default="burgers")
    ap.add_argument("--op", choices=["exact", "stepper"], default="exact")
    ap.add_argument("--degree", type=int, default=None, help="default 2 burgers, 3 heat")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=None,
                    help="subspace size scored; default: the algebra's dimension")
    ap.add_argument("--stepper", default=None,
                    help="checkpoint, or 'random' for the untrained-stepper null")
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="data (and stepper) seed")
    ap.add_argument("--ut", choices=["central", "central4", "forward", "stepper", "stencil"], default=None,
                    help="default: stencil for kdv, central otherwise")
    ap.add_argument("--stencil-delta", type=float, default=2e-5)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="seed of the evaluation trajectories (default --seed: they then lie in the "
                         "stepper's training set); use another value for unseen trajectories")
    ap.add_argument("--aligned", action="store_true",
                    help="burgers: snapshots on the solver's internal grid (saves not rounded)")
    ap.add_argument("--train-if-missing", action="store_true")
    ap.add_argument("--stepper-epochs", type=int, default=400)
    ap.add_argument("--stepper-n-train", type=int, default=None)
    ap.add_argument("--stepper-loss", choices=["abs", "incr"], default="abs",
                    help="training objective when a stepper is trained here")
    ap.add_argument("--stepper-pos-emb", default="grid", choices=["grid", "none"],
                    help="FNO positional embedding; 'none' keeps the stepper shift-equivariant on the torus")
    ap.add_argument("--stepper-modes", type=int, default=24,
                    help="FNO n_modes of the stepper; neuralop keeps n_modes // 2 + 1 "
                         "real-FFT frequencies, so 24 keeps 13 of 33 on a 64-point grid")
    ap.add_argument("--stepper-tag", default="",
                    help="suffix of the stepper checkpoint name, e.g. _incr")
    ap.add_argument("--reliability-filter", type=float, default=None,
                    help="drop trajectories whose stepper one-step error exceeds this "
                         "multiple of the median (equation-free)")
    ap.add_argument("--only-L", type=float, default=None,
                    help="heat: keep only samples with this domain length (single-scale)")
    args = ap.parse_args()
    d = args.degree or (3 if args.pde == "heat" else 2)
    cfg = ExperimentConfig()
    ut_given = None
    # evaluation trajectories; with dseed == seed they are the first n of the stepper's training set
    dseed = args.seed if args.data_seed is None else args.data_seed
    if args.pde == "kdv":
        xs, ts, u, ut_given = kdv_stencil(cfg, args.n, dseed, args.stencil_delta)
    elif args.pde == "burgers" and args.aligned:
        xs, ts, u = burgers_coarse(cfg, args.n, dseed)
    else:
        n_gen = args.n * (5 if args.only_L else 1)
        _, sols, xs, ts = generate_pde_data(args.pde, cfg, n_gen, seed=dseed)
        xs, ts = xs[:, 0].numpy().astype(np.float64), ts[:, 0].numpy().astype(np.float64)
        u = sols[:, 0].numpy().astype(np.float64)
    if args.only_L:
        # single-scale subset: heat cycles L over five values, so every fifth
        # sample shares one domain length and one final time
        sel = np.isclose(lengths(xs), args.only_L)[:args.n * 5]
        xs, ts, u = xs[sel][:args.n], ts[sel][:args.n], u[sel][:args.n]
        print(f"single-scale subset: {len(u)} samples with L = {args.only_L}, "
              f"T = {ts[0, -1]:.3f}")
    Nx, Nt = u.shape[1:]
    frac = np.arange(Nx) / Nx                    # same fractional grid for all samples
    keep = (frac >= args.seam) & (frac <= 1 - args.seam)
    scale = (lengths(xs).max(), ts[:, -1].max(), u.std())
    stp = load_stepper(args, cfg) if (args.op == "stepper" or args.ut == "stepper") else None
    if args.reliability_filter and stp is not None:
        # equation-free reliability: the stepper's one-step prediction error on
        # each trajectory relative to the data's own increment; drop outliers
        errs = []
        with torch.no_grad():
            for r in range(u.shape[0]):
                cur = torch.tensor(u[r, :, :-1].T, dtype=torch.float32)[:, None, :]
                nxt = u[r, :, 1:].T
                pred = stp(cur)[:, 0].double().numpy()
                errs.append(np.linalg.norm(pred - nxt) / np.linalg.norm(nxt - u[r, :, :-1].T))
        errs = np.array(errs)
        keep_r = errs <= args.reliability_filter * np.median(errs)
        print(f"reliability filter: step errors " + " ".join(f"{e:.3f}" for e in errs)
              + f"; kept {keep_r.sum()} of {len(errs)} (<= {args.reliability_filter} x median)")
        xs, ts, u = xs[keep_r], ts[keep_r], u[keep_r]
        if ut_given is not None:
            ut_given = ut_given[keep_r]
    ut_mode = args.ut or ("stencil" if args.pde == "kdv" else "central")
    if ut_mode == "stencil":
        if ut_given is None:
            raise SystemExit("--ut stencil is implemented for kdv")
        ut = ut_given
    else:
        ut = time_derivative(u, ts, ut_mode, stp)
    F, Qb, mons = basis(u, xs, ts, d, ut, scale)
    starts = list(range(0, Nt - args.K, args.every))
    if args.op == "stepper":
        A, B = blocks_stepper(Qb, u, starts, args.K, stp, keep)
    elif args.pde == "burgers" and args.aligned:
        A, B = blocks_exact_burgers_aligned(Qb, u, ts, starts, args.K, cfg.nu_burgers, keep)
    elif args.pde == "burgers":
        A, B = blocks_exact(Qb, u, ts[0], starts, args.K, cfg.nu_burgers, keep)
    elif args.pde == "kdv":
        A, B = blocks_exact_kdv(Qb, u, ts, starts, args.K, keep)
    else:
        A, B = blocks_exact_heat(Qb, xs, ts, starts, args.K, cfg.alpha_heat, keep)
    lam, C, Nm, Dm, sc = discover(A, B)

    names, gt = algebra(args.pde, xs, ts, u, cfg.alpha_heat, d)
    k = args.k or len(names)
    reg = (slice(None), slice(None), keep, slice(8, -8))
    gtv = np.stack([g[reg].ravel() for g in gt])
    Fv = F[(slice(None),) + reg].reshape(F.shape[0], -1)
    E = C[:, :k].T @ Fv
    Qe, _ = np.linalg.qr(E.T)
    cont = [np.linalg.norm(Qe.T @ g) / np.linalg.norm(g) for g in gtv]
    # the loss of each algebra generator from its exact coefficients (the
    # algebra lies in the class), and the share of each discovered
    # eigen-generator outside the algebra
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T,
                             np.stack([g.ravel() for g in gt]).T, rcond=None)
    rq = [float((g * sc) @ Nm @ (g * sc) / ((g * sc) @ Dm @ (g * sc))) for g in cg.T]
    Qg, _ = np.linalg.qr(gtv.T)
    outside = [np.linalg.norm(e - Qg @ (Qg.T @ e)) / np.linalg.norm(e) for e in E]
    ratios = lam[1:k + 6] / np.maximum(lam[:k + 5], 1e-15)
    rank = int(np.argmax(ratios)) + 1

    op = args.op if args.op == "exact" else f"stepper {args.stepper or 'seed ' + str(args.seed)}"
    print(f"[{args.pde}, {op}] seed {args.seed}, data seed {dseed}, {u.shape[0]} trajectories, u_t {ut_mode}, "
          f"degree {d} ({len(mons)} monomials, {F.shape[0]} coefficients), K = {args.K}, "
          f"starts every {args.every}, {len(starts)} starts; algebra in the class: {len(names)}")
    print("  lowest eigenvalues : " + " ".join(f"{v:.1e}" for v in lam[:k + 4]))
    print(f"  rank estimate      : {rank}  (largest gap lambda{rank + 1}/lambda{rank} = "
          f"{ratios[rank - 1]:.1f})")
    print("  loss of each algebra generator through this operator: "
          + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq))
          + f"   (lambda{k + 1} = {lam[k]:.1e})")
    print(f"  share of each of the {k} lowest eigen-generators OUTSIDE the algebra: "
          + "  ".join(f"{v:.2f}" for v in outside))
    print(f"  share of each algebra generator inside the {k} lowest eigen-generators: "
          + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))


if __name__ == "__main__":
    main()
