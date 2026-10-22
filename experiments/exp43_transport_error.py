"""Experiment 43 -- the transport error of a learned operator (Proposition 2 of the paper).

On the protocol's evaluation data, the transport blocks are built twice, through the exact operator
(A) and through the learned one (A_theta = A + E), with the same basis, starts, horizon and scored
region. With the exact normaliser D = A'A + B'B + mu (tr B'B / tr G0) G0:
  eta     = ||E D^{-1/2}||_2                    the operator's transport error over the class
  eta_V   = max over the algebra of ||E c||/||c||_D   (its part along the known generators)
  sigma_i = singular values of (A - B) D^{-1/2};  sigma~_i the same with A_theta
  floor   = ||(A - B) P_V D^{-1/2}||_2          the exact operator's discretisation floor on the algebra
Checks printed: max_i |sigma~_i - sigma_i| <= eta (Weyl); the sufficient condition
eta' < sigma'_(k+1) / 2 with eta' = ||(M P_V + E) D^{-1/2}|| and sigma' the spectrum of M (I - P_V);
the rescaling by the implemented normaliser D_theta; and (iii) per known generator, its share outside
the k lowest learned eigenvectors against (loss / lambda_(k+1))^{1/2}. Heat: exact heat-kernel window
operator vs the CNO on four-period windows (exp40); KdV: exact spectral solver vs the torus FNO
stepper on periodised solitons (exp39). Ground truth is used only to define the algebra subspace V.

Run:  python exp43_transport_error.py --pde kdv --seed 42                  (data seed 1000 + seed)
      python exp43_transport_error.py --pde heat --seed 42 --stepper S_theta_heat_s42_..._cno....pt
"""

import argparse
import glob
import itertools
import os
import sys
from types import SimpleNamespace

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp40_window as e40                                 # noqa: E402
from window_stepper import GlobalStepper, to_float64       # noqa: E402


def inv_sqrt(D):
    """D^{-1/2} of a symmetric positive definite matrix, with Jacobi scaling for conditioning."""
    s = np.sqrt(np.clip(np.diag(D), 1e-300, None))
    w, V = np.linalg.eigh(D / np.outer(s, s))
    w = np.clip(w, w.max() * 1e-15, None)
    return (V / np.sqrt(w)) @ V.T / s[:, None]                 # R with R' D R = I


def checksum(net):
    """Sum of absolute parameter values: tells checkpoints apart in the log."""
    return float(sum(p.detach().double().abs().sum() for p in net.parameters()))


def window_blocks(stp, uw, Qb, starts, K, shrink, W):
    """Transport blocks of a window operator: [rows, P] increments D S^k Q_j - Q_j on the scored region."""
    reg = slice(shrink, W - shrink)
    u0 = torch.tensor(np.concatenate([uw[:, :, j] for j in starts]))[:, None]
    A = []
    for p in range(Qb.shape[0]):
        q0 = np.concatenate([Qb[p][:, :, j] for j in starts])
        s, t = u0, torch.tensor(q0)[:, None]
        a_ = []
        for k in range(1, K + 1):
            s, t = jvp(stp, (s,), (t,))
            off = shrink - k * stp.r
            a_.append(t[:, 0, off:off + (W - 2 * shrink)].numpy() - q0[:, reg])
        A.append(np.stack(a_, -1).ravel())
    return np.stack(A, 1)


def heat_setup(args, cfg):
    xs, ts, u, ut = e40.trajectories("heat", cfg, args.n, args.data_seed, True, 1.0)
    N, Nx, Nt = u.shape
    W = args.periods * Nx
    idx = np.arange(W) % Nx
    dx = e31.lengths(xs) / Nx
    xw = dx[:, None] * np.arange(W)[None]
    uw, utw = u[:, idx], ut[:, idx]
    uxw = e40.dx4(uw, dx)
    Lw, T = (W * dx).max(), ts[:, -1].max()
    X = (xw[:, :, None] - xw[:, W // 2:W // 2 + 1, None]) / Lw
    Tn = ts[:, None, :] / T
    Un = uw / uw.std()
    mons = [mm for mm in itertools.product(range(4), repeat=3) if sum(mm) <= 3]
    F, Qb = [], []
    for comp in range(3):
        for a, b, c in mons:
            f = np.broadcast_to(X ** a * Tn ** b, uw.shape) * Un ** c
            fld = np.zeros((3,) + uw.shape)
            fld[comp] = f
            F.append(fld)
            Qb.append(-f * uxw if comp == 0 else (-f * utw if comp == 1 else f))
    F, Qb = np.stack(F), np.stack(Qb)
    starts = list(range(0, Nt - args.K, args.every))
    shrink = 6 * args.K
    reg = slice(shrink, W - shrink)
    # exact operator: the 13-tap moment-matched heat kernel (r = 6 per step)
    dx0, dt0 = e31.lengths(xs)[0] / Nx, ts[0, 1] - ts[0, 0]
    exact = e40.HeatKernelWindow(e40.moment_kernel(cfg.alpha_heat * dt0 / dx0 ** 2, 6)).double()
    learned = GlobalStepper("cno", W)
    sd = torch.load(args.stepper_path, map_location="cpu", weights_only=False)
    learned.load_state_dict({k: v for k, v in sd.items() if torch.is_tensor(v)})
    learned = to_float64(learned).eval()
    for p in learned.parameters():
        p.requires_grad_(False)
    print(f"  learned operator {args.stepper_path}: parameter checksum {checksum(learned):.10e}")
    A = window_blocks(exact, uw, Qb, starts, args.K, shrink, W)
    At = window_blocks(learned, uw, Qb, starts, args.K, shrink, W)
    B, Q0 = [], []
    for p in range(Qb.shape[0]):
        B.append(np.stack([np.concatenate([Qb[p][:, reg, j + k] - Qb[p][:, reg, j] for j in starts])
                           for k in range(1, args.K + 1)], -1).ravel())
        Q0.append(np.concatenate([Qb[p][:, reg, j] for j in starts]).ravel())
    names, gt = e31.algebra("heat", xw - xw[:, W // 2:W // 2 + 1], ts, uw, cfg.alpha_heat, 3)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T, np.stack([g.ravel() for g in gt]).T, rcond=None)
    return A, At, np.stack(B, 1), np.stack(Q0, 1), cg, names


def kdv_setup(args, cfg):
    T, Nx, Nt = 0.05, 128, 128
    L, x, t = e39.grid("kdv", cfg, Nx, Nt, T)
    u, ut, ts = e39.evaluation_data("kdv", "periodic", args.n, args.data_seed, x, t, 0.002)
    N = u.shape[0]
    xs = np.tile(x, (N, 1))
    frac = np.arange(Nx) / Nx
    keep = (frac >= 0.2) & (frac <= 0.8)
    F, Qb, mons = e31.basis(u, xs, ts, 2, ut, (L, T, u.std()))
    starts = list(range(0, Nt - args.K, args.every))
    ns = SimpleNamespace(pde="kdv", K=args.K, nu=0.002, modes=64, stepper=args.stepper_path,
                         train_ic="mixed", tag=args.tag, seed=args.seed, train_if_missing=False)
    stp = e39.load_or_train(ns, x, T, Nt)
    print(f"  learned operator {args.stepper_path or 'exp39 default'}: parameter checksum {checksum(stp):.10e}")
    A = e39.blocks_exact(ns, Qb, u, starts, args.K * (t[1] - t[0]), keep)
    At = e39.blocks_stepper(ns, Qb, u, starts, stp, keep)
    B, Q0 = [], []
    for i in range(Qb.shape[0]):
        qb = np.concatenate([Qb[i][:, :, j:j + args.K + 1] for j in starts])
        B.append((qb[:, :, 1:] - qb[:, :, :1])[:, keep].ravel())
        Q0.append(qb[:, keep, 0].ravel())
    names, gt = e31.algebra("kdv", xs, ts, u, cfg.alpha_heat)
    cg, *_ = np.linalg.lstsq(F.reshape(F.shape[0], -1).T, np.stack([g.ravel() for g in gt]).T, rcond=None)
    return A, At, np.stack(B, 1), np.stack(Q0, 1), cg, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["heat", "kdv"])
    ap.add_argument("--seed", type=int, default=42, help="operator seed")
    ap.add_argument("--data-seed", type=int, default=None, help="default 1000 + seed")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--stepper", default=None, help="checkpoint (heat: CNO trained at L = 1; kdv: default exp39 name)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.data_seed = 1000 + args.seed if args.data_seed is None else args.data_seed
    cfg = e31.ExperimentConfig()
    if args.pde == "heat":
        name = args.stepper or f"S_theta_heat_s{args.seed}_g{cfg.grid_size}x{cfg.time_steps}_cno{args.tag}.pt"
        args.stepper_path = name if os.path.isabs(name) else os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), name)
        if not os.path.exists(args.stepper_path):
            found = sorted(glob.glob(os.path.join(os.path.dirname(args.stepper_path),
                                                  f"S_theta_heat_s{args.seed}_*cno*.pt")))
            raise SystemExit(f"no CNO at {args.stepper_path}; pass --stepper or --tag (the protocol's CNO trained at "
                             f"L = 1 on smooth data). Candidates: " + (", ".join(map(os.path.basename, found)) or "none"))
        A, At, B, Q0, cg, names = heat_setup(args, cfg)
    else:
        args.stepper_path = args.stepper
        A, At, B, Q0, cg, names = kdv_setup(args, cfg)
    analyse(A, At, B, Q0, cg, names, args.K, args.mu,
            f"[{args.pde}] operator seed {args.seed}, data seed {args.data_seed}, {args.n} trajectories, K = {args.K}")


def analyse(A, At, B, Q0, cg, names, K, mu, header):
    """Proposition 2 on measured blocks: exact A, learned At, data B, starting characteristics Q0, the
    algebra's coefficient vectors cg [P, k]. Prints the checks and returns the main numbers."""
    k = len(names)
    G0 = K * Q0.T @ Q0
    unit = np.trace(B.T @ B) / np.trace(G0)
    D = A.T @ A + B.T @ B + mu * unit * G0
    Dt = At.T @ At + B.T @ B + mu * unit * G0
    R = inv_sqrt(D)
    M, Mt, E = A - B, At - B, At - A
    MR, ER = M @ R, E @ R
    sig = np.linalg.svd(MR, compute_uv=False)[::-1]
    sigt = np.linalg.svd(Mt @ R, compute_uv=False)[::-1]
    eta = float(np.linalg.norm(ER, 2))
    # the algebra in D-orthonormal coordinates y = R^{-1} c: columns of Vd orthonormal, P_V R = R Vd Vd'
    Vd = np.linalg.qr(np.linalg.solve(R, cg))[0]
    etaV = float(np.linalg.norm(ER @ Vd, 2))
    floor = float(np.linalg.norm(MR @ Vd, 2))                      # ||M P_V D^{-1/2}||
    # Proposition 2 applied to M' = M (I - P_V), which vanishes on V, with perturbation M P_V + E
    Pc = np.eye(R.shape[1]) - Vd @ Vd.T
    sigp = np.linalg.svd(MR @ Pc, compute_uv=False)[::-1]
    eta_eff = float(np.linalg.norm(MR @ Vd @ Vd.T + ER, 2))
    ex_loss = [float(c @ M.T @ M @ c / (c @ D @ c)) for c in cg.T]
    le_loss = [float(c @ Mt.T @ Mt @ c / (c @ D @ c)) for c in cg.T]
    # implemented learned pencil (normaliser D_theta)
    Rt = inv_sqrt(Dt)
    sig_impl = np.linalg.svd(Mt @ Rt, compute_uv=False)[::-1]
    im_loss = [float(c @ Mt.T @ Mt @ c / (c @ Dt @ c)) for c in cg.T]
    weyl = float(np.abs(sigt - sig).max())
    # (iii) per generator, in the implemented geometry: share of c outside the k lowest learned eigenvectors
    # (D_theta-norm) against its bound (L_theta(c) / lambda_(k+1))^{1/2}, the inverse square root of its margin
    Wt = Mt @ Rt
    Yk = np.linalg.svd(Wt, full_matrices=False)[2][::-1][:k].T       # right singular vectors, k smallest
    Yc = np.linalg.solve(Rt, cg)
    out = [float(np.linalg.norm(y - Yk @ (Yk.T @ y)) / np.linalg.norm(y)) for y in Yc.T]
    cert = [float(np.sqrt(v) / sig_impl[k]) for v in im_loss]
    lo, hi = 1 / (1 + eta), (1 / np.sqrt(1 - 2 * eta - eta ** 2) if 2 * eta + eta ** 2 < 1 else np.inf)
    ratio = sig_impl[sigt > 1e-12 * sigt.max()] / sigt[sigt > 1e-12 * sigt.max()]
    fmt = lambda v: " ".join(f"{x:.1e}" for x in v)
    print(f"{header}, {A.shape[1]} basis fields, algebra {k}: " + " ".join(names))
    print(f"  transport error eta = ||E D^-1/2|| = {eta:.6e}; along the algebra eta_V = {etaV:.6e}")
    print(f"  exact operator: losses of the algebra {fmt(ex_loss)}; floor ||M P_V D^-1/2|| = {floor:.3e}")
    print(f"  exact sigma_k = {sig[k - 1]:.3e}, sigma_(k+1) = {sig[k]:.3e} (lambda_(k+1) = {sig[k] ** 2:.3e}); "
          f"floor removed sigma'_(k+1) = {sigp[k]:.3e}")
    print(f"  learned, exact normaliser: sigma~_k = {sigt[k - 1]:.3e}, sigma~_(k+1) = {sigt[k]:.3e}; losses of the "
          f"algebra {fmt(le_loss)} (max {max(le_loss):.2e}, eta_V^2 = {etaV ** 2:.2e})")
    print(f"  Weyl: max_i |sigma~_i - sigma_i| = {weyl:.3e} <= eta: {weyl <= eta * (1 + 1e-9)}")
    ok = eta_eff < sigp[k] / 2
    print(f"  sufficient condition: eta' = ||(M P_V + E) D^-1/2|| = {eta_eff:.3e} (eta + floor {eta + floor:.3e}) "
          f"vs sigma'_(k+1) / 2 = {sigp[k] / 2:.3e} -> {'holds' if ok else 'fails'}; angle bound "
          + (f"{np.degrees(np.arcsin(min(1.0, eta_eff / (sigp[k] - eta_eff)))):.1f} deg" if sigp[k] > eta_eff else "none"))
    print(f"  implemented (normaliser D_theta): lambda_k = {sig_impl[k - 1] ** 2:.3e}, lambda_(k+1) = "
          f"{sig_impl[k] ** 2:.3e}; losses of the algebra max {max(im_loss):.3e}; sigma_theta / sigma~ in "
          f"[{ratio.min():.3f}, {ratio.max():.3f}], bound [{lo:.3f}, {hi:.3f}]")
    print("  (iii) share outside the k lowest learned eigenvectors / bound (loss / lambda_(k+1))^1/2: "
          + "  ".join(f"{n} {o:.3f}/{c:.3f}" for n, o, c in zip(names, out, cert))
          + f"   holds: {all(o <= c * (1 + 1e-9) + 1e-12 for o, c in zip(out, cert))}")
    return dict(eta=eta, eta_V=etaV, floor=floor, eta_eff=eta_eff, sigma=sig, sigma_learned=sigt,
                sigma_floorless=sigp, sigma_impl=sig_impl, weyl=weyl, out=out, cert=cert,
                losses_exact=ex_loss, losses_learned=le_loss, losses_impl=im_loss)

if __name__ == "__main__":
    main()
