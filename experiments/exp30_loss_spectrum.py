"""Experiment 30 -- what does the short-horizon loss actually minimise?

The increment residual is linear in the generator, so on the 27-dim
monomial class (xi, eta, phi each on {1, x, t, u, xt, x^2, t^2, tu, xu}) the
loss is a ratio of quadratic forms c'Nc / c'Dc. Its stationary values are
generalised eigenvalues: a good loss has five near-zero ones spanning the
Burgers algebra and a clear gap to the sixth -- the best spurious symmetry
an optimiser can reach inside the class. No training involved.

Compared: a single start at t = 0 (the v4 loss) versus multiple shooting
from data states at many t_j, each through the exact solver and the
trained stepper.

Run:  python exp30_loss_spectrum.py --stepper S_theta_burgers_s42_g64x128_stepper.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.linalg import eigh
from torch.func import jvp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pde"))
from config import ExperimentConfig                      # noqa: E402
from data_generation import PDESolver, generate_pde_data  # noqa: E402
from determining import MONO, gt_coefficients              # noqa: E402
from exp29_verify_generators import monomials               # noqa: E402
from stepper_surrogate import RolloutOperator, Stepper      # noqa: E402


def basis_Q(u, x, t):
    """Characteristic of each of the 27 basis generators, [27, N, Nx, Nt]."""
    L = x[-1] + (x[1] - x[0])
    k = 2 * np.pi * np.fft.fftfreq(u.shape[1], d=L / u.shape[1])
    ux = np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(u, axis=1), axis=1))
    ut = np.gradient(u, t[1] - t[0], axis=2, edge_order=2)
    X, T = np.meshgrid(x, t, indexing="ij")
    mono = monomials(np.broadcast_to(X, u.shape), np.broadcast_to(T, u.shape), u)
    out = []
    for comp in range(3):
        for m in range(len(MONO)):
            f = mono[..., m]
            out.append(-f * ux if comp == 0 else (-f * ut if comp == 1 else f))
    return np.stack(out)


def blocks_exact(Qb, u, t, starts, K, nu, keep):
    """Increment blocks a (operator) and b (data) through the exact solver."""
    nb, N = Qb.shape[0], u.shape[0]
    u0 = np.concatenate([u[:, :, j] for j in starts])              # [N*J, Nx]
    q0 = np.stack([np.concatenate([Qb[i][:, :, j] for j in starts])
                   for i in range(nb)])                            # [nb, N*J, Nx]
    qn = np.linalg.norm(q0, axis=2, keepdims=True)
    eps = 1e-4 * np.linalg.norm(u0, axis=1)[None, :, None] / np.maximum(qn, 1e-30)
    eps = np.where(qn > 1e-12, eps, 0.0)
    batch = np.concatenate([np.concatenate([u0 + eps[i] * q0[i],
                                            u0 - eps[i] * q0[i]])
                            for i in range(nb)])
    sol = np.asarray(PDESolver.solve_burgers(batch, nu, t[K], K + 1,
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


def blocks_stepper(Qb, u, starts, K, stp, keep):
    """Increment blocks through the stepper, jvp over K steps."""
    S = RolloutOperator(stp, K + 1).eval()
    for p in S.parameters():
        p.requires_grad_(False)
    nb = Qb.shape[0]
    u0 = torch.tensor(np.concatenate([u[:, :, j] for j in starts]),
                      dtype=torch.float32)[:, None, :]
    A, B = [], []
    for i in range(nb):
        q0 = torch.tensor(np.concatenate([Qb[i][:, :, j] for j in starts]),
                          dtype=torch.float32)[:, None, :]
        _, dS = jvp(lambda v: S(v, None, None), (u0,), (q0,))
        dS = dS[:, 0].double().numpy()
        a = dS[:, :, 1:K + 1] - q0[:, 0].double().numpy()[:, :, None]
        qb = np.concatenate([Qb[i][:, :, j:j + K + 1] for j in starts])
        b = qb[:, :, 1:] - qb[:, :, :1]
        A.append(a[:, keep].ravel())
        B.append(b[:, keep].ravel())
    return np.stack(A, 1), np.stack(B, 1)


def spectrum(A, B, G):
    """Generalised eigenvalues of (A-B)'(A-B) against A'A + B'B, the GT
    Rayleigh quotients, and the overlap of the five lowest eigenvectors with
    span(GT) in a column-normalised basis."""
    s = np.sqrt((A ** 2).sum(0) + (B ** 2).sum(0))
    s = np.where(s > 1e-30, s, 1.0)
    An, Bn = A / s, B / s
    Nm = (An - Bn).T @ (An - Bn)
    Dm = An.T @ An + Bn.T @ Bn + 1e-10 * np.eye(A.shape[1])
    lam, V = eigh(Nm, Dm)
    Gn = G * s[:, None]
    rq = [float(g @ Nm @ g / (g @ Dm @ g)) for g in Gn.T]
    # worst combination inside the algebra: combinations can score above
    # every single generator, so the conservative gap is lambda6 / this
    span_max = float(eigh(Gn.T @ Nm @ Gn, Gn.T @ Dm @ Gn, eigvals_only=True)[-1])
    Qv, _ = np.linalg.qr(V[:, :5])
    Qg, _ = np.linalg.qr(Gn)
    cos = np.linalg.svd(Qv.T @ Qg, compute_uv=False)
    return lam, rq, cos, span_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--only-stepper", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig()
    _, sols, xs, ts = generate_pde_data("burgers", cfg, args.n, seed=42)
    x, t = xs[0, 0].numpy(), ts[0, 0].numpy()
    u = sols[:, 0].numpy().astype(np.float64)
    L = x[-1] + (x[1] - x[0])
    keep = (x >= args.seam * L) & (x <= (1 - args.seam) * L)
    Qb = basis_Q(u, x, t)
    G = gt_coefficients(2 * np.pi)
    stp = None
    if args.stepper:
        path = args.stepper if os.path.isabs(args.stepper) else os.path.join(
            os.environ.get("FINS_RESULTS_DIR", "."), args.stepper)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        stp = Stepper()
        stp.load_state_dict({k: v for k, v in sd.items() if k != "_metadata"})
        stp.eval()
    Nt = len(t)
    configs = [("single start t=0, K=8 (v4)", [0], 8),
               ("multiple shooting, K=8, starts every 16", list(range(0, Nt - 9, 16)), 8),
               ("multiple shooting, K=8, starts every 8 (full cover)", list(range(0, Nt - 9, 8)), 8),
               ("multiple shooting, K=16, starts every 16 (full cover)", list(range(0, Nt - 17, 16)), 16)]
    if args.only_stepper:
        configs = configs[1:]
    print("generalised eigenvalues of the increment loss on the 27-dim monomial "
          "class (a good loss: five ~0 spanning the algebra, then a gap)")
    for op in ("exact", "stepper"):
        if op == "stepper" and stp is None:
            continue
        if op == "exact" and args.only_stepper:
            continue
        for lab, starts, K in configs:
            if op == "exact":
                A, B = blocks_exact(Qb, u, t, starts, K, cfg.nu_burgers, keep)
            else:
                A, B = blocks_stepper(Qb, u, starts, K, stp, keep)
            lam, rq, cos, smax = spectrum(A, B, G)
            print(f"\n  [{op}] {lab}, {len(starts)} start(s)")
            print("    lowest eigenvalues : " + " ".join(f"{v:.1e}" for v in lam[:8]))
            print("    GT Rayleigh values : " + " ".join(f"{v:.1e}" for v in rq)
                  + "   (d_x d_t Gal scal proj)")
            print(f"    worst algebra combination {smax:.1e}, gap lambda6 / it = "
                  f"{lam[5] / max(smax, 1e-12):.1f}"
                  f"   | 5 lowest vs span(GT), principal cosines "
                  + " ".join(f"{c:.2f}" for c in cos), flush=True)


if __name__ == "__main__":
    main()
