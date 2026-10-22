"""oracle_jacobian — an EXACT operator for the discovery loop, at surrogate speed.

Purpose. On KdV every ground-truth generator fails through the trained
surrogate (all >= 0.50 after ||Q||-normalisation), so FINS-v3 finds
nothing there (cos 0.22). The question is:
does the DISCOVERY LOOP recover the KdV algebra when the operator is
exact? If yes, the surrogate is the sole blocker; if no, KdV is closed at
the method level and no surrogate work will change that.

Running FINS through the differentiable solver directly is ~100x slower
than through a surrogate (10^4 sequential ETDRK4 steps per forward). But
the equivariance criterion only ever uses the Jacobian ACTION dS[u0](Q0)
at the training ICs, and dS[u0] is a fixed linear map per sample. So:

  1. precompute J[b] in R^{(Nx*Nt) x Nx} for N samples by central finite
     differences of the data-generating solver along the Nx basis
     directions -- 2*Nx batched solves in total, independent of N;
  2. wrap it as a first-order Taylor operator
        S(u0) = u_data[b] + J[b] (u0 - u0_data[b])
     whose forward at a training IC is exact and whose jvp is exactly
     J[b] delta -- the same (u0, xc, tc) -> u signature train_fins expects.

The solver runs in float64, so the FD Jacobian is accurate to ~1e-6
relative; the resulting residual floor is far below anything the loop
resolves. This operator uses the solver, so a FINS run through it is NOT
equation-free: it is the loop's ceiling, reported as a diagnostic row,
like LieNLSD, never as the method.

Run:  python oracle_jacobian.py --pde kdv --n 128     (precompute, ~10 min GPU)
      python run_single_pde.py --pde kdv --oracle-operator --fins-only ...
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import ExperimentConfig
from data_generation import PDESolver, generate_pde_data


def _solve(pde, u0_np, cfg, Ls=None, Ts=None):
    """Solve a batch with the SAME solver and per-sample (L, T) as the data.

    Heat data are multi-scale (L in {0.5..2.0}, T ~ L^2 per sample); a
    Jacobian taken at the wrong L is the Jacobian of a different operator.
    Burgers/KdV use the fixed L = 2 pi, T = T_final of the generator.
    """
    Nt = cfg.time_steps
    if pde == "heat":
        return np.stack([PDESolver.solve_heat(z, cfg.alpha_heat, float(T),
                                              Nt, L=float(L))
                         for z, L, T in zip(u0_np, Ls, Ts)])
    if pde == "burgers":
        return np.asarray(PDESolver.solve_burgers(u0_np, cfg.nu_burgers,
                                                  cfg.T_final, Nt, verbose=False))
    return np.asarray(PDESolver.solve_kdv(u0_np, cfg.T_final, Nt, verbose=False))


def precompute_jacobian(pde, ics, cfg, xs=None, ts=None, eps_rel=1e-4,
                        verbose=True):
    """J[b, :, i] = d S(u0_b)[:, :] / d u0_b[i], flattened over (Nx, Nt)."""
    u0 = np.asarray(ics[:, 0], dtype=np.float64)
    N, Nx = u0.shape
    Ls = Ts = None
    if pde == "heat":
        x = np.asarray(xs[:, 0]); t = np.asarray(ts[:, 0])
        Ls = x[:, -1] + (x[:, 1] - x[:, 0])          # periodic domain length
        Ts = t[:, -1]
    Nt = cfg.time_steps
    scale = np.sqrt(np.mean(u0 ** 2, axis=1, keepdims=True)) + 1e-12
    J = np.zeros((N, Nx * Nt, Nx), dtype=np.float32)
    t0 = time.time()
    for i in range(Nx):
        e = np.zeros((N, Nx))
        e[:, i] = eps_rel * scale[:, 0]
        plus = _solve(pde, u0 + e, cfg, Ls, Ts)
        minus = _solve(pde, u0 - e, cfg, Ls, Ts)
        J[:, :, i] = ((plus - minus) / (2 * e[:, i, None, None])
                      ).reshape(N, Nx * Nt).astype(np.float32)
        if verbose and (i + 1) % max(Nx // 8, 1) == 0:
            print(f"    jacobian column {i + 1}/{Nx}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return J


class LinearizedOracle(nn.Module):
    """First-order Taylor operator around the stored training ICs.

    forward(u0, xc, tc): locates each incoming u0 among the stored ICs by
    nearest neighbour (exact matches in practice, since the loop evaluates
    at training ICs), and returns u_data + J (u0 - u0_data). Its jvp is
    exactly J delta.
    """

    def __init__(self, u0_data, u_data, J):
        super().__init__()
        self.register_buffer("u0_data", torch.as_tensor(u0_data,
                                                        dtype=torch.float32))
        self.register_buffer("u_data", torch.as_tensor(u_data,
                                                       dtype=torch.float32))
        self.register_buffer("J", torch.as_tensor(J, dtype=torch.float32))
        self.Nx = self.u0_data.shape[1]
        self.Nt = self.u_data.shape[2]

    def _match(self, u0):
        d = torch.cdist(u0.detach(), self.u0_data)
        return d.argmin(dim=1)

    def forward(self, u0, x_coords, t_coords):
        if u0.dim() == 3:
            u0 = u0[:, 0]
        idx = self._match(u0)
        base = self.u0_data[idx]
        out = self.u_data[idx].reshape(u0.shape[0], -1) + torch.einsum(
            "bij,bj->bi", self.J[idx], u0 - base)
        return out.view(u0.shape[0], 1, self.Nx, self.Nt)


def build_oracle(pde, cfg, n, seed, cache_dir=None, verbose=True):
    cache_dir = cache_dir or os.environ.get("FINS_RESULTS_DIR", ".")
    path = os.path.join(cache_dir, f"oracle_J_{pde}_n{n}_s{seed}"
                                   f"_g{cfg.grid_size}x{cfg.time_steps}.npz")
    ics, sols, xs, ts = generate_pde_data(pde, cfg, n, seed=seed)
    if os.path.exists(path):
        J = np.load(path)["J"]
        if verbose:
            print(f"  oracle Jacobian loaded: {path}")
    else:
        if verbose:
            print(f"  precomputing exact Jacobian for {n} ICs "
                  f"({2 * cfg.grid_size} batched solves)...")
        J = precompute_jacobian(pde, ics.numpy(), cfg, xs=xs.numpy(),
                                ts=ts.numpy(), verbose=verbose)
        np.savez(path, J=J)
        if verbose:
            print(f"  saved -> {path}")
    S = LinearizedOracle(ics[:, 0].numpy(), sols[:, 0].numpy(), J)
    return S, (ics, sols, xs, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="kdv")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = ExperimentConfig()
    S, _ = build_oracle(args.pde, cfg, args.n, args.seed)
    print(f"oracle ready: J {tuple(S.J.shape)}")


if __name__ == "__main__":
    main()
