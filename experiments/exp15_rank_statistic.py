"""Experiment 15 — screening replacement rank statistics (CPU, minutes).

Why
---
exp6 showed the paper's rank criterion is self-defeating: the norm anchor
is applied to the DEFLATED generator, so the optimiser is instructed to
produce ||v_perp|| = 1 at every slot. A norm collapse then signals an
optimisation failure, not exhaustion of available directions, and the
spectral gap of Section 3.3 cannot detect rank.

Candidate replacements, all read off the SAME deflation run:

  norm       ||v_perp|| after training           (the paper's statistic)
  rayleigh   the converged normalised residual   ||dS Q0 - Q||^2 / ||v||^2
  resid_fix  the residual re-evaluated with the deflated generator
             rescaled to unit norm (removes any residual scale freedom)

The last two must RISE when no genuine direction is left: the optimiser
can always find something orthogonal with norm 1, but it cannot make that
something satisfy the equivariance condition. The screening question is
whether a gap appears after slot 3 (the essential rank of the periodic
problem) in the constrained run.

This is a screening tool, not a result: small grid, few epochs, one seed.

Run:  python exp15_rank_statistic.py --lam 30 --n-sym 6
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import exp_common as C
import exp10_leak_vs_delta as E10
from config import ExperimentConfig
from exp6_trivial_subalgebra import ConstrainedGeneratorFNO
from method_fins import train_solution_operator, equivariance_loss
from method_fins_components import deflate_generator, create_grids

L = E10.L


class OracleHeatOperator(nn.Module):
    """Exact periodic heat solution operator with the surrogate's signature.

    Lets the deflation run at delta_J = 0, which decouples the rank
    statistic from surrogate quality: genuine directions must hit machine
    zero and spurious ones must not. If no gap appears even here, the
    deflation scheme itself is at fault; if a gap appears, rank detection
    is a surrogate-quality requirement and the lemma R <= C delta_J^2
    turns it into a concrete budget.
    """

    def __init__(self, alpha, L):
        super().__init__()
        self.alpha, self.L = alpha, L

    def forward(self, u0, x_coords, t_coords):
        if u0.dim() == 3:
            u0 = u0.unsqueeze(1) if u0.shape[1] != 1 else u0
        B, _, Nx = u0.shape
        t = t_coords[:, 0, :]                                  # [B, Nt]
        k = (2 * np.pi / self.L) * torch.fft.fftfreq(
            Nx, d=1.0 / Nx, device=u0.device, dtype=torch.float32)
        decay = torch.exp(-self.alpha * (k[None, :, None] ** 2)
                          * t[:, None, :])                     # [B,Nx,Nt]
        uh = torch.fft.fft(u0[:, 0, :], dim=1)                 # [B, Nx]
        out = torch.fft.ifft(uh[:, :, None] * decay, dim=1).real
        return out.unsqueeze(1)                                # [B,1,Nx,Nt]


def slot_statistics(gen, frozen, sol, xs, ts, S, cfg, device, dx, dt,
                    loss_mode="jvp", n_eval=16):
    """norm / rayleigh / residual-at-unit-norm for one converged slot."""
    u = sol[:n_eval].to(device)
    xg, tg = create_grids(xs[:n_eval].to(device), ts[:n_eval].to(device),
                          device)
    with torch.no_grad():
        xi, eta, phi, b = gen(xg, tg, u)
        xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
        v_d = torch.cat([xi_d, eta_d, phi_d], dim=1)
        nrm = float(v_d.pow(2).mean().sqrt())
    with torch.enable_grad():
        ray = float(equivariance_loss(
            S, xi_d, eta_d, phi_d, u[:, :, :, 0].contiguous(), u,
            xs[:n_eval].to(device), ts[:n_eval].to(device), dx, dt, device,
            mode=loss_mode, normalize=True))
        s = 1.0 / max(nrm, 1e-12)
        fix = float(equivariance_loss(
            S, xi_d * s, eta_d * s, phi_d * s,
            u[:, :, :, 0].contiguous(), u,
            xs[:n_eval].to(device), ts[:n_eval].to(device), dx, dt, device,
            mode=loss_mode, normalize=False))
    return nrm, ray, fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--n-sym", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--ep-sur", type=int, default=60)
    ap.add_argument("--ep-gen", type=int, default=12)
    ap.add_argument("--oracle", action="store_true",
                    help="use the exact heat operator as S_theta "
                         "(delta_J = 0): tests the rank statistic itself, "
                         "free of surrogate noise")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    cfg.grid_size, cfg.time_steps = 32, 32
    cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ics, sols, xs, ts, x_np, t_np = E10.make_dataset(
        args.n_train, cfg.grid_size, cfg.time_steps, rng)
    print("=" * 78)
    print(f"EXP 15 - rank-statistic screening (single scale L = {L:.3f}, "
          f"lam = {args.lam:g}, device = {device})")
    print("=" * 78)
    if args.oracle:
        S = OracleHeatOperator(cfg.alpha_heat, L).to(device)
        delta = E10.rel_l2(S, ics, sols, xs, ts, device)
        print(f"  ORACLE operator: delta = {delta:.2e} (delta_J = 0)")
    else:
        S = train_solution_operator(ics, sols, xs, ts, cfg, device,
                                    epochs=args.ep_sur, verbose=False)
        delta = E10.rel_l2(S, ics, sols, xs, ts, device)
        print(f"  screening surrogate: delta = {delta:.4f}  "
              f"(floor C*delta_J^2 ~ {0.108 * delta ** 2:.1e} if "
              f"delta_J ~ delta)")

    loader = DataLoader(TensorDataset(sols, xs, ts),
                        batch_size=cfg.batch_size, shuffle=True)
    dx = (xs[0, 0, 1] - xs[0, 0, 0]).item()
    dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()

    frozen, rows = [], []
    for k in range(args.n_sym):
        t0 = time.time()
        torch.manual_seed(args.seed * 1000 + k)
        gen = ConstrainedGeneratorFNO(cfg).to(device)
        opt = optim.AdamW(gen.parameters(), lr=cfg.lr_symmetry,
                          weight_decay=1e-4)
        for _ in range(args.ep_gen):
            for u_b, x_b, t_b in loader:
                u = u_b.to(device)
                xg, tg = create_grids(x_b.to(device), t_b.to(device), device)
                xi, eta, phi, b = gen(xg, tg, u)
                xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
                l_eq = equivariance_loss(
                    S, xi_d, eta_d, phi_d, u[:, :, :, 0].contiguous(), u,
                    x_b.to(device), t_b.to(device), dx, dt, device,
                    mode="jvp")
                v_d = torch.cat([xi_d, eta_d, phi_d], dim=1)
                l_norm = (v_d.pow(2).mean().sqrt() + 1e-8 - 1.0).pow(2)
                share_b = b.pow(2).mean() / (phi.pow(2).mean() + 1e-12)
                loss = l_eq + l_norm + args.lam * share_b
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
                opt.step()
        nrm, ray, fix = slot_statistics(gen, frozen, sols, xs, ts, S, cfg,
                                        device, dx, dt)
        with torch.no_grad():
            u1 = sols[:1].to(device)
            xg1, tg1 = create_grids(xs[:1].to(device), ts[:1].to(device),
                                    device)
            xi, eta, phi, _ = gen(xg1, tg1, u1)
            xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
        frozen.append((xi_d.detach(), eta_d.detach(), phi_d.detach()))
        rows.append([k + 1, nrm, ray, fix])
        print(f"  slot {k + 1}: norm = {nrm:.3f}  rayleigh = {ray:.3e}  "
              f"resid@unit = {fix:.3e}  [{time.time() - t0:.0f} s]")

    print()
    tab = [[str(r[0]), f"{r[1]:.3f}", f"{r[2]:.3e}", f"{r[3]:.3e}"]
           for r in rows]
    print(C.fmt_table(tab, ["slot", "norm", "rayleigh", "resid@unit"]))
    print("\nratios slot(k+1)/slot(k) — a rank-r statistic should jump "
          "after slot r:")
    for name, j in (("norm(k)/norm(k+1)", 1), ("rayleigh(k+1)/(k)", 2),
                    ("resid@unit(k+1)/(k)", 3)):
        vals = [r[j] for r in rows]
        if j == 1:
            rat = [vals[i] / max(vals[i + 1], 1e-12)
                   for i in range(len(vals) - 1)]
        else:
            rat = [vals[i + 1] / max(vals[i], 1e-12)
                   for i in range(len(vals) - 1)]
        print(f"  {name:<22}: "
              + "  ".join(f"{i+1}->{i+2}: {v:6.2f}" for i, v in
                          enumerate(rat)))

    path = os.path.join(
        C.RESULTS_DIR,
        f"exp15_rank_statistic_lam{args.lam:g}"
        f"{'_oracle' if args.oracle else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "norm", "rayleigh", "resid_at_unit_norm"])
        w.writerows(rows)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
