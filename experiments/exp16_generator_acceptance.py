"""Experiment 16 — generator acceptance test: why does a slot never reach zero?

exp15 (--oracle) showed that with the EXACT operator, so delta_J = 0, a
deflation slot still plateaus at a residual of ~1e-4 instead of machine
zero, even though d_x, d_t and u d_u are all inside the hypothesis space.
The generator network is therefore the binding constraint. This script
decomposes that floor into its three possible causes, using known
symmetries as targets:

  R_analytic  residual of the ANALYTIC ground-truth field through the
              oracle operator — the true floor (exp1: ~1e-28 for d_x)
  eps_fit     relative field error after SUPERVISED regression of the
              network onto that same analytic field — pure expressivity
  R_fitted    residual of the fitted network's field — how much a field
              error of eps_fit costs in the equivariance residual
  R_scratch   residual after training the network with the equivariance
              loss alone, from scratch — the optimisation path

Reading the four numbers:
  R_fitted >> R_analytic at small eps_fit  ->  the residual amplifies field
      error; the floor is a discretisation/conditioning property, and the
      amplification factor R_fitted / eps_fit^2 is the constant to report
  eps_fit large  ->  the FNO parameterisation cannot represent the
      generator; an architecture problem
  R_scratch >> R_fitted  ->  the loss landscape, not expressivity, is what
      stops the optimiser

Acceptance criterion: a
generator must reach R_scratch within an order of magnitude of R_fitted on
at least one known symmetry.

Run:  python exp16_generator_acceptance.py [--targets d_x u_du] [--net both]
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import exp_common as C
import exp10_leak_vs_delta as E10
from config import ExperimentConfig
from exp6_trivial_subalgebra import ConstrainedGeneratorFNO
from exp15_rank_statistic import OracleHeatOperator
from method_fins import equivariance_loss
from method_fins_components import SingleGeneratorFNO, create_grids

L, ALPHA = E10.L, 0.01


def gt_field(target, u):
    """Analytic (xi, eta, phi) for a torus-essential heat symmetry."""
    z, o = torch.zeros_like(u), torch.ones_like(u)
    return {"d_x": (o, z, z), "d_t": (z, o, z), "u_du": (z, z, u)}[target]


def residual_of(S, xi, eta, phi, u, xb, tb, dx, dt, device):
    with torch.enable_grad():
        return float(equivariance_loss(
            S, xi, eta, phi, u[:, :, :, 0].contiguous(), u, xb, tb,
            dx, dt, device, mode="jvp", normalize=True))


def make_net(kind, cfg, device):
    return (SingleGeneratorFNO(cfg) if kind == "single"
            else ConstrainedGeneratorFNO(cfg)).to(device)


def net_fields(net, kind, xg, tg, u):
    out = net(xg, tg, u)
    return out[0], out[1], out[2]                    # both nets: xi, eta, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*",
                    default=["d_x", "d_t", "u_du"])
    ap.add_argument("--net", default="both",
                    choices=["single", "constrained", "both"])
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--ep-fit", type=int, default=150)
    ap.add_argument("--ep-scratch", type=int, default=150)
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.full_grid:
        cfg.grid_size, cfg.time_steps = 64, 128
    else:
        cfg.grid_size, cfg.time_steps = 32, 32
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ics, sols, xs, ts, _, _ = E10.make_dataset(
        args.n_train, cfg.grid_size, cfg.time_steps, rng)
    S = OracleHeatOperator(ALPHA, L).to(device)
    dx = (xs[0, 0, 1] - xs[0, 0, 0]).item()
    dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()
    loader = DataLoader(TensorDataset(sols, xs, ts),
                        batch_size=cfg.batch_size, shuffle=True)

    n_ev = min(16, args.n_train)
    u_ev = sols[:n_ev].to(device)
    xb_ev, tb_ev = xs[:n_ev].to(device), ts[:n_ev].to(device)
    xg_ev, tg_ev = create_grids(xb_ev, tb_ev, device)

    kinds = (["single", "constrained"] if args.net == "both"
             else [args.net])
    print("=" * 92)
    print(f"EXP 16 - generator acceptance (oracle operator, delta_J = 0, "
          f"grid {cfg.grid_size}x{cfg.time_steps}, device = {device})")
    print("=" * 92)

    rows, tab = [], []
    for target in args.targets:
        xi_a, eta_a, phi_a = gt_field(target, u_ev)
        r_analytic = residual_of(S, xi_a, eta_a, phi_a, u_ev, xb_ev, tb_ev,
                                 dx, dt, device)
        for kind in kinds:
            t0 = time.time()
            # --- supervised fit onto the analytic field ------------------
            torch.manual_seed(args.seed)
            net = make_net(kind, cfg, device)
            opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(args.ep_fit):
                for u_b, x_b, t_b in loader:
                    u = u_b.to(device)
                    xg, tg = create_grids(x_b.to(device), t_b.to(device),
                                          device)
                    tgt = gt_field(target, u)
                    pred = net_fields(net, kind, xg, tg, u)
                    loss = sum((p - g).pow(2).mean()
                               for p, g in zip(pred, tgt))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                pf = net_fields(net, kind, xg_ev, tg_ev, u_ev)
                num = sum((p - g).pow(2).sum()
                          for p, g in zip(pf, (xi_a, eta_a, phi_a)))
                den = sum(g.pow(2).sum() for g in (xi_a, eta_a, phi_a))
                eps_fit = float((num / den).sqrt())
            r_fitted = residual_of(S, *pf, u_ev, xb_ev, tb_ev, dx, dt,
                                   device)

            # --- equivariance training from scratch ----------------------
            torch.manual_seed(args.seed + 1)
            net2 = make_net(kind, cfg, device)
            opt2 = optim.AdamW(net2.parameters(), lr=cfg.lr_symmetry,
                               weight_decay=1e-4)
            for _ in range(args.ep_scratch):
                for u_b, x_b, t_b in loader:
                    u = u_b.to(device)
                    xg, tg = create_grids(x_b.to(device), t_b.to(device),
                                          device)
                    xi, eta, phi = net_fields(net2, kind, xg, tg, u)
                    l_eq = equivariance_loss(
                        S, xi, eta, phi, u[:, :, :, 0].contiguous(), u,
                        x_b.to(device), t_b.to(device), dx, dt, device,
                        mode="jvp")
                    v = torch.cat([xi, eta, phi], dim=1)
                    loss = l_eq + (v.pow(2).mean().sqrt() + 1e-8 - 1).pow(2)
                    opt2.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net2.parameters(), 5.0)
                    opt2.step()
            with torch.no_grad():
                ps = net_fields(net2, kind, xg_ev, tg_ev, u_ev)
            r_scratch = residual_of(S, *ps, u_ev, xb_ev, tb_ev, dx, dt,
                                    device)

            amp = r_fitted / max(eps_fit ** 2, 1e-30)
            tab.append([target, kind, f"{r_analytic:.2e}", f"{eps_fit:.3f}",
                        f"{r_fitted:.2e}", f"{amp:.2f}", f"{r_scratch:.2e}",
                        f"{time.time() - t0:.0f}s"])
            rows.append([target, kind, f"{r_analytic:.6e}",
                         f"{eps_fit:.6f}", f"{r_fitted:.6e}",
                         f"{r_scratch:.6e}"])
            print(f"  {target:>5} / {kind:<12} eps_fit = {eps_fit:.3f}  "
                  f"R_fit = {r_fitted:.2e}  R_scratch = {r_scratch:.2e}  "
                  f"(analytic {r_analytic:.1e})")

    print()
    print(C.fmt_table(tab, ["target", "net", "R_analytic", "eps_fit",
                            "R_fitted", "R_fit/eps^2", "R_scratch",
                            "time"]))
    print("\nAcceptance: R_scratch within ~10x of R_fitted on at least one "
          "target.\nIf eps_fit is large the parameterisation cannot "
          "represent the generator;\nif R_fitted >> R_analytic the residual "
          "amplifies field error.")

    path = os.path.join(C.RESULTS_DIR, "exp16_generator_acceptance.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "net", "R_analytic", "eps_fit", "R_fitted",
                    "R_scratch"])
        w.writerows(rows)
    print(f"CSV -> {path}")


if __name__ == "__main__":
    main()
