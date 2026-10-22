"""Experiment 7 — locality of the generator parameterisation.

A Lie POINT symmetry has coefficients that are functions of
(x, t, u) only, but `SingleGeneratorFNO` maps the whole space-time field u
through spectral layers, so xi(x, t) can depend on u everywhere — the
hypothesis space is generalised (nonlocal) vector fields.

This script measures the sensitivity row J(x') = d out_c(x_c, t_c) /
d u(x', t_c) by autograd, for the xi and phi output channels, on three
networks: SingleGeneratorFNO at random init, the same network after a short
equivariance-loss fit, and a pointwise 1x1-conv MLP (the architecture a
point symmetry actually requires; delta-like by construction).

Caveat: `SpectralConv2d` initialises its spectral
weights with scale = 1/(in_ch*out_ch), which at width 16 makes the spectral
path ~an order weaker than the local 1x1 path — so the RANDOM-INIT map is
near-delta and proves nothing. The architectural statement requires the
TRAINED map, which is what the quick-train stage provides.

Run:  python exp7_receptive_field.py [--checkpoint path.pt]
"""

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
from method_fins import train_solution_operator, equivariance_loss
from method_fins_components import SingleGeneratorFNO, create_grids


class PointwiseGeneratorMLP(nn.Module):
    """(x, t, u) -> (xi, eta, phi) applied independently at every point."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1))

    def forward(self, x_grid, t_grid, u):
        out = self.net(torch.cat([x_grid, t_grid, u], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def sensitivity_row(gen, Nx, Nt, device, channel, seed=0):
    torch.manual_seed(seed)
    x = torch.linspace(0, 1, Nx).view(1, 1, Nx).to(device)
    t = torch.linspace(0, 1, Nt).view(1, 1, Nt).to(device)
    xg, tg = create_grids(x, t, device)
    base = torch.randn(1, 1, Nx, 1) * torch.hann_window(Nx).view(1, 1, Nx, 1)
    u = (base.expand(1, 1, Nx, Nt) * torch.linspace(1, 0.5, Nt)).clone() \
        .to(device).requires_grad_(True)
    out = gen(xg, tg, u)[channel]
    s = out[0, 0, Nx // 2, Nt // 2]
    (J,) = torch.autograd.grad(s, u)
    row = J[0, 0, :, Nt // 2].abs().cpu().numpy()
    raw = float(row.max())
    row = row / (row.max() + 1e-30)
    width10 = int((row >= 0.1).sum())
    pr = float(row.sum() ** 2 / (Nx * (row ** 2).sum() + 1e-30))
    return row, width10, pr, raw


def quick_train_generator(cfg, device, seed):
    torch.manual_seed(seed)
    ics, sols, xs, ts = generate_pde_data("heat", cfg, 24, seed=seed)
    S = train_solution_operator(ics, sols, xs, ts, cfg, device,
                                epochs=12, verbose=False)
    gen = SingleGeneratorFNO(cfg).to(device)
    opt = optim.AdamW(gen.parameters(), lr=cfg.lr_symmetry)
    loader = DataLoader(TensorDataset(sols, xs, ts),
                        batch_size=cfg.batch_size, shuffle=True)
    dx = (xs[0, 0, 1] - xs[0, 0, 0]).item()
    dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()
    for ep in range(8):
        for u_b, x_b, t_b in loader:
            u = u_b.to(device)
            xg, tg = create_grids(x_b.to(device), t_b.to(device), device)
            xi, eta, phi = gen(xg, tg, u)
            l_eq = equivariance_loss(
                S, xi, eta, phi, u[:, :, :, 0].contiguous(), u,
                x_b.to(device), t_b.to(device), dx, dt, device, mode="jvp")
            v = torch.cat([xi, eta, phi], dim=1)
            loss = l_eq + (v.pow(2).mean().sqrt() - 1.0).pow(2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    cfg.grid_size, cfg.time_steps = 32, 32
    cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
    Nx, Nt = cfg.grid_size, cfg.time_steps

    fno_rand = SingleGeneratorFNO(cfg).to(device)
    if args.checkpoint:
        fno_tr = SingleGeneratorFNO(cfg).to(device)
        fno_tr.load_state_dict(torch.load(args.checkpoint,
                                          map_location=device))
        tag = f"checkpoint {args.checkpoint}"
    else:
        fno_tr = quick_train_generator(cfg, device, args.seed)
        tag = "quick-trained, 8 ep"
    mlp = PointwiseGeneratorMLP().to(device)

    print("=" * 74)
    print(f"EXP 7 - receptive field of out(x_c, t_c) w.r.t. u(x', t_c)  "
          f"(Nx={Nx})")
    print("=" * 74)
    rows = []
    for name, gen in [("SingleGeneratorFNO (random init)", fno_rand),
                      (f"SingleGeneratorFNO ({tag})", fno_tr),
                      ("Pointwise 1x1 MLP", mlp)]:
        for ch, ch_name in [(0, "xi"), (2, "phi")]:
            prof, w10, pr, raw = sensitivity_row(gen, Nx, Nt, device, ch)
            rows.append((name, ch_name, f"{w10}/{Nx}", f"{pr:.3f}",
                         f"{raw:.1e}"))
            marks = "".join("#" if v >= 0.1 else ("+" if v >= 0.01 else ".")
                            for v in prof)
            print(f"\n  {name}, d {ch_name}/d u\n  |J| profile: {marks}")
    print()
    print(C.fmt_table(rows, ["generator net", "out", "width at 10%",
                             "participation", "max|J|"]))
    print("\nA point-symmetry parameterisation must be delta-like "
          "(width 1, PR -> 1/Nx = "
          f"{1.0 / Nx:.3f}); large width on the trained FNO means the "
          "hypothesis space being searched is generalised (nonlocal) "
          "vector fields.")


if __name__ == "__main__":
    main()
