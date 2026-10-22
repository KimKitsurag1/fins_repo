"""Experiment 28 -- does a TIME-STEPPING surrogate carry the Galilean symmetry?

Measured bottleneck on Burgers: the one-shot surrogate u0 -> u(., t) cannot
tell the true Galilean generator from the non-symmetry u d_u (probe 0.81 vs
0.86), and tangent training did not change that (exp27). Hypothesis: the
one-shot map must learn how the MEAN of u0 sets the advection speed over
the whole horizon at once; a stepper u(t) -> u(t + dt) only has to learn it
over one short step, and Galilean covariance of a single step,
step(u + c)(x) = step(u)(x - c dt) + c, is a local statement a
translation-equivariant network can represent. The trajectory Jacobian is
then the product of step Jacobians, which jvp differentiates exactly.

Same data, seeds and scoring as exp26/27: small nonlinear Burgers, 16 ICs,
||Q||-normalised equivariance residuals through each surrogate. Exact
operator: R(Galilean) = 0, R(u d_u) = 1.24.

Run:  python exp28_timestepper.py            (~3 min CPU)
"""

import contextlib
import io
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO

sys.path.insert(0, "../pde")
from config import ExperimentConfig                                   # noqa: E402
from data_generation import PDESolver                                 # noqa: E402
from method_fins import equivariance_loss, train_solution_operator    # noqa: E402

L, NX, NT, N, NU, T = 2 * np.pi, 32, 32, 16, 0.02, 0.5


class Stepper(nn.Module):
    """u(t) -> u(t + dt) as u + F(u); F is a 1-D FNO with no coordinate
    channel, so every step is translation-equivariant by construction."""

    def __init__(self, modes=12, width=32, layers=3):
        super().__init__()
        self.f = FNO(n_modes=(modes,), hidden_channels=width, in_channels=1,
                     out_channels=1, n_layers=layers)

    def forward(self, u):
        return u + self.f(u)


class RolloutOperator(nn.Module):
    """Wraps a stepper in the (u0, x_coords, t_coords) -> [B,1,Nx,Nt]
    signature the FINS loss expects."""

    def __init__(self, stepper, nt):
        super().__init__()
        self.step, self.nt = stepper, nt

    def forward(self, u0, x_coords=None, t_coords=None):
        u = u0 if u0.dim() == 3 else u0.unsqueeze(1)
        out = [u]
        for _ in range(self.nt - 1):
            u = self.step(u)
            out.append(u)
        return torch.stack(out, dim=-1)


def train_stepper(sols, epochs=400, unroll=4, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    stp = Stepper()
    opt = torch.optim.Adam(stp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, _, nx, nt = sols.shape
    for ep in range(epochs):
        j0 = torch.randint(0, nt - unroll, (n,))
        u = sols[torch.arange(n), :, :, j0]
        loss = 0.0
        for k in range(1, unroll + 1):
            u = stp(u)
            loss = loss + (u - sols[torch.arange(n), :, :, j0 + k]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return stp


def main():
    cfg = ExperimentConfig()
    cfg.device = "cpu"
    cfg.grid_size, cfg.time_steps, cfg.batch_size = NX, NT, 2
    cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
    cfg.nu_burgers, cfg.T_final = NU, T
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    rng = np.random.default_rng(0)
    u0 = 2.0 * np.sin(x)[None] * rng.uniform(.5, 1.5, (N, 1)) \
        + rng.uniform(-.5, .5, (N, 1)) \
        + 0.5 * np.cos(2 * x + rng.uniform(0, 6.28, (N, 1)))
    u = np.asarray(PDESolver.solve_burgers(u0, NU, T, NT, verbose=False))
    Tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    ics, sols = Tt(u0).view(N, 1, NX), Tt(u).view(N, 1, NX, NT)
    xs = Tt(np.tile(x, (N, 1))).view(N, 1, NX)
    ts = Tt(np.tile(t, (N, 1))).view(N, 1, NT)
    dx, dt = xs[:, 0, 1] - xs[:, 0, 0], ts[:, 0, 1] - ts[:, 0, 0]
    Z = torch.zeros_like(sols)
    _, Tm = np.meshgrid(x, t, indexing="ij")
    gal = (Tt(np.tile(Tm, (N, 1, 1, 1))).view(N, 1, NX, NT), Z,
           torch.ones_like(sols))
    ref = float(sols.pow(2).mean())

    def probe(S):
        S.eval()
        r_u = float(equivariance_loss(S, Z, Z, sols, ics, sols, xs, ts, dx, dt,
                                      "cpu", mode="jvp", channel_norm="q"))
        r_g = float(equivariance_loss(S, *gal, ics, sols, xs, ts, dx, dt,
                                      "cpu", mode="jvp", channel_norm="q"))
        with torch.no_grad():
            e = float((S(ics, xs, ts) - sols).pow(2).mean())
        return np.sqrt(e / ref), np.sqrt(r_g), np.sqrt(r_u)

    print(f"{'surrogate':<34}{'rel L2':>8}{'R(Galilean) TRUE':>18}"
          f"{'R(u d_u) SPURIOUS':>19}{'time':>8}")
    print("-" * 87)
    t0 = time.time()
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        S1 = train_solution_operator(ics, sols, xs, ts, cfg, "cpu",
                                     epochs=700, verbose=False)
    e, rg, ru = probe(S1)
    print(f"{'one-shot FNO u0 -> u(.,t) (repo)':<34}{e:>8.3f}{rg:>18.3f}"
          f"{ru:>19.3f}{time.time() - t0:>7.0f}s", flush=True)
    t0 = time.time()
    S2 = RolloutOperator(train_stepper(sols), NT)
    e, rg, ru = probe(S2)
    print(f"{'stepper u(t) -> u(t+dt), rollout':<34}{e:>8.3f}{rg:>18.3f}"
          f"{ru:>19.3f}{time.time() - t0:>7.0f}s", flush=True)
    print(f"{'exact operator':<34}{'-':>8}{0.0:>18.3f}{1.244:>19.3f}")


if __name__ == "__main__":
    main()
