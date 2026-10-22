"""Experiment 26 — does joint surrogate + equivariance training manufacture
symmetries?

Proposal under test: alternate discovery and surrogate refinement, adding
the Olver/equivariance residual of the discovered generators to the
surrogate loss. Danger: the data do not pin the surrogate's Jacobian (L2
1.7 percent with an O(1) Galilean residual on Burgers), so many surrogates
share the same L2 and differ in symmetry; a joint objective picks the one
most symmetric under the CURRENT generators -- including spurious ones from
early rounds -- and the data term cannot object in flat directions.

Design. Small Burgers in a genuinely nonlinear regime (amplitude 2,
nu = 0.02; the exact operator gives a relative residual of 1.24 for u d_u
and 0 for Galilean). Phase 1 trains the repo surrogate to plateau. Phase 2
branches from that checkpoint into arms of equal length: MSE only
(control), and MSE + lambda * R(u d_u) with u d_u standing in for a
spurious early discovery. Read-outs: relative L2, R(u d_u), R(Galilean).

Result (CPU, one seed, 150 steps per arm):
    checkpoint           L2 0.240   R(u d_u) 0.299   R(Galilean) 0.421
    control (MSE only)      0.208            0.311               0.394
    joint lambda = 0.1      0.207            0.240               0.400
    joint lambda = 1.0      0.227            0.174               0.420
The injected non-symmetry is manufactured (residual halved at lambda = 1,
and at lambda = 0.1 at NO L2 cost: 0.207 vs 0.208), while the true
Galilean symmetry does not move. An EM-style loop would therefore bake in
whatever the first round found. The plain surrogate is also
already 4x more "u d_u-symmetric" than the exact operator (0.30 vs 1.24)
before any joint training: the Jacobian is under-constrained by the data.

Run:  python exp26_joint_training.py            (~2 min CPU)
"""

import io
import contextlib

import numpy as np
import torch

import exp_common as C
import exp1_oracle_torus_audit as E1

import sys
sys.path.insert(0, "../pde")
from config import ExperimentConfig                       # noqa: E402
from method_fins import equivariance_loss, train_solution_operator  # noqa: E402

L, T, NX, NT, N, NU = 2 * np.pi, 0.5, 32, 32, 16, 0.02
STEPS, LR = 150, 5e-4


def main():
    torch.manual_seed(0)
    cfg = ExperimentConfig()
    cfg.device = "cpu"
    cfg.grid_size, cfg.time_steps, cfg.batch_size = NX, NT, 2
    cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    rng = np.random.default_rng(0)
    u0 = 2.0 * C.ic_burgers_sine(N, x, rng) + rng.uniform(-0.5, 0.5, (N, 1))
    solve = lambda z: C.burgers_wide_solve(z, L, T, NT, nu=NU, dt=1e-4)
    u = solve(u0)

    q0 = {"u d_u": u[:, :, 0], "Galilean": np.ones_like(u[:, :, 0])}
    _, dS = E1.batched_fd_jvp_multi(solve, u[:, :, 0], q0)
    Q_gal = 1.0 - t[None, None, :] * C.dx_spectral(u, L)
    for nm, Q in (("u d_u", u), ("Galilean", Q_gal)):
        print(f"exact operator: R({nm}) = "
              f"{np.sqrt(np.mean((dS[nm] - Q) ** 2) / np.mean(Q ** 2)):.3f}")

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
        for p in S.parameters():
            p.requires_grad_(True)
        r_u = float(equivariance_loss(S, Z, Z, sols, ics, sols, xs, ts, dx, dt,
                                      "cpu", mode="jvp", channel_norm="q"))
        r_g = float(equivariance_loss(S, *gal, ics, sols, xs, ts, dx, dt,
                                      "cpu", mode="jvp", channel_norm="q"))
        with torch.no_grad():
            mse = float((S(ics, xs, ts) - sols).pow(2).mean())
        return np.sqrt(mse / ref), np.sqrt(r_u), np.sqrt(r_g)

    with contextlib.redirect_stdout(io.StringIO()):
        S0 = train_solution_operator(ics, sols, xs, ts, cfg, "cpu",
                                     epochs=900, verbose=False)
    sd = {k: v.clone() for k, v in S0.state_dict().items()}
    sd.pop("_metadata", None)

    def arm(lam):
        torch.manual_seed(1)
        S = type(S0)(cfg)
        S.load_state_dict(sd)
        for p in S.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(S.parameters(), lr=LR)
        for _ in range(STEPS):
            loss = (S(ics, xs, ts) - sols).pow(2).mean()
            if lam > 0:
                loss = loss + lam * equivariance_loss(
                    S, Z, Z, sols, ics, sols, xs, ts, dx, dt, "cpu",
                    mode="jvp", channel_norm="q")
            opt.zero_grad()
            loss.backward()
            opt.step()
        return probe(S)

    print(f"\n{'arm':<28}{'rel L2':>9}{'R(u d_u) spurious':>20}"
          f"{'R(Galilean) true':>18}")
    print("-" * 75)
    e, ru, rg = probe(S0)
    print(f"{'checkpoint':<28}{e:>9.3f}{ru:>20.3f}{rg:>18.3f}")
    for lam, label in ((0.0, "control: MSE only"), (0.1, "joint lambda=0.1"),
                       (1.0, "joint lambda=1.0")):
        e, ru, rg = arm(lam)
        print(f"{label:<28}{e:>9.3f}{ru:>20.3f}{rg:>18.3f}")


if __name__ == "__main__":
    main()
