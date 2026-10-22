"""Experiment 25 — how much of the Burgers/KdV failure is the DATA?

Before improving the method on the two nonlinear equations, measure the
floor the data itself imposes. At the campaign grid (Nx = 64, nu = 0.01)
a Burgers shock is ~0.01 wide against a spacing of ~0.1, so the discrete
field does not satisfy the symmetry condition to any useful accuracy: the
ground-truth generators themselves fail the residual test, and no method
scoring against this data can do better than that.

For each resolution the equation is solved with the exact spectral
integrator, the GT characteristics are built, and the oracle residual
||dS[u0](Q0) - Q|| / ||Q|| is evaluated by finite-difference JVP through
the exact operator -- globally (m = 0) and with a seam margin (m = 0.1L).
The number of generators
passing the papers' 1e-2 threshold is the "data floor rank" at that Nx.

Run:  python exp25_resolution_floor.py
"""

import numpy as np

import exp_common as C
import exp1_oracle_torus_audit as E1
import exp8_seam_window as E8

NT, THRESH, L = 128, 1e-2, 2 * np.pi


def audit(pde, x, t, u, solve, margins, ut):
    gens = C.gt_characteristics(pde, x, t, u, alpha=0.01, nu=0.01, ut=ut)
    q0 = {n: g[3][:, :, 0] for n, g in gens.items()}
    _, dS = E1.batched_fd_jvp_multi(solve, u[:, :, 0], q0)
    rows = {}
    for n, (xi, eta, phi, Q) in gens.items():
        rows[n] = [E8.windowed_residual(dS[n], Q, xi, eta, phi, x, L, m * L)
                   for m in margins]
    return rows


def main():
    rng = np.random.default_rng(0)
    margins = [0.0, 0.10]
    print("=" * 84)
    print("EXP 25 - data-resolution floor: oracle residuals of the GT "
          "generators vs Nx")
    print("=" * 84)

    print("\nBURGERS  nu = 0.01, L = 2pi, T = 1.0 (the campaign horizon), "
          "6 samples")
    for Nx in (64, 128, 256, 512):
        x = np.linspace(0, L, Nx, endpoint=False)
        t = np.linspace(0, 1.0, NT)
        rng = np.random.default_rng(0)          # SAME ICs at every Nx
        u0 = C.ic_burgers_sine(6, x, rng)
        dt = min(2.5e-4, 0.2 * (L / Nx) ** 2 / 0.01)
        solve = lambda z, dt=dt: C.burgers_wide_solve(z, L, 1.0, NT, nu=0.01,
                                                      dt=dt)
        u = solve(u0)
        rows = audit("burgers", x, t, u, solve, margins,
                     C.discrete_ut("burgers", u, L, nu=0.01))
        passed = {m: sum(1 for r in rows.values() if r[i] < THRESH)
                  for i, m in enumerate(margins)}
        print(f"  Nx = {Nx:>3}: "
              + "  ".join(f"{n} {r[0]:.1e}/{r[1]:.1e}"
                          for n, r in rows.items())
              + f"   | pass 1e-2: global {passed[0.0]}/5, "
                f"m=0.1L {passed[0.10]}/5")

    print("\nKdV  L = 2pi, T = 1.0, torus, 6 sech-type samples")
    for Nx in (64, 128):
        x = np.linspace(0, L, Nx, endpoint=False)
        t = np.linspace(0, 1.0, NT)
        rng = np.random.default_rng(1)          # SAME ICs at every Nx
        u0 = np.zeros((6, Nx))
        for i in range(6):
            for _ in range(rng.integers(1, 4)):
                u0[i] += rng.uniform(0.2, 1.5) / np.cosh(
                    (x - rng.uniform(0.5, 5.5)) / rng.uniform(0.5, 1.0)) ** 2
        dt = min(1e-5, 0.05 * (L / Nx) ** 3)
        solve = lambda z, dt=dt: C.kdv_wide_solve(z, L, 1.0, NT, dt=dt)
        u = solve(u0)
        rows = audit("kdv", x, t, u, solve, margins,
                     C.discrete_ut("kdv", u, L))
        passed = {m: sum(1 for r in rows.values() if r[i] < THRESH)
                  for i, m in enumerate(margins)}
        print(f"  Nx = {Nx:>3}: "
              + "  ".join(f"{n} {r[0]:.1e}/{r[1]:.1e}"
                          for n, r in rows.items())
              + f"   | pass 1e-2: global {passed[0.0]}/4, "
                f"m=0.1L {passed[0.10]}/4")
    print("\nformat: generator global/windowed. A generator that fails here "
          "cannot be recovered by ANY method from this data.")


if __name__ == "__main__":
    main()
