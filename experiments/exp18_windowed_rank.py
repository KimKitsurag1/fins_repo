"""Experiment 18 — what rank is detectable under the WINDOWED protocol?

The global residual on the torus can only see symmetries of the periodic
BOUNDARY VALUE PROBLEM (essential rank 3, exp1/exp17). But the paper's aim
is to recover symmetries of the underlying LOCAL law, of which there are
6/5/4. The seam-windowed residual tests exactly that, and is legitimate up
to the contamination the equation's domain of dependence lets into the
window (exp8).

This script produces the protocol comparison: for every
classical generator of every equation, the oracle residual with margins
m/L = 0, 0.05, 0.10, on grids that resolve the data.

Run:  python exp18_windowed_rank.py
"""
import csv, os
import numpy as np
import exp_common as C
import exp1_oracle_torus_audit as E1
import exp8_seam_window as E8

NT, THRESH = 128, 1e-2
MARG = [0.0, 0.05, 0.10]


def report(pde, x, t, u, solve, L, linear=False, ut=None, label=""):
    gens = C.gt_characteristics(pde, x, t, u, alpha=0.01, nu=0.01, ut=ut)
    if linear:
        dS = {n: solve(g[3][:, :, 0]) for n, g in gens.items()}
    else:
        q0 = {n: g[3][:, :, 0] for n, g in gens.items()}
        _, dS = E1.batched_fd_jvp_multi(solve, u[:, :, 0], q0)
    rows, det = [], {m: 0 for m in MARG}
    for n, (xi, eta, phi, Q) in gens.items():
        vals = [E8.windowed_residual(dS[n], Q, xi, eta, phi, x, L, m * L)
                for m in MARG]
        for m, v in zip(MARG, vals):
            if v < THRESH:
                det[m] += 1
        rows.append([n] + [f"{v:.2e}" for v in vals])
    print(f"\n  {label}")
    print("  " + C.fmt_table(rows, ["generator"] + [f"m/L={m}" for m in MARG])
          .replace("\n", "\n  "))
    print(f"  detected rank: " +
          ", ".join(f"m/L={m}: {det[m]}/{len(gens)}" for m in MARG))
    return det, len(gens)


def main():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("EXP 18 - detectable rank: global vs seam-windowed protocol")
    print("=" * 78)
    out = []

    # heat, the paper's own grid
    L, Nx, T = 2 * np.pi, 64, 1.0
    x = np.linspace(0, L, Nx, endpoint=False); t = np.linspace(0, T, NT)
    u0 = np.concatenate([C.ic_heat_gauss(8, x, L, rng),
                         C.ic_heat_sine(8, x, L, rng)])
    solve = lambda z: C.heat_solve(z, 0.01, L, t)
    d, n = report("heat", x, t, solve(u0), solve, L, linear=True,
                  label=f"HEAT  (paper grid Nx={Nx}, L=2pi, classical rank 6)")
    out.append(["heat", n] + [d[m] for m in MARG])

    # burgers, resolved grid, shorter horizon
    Nx, T = 512, 0.5
    x = np.linspace(0, L, Nx, endpoint=False); t = np.linspace(0, T, NT)
    u0 = C.ic_burgers_sine(6, x, rng)
    solve = lambda z: C.burgers_wide_solve(z, L, T, NT, nu=0.01, dt=2.5e-4)
    u = solve(u0)
    d, n = report("burgers", x, t, u, solve, L,
                  ut=C.discrete_ut("burgers", u, L, nu=0.01),
                  label=f"BURGERS (resolved Nx={Nx}, T={T}, classical rank 5)")
    out.append(["burgers", n] + [d[m] for m in MARG])

    # kdv, resolved torus
    Nx, T = 128, 1.0
    x = np.linspace(0, L, Nx, endpoint=False); t = np.linspace(0, T, NT)
    u0 = np.zeros((6, Nx))
    for i in range(6):
        for _ in range(rng.integers(1, 4)):
            u0[i] += rng.uniform(0.2, 1.5) / np.cosh(
                (x - rng.uniform(0.5, 5.5)) / rng.uniform(0.5, 1.0)) ** 2
    solve = lambda z: C.kdv_wide_solve(z, L, T, NT, dt=1e-5)
    u = solve(u0)
    d, n = report("kdv", x, t, u, solve, L, ut=C.discrete_ut("kdv", u, L),
                  label=f"KdV (resolved torus Nx={Nx}, T={T}, classical rank 4)")
    out.append(["kdv_torus", n] + [d[m] for m in MARG])

    # kdv, wide domain + short horizon (the escape route of exp12)
    Lw, Nxw, Tw = 32 * np.pi, 512, 0.2
    xw = np.linspace(0, Lw, Nxw, endpoint=False); tw = np.linspace(0, Tw, NT)
    u0w = C.ic_localised(6, xw, Lw, rng, width=(0.6, 1.2), amp=(0.3, 1.0),
                         sech=True)
    solvew = lambda z: C.kdv_wide_solve(z, Lw, Tw, NT, dt=5e-5)
    uw = solvew(u0w)
    d, n = report("kdv", xw, tw, uw, solvew, Lw,
                  ut=C.discrete_ut("kdv", uw, Lw),
                  label=f"KdV (wide L=32pi, short T={Tw}, classical rank 4)")
    out.append(["kdv_wide_shortT", n] + [d[m] for m in MARG])

    print("\n" + "=" * 78)
    print(C.fmt_table([[r[0], str(r[1])] + [str(v) for v in r[2:]]
                       for r in out],
                      ["setting", "classical rank"]
                      + [f"detected m/L={m}" for m in MARG]))
    path = os.path.join(C.RESULTS_DIR, "exp18_windowed_rank.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "classical_rank"]
                                      + [f"detected_m{m}" for m in MARG])
        w.writerows(out)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
