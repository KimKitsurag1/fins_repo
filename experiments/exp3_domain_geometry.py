"""Experiment 3 — domain geometry as a controlled parameter (heat, exact op).

The torus audit (exp1) shows the x-weighted heat generators fail on the
periodic domain and pass on a wide domain with localised data. This script
turns that dichotomy into a curve: fixed bump width w = 0.5, domain length
L swept over two decades, Rayleigh residual of Galilean / scaling /
projective recorded per L. Grid resolution is scaled with L so dx stays
constant; the operator is exact, so every number is pure geometry.

Outputs the L/w threshold at which each generator crosses the paper's 1e-2
discoverability threshold and the machine floor, plus a CSV for plotting.

Run:  python exp3_domain_geometry.py
"""

import csv
import os

import numpy as np

import exp_common as C
import argparse as _ap

NT, T, ALPHA, W = 128, 1.0, 0.01, 0.5
L_LIST = [2, 4, 8, 16, 32, 64, 128, 256]
N_SAMPLES = 8
THRESH = 1e-2


def main():
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    t = np.linspace(0, T, NT)
    names = ["Galilean", "scaling", "projective"]
    path = os.path.join(C.RESULTS_DIR, f"exp3_domain_geometry{('_s' + str(SEED)) if SEED else ''}.csv")
    rows_csv = []
    print("=" * 76)
    print(f"EXP 3 - heat, exact operator: residual vs domain length "
          f"(w = {W}, T = {T})")
    print("=" * 76)
    table = []
    for L in L_LIST:
        Nx = int(max(256, 16 * L))
        x = np.linspace(0, L, Nx, endpoint=False)
        u0 = C.ic_localised(N_SAMPLES, x, L, rng, width=(W, W),
                            amp=(0.3, 1.0))
        solve = lambda z: C.heat_solve(z, ALPHA, L, t)
        u = solve(u0)
        gens = C.gt_characteristics("heat", x, t, u, alpha=ALPHA)
        vals = []
        for n in names:
            xi, eta, phi, Q = gens[n]
            dSQ = solve(Q[:, :, 0])
            rv = C.rayleigh_residual(dSQ, Q, xi, eta, phi)
            Ntt = Q.shape[-1]
            sl = slice(2, Ntt - 2)
            rq = float(((dSQ[..., sl] - Q[..., sl]) ** 2).mean()
                       / ((Q[..., sl] ** 2).mean() + 1e-30))
            vals.append((rv, rq))
            rows_csv.append([L, L / W, n, f"{rv:.6e}", f"{rq:.6e}"])
        table.append([f"{L:g}", f"{L / W:g}"]
                     + [f"{rv:.2e}/{rq:.2e}" for rv, rq in vals])
    print(C.fmt_table(table, ["L", "L/w"] + [f"{n} (v/Q)" for n in names]))

    # control: same wide domain, bumps uniform over [0, L] as in the repo
    # family — data straddles the seam, so x-weighted residuals must NOT
    # collapse no matter how large L is.
    Lc, Nxc = 64, 1024
    xc = np.linspace(0, Lc, Nxc, endpoint=False)
    u0c = np.zeros((N_SAMPLES, Nxc))
    for i in range(N_SAMPLES):
        for _ in range(rng.integers(1, 4)):
            u0c[i] += rng.uniform(0.3, 1.0) * np.exp(
                -((xc - rng.uniform(0.0, Lc)) / W) ** 2)
    # deterministic worst case: one bump exactly on the seam (periodic image)
    d = np.minimum(xc, Lc - xc)
    u0c[0] += 0.7 * np.exp(-(d / W) ** 2)
    solvec = lambda z: C.heat_solve(z, ALPHA, Lc, t)
    uc = solvec(u0c)
    gensc = C.gt_characteristics("heat", xc, t, uc, alpha=ALPHA)
    print(f"\nControl at L = {Lc} (L/w = {Lc / W:g}) with bump positions "
          "uniform over [0, L] + one bump ON the seam (repo-style):")
    ctrl = []
    for n in names:
        xi, eta, phi, Q = gensc[n]
        dSQ = solvec(Q[:, :, 0])
        rv = C.rayleigh_residual(dSQ, Q, xi, eta, phi)
        ctrl.append((n, f"{rv:.3e}"))
        rows_csv.append([Lc, Lc / W, n + "_uniform_positions",
                         f"{rv:.6e}", ""])
    print(C.fmt_table(ctrl, ["generator", "resid (v-norm)"]))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "L_over_w", "generator", "resid_v_norm",
                    "resid_q_norm"])
        w.writerows(rows_csv)
    print(f"\nCSV -> {path}")

    # thresholds
    print(f"\nSmallest L/w with residual below {THRESH:g} "
          f"(repo discoverability threshold), v-norm:")
    for j, n in enumerate(names):
        found = next((f"{L / W:g}" for i, L in enumerate(L_LIST)
                      if float(rows_csv[i * 3 + j][3]) < THRESH), "none")
        print(f"  {n:>10}: L/w >= {found}")


if __name__ == "__main__":
    main()
