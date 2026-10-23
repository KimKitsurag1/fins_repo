"""Experiment 13 — the Burgers projective generator of the paper is wrong.

Claim
-----
The paper (Section 5.1) and `pde/ground_truth.py` list the fifth Burgers
generator as

    v5 = x t d_x + t^2 d_t - (x + t u) d_u          (repo variant)

The correct projective symmetry of u_t + u u_x = nu u_xx is

    v5 = x t d_x + t^2 d_t + (x - t u) d_u          (corrected)

Proof (three lines). A vector field with characteristic Q is a symmetry iff
Q solves the linearised equation M[Q] = Q_t + (u Q)_x - nu Q_xx = 0 on
solutions. For the corrected variant, Q = (x - t u) - x t u_x - t^2 u_t and
direct substitution collapses to

    M[Q] = -3 t E - x t (d_x E) - t^2 (d_t E) = 0,   E := u_t + u u_x - nu u_xx,

identically on solutions, for every nu. The repo variant differs by
Q_repo = Q - 2x, and M is linear with M[x] = u + x u_x, so

    M[Q_repo] = -2 (u + x u_x)  !=  0.

The defect is O(u): the repo v5 is not a symmetry of Burgers at all.

Consequences for the paper: the Burgers ground-truth subspace in Table 1 is
partly wrong for ALL six methods; the v5 column of Table 3 tracked a
non-symmetry, so its plateau above the threshold under IC enrichment was
guaranteed a priori and says nothing about orbit coverage (the v3 story is
unaffected — v3 is correct); the windowed test of exp8/exp12 could not
rescue v5 for the same reason, and does rescue the corrected generator.

This script verifies all of it numerically:
  A. determining-equation residual with LOCAL finite differences on a wide
     domain (no seam, no spectral Gibbs), two viscosities;
  B. the discrete defect matches -2(u + x u_x) pointwise;
  C. windowed JVP test on the torus: corrected v5 reaches the scaling
     control's floor, repo v5 stays O(0.1) at every margin.

Run:  python exp13_projective_correction.py
"""

import csv
import os

import numpy as np

import exp_common as C
import exp1_oracle_torus_audit as E1
import exp8_seam_window as E8
import argparse as _ap

NT = 128


def fd6_axis(u, h, axis):
    """6th-order central first derivative; 3-point borders left zero."""
    c = np.array([-1., 9., -45., 0., 45., -9., 1.]) / (60. * h)
    acc = np.zeros_like(u)
    n = u.shape[axis]
    for j, cj in enumerate(c):
        if cj == 0.0:
            continue
        s = [slice(None)] * u.ndim
        s[axis] = slice(j, n - 6 + j)
        d = [slice(None)] * u.ndim
        d[axis] = slice(3, n - 3)
        acc[tuple(d)] += cj * u[tuple(s)]
    return acc


def determining_residual(u, x, t, nu, cands, xw, tw):
    """|M[Q]|^2 / |Q|^2 on the window, all derivatives local FD."""
    dx, dt = x[1] - x[0], t[1] - t[0]
    ux = fd6_axis(u, dx, 1)
    ut = fd6_axis(u, dt, 2)
    X = np.broadcast_to(x[None, :, None], u.shape)
    Tm = np.broadcast_to(t[None, None, :], u.shape)
    out = {}
    for name, mk in cands.items():
        xi, eta, phi = mk(X, Tm, u)
        Q = phi - xi * ux - eta * ut
        Z = (fd6_axis(Q, dt, 2) + fd6_axis(u * Q, dx, 1)
             - nu * fd6_axis(fd6_axis(Q, dx, 1), dx, 1))
        num = (Z[:, xw][..., tw] ** 2).mean()
        den = (Q[:, xw][..., tw] ** 2).mean() + 1e-30
        out[name] = (num / den, Q, Z)
    return out, ux


CANDS = {
    "d_x (floor)": lambda X, T, u: (np.ones_like(u), np.zeros_like(u),
                                    np.zeros_like(u)),
    "scaling (control)": lambda X, T, u: (X + 0 * u, 2 * T + 0 * u, -u),
    "repo v5: phi = -(x+tu)": lambda X, T, u: (X * T + 0 * u, T ** 2 + 0 * u,
                                               -(X + T * u)),
    "corrected v5: phi = x-tu": lambda X, T, u: (X * T + 0 * u,
                                                 T ** 2 + 0 * u, X - T * u),
}


def main():
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    rows_csv = []

    # ---------------- A + B: determining equation, wide domain -----------
    L, Nx, T = 16 * np.pi, 1024, 1.0
    x = np.linspace(0, L, Nx, endpoint=False)
    t = np.linspace(0, T, NT)
    xw = (x >= 0.30 * L) & (x <= 0.70 * L)
    tw = slice(8, NT - 8)
    print("=" * 82)
    print("EXP 13A - determining equation M[Q] = Q_t + (uQ)_x - nu Q_xx, "
          "local FD, wide domain")
    print(f"L = {L:.1f}, Nx = {Nx}, window x in [0.3, 0.7] L")
    print("=" * 82)
    for nu in (0.01, 0.1):
        u0 = C.ic_localised(6, x, L, rng, width=(0.8, 1.6), amp=(0.1, 0.3))
        u = C.burgers_wide_solve(u0, L, T, NT, nu=nu, dt=5e-4)
        res, ux = determining_residual(u, x, t, nu, CANDS, xw, tw)
        print(f"\n  nu = {nu}")
        for name, (val, Q, Z) in res.items():
            print(f"    {name:<26} |M[Q]|^2/|Q|^2 = {val:.3e}")
            rows_csv.append(["determining", f"{nu}", name, f"{val:.6e}"])
        # B: defect identity for the repo variant
        _, _, Zr = res["repo v5: phi = -(x+tu)"]
        X = np.broadcast_to(x[None, :, None], u.shape)
        pred = -2.0 * (u + X * ux)
        rel = (np.linalg.norm((Zr - pred)[:, xw][..., tw])
               / (np.linalg.norm(pred[:, xw][..., tw]) + 1e-30))
        print(f"    defect check: ||M[Q_repo] + 2(u + x u_x)|| / "
              f"||2(u + x u_x)|| = {rel:.2e}")
        rows_csv.append(["defect_identity", f"{nu}", "repo v5",
                         f"{rel:.6e}"])

    # ---------------- C: windowed JVP on the torus -----------------------
    L2, Nx2, NU = 2 * np.pi, 512, 0.01
    x2 = np.linspace(0, L2, Nx2, endpoint=False)
    u0 = C.ic_burgers_sine(6, x2, rng)
    print("\n" + "=" * 82)
    print("EXP 13C - windowed JVP test on the torus "
          f"(L = {L2:.3f}, Nx = {Nx2}, nu = {NU})")
    print("=" * 82)
    for T2 in (0.5, 1.0):
        t2 = np.linspace(0, T2, NT)
        solve = lambda z: C.burgers_wide_solve(z, L2, T2, NT, nu=NU,
                                               dt=2.5e-4)
        u = solve(u0)
        ux = C.dx_spectral(u, L2)
        ut = C.discrete_ut("burgers", u, L2, nu=NU)
        X = np.broadcast_to(x2[None, :, None], u.shape)
        Tm = np.broadcast_to(t2[None, None, :], u.shape)
        gens = {
            "repo v5": (X * Tm, Tm ** 2, -(X + Tm * u)),
            "corrected v5": (X * Tm, Tm ** 2, X - Tm * u),
            "scaling (control)": (X + 0 * u, 2 * Tm + 0 * u, -u),
        }
        full = {n: (xi, eta, phi, phi - xi * ux - eta * ut)
                for n, (xi, eta, phi) in gens.items()}
        q0s = {n: g[3][:, :, 0] for n, g in full.items()}
        _, dS = E1.batched_fd_jvp_multi(solve, u0, q0s)
        tab = []
        for m in (0.0, 0.1, 0.2):
            row = [f"{m:.1f}"]
            for n, (xi, eta, phi, Q) in full.items():
                v = E8.windowed_residual(dS[n], Q, xi, eta, phi, x2, L2,
                                         m * L2)
                row.append(f"{v:.2e}")
                rows_csv.append(["windowed_jvp", f"T={T2},m={m}", n,
                                 f"{v:.6e}"])
            tab.append(row)
        print(f"\n  T = {T2}")
        print("  " + C.fmt_table(tab, ["m/L"] + list(full))
              .replace("\n", "\n  "))

    path = os.path.join(C.RESULTS_DIR, f"exp13_projective_correction{('_s' + str(SEED)) if SEED else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "setting", "candidate", "value"])
        w.writerows(rows_csv)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
