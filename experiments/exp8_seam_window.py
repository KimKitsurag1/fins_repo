"""Experiment 8 — escaping the torus without leaving the torus:
the seam-windowed (local) equivariance test.

Motivation
----------
The x-weighted generators fail on the torus for one reason only: xi = x is
not a periodic function, so Q0 carries a jump at the seam x = 0 ~ L. That
jump is a boundary artefact of the coordinate chart, not a statement about
the PDE. Lie POINT symmetries are a local property of L[u]; the FINS
condition is global only because S is global.

So: evaluate the residual on a window W = [m, L - m] that excludes the seam.
The test is then valid up to the contamination that the seam jump propagates
INTO the window within the horizon T, which is governed by the equation's
domain of dependence:

  heat      parabolic, Green kernel exp(-d^2 / 4 alpha t)
            -> contamination ~ exp(-m^2 / 4 alpha T), exponentially small
  Burgers   characteristic speed ~ max|u| plus nu-diffusion
            -> contamination shut off for m > max|u| T, algebraically
  KdV       dispersive, group velocity ~ -3 k^2 unbounded on the grid
            -> NO finite domain of dependence, windowing cannot work

This script measures the windowed residual against the margin m, so the
predicted hierarchy heat > Burgers > KdV is tested rather than asserted.

Run:  python exp8_seam_window.py
"""

import csv
import os

import numpy as np

import exp_common as C

NT, T, ALPHA, NU = 128, 1.0, 0.01, 0.01
THRESH = 1e-2
XW_HEAT = ["Galilean", "scaling", "projective"]
XW_BURG = ["scaling", "projective"]
XW_KDV = ["scaling"]


def windowed_residual(dSQ, Q, xi, eta, phi, x, L, m, t_margin=2):
    """Rayleigh residual restricted to x in [m, L - m]."""
    Nt = Q.shape[-1]
    ts = slice(t_margin, Nt - t_margin) if Nt > 2 * t_margin else slice(None)
    xs = (x >= m) & (x <= L - m)
    if xs.sum() < 8:
        return np.nan
    r = (dSQ[:, xs][..., ts] - Q[:, xs][..., ts])
    v = (xi ** 2 + eta ** 2 + phi ** 2)[:, xs][..., ts]
    return float(((r ** 2).mean(axis=(1, 2)) / (v.mean(axis=(1, 2)) + 1e-6))
                 .mean())


def sweep(pde, x, t, u, solve, names, L, margins, linear=False, ut=None):
    gens = C.gt_characteristics(pde, x, t, u, alpha=ALPHA, nu=NU, ut=ut)
    if linear:
        dS = {n: solve(g[3][:, :, 0]) for n, g in gens.items()}
    else:
        q0s = {n: g[3][:, :, 0] for n, g in gens.items()}
        B = u.shape[0]
        dS = {}
        for n, q0 in q0s.items():
            dS[n] = C.fd_jvp(solve, u[:, :, 0], q0)
    rows = []
    for m in margins:
        vals = []
        for n in names:
            xi, eta, phi, Q = gens[n]
            vals.append(windowed_residual(dS[n], Q, xi, eta, phi, x, L, m))
        rows.append((m, vals))
    return rows


def report(title, rows, names, L, diff_len=None):
    print(f"\n  {title}")
    tab = []
    for m, vals in rows:
        cells = [f"{m:.2f}", f"{m / L:.3f}"]
        if diff_len:
            cells.append(f"{m / diff_len:.1f}")
        cells += [f"{v:.2e}" for v in vals]
        tab.append(cells)
    head = ["margin m", "m/L"] + (["m/l_diff"] if diff_len else []) + names
    print("  " + C.fmt_table(tab, head).replace("\n", "\n  "))


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    t = np.linspace(0, T, NT)
    out_rows = []
    print("=" * 78)
    print("EXP 8 - seam-windowed equivariance test on the torus")
    print(f"threshold = {THRESH:g}; full-domain values are the m = 0 row")
    print("=" * 78)

    # ---- heat, exact operator, the paper's own grid --------------------
    for L, Nx in [(2 * np.pi, 64), (1.0, 64)]:
        x = np.linspace(0, L, Nx, endpoint=False)
        u0 = np.concatenate([C.ic_heat_gauss(8, x, L, rng),
                             C.ic_heat_sine(8, x, L, rng)])
        solve = lambda z: C.heat_solve(z, ALPHA, L, t)
        u = solve(u0)
        l_diff = np.sqrt(4 * ALPHA * T)
        margins = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]) * L
        rows = sweep("heat", x, t, u, solve, XW_HEAT, L, margins, linear=True)
        report(f"HEAT torus L = {L:.3f}, Nx = {Nx}, repo ICs "
               f"(diffusion length sqrt(4 alpha T) = {l_diff:.3f})",
               rows, XW_HEAT, L, diff_len=l_diff)
        for m, vals in rows:
            for n, v in zip(XW_HEAT, vals):
                out_rows.append(["heat", f"{L:.4f}", f"{m:.4f}", n,
                                 f"{v:.6e}"])

    # ---- Burgers, resolved grid ----------------------------------------
    L, Nx = 2 * np.pi, 512
    x = np.linspace(0, L, Nx, endpoint=False)
    u0 = C.ic_burgers_sine(6, x, rng)
    solve = lambda z: C.burgers_wide_solve(z, L, T, NT, nu=NU, dt=2.5e-4)
    u = solve(u0)
    ut = C.discrete_ut("burgers", u, L, nu=NU)
    umax = float(np.abs(u).max())
    margins = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]) * L
    rows = sweep("burgers", x, t, u, solve, XW_BURG, L, margins, ut=ut)
    report(f"BURGERS torus L = {L:.3f}, Nx = {Nx} resolved "
           f"(max|u| T = {umax * T:.3f})", rows, XW_BURG, L,
           diff_len=umax * T)
    for m, vals in rows:
        for n, v in zip(XW_BURG, vals):
            out_rows.append(["burgers", f"{L:.4f}", f"{m:.4f}", n,
                             f"{v:.6e}"])

    # ---- KdV, resolved grid, moderate amplitudes ------------------------
    L, Nx = 2 * np.pi, 128
    x = np.linspace(0, L, Nx, endpoint=False)
    u0 = np.zeros((6, Nx))
    for i in range(6):
        for _ in range(rng.integers(1, 4)):
            u0[i] += rng.uniform(0.2, 1.5) / np.cosh(
                (x - rng.uniform(0.5, 5.5)) / rng.uniform(0.5, 1.0)) ** 2
    solve = lambda z: C.kdv_wide_solve(z, L, T, NT, dt=1e-5)
    u = solve(u0)
    ut = C.discrete_ut("kdv", u, L)
    rows = sweep("kdv", x, t, u, solve, XW_KDV, L, margins, ut=ut)
    report(f"KdV torus L = {L:.3f}, Nx = {Nx} resolved "
           "(dispersive: no finite domain of dependence)",
           rows, XW_KDV, L)
    for m, vals in rows:
        for n, v in zip(XW_KDV, vals):
            out_rows.append(["kdv", f"{L:.4f}", f"{m:.4f}", n, f"{v:.6e}"])

    path = os.path.join(C.RESULTS_DIR, f"exp8_seam_window{('_s' + str(SEED)) if SEED else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pde", "L", "margin", "generator", "resid_v_norm"])
        w.writerows(out_rows)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
