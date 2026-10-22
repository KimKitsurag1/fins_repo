"""Experiment 1 — oracle audit of the ground-truth algebras on the torus.

Question: do the tabulated ground-truth generators
actually satisfy the FINS equivariance condition on the periodic domains the
experiments use, when ALL model error is removed?

Design
------
For each PDE the solution operator is evaluated without any surrogate:
  heat     — exact Fourier multiplier (linear, dS = S), repo IC families;
  Burgers  — the repository solver `PDESolver.solve_burgers` (the paper's own
             data pipeline), dS via batched central finite differences;
  KdV      — dealiased ETDRK4 (same scheme family as the repo solver, save
             grid snapped to linspace(0,T,Nt)), cross-validated against
             `PDESolver.solve_kdv` on the base trajectories.
The residual is the Rayleigh form of `method_fins.equivariance_loss_jvp`,
so numbers are directly comparable with the paper's sanity tables and the
1e-2 discoverability threshold.

Controls: the same physics on a wide domain with localised data, where the
x-weighted characteristics are effectively compactly supported. A residual
that is large on the torus and collapses on the wide domain is a property of
the domain geometry, not of any method.

Run:  python exp1_oracle_torus_audit.py --pde all [--validate-repo-kdv]
"""

import argparse
import csv
import os
import time

import numpy as np

import exp_common as C

NX, NT, T = 64, 128, 1.0
ALPHA, NU = 0.01, 0.01
THRESH = 1e-2


def batched_fd_jvp_multi(solve_fn, u0, q0_dict, eps_rel=1e-5):
    """One solver call for the base batch and every (gen, +/-) perturbation."""
    B, Nx = u0.shape
    names = list(q0_dict)
    blocks, eps_list = [u0], []
    for n in names:
        q0 = q0_dict[n]
        rms_u = np.sqrt(np.mean(u0 ** 2, axis=1, keepdims=True))
        rms_q = np.sqrt(np.mean(q0 ** 2, axis=1, keepdims=True))
        eps = eps_rel * (rms_u + 1.0) / (rms_q + 1e-12)
        eps_list.append(eps)
        blocks += [u0 + eps * q0, u0 - eps * q0]
    sol = solve_fn(np.concatenate(blocks, axis=0))
    base = sol[:B]
    out = {}
    for i, n in enumerate(names):
        p = sol[B + 2 * i * B: B + (2 * i + 1) * B]
        m = sol[B + (2 * i + 1) * B: B + (2 * i + 2) * B]
        out[n] = (p - m) / (2.0 * eps_list[i][:, :, None])
    return base, out


def q_normalised(dSQ, Q, t_margin=2):
    """mean(resid^2)/mean(Q^2) on the interior: invariant to generator scale."""
    Nt = Q.shape[-1]
    sl = slice(t_margin, Nt - t_margin) if Nt > 2 * t_margin else slice(None)
    resid = dSQ[..., sl] - Q[..., sl]
    return float((resid ** 2).mean() / ((Q[..., sl] ** 2).mean() + 1e-30))


def audit(pde, x, t, u, solve_fn, linear=False, label="", nu=NU,
          ut_mode="pde"):
    L = x[-1] + (x[1] - x[0])
    if ut_mode == "fd6":
        ut = C.fd_t6(u, t[1] - t[0])
    elif ut_mode == "discrete":
        ut = C.discrete_ut(pde, u, L, alpha=ALPHA, nu=nu)
    else:
        ut = None
    gens = C.gt_characteristics(pde, x, t, u, alpha=ALPHA, nu=nu, ut=ut)
    rows = []
    if linear:
        dS = {n: solve_fn(g[3][:, :, 0]) for n, g in gens.items()}
    else:
        q0s = {n: g[3][:, :, 0] for n, g in gens.items()}
        _, dS = batched_fd_jvp_multi(solve_fn, u[:, :, 0], q0s)
    for name, (xi, eta, phi, Q) in gens.items():
        rows.append((name, C.rayleigh_residual(dS[name], Q, xi, eta, phi),
                     q_normalised(dS[name], Q)))
    print(f"\n  {label}  (L = {L:.3f})")
    tab = [(n, f"{r:.3e}", f"{rq:.3e}", "below" if r < THRESH else "ABOVE")
           for n, r, rq in rows]
    print("  " + C.fmt_table(tab, ["generator", "resid/|v|^2", "resid/|Q|^2",
                                   f"vs {THRESH:g}"]).replace("\n", "\n  "))
    return {n: (r, rq) for n, r, rq in rows}


def run_heat(rng, writer):
    t = np.linspace(0, T, NT)
    L = 1.0
    x = np.linspace(0, L, NX, endpoint=False)
    u0 = np.concatenate([C.ic_heat_gauss(16, x, L, rng),
                         C.ic_heat_sine(16, x, L, rng)])
    solve = lambda z: C.heat_solve(z, ALPHA, L, t)
    r = audit("heat", x, t, solve(u0), solve, linear=True,
              label="HEAT torus, repo IC families (exact operator)")
    for n, (v, vq) in r.items():
        writer.writerow(["heat", "torus", n, f"{v:.6e}", f"{vq:.6e}"])

    Lw, Nxw = 64.0, 1024
    xw = np.linspace(0, Lw, Nxw, endpoint=False)
    u0w = C.ic_localised(8, xw, Lw, rng)
    solvew = lambda z: C.heat_solve(z, ALPHA, Lw, t)
    r = audit("heat", xw, t, solvew(u0w), solvew, linear=True,
              label="HEAT wide control, localised data (exact operator)")
    for n, (v, vq) in r.items():
        writer.writerow(["heat", "wide", n, f"{v:.6e}", f"{vq:.6e}"])


def run_burgers(rng, writer):
    from data_generation import PDESolver
    t = np.linspace(0, T, NT)
    L = 2 * np.pi
    x = np.linspace(0, L, NX, endpoint=False)
    u0 = C.ic_burgers_sine(8, x, rng)

    def solve_repo(z):
        return np.asarray(PDESolver.solve_burgers(z, NU, T, NT,
                                                  dt_internal=1e-4))

    t0 = time.time()
    r = audit("burgers", x, t, solve_repo(u0), solve_repo, ut_mode="fd6",
              label="BURGERS torus, repo grid + repo Q pipeline (fd6 u_t)")
    print(f"  [{time.time()-t0:.0f} s]")
    for n, (v, vq) in r.items():
        writer.writerow(["burgers", "torus_repo_grid", n, f"{v:.6e}", f"{vq:.6e}"])

    # Resolved torus: same L and IC family, shock-resolving grid, exp_common's solver.
    Nxr = 512
    xr = np.linspace(0, L, Nxr, endpoint=False)
    u0r = C.ic_burgers_sine(8, xr, rng)
    solver = lambda z: C.burgers_wide_solve(z, L, T, NT, nu=NU, dt=2.5e-4)
    r = audit("burgers", xr, t, solver(u0r), solver, ut_mode="discrete",
              label=f"BURGERS torus RESOLVED (Nx={Nxr}, same L, nu={NU})")
    for n, (v, vq) in r.items():
        writer.writerow(["burgers", "torus_resolved", n, f"{v:.6e}", f"{vq:.6e}"])

    Lw, Nxw, NUW = 16 * np.pi, 256, 0.1
    xw = np.linspace(0, Lw, Nxw, endpoint=False)
    u0w = C.ic_localised(8, xw, Lw, rng, width=(0.8, 1.6), amp=(0.1, 0.3))
    solvew = lambda z: C.burgers_wide_solve(z, Lw, T, NT, nu=NUW, dt=1e-3)
    uw = solvew(u0w)
    # nu enters only through u_t in the characteristics
    gens = C.gt_characteristics("burgers", xw, t, uw, nu=NUW,
                                ut=C.discrete_ut("burgers", uw, Lw, nu=NUW))
    q0s = {n: g[3][:, :, 0] for n, g in gens.items()}
    _, dS = batched_fd_jvp_multi(solvew, u0w, q0s)
    print(f"\n  BURGERS wide control, localised data  (L = {Lw:.3f}, nu = {NUW})")
    rows = []
    for n, (xi, eta, phi, Q) in gens.items():
        v = C.rayleigh_residual(dS[n], Q, xi, eta, phi)
        vq = q_normalised(dS[n], Q)
        rows.append((n, f"{v:.3e}", f"{vq:.3e}",
                     "below" if v < THRESH else "ABOVE"))
        writer.writerow(["burgers", "wide", n, f"{v:.6e}", f"{vq:.6e}"])
    print("  " + C.fmt_table(rows, ["generator", "resid/|v|^2", "resid/|Q|^2",
                                    f"vs {THRESH:g}"]).replace("\n", "\n  "))


def run_kdv(rng, writer, validate_repo=False):
    t = np.linspace(0, T, NT)
    L = 2 * np.pi
    x = np.linspace(0, L, NX, endpoint=False)
    u0 = C.ic_kdv_sech(6, x, rng)
    solve = lambda z: C.kdv_wide_solve(z, L, T, NT, dt=5e-5)

    t0 = time.time()
    base = solve(u0)
    conv = C.kdv_wide_solve(u0, L, T, NT, dt=2.5e-5)
    print(f"\n  KdV torus solver self-convergence (dt 5e-5 vs 2.5e-5): "
          f"{np.abs(base-conv).max()/np.abs(conv).max():.2e}")
    if validate_repo:
        from data_generation import PDESolver
        rep = np.asarray(PDESolver.solve_kdv(u0, T, NT))
        print(f"  KdV vs repository ETDRK4 on base trajectories:      "
              f"{np.abs(base-rep).max()/np.abs(rep).max():.2e}")
    r = audit("kdv", x, t, base, solve, ut_mode="fd6",
              label="KdV torus, repo grid Nx=64, repo Q pipeline (fd6 u_t)")
    print(f"  [{time.time()-t0:.0f} s]")
    for n, (v, vq) in r.items():
        writer.writerow(["kdv", "torus_repo_grid", n, f"{v:.6e}", f"{vq:.6e}"])

    # Resolved torus: soliton-resolving grid, same L, moderate amplitudes.
    # ETDRK4 with a purely dispersive symbol is stable for |h k^3| <~ 1 on
    # the dealiased band, which caps Nx at fixed dt; Nx=128 with dt=1e-5
    # keeps |z| <= 0.75 while giving >= 10 points across the narrowest pulse.
    Nxr = 128
    xr = np.linspace(0, L, Nxr, endpoint=False)
    u0r = np.zeros((6, Nxr))
    for i in range(6):
        for _ in range(rng.integers(1, 4)):
            u0r[i] += rng.uniform(0.2, 1.5) / np.cosh(
                (xr - rng.uniform(0.5, 5.5)) / rng.uniform(0.5, 1.0)) ** 2
    solver = lambda z: C.kdv_wide_solve(z, L, T, NT, dt=1e-5)
    t0 = time.time()
    r = audit("kdv", xr, t, solver(u0r), solver, ut_mode="discrete",
              label=f"KdV torus RESOLVED (Nx={Nxr}, same L, "
                    f"moderate amplitudes)")
    print(f"  [{time.time()-t0:.0f} s]")
    for n, (v, vq) in r.items():
        writer.writerow(["kdv", "torus_resolved", n, f"{v:.6e}", f"{vq:.6e}"])

    Lw, Nxw = 32 * np.pi, 512
    xw = np.linspace(0, Lw, Nxw, endpoint=False)
    u0w = C.ic_localised(6, xw, Lw, rng, width=(0.6, 1.2), amp=(0.3, 1.0),
                         sech=True)
    solvew = lambda z: C.kdv_wide_solve(z, Lw, T, NT, dt=1e-4)
    r = audit("kdv", xw, t, solvew(u0w), solvew, ut_mode="discrete",
              label="KdV wide control, localised data")
    for n, (v, vq) in r.items():
        writer.writerow(["kdv", "wide", n, f"{v:.6e}", f"{vq:.6e}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="all",
                    choices=["all", "heat", "burgers", "kdv"])
    ap.add_argument("--validate-repo-kdv", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    path = os.path.join(C.RESULTS_DIR, f"exp1_oracle_audit_{args.pde}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pde", "setting", "generator", "resid_v_norm", "resid_q_norm"])
        print("=" * 74)
        print("EXP 1 - oracle audit of ground-truth algebras "
              "(zero surrogate error)")
        print(f"Nx={NX}, Nt={NT}, T={T}, alpha={ALPHA}, nu={NU}, "
              f"threshold={THRESH:g}")
        print("=" * 74)
        if args.pde in ("all", "heat"):
            run_heat(rng, w)
        if args.pde in ("all", "burgers"):
            run_burgers(rng, w)
        if args.pde in ("all", "kdv"):
            run_kdv(rng, w, validate_repo=args.validate_repo_kdv)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
