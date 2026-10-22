"""Experiment 12 — two falsifiable short-horizon predictions.

Both open ends of exp8 have the same proposed mechanism: seam or wrap-around
contamination travelling a distance ~ (speed) * T. If that is the right
mechanism, shortening the horizon must rescue them; if the residuals do not
move with T, the mechanism is wrong.

A. Burgers projective on the torus. Q0 = -x is smooth inside the window;
   the only defect is the seam jump of height L, advected at characteristic
   speed ~ max|u|. Prediction: the windowed residual at margin m falls below
   the 1e-2 threshold once max|u| * T < m, so at T = 0.2 (range ~ 0.9) the
   generator is rescued at m >= 0.2 L, while at T = 1.0 (range ~ 4.6) no
   margin on L = 2 pi can save it — which is what exp8 measured.

B. KdV scaling on the wide domain. The residual floor of 3.8e-2 (Q-norm)
   was attributed to dispersive radiation wrapping the domain: group speed
   up to ~ 3 k_dealias^2 = 343 against L = 100. Prediction: the Q-norm
   residual falls monotonically with T (ranges 17 / 69 / 343 for
   T = 0.05 / 0.2 / 1.0).

Run:  python exp12_short_horizon.py
"""

import csv
import os

import numpy as np

import exp_common as C
import exp1_oracle_torus_audit as E1
import exp8_seam_window as E8

NT = 128
THRESH = 1e-2


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    out = []

    # ---------------- A: Burgers projective vs horizon -------------------
    L, Nx, NU = 2 * np.pi, 512, 0.01
    x = np.linspace(0, L, Nx, endpoint=False)
    u0 = C.ic_burgers_sine(6, x, rng)
    margins = np.array([0.0, 0.1, 0.2, 0.3]) * L
    print("=" * 84)
    print("EXP 12A - Burgers torus, resolved: windowed residual vs horizon T")
    print(f"L = {L:.3f}, Nx = {Nx}, nu = {NU}, threshold = {THRESH:g}")
    print("=" * 84)
    for T in (0.2, 0.5, 1.0):
        t = np.linspace(0, T, NT)
        solve = lambda z: C.burgers_wide_solve(z, L, T, NT, nu=NU, dt=2.5e-4)
        u = solve(u0)
        umax = float(np.abs(u).max())
        ut = C.discrete_ut("burgers", u, L, nu=NU)
        gens = C.gt_characteristics("burgers", x, t, u, nu=NU, ut=ut)
        q0s = {n: gens[n][3][:, :, 0] for n in ("d_x", "scaling",
                                                "projective")}
        _, dS = E1.batched_fd_jvp_multi(solve, u0, q0s)
        print(f"\n  T = {T}: max|u| = {umax:.2f}, contamination range "
              f"max|u|*T = {umax * T:.2f} "
              f"(margins in length units: "
              + ", ".join(f"{m:.2f}" for m in margins) + ")")
        tab = []
        for m in margins:
            row = [f"{m / L:.1f}"]
            for n in ("d_x", "scaling", "projective"):
                xi, eta, phi, Q = gens[n]
                v = E8.windowed_residual(dS[n], Q, xi, eta, phi, x, L, m)
                row.append(f"{v:.2e}")
                out.append(["burgers", f"{T}", f"{m / L:.2f}", n,
                            f"{v:.6e}"])
            tab.append(row)
        print("  " + C.fmt_table(tab, ["m/L", "d_x (floor)", "scaling",
                                       "projective"]).replace("\n",
                                                              "\n  "))

    # ---------------- B: KdV wide domain vs horizon ----------------------
    Lw, Nxw = 32 * np.pi, 512
    xw = np.linspace(0, Lw, Nxw, endpoint=False)
    u0w = C.ic_localised(6, xw, Lw, rng, width=(0.6, 1.2), amp=(0.3, 1.0),
                         sech=True)
    k = C.wavenumbers(Nxw, Lw)
    vg = 3.0 * ((2.0 / 3.0) * np.abs(k).max()) ** 2
    print("\n" + "=" * 84)
    print("EXP 12B - KdV wide domain: scaling residual vs horizon T")
    print(f"L = {Lw:.1f}, Nx = {Nxw}, max group speed ~ 3 k_dealias^2 = "
          f"{vg:.0f}")
    print("=" * 84)
    tab = []
    for T in (0.05, 0.2, 1.0):
        t = np.linspace(0, T, NT)
        solve = lambda z: C.kdv_wide_solve(z, Lw, T, NT, dt=1e-4)
        u = solve(u0w)
        ut = C.discrete_ut("kdv", u, Lw)
        gens = C.gt_characteristics("kdv", xw, t, u, ut=ut)
        q0s = {n: gens[n][3][:, :, 0] for n in ("d_x", "scaling")}
        _, dS = E1.batched_fd_jvp_multi(solve, u0w, q0s)
        row = [f"{T}", f"{vg * T:.0f}"]
        for n in ("d_x", "scaling"):
            xi, eta, phi, Q = gens[n]
            rv = C.rayleigh_residual(dS[n], Q, xi, eta, phi)
            rq = E1.q_normalised(dS[n], Q)
            row.append(f"{rv:.1e}/{rq:.1e}")
            out.append(["kdv_wide", f"{T}", "", n, f"{rv:.6e}",
                        f"{rq:.6e}"])
        tab.append(row)
    print(C.fmt_table(tab, ["T", "range vg*T", "d_x (v/Q)",
                            "scaling (v/Q)"]))
    print(f"\ndomain length for reference: L = {Lw:.1f}")

    path = os.path.join(C.RESULTS_DIR, f"exp12_short_horizon{('_s' + str(SEED)) if SEED else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "T", "m_over_L", "generator",
                    "resid_v_norm", "resid_q_norm"])
        for r in out:
            w.writerow(r + [""] * (6 - len(r)))
    print(f"CSV -> {path}")


if __name__ == "__main__":
    main()
