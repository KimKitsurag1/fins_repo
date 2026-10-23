"""Experiment 14 — detection curve on a NONLINEAR operator.

exp4 validated the eps^2 detection law on perturbed heat, where the
split-step operator is linear and dS = S exactly. Whether the
criterion still tracks breaking when the operator is nonlinear and dS is a
genuine linearisation is tested here on forced Burgers:

    u_t + u u_x = nu u_xx + eps f(x) u,   f(x) = sin x.

The x-dependent forcing breaks d_x and the Galilean boost while keeping d_t
(the equation stays autonomous). dS is taken by central FD through the
solver, u_t in the characteristics is consistent with the forced discrete
dynamics. Expected: broken generators' Rayleigh residual ~ eps^2 (slope 2
in log-log), d_t pinned at the eps = 0 floor throughout.

Run:  python exp14_nonlinear_breaking.py
"""

import csv
import os

import numpy as np

import exp_common as C
import exp1_oracle_torus_audit as E1
import argparse as _ap

L, Nx, NT, T, NU = 2 * np.pi, 256, 128, 1.0, 0.05
EPS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
THRESH = 1e-2


def forced_solve(u0, eps, dt=5e-4):
    x = np.linspace(0, L, u0.shape[1], endpoint=False)
    f = np.sin(x)

    def lin(k):
        return -NU * k ** 2 + 0j

    def nl_factory(k, mask):
        ik = 1j * k

        def N(v):
            um = np.real(np.fft.ifft(v * mask[None, :], axis=1))
            return (-0.5 * ik[None, :] * np.fft.fft(um ** 2, axis=1)
                    + eps * np.fft.fft(f[None, :] * um, axis=1)) * mask[None, :]

        return N

    return C.etdrk4_solve(u0, L, T, NT, dt, lin, nl_factory)


def forced_ut(u, eps):
    x = np.linspace(0, L, u.shape[1], endpoint=False)
    f = np.sin(x)[None, :, None]
    k = C.wavenumbers(u.shape[1], L)
    kc = k[None, :, None]
    mask = (np.abs(k) <= (2 / 3) * np.abs(k).max())[None, :, None]
    v = np.fft.fft(u, axis=1)
    um = np.real(np.fft.ifft(v * mask, axis=1))
    rhs = (-NU * kc ** 2 * v
           + mask * (-0.5j * kc * np.fft.fft(um ** 2, axis=1)
                     + eps * np.fft.fft(f[:, :, 0][..., None] * um
                                        * 0 + f * um, axis=1)))
    return np.real(np.fft.ifft(rhs, axis=1))


def main():
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    x = np.linspace(0, L, Nx, endpoint=False)
    t = np.linspace(0, T, NT)
    u0 = 0.5 * C.ic_burgers_sine(6, x, rng)

    print("=" * 78)
    print("EXP 14 - nonlinear detection: forced Burgers "
          f"u_t + u u_x = nu u_xx + eps sin(x) u  (nu = {NU})")
    print("broken: d_x, Galilean; kept: d_t")
    print("=" * 78)
    rows, tab = [], []
    names = ["d_x", "Galilean", "d_t"]
    for eps in EPS:
        solve = lambda z: forced_solve(z, eps)
        u = solve(u0)
        ut = forced_ut(u, eps)
        gens = C.gt_characteristics("burgers", x, t, u, nu=NU, ut=ut)
        q0s = {n: gens[n][3][:, :, 0] for n in names}
        _, dS = E1.batched_fd_jvp_multi(solve, u0, q0s)
        vals = []
        for n in names:
            xi, eta, phi, Q = gens[n]
            vals.append(C.rayleigh_residual(dS[n], Q, xi, eta, phi))
        tab.append([f"{eps:.0e}" if eps else "0"]
                   + [f"{v:.2e}" for v in vals])
        rows += [[f"{eps:.6e}", n, f"{v:.6e}"] for n, v in zip(names, vals)]
    print(C.fmt_table(tab, ["eps", "d_x (broken)", "Galilean (broken)",
                            "d_t (kept)"]))

    e = np.array(EPS[1:])
    for j, n in enumerate(names[:2]):
        r = np.array([float(row[j + 1].replace("e", "e"))
                      for row in [t_[0:] for t_ in tab[1:]]] if False else
                     [float(tab[i + 1][j + 1]) for i in range(len(e))])
        floor = float(tab[0][j + 1])
        m = r > 3 * floor
        if m.sum() >= 3:
            sl = np.polyfit(np.log10(e[m]), np.log10(r[m]), 1)[0]
            cross = np.interp(np.log10(THRESH), np.log10(r[m]),
                              np.log10(e[m]))
            print(f"{n}: log-log slope = {sl:.2f} (expected 2.00), "
                  f"eps* at {THRESH:g} = {10 ** cross:.2e}")

    path = os.path.join(C.RESULTS_DIR, f"exp14_nonlinear_breaking{('_s' + str(SEED)) if SEED else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "generator", "resid_v_norm"])
        w.writerows(rows)
    print(f"CSV -> {path}")


if __name__ == "__main__":
    main()
