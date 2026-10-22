"""Experiment 4 — controlled symmetry breaking with an exact linear operator.

The paper's rank claim needs a positive control: a family of problems where
a KNOWN generator is broken by a KNOWN amount, and the equivariance residual
tracks it. Perturbed heat equations provide this exactly:

    u_t = alpha u_xx + eps f(x) u    breaks d_x,  keeps d_t and u d_u;
    u_t = alpha u_xx + eps g(t) u    breaks d_t,  keeps d_x and u d_u.

Both are LINEAR in u, so the split-step (Strang) propagator is itself the
exact solution operator of a discrete linear system and dS = S with zero
model error. The residual of the broken generator must scale as eps^2 in the
Rayleigh form (the residual field is linear in eps); the kept generators
must stay at machine floor for every eps.

The eps* at which the broken generator crosses the 1e-2 threshold is the
detection limit of the criterion itself — the number a stability lemma has
to reproduce.

Run:  python exp4_symmetry_breaking.py
"""

import csv
import os

import numpy as np

import exp_common as C

NX, NT, T, L, ALPHA = 64, 128, 1.0, 1.0, 0.01
N_STEPS = 32 * (NT - 1)          # multiple of the save grid
N_SAMPLES = 8
THRESH = 1e-2
EPS = np.logspace(-4, 0, 9)


def make_solver(mode, eps):
    """Strang split-step for u_t = alpha u_xx + eps m u, m = f(x) or g(t)."""
    x = np.linspace(0, L, NX, endpoint=False)
    k = C.wavenumbers(NX, L)
    dt = T / N_STEPS
    diff = np.exp(-ALPHA * k ** 2 * dt)[None, :]
    per = N_STEPS // (NT - 1)

    def solve(u0):
        u = u0.astype(np.float64).copy()
        out = np.zeros((u0.shape[0], NX, NT))
        out[:, :, 0] = u
        sc = 1
        for s in range(N_STEPS):
            t0 = s * dt
            if mode == "fx":
                r1 = np.exp(0.5 * eps * np.sin(2 * np.pi * x / L) * dt)[None, :]
                r2 = r1
            else:
                r1 = np.exp(0.5 * eps * np.sin(2 * np.pi * t0 / T) * dt)
                r2 = np.exp(0.5 * eps * np.sin(2 * np.pi * (t0 + dt) / T) * dt)
            u = np.real(np.fft.ifft(np.fft.fft(u * r1, axis=1) * diff,
                                    axis=1)) * r2
            if (s + 1) % per == 0:
                out[:, :, sc] = u
                sc += 1
        return out

    return solve


def residual(gen_name, u, x, t, solve, mode, eps):
    """Rayleigh residual with u_t consistent with the perturbed equation."""
    ux = C.dx_spectral(u, L)
    uxx = C.dx_spectral(u, L, 2)
    if mode == "fx":
        m = np.sin(2 * np.pi * x / L)[None, :, None]
    else:
        m = np.sin(2 * np.pi * t / T)[None, None, :]
    ut = ALPHA * uxx + eps * m * u
    O, Z = np.ones_like(u), np.zeros_like(u)
    xi, eta, phi = {"d_x": (O, Z, Z), "d_t": (Z, O, Z),
                    "u d_u": (Z, Z, u)}[gen_name]
    Q = phi - xi * ux - eta * ut
    dSQ = solve(Q[:, :, 0])
    return C.rayleigh_residual(dSQ, Q, xi, eta, phi)


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=0)
    SEED = _p.parse_args().seed
    rng = np.random.default_rng(SEED)
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    u0 = C.ic_heat_gauss(N_SAMPLES, x, L, rng)

    # validation: eps = 0 must reproduce the exact heat operator
    s0 = make_solver("fx", 0.0)(u0)
    ex = C.heat_solve(u0, ALPHA, L, t)
    print(f"split-step vs exact heat at eps=0: "
          f"{np.abs(s0 - ex).max() / np.abs(ex).max():.2e}")

    path = os.path.join(C.RESULTS_DIR, f"exp4_symmetry_breaking{('_s' + str(SEED)) if SEED else ''}.csv")
    rows_csv = []
    for mode, broken, kept in [("fx", "d_x", ["d_t", "u d_u"]),
                               ("gt", "d_t", ["d_x", "u d_u"])]:
        print("\n" + "=" * 70)
        print(f"perturbation eps * {'f(x)' if mode == 'fx' else 'g(t)'} * u"
              f"   ->  broken: {broken}, kept: {', '.join(kept)}")
        print("=" * 70)
        tab = []
        for eps in EPS:
            solve = make_solver(mode, eps)
            u = solve(u0)
            rb = residual(broken, u, x, t, solve, mode, eps)
            rk = [residual(n, u, x, t, solve, mode, eps) for n in kept]
            tab.append([f"{eps:.1e}", f"{rb:.3e}"]
                       + [f"{v:.1e}" for v in rk])
            rows_csv.append([mode, f"{eps:.6e}", broken, f"{rb:.6e}"]
                            )
            for n, v in zip(kept, rk):
                rows_csv.append([mode, f"{eps:.6e}", n, f"{v:.6e}"])
        print(C.fmt_table(tab, ["eps", f"{broken} (broken)"]
                          + [f"{n} (kept)" for n in kept]))
        lb = np.array([float(r[1]) for r in tab])
        sl = np.polyfit(np.log10(EPS[lb > 1e-20]),
                        np.log10(lb[lb > 1e-20]), 1)[0]
        cross = np.interp(np.log10(THRESH), np.log10(lb), np.log10(EPS))
        print(f"log-log slope of broken-generator residual: {sl:.2f} "
              f"(expected 2.00)")
        print(f"detection limit eps* at residual {THRESH:g}: "
              f"{10 ** cross:.2e}")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "eps", "generator", "resid_v_norm"])
        w.writerows(rows_csv)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
