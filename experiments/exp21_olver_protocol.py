"""Experiment 21 — the local (Olver) formulation on single-scale torus data.

LieSym (§5.2) claims single-domain-length heat data is "physically
degenerate" for the scaling generator x d_x + 2t d_t and that no
operator-loss method can recover it without multi-scale + coordinate-aware
data. This script tests the claim at the level of the CONDITION itself,
with the exact operator (no surrogate), on exactly such data.

The Olver residual for an evolution PDE u_t = F[u] and characteristic
Q = phi - xi u_x - eta u_t is M[Q] = D_t Q - dF[u](Q). It is a LOCAL
statement: it holds pointwise wherever the generator is a symmetry. On a
periodic domain the x-weighted characteristics are discontinuous at the
seam x = L -> 0, so any derivative that crosses the seam is contaminated:

  spectral   periodic FFT derivatives, what a global spatial operator
             (FNO) effectively applies -- the seam pollutes every point
  local FD   non-periodic finite differences, evaluated only outside a
             margin m around the seam -- what a seam-aware protocol does

If the scaling residual is small under the local protocol on single-L
data, the degeneracy is a property of the PERIODIC-GLOBAL evaluation, not
of the data, and the multi-scale construction is one fix among others.

The table also includes LieSym's ground-truth generators as coded
(Galilean with phi = +x u and no alpha; projective without alpha): a
correct residual test rejects them under every protocol, which is how it
serves as a ground-truth linter.

numpy only.  Run:  python exp21_olver_protocol.py
"""

import numpy as np

AL, L, T, NX, NT, KMAX = 0.01, 2 * np.pi, 1.0, 64, 128, 6


def heat_modes(rng, n):
    """Exact heat solutions on the torus as random Fourier sums."""
    ks = np.arange(1, KMAX + 1)
    a = rng.standard_normal((n, KMAX)) / ks[None, :] ** 1.5
    ph = rng.uniform(0, 2 * np.pi, (n, KMAX))
    return ks, a, ph


def field(ks, a, ph, x, t, dx=0, dt=0):
    """d^dx/dx^dx d^dt/dt^dt of u, analytic (dt in {0,1,2})."""
    X, Tm = np.meshgrid(x, t, indexing="ij")
    out = np.zeros_like(X)
    for k, ak, pk in zip(ks, a, ph):
        decay = np.exp(-AL * k ** 2 * Tm) * (-AL * k ** 2) ** dt
        arg = k * X + pk
        # x-derivatives of cos cycle through cos, -sin, -cos, sin
        trig = [np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z),
                np.sin][dx % 4](arg) * k ** dx
        out += ak * decay * trig
    return out


def generators(x, t, U):
    """Correct heat algebra, plus LieSym's two mis-coded generators."""
    X, Tm = np.meshgrid(x, t, indexing="ij")
    Z, O = np.zeros_like(X), np.ones_like(X)
    u = U[0, 0]
    return {
        "d_x":              (O, Z, Z),
        "d_t":              (Z, O, Z),
        "u d_u":            (Z, Z, u),
        "Galilean":         (2 * AL * Tm, Z, -X * u),
        "scaling":          (X, 2 * Tm, Z),
        "projective":       (4 * AL * Tm * X, 4 * AL * Tm ** 2,
                             -(X ** 2 + 2 * AL * Tm) * u),
        "[LieSym] Galilean (2t,0,+xu)":
                            (2 * Tm, Z, X * u),
        "[LieSym] projective (no alpha)":
                            (4 * Tm * X, 4 * Tm ** 2, -(X ** 2 + 2 * Tm) * u),
    }


def d_spectral(f, axis, order, L_):
    n = f.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L_ / n)
    shape = [1, 1]
    shape[axis] = n
    return np.real(np.fft.ifft((1j * k.reshape(shape)) ** order
                               * np.fft.fft(f, axis=axis), axis=axis))


def d_local(f, h, axis, order):
    """Non-periodic central differences (2nd order), one-sided at edges."""
    f = np.moveaxis(f, axis, 0)
    out = np.zeros_like(f)
    if order == 1:
        out[1:-1] = (f[2:] - f[:-2]) / (2 * h)
        out[0] = (f[1] - f[0]) / h
        out[-1] = (f[-1] - f[-2]) / h
    else:
        out[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2
        out[0] = out[1]
        out[-1] = out[-2]
    return np.moveaxis(out, 0, axis)


def olver_residual(Q, x, t, mode, margin):
    dx, dt = x[1] - x[0], t[1] - t[0]
    if mode == "spectral":
        Qt = d_local(Q, dt, 1, 1)
        Qxx = d_spectral(Q, 0, 2, L)
        i0, i1 = 0, len(x)
    else:
        Qt = d_local(Q, dt, 1, 1)
        Qxx = d_local(Q, dx, 0, 2)
        i0, i1 = int(margin * len(x)), len(x) - int(margin * len(x))
    M = Qt - AL * Qxx
    sl = (slice(i0, i1), slice(2, len(t) - 2))
    return np.sqrt(np.mean(M[sl] ** 2)) / (np.sqrt(np.mean(Q[sl] ** 2))
                                            + 1e-30)


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    ks, a, ph = heat_modes(rng, 1)
    u = field(ks, a[0], ph[0], x, t)
    ux = field(ks, a[0], ph[0], x, t, dx=1)
    ut = field(ks, a[0], ph[0], x, t, dt=1)
    gens = generators(x, t, u[None, None])
    protocols = [("spectral (global)", "spectral", 0.0),
                 ("local FD, m = 0", "local", 0.0),
                 ("local FD, m = 0.05L", "local", 0.05),
                 ("local FD, m = 0.10L", "local", 0.10)]
    print("=" * 96)
    print(f"EXP 21 - Olver residual ||D_t Q - alpha D_xx Q|| / ||Q||, exact "
          f"heat data, SINGLE scale L = 2pi, Nx = {NX}, alpha = {AL}")
    print("=" * 96)
    head = f"{'generator':<34}" + "".join(f"{p[0]:>19}" for p in protocols)
    print(head)
    print("-" * len(head))
    for name, (xi, eta, phi) in gens.items():
        Q = phi - xi * ux - eta * ut
        vals = [olver_residual(Q, x, t, m, mg) for _, m, mg in protocols]
        print(f"{name:<34}" + "".join(f"{v:>19.2e}" for v in vals))
    print("\nthreshold 1e-2 as in the papers; time-derivative FD floor ~1e-5")


if __name__ == "__main__":
    main()
