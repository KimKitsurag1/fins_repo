"""Shared utilities for the experiment suite.

Oracle solution operators (exact or
high-resolution numerical), finite-difference JVPs through them, per-sample
ground-truth characteristics, and the Rayleigh residual in exactly the form
used by `pde/method_fins.equivariance_loss_jvp` — so that oracle numbers are
directly comparable with the surrogate sanity tables in the paper.

Conventions
-----------
Fields are numpy float64 arrays of shape [B, Nx, Nt] unless stated otherwise.
Generators follow the repo convention v = xi d_x + eta d_t + phi d_u with the
characteristic Q = phi - xi u_x - eta u_t and Q0 = Q(., t=0).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDE_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "pde"))
if PDE_DIR not in sys.path:
    sys.path.insert(0, PDE_DIR)

# All experiment scripts write through this path. Point the environment
# variable FINS_RESULTS_DIR at a mounted Drive folder in Colab so artefacts
# survive VM recycling; default is the local experiments/results.
RESULTS_DIR = os.environ.get("FINS_RESULTS_DIR",
                             os.path.join(HERE, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------
def wavenumbers(nx, L):
    return 2.0 * np.pi * np.fft.fftfreq(nx, d=L / nx)


def dx_spectral(f, L, order=1):
    """Spectral x-derivative along axis 1 of [B, Nx] or [B, Nx, Nt]."""
    k = wavenumbers(f.shape[1], L)
    mul = (1j * k) ** order
    shape = (1, -1) + (1,) * (f.ndim - 2)
    return np.real(np.fft.ifft(mul.reshape(shape) * np.fft.fft(f, axis=1), axis=1))


# ---------------------------------------------------------------------------
# Oracle solution operators
# ---------------------------------------------------------------------------
def heat_solve(u0, alpha, L, t):
    """Exact periodic heat operator. u0 [B,Nx] -> [B,Nx,Nt]. Linear: dS = S."""
    k = wavenumbers(u0.shape[1], L)
    decay = np.exp(-alpha * (k[None, :, None] ** 2) * t[None, None, :])
    return np.real(np.fft.ifft(np.fft.fft(u0, axis=1)[:, :, None] * decay, axis=1))


def _etdrk4_phis(z, h, M=32):
    """ETDRK4 phi-coefficients via the Kassam-Trefethen contour trick."""
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = z[:, None] + r[None, :]
    Q = h * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = h * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1)
    f2 = h * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1)
    f3 = h * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1)
    return Q.real, f1.real, f2.real, f3.real


def etdrk4_solve(u0, L, T, Nt, dt, lin_symbol, nonlin_factory):
    """Generic dealiased ETDRK4 on the torus of length L (numpy, float64).

    lin_symbol(k) -> complex array Lk; nonlin_factory(k, mask) -> N(v_hat).
    Used for the wide-domain controls; the torus (L = 2 pi) runs go through
    the repository solvers instead, so the audit uses the conference version's data
    pipeline.
    """
    B, Nx = u0.shape
    k = wavenumbers(Nx, L)
    Lk = lin_symbol(k)
    kmax = np.max(np.abs(k))
    mask = (np.abs(k) <= (2.0 / 3.0) * kmax)
    N = nonlin_factory(k, mask)

    # Snap the internal step count to a multiple of (Nt - 1) so the save
    # times coincide exactly with linspace(0, T, Nt); the repository solvers
    # round instead, which floors their comparisons at O(dt_internal * u_t).
    per = max(int(np.ceil(T / dt / (Nt - 1))), 1)
    n_int = per * (Nt - 1)
    h = T / n_int
    save_idx = np.arange(Nt) * per

    E = np.exp(h * Lk)[None, :]
    E2 = np.exp(h * Lk / 2)[None, :]
    Qc, f1, f2, f3 = _etdrk4_phis(h * Lk, h)
    Qc, f1, f2, f3 = (a[None, :] for a in (Qc, f1, f2, f3))

    v = np.fft.fft(u0, axis=1) * mask[None, :]
    out = np.zeros((B, Nx, Nt))
    sc = 0
    for step in range(n_int + 1):
        if sc < Nt and step == save_idx[sc]:
            out[:, :, sc] = np.real(np.fft.ifft(v, axis=1))
            sc += 1
            if sc == Nt:
                break
        Nv = N(v)
        a = E2 * v + Qc * Nv
        Na = N(a)
        b = E2 * v + Qc * Na
        Nb = N(b)
        c = E2 * a + Qc * (2 * Nb - Nv)
        Nc = N(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    return out


def burgers_wide_solve(u0, L, T, Nt, nu, dt):
    def lin(k):
        return -nu * k ** 2 + 0j

    def nl_factory(k, mask):
        ik = 1j * k

        def N(v):
            u = np.real(np.fft.ifft(v * mask[None, :], axis=1))
            return -0.5 * ik[None, :] * np.fft.fft(u ** 2, axis=1) * mask[None, :]

        return N

    return etdrk4_solve(u0, L, T, Nt, dt, lin, nl_factory)


def kdv_wide_solve(u0, L, T, Nt, dt):
    def lin(k):
        return 1j * k ** 3

    def nl_factory(k, mask):
        ik = 1j * k

        def N(v):
            u = np.real(np.fft.ifft(v * mask[None, :], axis=1))
            return -3.0 * ik[None, :] * np.fft.fft(u ** 2, axis=1) * mask[None, :]

        return N

    return etdrk4_solve(u0, L, T, Nt, dt, lin, nl_factory)


def fd_jvp(solve_fn, u0, q0, eps_rel=1e-5):
    """Central finite-difference JVP dS[u0](q0) through a batched solver.

    Both perturbed batches are stacked into a single solver call so the base
    and perturbed trajectories share the exact same time discretisation.
    """
    B = u0.shape[0]
    rms_u = np.sqrt(np.mean(u0 ** 2, axis=1, keepdims=True))
    rms_q = np.sqrt(np.mean(q0 ** 2, axis=1, keepdims=True))
    eps = eps_rel * (rms_u + 1.0) / (rms_q + 1e-12)          # [B, 1]
    stacked = np.concatenate([u0 + eps * q0, u0 - eps * q0], axis=0)
    sol = solve_fn(stacked)
    return (sol[:B] - sol[B:]) / (2.0 * eps[:, :, None])


def fd_t6(u, dt):
    """Numpy mirror of method_fins_components.fd_t_high_order (axis -1)."""
    ut = np.zeros_like(u)
    c6 = np.array([-1., 9., -45., 0., 45., -9., 1.]) / (60. * dt)
    for j, c in enumerate(c6):
        if c != 0.0:
            ut[..., 3:-3] += c * u[..., j:u.shape[-1] - 6 + j]
    ut[..., 2] = (u[..., 0] - 8*u[..., 1] + 8*u[..., 3] - u[..., 4]) / (12*dt)
    ut[..., -3] = (u[..., -5] - 8*u[..., -4] + 8*u[..., -2] - u[..., -1]) / (12*dt)
    ut[..., 1] = (u[..., 2] - u[..., 0]) / (2*dt)
    ut[..., -2] = (u[..., -1] - u[..., -3]) / (2*dt)
    ut[..., 0] = (u[..., 1] - u[..., 0]) / dt
    ut[..., -1] = (u[..., -1] - u[..., -2]) / dt
    return ut


def discrete_ut(pde, u, L, alpha=0.01, nu=0.01):
    """u_t consistent with the dealiased spectral dynamics of the solvers."""
    Nx = u.shape[1]
    k = wavenumbers(Nx, L)
    kc = k[None, :, None]
    mask = (np.abs(k) <= (2.0 / 3.0) * np.abs(k).max())[None, :, None]
    v = np.fft.fft(u, axis=1)
    um = np.real(np.fft.ifft(v * mask, axis=1))
    if pde == "heat":
        rhs = -alpha * kc ** 2 * v
    elif pde == "burgers":
        rhs = -nu * kc ** 2 * v - 0.5j * kc * np.fft.fft(um ** 2, axis=1) * mask
    elif pde == "kdv":
        rhs = 1j * kc ** 3 * v - 3j * kc * np.fft.fft(um ** 2, axis=1) * mask
    else:
        raise ValueError(pde)
    return np.real(np.fft.ifft(rhs, axis=1))


# ---------------------------------------------------------------------------
# Ground-truth characteristics (per-sample, exact u_t from the PDE)
# ---------------------------------------------------------------------------
def gt_characteristics(pde, x, t, u, alpha=0.01, nu=0.01, ut=None):
    """Return {name: (xi, eta, phi, Q)} with fields [B, Nx, Nt].

    xi/eta/phi follow `pde/ground_truth.py` (alpha-correct heat algebra).
    u_t defaults to the raw PDE formula; pass ut=fd_t6(u, dt) to reproduce
    the paper pipeline, or ut=discrete_ut(...) for consistency with the
    dealiased discrete dynamics (needed at marginal shock resolution).
    """
    L = x[-1] + (x[1] - x[0])
    B = u.shape[0]
    ux = dx_spectral(u, L)
    if ut is None:
        if pde == "heat":
            ut = alpha * dx_spectral(u, L, 2)
        elif pde == "burgers":
            ut = nu * dx_spectral(u, L, 2) - u * ux
        elif pde == "kdv":
            ut = -6.0 * u * ux - dx_spectral(u, L, 3)
        else:
            raise ValueError(pde)

    X = np.broadcast_to(x[None, :, None], u.shape)
    T = np.broadcast_to(t[None, None, :], u.shape)
    O = np.ones_like(u)
    Z = np.zeros_like(u)
    a = alpha

    if pde == "heat":
        gens = {
            "d_x": (O, Z, Z),
            "d_t": (Z, O, Z),
            "u d_u": (Z, Z, u),
            "Galilean": (2 * a * T, Z, -X * u),
            "scaling": (X, 2 * T, Z),
            "projective": (4 * a * T * X, 4 * a * T ** 2, -(X ** 2 + 2 * a * T) * u),
        }
    elif pde == "burgers":
        gens = {
            "d_x": (O, Z, Z),
            "d_t": (Z, O, Z),
            "Galilean": (T, Z, O),
            "scaling": (X, 2 * T, -u),
            # Projective generator: the conference paper's phi = -(x + t u)
            # fails the determining equation with defect -2(u + x u_x)
            # (exp13 numerically, exp17 symbolically); the symmetry of
            # u_t + u u_x = nu u_xx has phi = x - t u.
            "projective": (X * T, T ** 2, X - T * u),
        }
    else:
        gens = {
            "d_x": (O, Z, Z),
            "d_t": (Z, O, Z),
            "Galilean": (T, Z, O / 6.0),
            "scaling": (X, 3 * T, -2 * u),
        }

    out = {}
    for name, (xi, eta, phi) in gens.items():
        Q = phi - xi * ux - eta * ut
        out[name] = (xi, eta, phi, Q)
    return out


def rayleigh_residual(dSQ, Q, xi, eta, phi, t_margin=2):
    """Repo-form Rayleigh residual (method_fins.equivariance_loss_jvp).

    mean_b [ mean_{x, t in interior} (dSQ - Q)^2 / mean(xi^2+eta^2+phi^2) ]
    """
    Nt = Q.shape[-1]
    sl = slice(t_margin, Nt - t_margin) if Nt > 2 * t_margin else slice(None)
    resid = dSQ[..., sl] - Q[..., sl]
    v_sq = (xi ** 2 + eta ** 2 + phi ** 2)[..., sl]
    v_norm = v_sq.mean(axis=(1, 2)) + 1e-6                    # [B]
    return float((resid ** 2).mean(axis=(1, 2)).__truediv__(v_norm).mean())


# ---------------------------------------------------------------------------
# Initial-condition families (mirroring pde/data_generation.py)
# ---------------------------------------------------------------------------
def ic_heat_gauss(n, x, L, rng):
    u0 = np.zeros((n, x.size))
    for i in range(n):
        for _ in range(rng.integers(1, 5)):
            x0 = rng.uniform(0.0, L)
            A = rng.uniform(0.1, 3.0)
            w = rng.uniform(0.02 * L, 0.30 * L)
            u0[i] += A * np.exp(-((x - x0) / w) ** 2)
    return u0


def ic_heat_sine(n, x, L, rng):
    u0 = np.zeros((n, x.size))
    for i in range(n):
        amp = rng.uniform(0.3, 4.0)
        for m in range(1, rng.integers(2, 6)):
            u0[i] += rng.uniform(-1, 1) / np.sqrt(m) * np.sin(
                2 * np.pi * m * x / L + rng.uniform(0, 2 * np.pi))
        u0[i] *= amp
    return u0


def ic_burgers_sine(n, x, rng):
    u0 = np.zeros((n, x.size))
    for i in range(n):
        ga = rng.uniform(0.2, 2.0)
        for m in range(1, rng.integers(3, 7)):
            u0[i] += rng.uniform(-1, 1) / m * np.sin(m * x + rng.uniform(0, 2 * np.pi))
        u0[i] *= ga
        if rng.random() < 0.3:
            u0[i] += rng.uniform(-1, 1)
    return u0


def ic_kdv_sech(n, x, rng):
    u0 = np.zeros((n, x.size))
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            x0 = rng.uniform(0.5, 5.5)
            A = rng.choice([rng.uniform(0.1, 0.5), rng.uniform(0.5, 2.0),
                            rng.uniform(2.0, 5.0)])
            w = rng.uniform(0.2, 1.0)
            u0[i] += A / np.cosh((x - x0) / w) ** 2
    return u0


def ic_localised(n, x, L, rng, width=(0.5, 1.5), amp=(0.2, 0.6), sech=False):
    """Compactly supported bumps in the middle of a wide domain."""
    u0 = np.zeros((n, x.size))
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            x0 = rng.uniform(0.38 * L, 0.62 * L)
            A = rng.uniform(*amp)
            w = rng.uniform(*width)
            if sech:
                u0[i] += A / np.cosh((x - x0) / w) ** 2
            else:
                u0[i] += A * np.exp(-((x - x0) / w) ** 2)
    return u0


# ---------------------------------------------------------------------------
# Superposition subalgebra (heat)
# ---------------------------------------------------------------------------
def superposition_directions(n_dir, x, t, L, alpha, rng, n_modes=8):
    """Random band-limited solutions beta of the heat equation itself."""
    beta0 = rng.normal(size=(n_dir, x.size))
    bh = np.fft.fft(beta0, axis=1)
    bh[:, n_modes:-n_modes] = 0.0
    beta0 = np.real(np.fft.ifft(bh, axis=1))
    return heat_solve(beta0, alpha, L, t)                      # [n_dir, Nx, Nt]


# ---------------------------------------------------------------------------
# Per-sample fixed sanity check (torch; drop-in replacement)
# ---------------------------------------------------------------------------
def sanity_check_equivariance_fixed(S_theta, pde_type, train_sol, train_x,
                                    train_t, cfg, device, alpha=0.01,
                                    loss_mode="jvp", n_samples=4,
                                    verbose=True):
    """Per-sample variant of method_fins.sanity_check_equivariance.

    Building all ground-truth fields from sample 0 and broadcasting them
    across the batch is biased: for generators whose phi depends on u
    (heat v3/v4/v6, Burgers v4/v5, KdV scaling) the broadcast evaluates the
    characteristic on the wrong trajectory, and the measured residual comes
    out systematically LOWER than the true one (about two orders of magnitude
    on the heat Galilean and projective generators at zero model error).
    Here every sample gets fields built from its own (x, t, u).
    """
    # lazy: this module is numpy-only for the oracle scripts; torch loads here
    import torch
    from ground_truth import GroundTruthAlgebra, get_gt_names
    from method_fins import equivariance_loss

    u_sample = train_sol[:n_samples].to(device)
    ic_sample = u_sample[:, :, :, 0].contiguous()
    xc = train_x[:n_samples].to(device)
    tc = train_t[:n_samples].to(device)
    Nx, Nt = u_sample.shape[2], u_sample.shape[3]

    builders = {
        "heat": lambda xn, tn, un: GroundTruthAlgebra.heat_generators(
            xn, tn, un, alpha=alpha),
        "burgers": GroundTruthAlgebra.burgers_generators,
        "kdv": GroundTruthAlgebra.kdv_generators,
    }
    per_sample = []
    for b in range(n_samples):
        per_sample.append(builders[pde_type](
            train_x[b, 0].cpu().numpy(),
            train_t[b, 0].cpu().numpy(),
            train_sol[b, 0].cpu().numpy()))

    gt_names = get_gt_names(pde_type)
    dx = (xc[0, 0, 1] - xc[0, 0, 0]).item()
    dt_scalar = (tc[0, 0, 1] - tc[0, 0, 0]).item()

    results = []
    if verbose:
        print(f"    [Per-sample equivariance sanity - {pde_type}, "
              f"mode='{loss_mode}']")
    for gi, name in enumerate(gt_names):
        comp = [np.stack([per_sample[b][gi][c] for b in range(n_samples)])
                for c in range(3)]
        xi_t, eta_t, phi_t = (
            torch.tensor(c, dtype=torch.float32, device=device)
                 .view(n_samples, 1, Nx, Nt) for c in comp)
        with torch.enable_grad():
            val = equivariance_loss(
                S_theta, xi_t, eta_t, phi_t,
                ic_sample, u_sample, xc, tc,
                dx, dt_scalar, device, mode=loss_mode, normalize=True,
            ).item()
        if verbose:
            print(f"      {name:>12}: equiv = {val:.4e}")
        results.append((name, val))
    return results


def fmt_table(rows, headers):
    """Plain fixed-width table for logs."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    sep = "-" * len(line)
    body = "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths))
                     for r in rows)
    return f"{line}\n{sep}\n{body}"
