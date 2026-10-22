"""Fractional Burgers equation on the torus [0, 2 pi): u_t + u u_x = -nu Lambda^alpha u, Lambda^alpha the Fourier
multiplier |k|^alpha (the fractional Laplacian (-d_xx)^(alpha / 2)), 1 <= alpha <= 2. Equation names fb100 ... fb200
(alpha = 1.00 ... 2.00).

alpha = 2 is viscous Burgers, u_t + u u_x = nu u_xx. Otherwise the dissipation is nonlocal: no expression in the jet
exists, so a library of local terms cannot represent it and Lie's algorithm on an identified equation does not apply as
written. Point symmetries: d_x, d_t, Galilean t d_x + d_u (Lambda^alpha annihilates constants), scaling
x d_x + alpha t d_t + (1 - alpha) u d_u; alpha = 2 adds the projective x t d_x + t^2 d_t + (x - t u) d_u.

nu = NU0 / K0^alpha: the damping rate at the humps' wavenumber K0 is the same for every alpha.
Data mirror bo.py: evaluation trajectories 'periodic' (two Gaussian humps of either sign, centres at [0.30, 0.34] L
and [0.48, 0.52] L), stepper training data 'mixed' (half one or two such humps on a zero or constant background, half
broadband humps anywhere on the torus), Nx = 128 over T = 1.
solve: ETDRK4 (Kassam and Trefethen's contour integrals), 2/3 dealiasing, torch, float64, PER_SAVE steps per save, on
UP times the data's grid (the state interpolated spectrally, the solution sampled back on the data's points): against
8 times the grid the sampled solution differs by 1.2e-4 of max |u| at alpha = 1 (critical: the fronts steepen until
nu balances them) and by below 1e-8 from alpha = 1.25 (at the data's grid alone: 9e-3 and 1.4e-3).
"""

import numpy as np
import torch

L = 2 * np.pi
T = 1.0
PER_SAVE = 8
UP = 4
NU0, K0 = 1.0, 3.0
AMP, WIDTH = (0.5, 1.0), (0.25, 0.4)        # hump amplitude |a| and Gaussian width (radians)
EQUATIONS = ("fb100", "fb125", "fb150", "fb175", "fb200")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def alpha(pde):
    """'fb150' -> 1.5."""
    return int(pde[2:]) / 100


def nu(a):
    return NU0 / K0 ** a


def names(pde):
    out = ["∂_x", "∂_t", "Galilean", "scaling"]
    return out + ["projective"] if alpha(pde) == 2 else out


def upsample(u0, up):
    """[B, n] -> [B, up n]: the trigonometric interpolant (the Nyquist mode split evenly)."""
    n = u0.shape[1]
    if up == 1:
        return u0
    c = np.fft.fft(u0, axis=1)
    out = np.zeros((len(u0), up * n), dtype=complex)
    h = n // 2
    out[:, :h] = c[:, :h]
    out[:, -h + 1:] = c[:, h + 1:]
    out[:, h] = out[:, -h] = c[:, h] / 2
    return np.real(np.fft.ifft(out, axis=1)) * up


def solve(u0, T_end, Nt, a, per_save=PER_SAVE, up=UP):
    """u0 [B, Nx] -> [B, Nx, Nt] on t = linspace(0, T_end, Nt), exponent a; solved on up Nx points."""
    u0 = upsample(np.atleast_2d(np.asarray(u0, dtype=np.float64)), up)
    Nx = u0.shape[1]
    n_int = (Nt - 1) * per_save
    h = T_end / n_int
    dev = DEVICE
    k = torch.fft.fftfreq(Nx, d=L / Nx, dtype=torch.float64, device=dev) * 2 * np.pi
    Lop = (-nu(a) * torch.abs(k) ** a).to(torch.complex128)
    E, E2 = torch.exp(h * Lop), torch.exp(h * Lop / 2)
    M = 32
    r = torch.exp(2j * np.pi * (torch.arange(1, M + 1, dtype=torch.float64, device=dev) - 0.5) / M)
    LR = h * Lop[:, None] + r[None, :]
    Q = h * torch.mean((torch.exp(LR / 2) - 1) / LR, dim=1).real
    f1 = h * torch.mean((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, dim=1).real
    f2 = h * torch.mean((2 + LR + torch.exp(LR) * (-2 + LR)) / LR ** 3, dim=1).real
    f3 = h * torch.mean((-4 - 3 * LR - LR ** 2 + torch.exp(LR) * (4 - LR)) / LR ** 3, dim=1).real
    keep = (torch.abs(k) <= 2.0 / 3.0 * torch.abs(k).max()).to(torch.float64)
    g = -0.5j * k * keep                                           # -(u^2 / 2)_x

    def nonlin(v):
        return g * torch.fft.fft(torch.fft.ifft(v * keep).real ** 2)

    E, E2 = E.real, E2.real
    v = torch.fft.fft(torch.tensor(u0, device=dev)) * keep
    out = torch.zeros((len(u0), Nx, Nt), dtype=torch.float64, device=dev)
    out[:, :, 0] = torch.fft.ifft(v).real
    for step in range(1, n_int + 1):
        Nv = nonlin(v)
        a_ = E2 * v + Q * Nv
        Na = nonlin(a_)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c = E2 * a_ + Q * (2 * Nb - Nv)
        Nc = nonlin(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % per_save == 0:
            out[:, :, step // per_save] = torch.fft.ifft(v).real
    return out[:, ::up].cpu().numpy()


def hump(x, a, w, x0):
    """Gaussian of amplitude a and width w centred at x0, summed over the periodic images."""
    return sum(a * np.exp(-0.5 * ((x - x0 - m * L) / w) ** 2) for m in (-1, 0, 1))


def initial(ic, n, x, rng):
    """'periodic': two humps of random sign, |a| in AMP, width in WIDTH, centres at [0.30, 0.34] L and [0.48, 0.52] L
    (evaluation data); 'broad': one to three humps of random sign, |a| in [0.3, 1.5], width in [0.15, 0.8], anywhere."""
    u0 = np.zeros((n, len(x)))
    for r in range(n):
        if ic == "periodic":
            for lo, hi in ((0.30, 0.34), (0.48, 0.52)):
                s = 1.0 if rng.random() < 0.5 else -1.0
                u0[r] += hump(x, s * rng.uniform(*AMP), rng.uniform(*WIDTH), rng.uniform(lo, hi) * L)
        else:
            for _ in range(rng.integers(1, 4)):
                s = 1.0 if rng.random() < 0.5 else -1.0
                u0[r] += hump(x, s * rng.uniform(0.3, 1.5), rng.uniform(0.15, 0.8), rng.uniform(0, L))
    return u0


def training_initial(n, x, rng, kind="mixed"):
    """'humps': one or two 'periodic' humps in the bands [0.29, 0.35] L and [0.47, 0.53] L on a zero (p = 0.5) or
    uniform [-0.5, 0.5] background; 'mixed': half those, half 'broad', shuffled."""
    bands = ((0.29, 0.35), (0.47, 0.53))
    n_h = n if kind == "humps" else n - n // 2
    u0 = np.zeros((n_h, len(x)))
    for r in range(n_h):
        u0[r] = 0.0 if rng.random() < 0.5 else rng.uniform(-0.5, 0.5)
        chosen = bands if rng.integers(1, 3) == 2 else (bands[rng.integers(0, 2)],)
        for lo, hi in chosen:
            s = 1.0 if rng.random() < 0.5 else -1.0
            u0[r] += hump(x, s * rng.uniform(*AMP), rng.uniform(*WIDTH), rng.uniform(lo, hi) * L)
    if kind == "mixed":
        u0 = np.concatenate([u0, initial("broad", n // 2, x, rng)])[rng.permutation(n)]
    return u0


def generators(x, t, u, pde):
    """(xi, tau, phi) of names(pde); each [Nx, Nt]."""
    a = alpha(pde)
    zeros, ones = np.zeros_like(u), np.ones_like(u)
    X, Tg = np.meshgrid(x, t, indexing="ij")
    out = [(ones, zeros, zeros), (zeros, ones, zeros), (Tg, zeros, ones), (X, a * Tg, (1 - a) * u)]
    if a == 2:
        out.append((X * Tg, Tg ** 2, X - Tg * u))
    return out
