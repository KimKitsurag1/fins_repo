"""Benjamin-Ono equation on the torus [0, 2 pi): u_t + u u_x = H u_xx, H the Hilbert transform (symbol -i sgn k).

The right-hand side is nonlocal: no expression in the jet (x, t, u, u_x, u_xx, ...) exists, so Lie's algorithm on an
identified equation and LieNLSD's jet-space determining system do not apply as such. Point symmetries: d_x, d_t,
Galilean t d_x + d_u, scaling x d_x + 2 t d_t - u d_u. Solitary wave u = 4 c / (1 + c^2 (x - c t)^2), speed c > 0;
summed over periodic images it is 2 sinh(1/c) / (cosh(1/c) - cos(x - c t)) on [0, 2 pi), smooth and periodic.

Data mirror exp39's KdV: evaluation trajectories 'periodic' (two periodised solitons, c in [3, 5], centres at
[0.30, 0.34] L and [0.48, 0.52] L), stepper training data 'mixed' (half one or two periodised solitons on a zero or
constant background, half broadband periodised Lorentzian humps), Nx = 128 over T = 0.2 (a soliton moves 0.6-1.0).
solve: ETDRK4 (Kassam and Trefethen's contour integrals), 2/3 dealiasing, torch, float64, PER_SAVE steps per save.
"""

import numpy as np
import torch

import ks

L = 2 * np.pi
T = 0.2
PER_SAVE = 20
C_RANGE = (3.0, 5.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = ["∂_x", "∂_t", "Galilean", "scaling"]


def periodic_soliton(x, c, x0):
    """4 c / (1 + c^2 (x - x0)^2) summed over the images x0 + m L (closed form, L = 2 pi)."""
    return 2 * np.sinh(1 / c) / (np.cosh(1 / c) - np.cos(x - x0))


def solve(u0, T_end, Nt, per_save=PER_SAVE):
    """u0 [B, Nx] -> [B, Nx, Nt] on t = linspace(0, T_end, Nt)."""
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    Nx = u0.shape[1]
    n_int = (Nt - 1) * per_save
    h = T_end / n_int
    dev = DEVICE
    k = torch.fft.fftfreq(Nx, d=L / Nx, dtype=torch.float64, device=dev) * 2 * np.pi
    Lop = 1j * k * torch.abs(k)                                    # H u_xx
    E, E2 = torch.exp(h * Lop), torch.exp(h * Lop / 2)
    M = 32
    # the full circle: a mean over the upper half equals the integral only for real L (and then only its real part)
    r = torch.exp(2j * np.pi * (torch.arange(1, M + 1, dtype=torch.float64, device=dev) - 0.5) / M)
    LR = h * Lop[:, None] + r[None, :]
    Q = h * torch.mean((torch.exp(LR / 2) - 1) / LR, dim=1)
    f1 = h * torch.mean((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, dim=1)
    f2 = h * torch.mean((2 + LR + torch.exp(LR) * (-2 + LR)) / LR ** 3, dim=1)
    f3 = h * torch.mean((-4 - 3 * LR - LR ** 2 + torch.exp(LR) * (4 - LR)) / LR ** 3, dim=1)
    keep = (torch.abs(k) <= 2.0 / 3.0 * torch.abs(k).max()).to(torch.float64)
    g = -0.5j * k * keep                                           # -(u^2 / 2)_x

    def nonlin(v):
        return g * torch.fft.fft(torch.fft.ifft(v * keep).real ** 2)

    v = torch.fft.fft(torch.tensor(u0, device=dev)) * keep
    out = torch.zeros((len(u0), Nx, Nt), dtype=torch.float64, device=dev)
    out[:, :, 0] = torch.fft.ifft(v).real
    for step in range(1, n_int + 1):
        Nv = nonlin(v)
        a = E2 * v + Q * Nv
        Na = nonlin(a)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlin(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % per_save == 0:
            out[:, :, step // per_save] = torch.fft.ifft(v).real
    return out.cpu().numpy()


def initial(ic, n, x, rng):
    """'periodic': two periodised solitons (evaluation data); 'broad': one to three periodised Lorentzian humps of
    random amplitude and width anywhere on the torus (not solitons: they disperse)."""
    u0 = np.zeros((n, len(x)))
    for r in range(n):
        if ic == "periodic":
            for lo, hi in ((0.30, 0.34), (0.48, 0.52)):
                u0[r] += periodic_soliton(x, rng.uniform(*C_RANGE), rng.uniform(lo, hi) * L)
        else:
            for _ in range(rng.integers(1, 4)):
                w, a, x0 = rng.uniform(0.2, 1.0), rng.uniform(0.5, 4.0), rng.uniform(0, L)
                u0[r] += a * w / 4 * periodic_soliton(x, 1 / w, x0)     # peak ~ a
    return u0


def training_initial(n, x, rng, kind="mixed"):
    """'solitons': one or two periodised solitons (c in C_RANGE) in the bands [0.29, 0.35] L and [0.47, 0.53] L on a
    zero (p = 0.5) or uniform [-0.5, 0.5] background (a soliton on a constant background b is a travelling wave of speed
    c + b); 'mixed': half those, half 'broad' humps, shuffled; 'narrow', 'wide' (exp72): 'mixed' with the backgrounds
    scaled by ks.SCALE."""
    bands = ((0.29, 0.35), (0.47, 0.53))
    n_sol = n if kind == "solitons" else n - n // 2
    u0 = np.zeros((n_sol, len(x)))
    for r in range(n_sol):
        u0[r] = 0.0 if rng.random() < 0.5 else rng.uniform(-0.5, 0.5)
        u0[r] *= ks.SCALE.get(kind, 1.0)
        chosen = bands if rng.integers(1, 3) == 2 else (bands[rng.integers(0, 2)],)
        for lo, hi in chosen:
            u0[r] += periodic_soliton(x, rng.uniform(*C_RANGE), rng.uniform(lo, hi) * L)
    if kind in ks.SCALE:
        u0 = np.concatenate([u0, initial("broad", n // 2, x, rng)])[rng.permutation(n)]
    return u0


def generators(x, t, u):
    """(xi, tau, phi) of d_x, d_t, Galilean t d_x + d_u, scaling x d_x + 2 t d_t - u d_u; each [Nx, Nt]."""
    zeros, ones = np.zeros_like(u), np.ones_like(u)
    X, Tg = np.meshgrid(x, t, indexing="ij")
    return [(ones, zeros, zeros), (zeros, ones, zeros), (Tg, zeros, ones), (X, 2 * Tg, -u)]
