"""Transport by a nonlocal velocity with fractional dissipation on the torus [0, 2 pi) (Cordoba, Cordoba and Fontelos
type): u_t + (H u) u_x = -nu Lambda^alpha u, H the Hilbert transform (symbol -i sgn k), Lambda^alpha the multiplier
|k|^alpha, nu = NU0 / K0^alpha as in fburgers.py. Equation names ccf125 ... ccf175 (alpha = 1.25 ... 1.75).

The velocity H u ignores constants: the shift d_u is a symmetry and the Galilean boost is not. Point symmetries: d_x,
d_t, d_u, x d_x + alpha t d_t + (1 - alpha) u d_u. Neither the bilinear nonlocal transport nor the dissipation has a jet
expression; alpha > 1 is subcritical (global smooth solutions). Data and solver as fburgers.py: two Gaussian humps,
128 x 128 over T = 1, ETDRK4 on UP times the data's grid with 2/3 dealiasing.
"""

import numpy as np
import torch

import fburgers as fb

L, T, PER_SAVE, UP = fb.L, fb.T, fb.PER_SAVE, fb.UP
EQUATIONS = ("ccf125", "ccf150", "ccf175")
DEVICE = fb.DEVICE


def alpha(pde):
    """'ccf150' -> 1.5."""
    return int(pde[3:]) / 100


def nu(a):
    return fb.nu(a)


def names(pde):
    return ["∂_x", "∂_t", "shift", "scaling"]


def initial(ic, n, x, rng):
    return fb.initial(ic, n, x, rng)


def training_initial(n, x, rng, kind="mixed"):
    return fb.training_initial(n, x, rng, kind)


def solve(u0, T_end, Nt, a, per_save=PER_SAVE, up=UP):
    """u0 [B, Nx] -> [B, Nx, Nt] on t = linspace(0, T_end, Nt), exponent a; solved on up Nx points."""
    u0 = fb.upsample(np.atleast_2d(np.asarray(u0, dtype=np.float64)), up)
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
    hil, dx = -1j * torch.sign(k) * keep, 1j * k * keep

    def nonlin(v):                                                 # -(H u) u_x
        return -keep * torch.fft.fft(torch.fft.ifft(hil * v).real * torch.fft.ifft(dx * v).real)

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


def generators(x, t, u, pde):
    """(xi, tau, phi) of names(pde); each [Nx, Nt]."""
    a = alpha(pde)
    zeros, ones = np.zeros_like(u), np.ones_like(u)
    X, Tg = np.meshgrid(x, t, indexing="ij")
    return [(ones, zeros, zeros), (zeros, ones, zeros), (zeros, zeros, ones), (X, a * Tg, (1 - a) * u)]
