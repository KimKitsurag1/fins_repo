"""Kuramoto-Sivashinsky equation on the torus [0, L): u_t + u u_x + u_xx + u_xxxx = 0, L = 22 (chaotic; the smallest
box with a chaotic attractor studied in detail, Cvitanovic et al.).

Point symmetries: d_x, d_t, Galilean t d_x + d_u (no scaling: u_xx and u_xxxx weigh differently). The equation lies in
a polynomial jet library (to u_xxxx), so the two-stage pipeline can identify it; the test here is chaotic dynamics.
Data: random smooth initial fields (Fourier modes 1..4) integrated through a transient of T_SPIN to the attractor,
then recorded over T = 20 (about one Lyapunov time: lambda_max ~ 0.043 for L = 22). Evaluation trajectories have zero
mean; training trajectories ('mixed') sit on a constant background in half the cases, as KdV's and BO's.
solve: ETDRK4 (Kassam and Trefethen), 2/3 dealiasing, torch, float64, PER_SAVE steps per save.
"""

import numpy as np
import torch

L = 22.0
T = 20.0
T_SPIN = 200.0
PER_SAVE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = ["∂_x", "∂_t", "Galilean"]
# the constant backgrounds of the training recipes with backgrounds (KdV, BO, KS): the same draws, scaled;
# 'narrow' and 'wide' for exp72
SCALE = {"mixed": 1.0, "narrow": 0.2, "wide": 6.0}


def _coefficients(Nx, h):
    dev = DEVICE
    k = torch.fft.fftfreq(Nx, d=L / Nx, dtype=torch.float64, device=dev) * 2 * np.pi
    Lop = (k ** 2 - k ** 4).to(torch.complex128)
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
    return E, E2, Q, f1, f2, f3, -0.5j * k * keep, keep


def solve(u0, T_end, Nt, per_save=PER_SAVE):
    """u0 [B, Nx] -> [B, Nx, Nt] on t = linspace(0, T_end, Nt)."""
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    Nx = u0.shape[1]
    n_int = (Nt - 1) * per_save
    E, E2, Q, f1, f2, f3, g, keep = _coefficients(Nx, T_end / n_int)

    def nonlin(v):
        return g * torch.fft.fft(torch.fft.ifft(v * keep).real ** 2)

    v = torch.fft.fft(torch.tensor(u0, device=DEVICE)) * keep
    out = torch.zeros((len(u0), Nx, Nt), dtype=torch.float64, device=DEVICE)
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


def on_attractor(n, x, rng, spin=T_SPIN):
    """n states on the attractor: random fields of Fourier modes 1..4 (amplitude ~1), zero mean, run for spin."""
    u0 = np.zeros((n, len(x)))
    for r in range(n):
        for m in range(1, 5):
            u0[r] += rng.standard_normal() * np.cos(2 * np.pi * m * x / L + rng.uniform(0, 2 * np.pi))
    return solve(u0, spin, 2, per_save=int(round(spin / 0.01)))[:, :, -1]     # h = 0.01: the nonlinear CFL ~ 0.4


def initial(ic, n, x, rng):
    return on_attractor(n, x, rng)


def training_initial(n, x, rng, kind="mixed"):
    """'mixed': states on the attractor on a zero (p = 0.5) or uniform [-0.5, 0.5] constant background, as KdV's and
    BO's training data (on a background b the pattern drifts at speed b); 'attractor': zero mean only (the mean is
    conserved, so no trajectory moves it); 'narrow', 'wide' (exp72): 'mixed' with the backgrounds scaled by SCALE."""
    u0 = on_attractor(n, x, rng)
    if kind in SCALE:
        u0 += SCALE[kind] * np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-0.5, 0.5, n))[:, None]
    return u0


def generators(x, t, u):
    """(xi, tau, phi) of d_x, d_t, Galilean t d_x + d_u; each [Nx, Nt]."""
    zeros, ones = np.zeros_like(u), np.ones_like(u)
    _, Tg = np.meshgrid(x, t, indexing="ij")
    return [(ones, zeros, zeros), (zeros, ones, zeros), (Tg, zeros, ones)]
