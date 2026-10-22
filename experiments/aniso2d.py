"""Anisotropic porous-medium flow with anomalous dissipation on the periodic box [-L/2, L/2)^2 (exp45's setting):
u_t = div_A(u grad_A u) - nu (-div_A grad_A)^(a/2) u, grad_A = (d_x, eps d_y): an anisotropic medium (permeability
diag(1, eps^2)) with a fractional (Levy) dissipation of the same anisotropy; symbols k_A^2 = k_x^2 + eps^2 k_y^2 and
(k_A^2)^(a/2). eps = 1, nu = 0 is exp45's u_t = Lap(u^2 / 2).

The equation is the isotropic one in the stretched coordinate y' = y / eps, so its point symmetries are the isotropic
algebra conjugated by the stretch: d_x, d_y, d_t, the stretched rotation -y / eps d_x + eps x d_y (eps read from the
direction in the plane (y d_x, x d_y)), and for a < 2 one scaling x d_x + y d_y + a t d_t + (2 - a) u d_u (the
dissipation fixes the time weight; exp45's second scaling t d_t - u d_u is broken). For a < 2 the dissipation has no jet
expression; a local library cannot represent it, and a nonlocal completion recovers the stretched rotation only if its
symbol family is elliptic.

Solver: exp45's pseudo-spectral RK4 with the circular 2/3 mask and the exact tangent; the linear part is a multiplier,
so RK4 is used as it is (stability checked in exp60's --check).
"""

import math

import numpy as np
import torch

import exp45_neural_2d as e45

EPS, NU = 0.6, 0.3
EXPONENTS = (1.25, 1.5, 1.75)


class Grid(e45.Grid):
    """exp45's grid with the anisotropic symbols; a = 2 is local anisotropic diffusion."""

    def __init__(self, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu", eps=EPS, a=1.5, nu=NU):
        super().__init__(n, L, score, dtype, device)
        self.eps, self.a, self.nu = eps, a, nu
        kx = 2 * math.pi * torch.fft.fftfreq(n, d=self.dx, dtype=dtype, device=device)
        ky = 2 * math.pi * torch.fft.rfftfreq(n, d=self.dx, dtype=dtype, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        KA2 = KX ** 2 + eps ** 2 * KY ** 2
        self.lapA = -KA2 * self.mask
        self.frac = -nu * KA2 ** (a / 2) * self.mask

    def cast(self, dtype, device=None):
        return Grid(self.n, self.L, self.score, dtype, self.device if device is None else device, self.eps, self.a,
                    self.nu)


def rhs(g, u):
    return torch.fft.irfft2(g.lapA * torch.fft.rfft2(0.5 * u * u) + g.frac * torch.fft.rfft2(u), s=(g.n, g.n))


def rhs_tan(g, u, d):
    return torch.fft.irfft2(g.lapA * torch.fft.rfft2(u * d) + g.frac * torch.fft.rfft2(d), s=(g.n, g.n))


def rk4(g, u, h):
    k1 = rhs(g, u)
    k2 = rhs(g, u + 0.5 * h * k1)
    k3 = rhs(g, u + 0.5 * h * k2)
    k4 = rhs(g, u + h * k3)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_tan(g, u, d, h):
    k1, l1 = rhs(g, u), rhs_tan(g, u, d)
    u2, d2 = u + 0.5 * h * k1, d + 0.5 * h * l1
    k2, l2 = rhs(g, u2), rhs_tan(g, u2, d2)
    u3, d3 = u + 0.5 * h * k2, d + 0.5 * h * l2
    k3, l3 = rhs(g, u3), rhs_tan(g, u3, d3)
    u4, d4 = u + h * k3, d + h * l3
    k4, l4 = rhs(g, u4), rhs_tan(g, u4, d4)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), d + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4)


def solve(g, u0, nt, dt, nsub):
    """u0 [B, n, n] -> [B, nt, n, n]."""
    out, u = [u0], u0
    for _ in range(nt - 1):
        for _ in range(nsub):
            u = rk4(g, u, dt / nsub)
        out.append(u)
    return torch.stack(out, 1)


class ExactOp(e45.ExactOp):
    def cast(self, dtype, device=None):
        return ExactOp(self.g.cast(dtype, device), self.dt, self.nsub)

    def transport(self, u0, d0, K):
        u, d, out = u0[:, None], d0, []
        for _ in range(K):
            for _ in range(self.nsub):
                u, d = rk4_tan(self.g, u, d, self.dt / self.nsub)
            out.append(d)
        return torch.stack(out, 1)


def trajectories(g, n_traj, seed, nt, dt, nsub):
    return solve(g, e45.initial(g, n_traj, np.random.default_rng(seed)), nt, dt, nsub)


NAMES = ("d_x", "d_y", "d_t", "stretched rotation", "scaling")


def true_coefficients(eps, a):
    """The algebra as coefficient vectors over exp45's 60 basis fields (scaled components, MONS order): d_x, d_y, d_t,
    -y / eps d_x + eps x d_y, x d_x + y d_y + a t d_t + (2 - a) u d_u."""
    nm = len(e45.MONS)
    C = np.zeros((4 * nm, 5))
    idx = {m: i for i, m in enumerate(e45.MONS)}
    put = [[(0, (0, 0, 0, 0), 1)], [(1, (0, 0, 0, 0), 1)], [(2, (0, 0, 0, 0), 1)],
           [(0, (0, 1, 0, 0), -1 / eps), (1, (1, 0, 0, 0), eps)],
           [(0, (1, 0, 0, 0), 1), (1, (0, 1, 0, 0), 1), (2, (0, 0, 1, 0), a), (3, (0, 0, 0, 1), 2 - a)]]
    for j, terms in enumerate(put):
        for c, mon, v in terms:
            C[c * nm + idx[mon], j] = v
    return C


def basis_vector(entries):
    """Coefficient vector of a field given as [(component, monomial, value)]."""
    nm = len(e45.MONS)
    idx = {m: i for i, m in enumerate(e45.MONS)}
    v = np.zeros(4 * nm)
    for c, mon, val in entries:
        v[c * nm + idx[mon]] = val
    return v
