"""Experiment 45 -- neural symmetry generators of a 2D PDE, trained through a solution operator's Jacobian.

u_t = div(u grad u) = Lap(u^2 / 2) on the periodic box [-8, 8)^2. Data: two or three anisotropic Gaussian bumps
(widths 0.9-1.6, centres in [-2, 2]^2) on a background b0 in [0.25, 0.45]; the field stays constant to ~1e-4 near
the seam, so the characteristics of the x-dependent generators have no jump there; scored block |x|, |y| < 4.5. Point-symmetry algebra (Ovsiannikov; the
exponent 1 is not the conformal case -4 / (n + 2) = -1 of n = 2), 6-dim:
  d_x, d_y, d_t, rotation -y d_x + x d_y, D1 = x d_x + y d_y + 2t d_t, D2 = t d_t - u d_u.

Operators: 'exact' -- pseudo-spectral RK4 with the circular 2/3 mask; the tangent goes through the same RK4
stages, so it is the exact Jacobian of the discrete map; 'fno' -- an FNO2d stepper (no positional input, so
translation-equivariant on the grid) trained on trajectories of its own data seed, tangent propagated through
its layers.

Generator classes, both scored on held-out trajectories:
  poly   -- FINS's class: monomials of degree <= 2 in X = x / 8, Y = y / 8, T = t / T_max, U = u / sigma in each
            of the four components (60 fields); generalized eigenproblem of the relative transport pencil (exp39);
  neural -- m point fields (x, y, t, u) -> (xi, eta, tau, phi): a shared swish trunk and one hidden layer per
            generator (the layout of Ko et al. 2024, narrower), trained by the Ky Fan loss Tr((D + eps)^-1 N),
            the sum of the generalized eigenvalues of the same pencil restricted to the networks' span (invariant
            to any invertible mixing of the m fields, so no orthogonality term is needed).
Scores: missing dimensions of the algebra, k_B - ||Qa' Qb||_F^2 (exp42), in characteristic space (the part any
data-based method can identify; primary) and in field space (scaled components xi / 8, eta / 8, tau / T_max,
phi / sigma); the pencil's eigenvalues inside the learned span on held-out pairs; nulls: Gaussian-coefficient
subspaces of the polynomial class and the untrained network.

  python exp45_neural_2d.py --check
  python exp45_neural_2d.py --seed 42 --m 6 --save neural2d_exact_m6_s42
  python exp45_neural_2d.py --op fno --train-fno-if-missing --seed 42 --m 6 --save neural2d_fno_m6_s42
"""

import argparse
import copy
import json
import math
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf
from scipy.linalg import eigh

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
MONS = [(a, b, c, d) for a in range(3) for b in range(3) for c in range(3) for d in range(3) if a + b + c + d <= 2]
NAMES = ("d_x", "d_y", "d_t", "rotation", "D1", "D2")
TOL = 1e-6


# ---------------------------------------------------------------------------------------------------------------
# Grid, solver, tangent
# ---------------------------------------------------------------------------------------------------------------
class Grid:
    """Periodic box [-L/2, L/2)^2 with n^2 points; rfft2 symbols, circular 2/3 mask, scored block |x|, |y| < score."""

    def __init__(self, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu"):
        self.n, self.L, self.score, self.dx, self.dtype, self.device = n, L, score, L / n, dtype, device
        self.x = -L / 2 + self.dx * torch.arange(n, dtype=dtype, device=device)
        self.X, self.Y = torch.meshgrid(self.x, self.x, indexing="ij")
        kx = 2 * math.pi * torch.fft.fftfreq(n, d=self.dx, dtype=dtype, device=device)
        ky = 2 * math.pi * torch.fft.rfftfreq(n, d=self.dx, dtype=dtype, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        self.mask = (K2.sqrt() < (2.0 / 3.0) * math.pi / self.dx).to(dtype)
        self.lap = -K2 * self.mask
        self.ikx, self.iky = 1j * KX, 1j * KY
        self.kmax2 = float((K2 * self.mask).max())
        i0 = int(round((L / 2 - score) / self.dx))
        self.sl = slice(i0, n - i0)

    def cast(self, dtype, device=None):
        return Grid(self.n, self.L, self.score, dtype, self.device if device is None else device)


def rhs(g, u):
    return torch.fft.irfft2(g.lap * torch.fft.rfft2(0.5 * u * u), s=(g.n, g.n))


def rhs_tan(g, u, d):
    return torch.fft.irfft2(g.lap * torch.fft.rfft2(u * d), s=(g.n, g.n))


def rk4(g, u, h):
    k1 = rhs(g, u)
    k2 = rhs(g, u + 0.5 * h * k1)
    k3 = rhs(g, u + 0.5 * h * k2)
    k4 = rhs(g, u + h * k3)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_tan(g, u, d, h):
    """One RK4 step of u and its derivative along d (the exact Jacobian of the step)."""
    k1, l1 = rhs(g, u), rhs_tan(g, u, d)
    u2, d2 = u + 0.5 * h * k1, d + 0.5 * h * l1
    k2, l2 = rhs(g, u2), rhs_tan(g, u2, d2)
    u3, d3 = u + 0.5 * h * k2, d + 0.5 * h * l2
    k3, l3 = rhs(g, u3), rhs_tan(g, u3, d3)
    u4, d4 = u + h * k3, d + h * l3
    k4, l4 = rhs(g, u4), rhs_tan(g, u4, d4)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), d + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4)


def solve(g, u0, nt, dt, nsub):
    """u0 [B, n, n] -> [B, nt, n, n], snapshots every dt, nsub RK4 steps each."""
    out, u = [u0], u0
    for _ in range(nt - 1):
        for _ in range(nsub):
            u = rk4(g, u, dt / nsub)
        out.append(u)
    return torch.stack(out, 1)


class ExactOp:
    """Discrete spectral map over one snapshot interval and its exact tangent."""

    def __init__(self, g, dt, nsub):
        self.g, self.dt, self.nsub = g, dt, nsub

    def cast(self, dtype, device=None):
        return ExactOp(self.g.cast(dtype, device), self.dt, self.nsub)

    def transport(self, u0, d0, K):
        """J_k d0 for k = 1..K along the operator's own rollout: u0 [P, n, n], d0 [P, m, n, n] -> [P, K, m, n, n]."""
        u, d, out = u0[:, None], d0, []
        for _ in range(K):
            for _ in range(self.nsub):
                u, d = rk4_tan(self.g, u, d, self.dt / self.nsub)
            out.append(d)
        return torch.stack(out, 1)


def initial(g, n_traj, rng, centre=2.0, width=(0.9, 1.6), amp=(0.4, 0.8), background=(0.25, 0.45)):
    """Two or three anisotropic Gaussian bumps on a constant background, projected onto the masked band."""
    X, Y = g.X.numpy(), g.Y.numpy()
    u0 = np.zeros((n_traj, g.n, g.n))
    for r in range(n_traj):
        u0[r] = rng.uniform(*background)
        for _ in range(rng.integers(2, 4)):
            cx, cy = rng.uniform(-centre, centre, 2)
            wa, wb = rng.uniform(*width, 2)
            th = rng.uniform(0, np.pi)
            xr = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
            yr = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
            u0[r] += rng.uniform(*amp) * np.exp(-(xr / wa) ** 2 - (yr / wb) ** 2)
    u0 = torch.tensor(u0, dtype=g.dtype)
    return torch.fft.irfft2(g.mask * torch.fft.rfft2(u0), s=(g.n, g.n))


def trajectories(g, n_traj, seed, nt, dt, nsub):
    return solve(g, initial(g, n_traj, np.random.default_rng(seed)), nt, dt, nsub)


def derivatives(g, u, dt):
    """Spectral u_x, u_y; u_t by fourth-order central differences in time (second order next to the ends)."""
    v = torch.fft.rfft2(u)
    ux = torch.fft.irfft2(g.ikx * v, s=(g.n, g.n))
    uy = torch.fft.irfft2(g.iky * v, s=(g.n, g.n))
    ut = torch.zeros_like(u)
    ut[:, 2:-2] = (u[:, :-4] - 8 * u[:, 1:-3] + 8 * u[:, 3:-1] - u[:, 4:]) / (12 * dt)
    ut[:, 1], ut[:, -2] = (u[:, 2] - u[:, 0]) / (2 * dt), (u[:, -1] - u[:, -3]) / (2 * dt)
    ut[:, 0], ut[:, -1] = (u[:, 1] - u[:, 0]) / dt, (u[:, -1] - u[:, -2]) / dt
    return ux, uy, ut


# ---------------------------------------------------------------------------------------------------------------
# Data, generator fields, characteristics
# ---------------------------------------------------------------------------------------------------------------
class Data:
    """Trajectories [B, nt, n, n] with derivatives; scale (x / Lx, t / T, u / sigma, network input range) is fixed by
    the training set and passed to every other set."""

    def __init__(self, g, u, dt, scale=None):
        self.g, self.u, self.dt = g, u, dt
        self.ux, self.uy, self.ut = derivatives(g, u, dt)
        self.t = dt * torch.arange(u.shape[1], dtype=u.dtype)
        if scale is None:
            flat = u.flatten()[::7]
            q = torch.quantile(flat.double(), torch.tensor([0.001, 0.999], dtype=torch.float64))
            scale = dict(Lx=g.L / 2, T=float(self.t[-1]), sigma=float(u.std()), lo=float(q[0]), hi=float(q[1]))
        self.s = scale

    def cast(self, dtype, device=None):
        d = copy.copy(self)
        d.g = self.g.cast(dtype, device)
        for k in ("u", "ux", "uy", "ut", "t"):
            setattr(d, k, getattr(self, k).to(dtype=dtype, device=d.g.device))
        return d

    def at(self, r, j, reg):
        """u, u_x, u_y, u_t, X, Y, T at snapshots (r, j) (index tensors of one shape S) on reg x reg: [*S, h, w]."""
        u, ux, uy, ut = (a[r, j][..., reg, reg] for a in (self.u, self.ux, self.uy, self.ut))
        X = (self.g.X[reg, reg] / self.s["Lx"]).expand_as(u)
        Y = (self.g.Y[reg, reg] / self.s["Lx"]).expand_as(u)
        T = (self.t[j] / self.s["T"])[..., None, None].expand_as(u)
        return u, ux, uy, ut, X, Y, T

    def factors(self, u, ux, uy, ut):
        """Q / sigma = sum_c o_c f_c for scaled components o = (xi / Lx, eta / Lx, tau / T, phi / sigma): [*S, 4, h, w]."""
        sx, st = self.s["Lx"] / self.s["sigma"], self.s["T"] / self.s["sigma"]
        return torch.stack([-sx * ux, -sx * uy, -st * ut, torch.ones_like(u)], -3)

    def net_input(self, u, X, Y, T):
        """Network input: x / score, y / score, 2 t / T - 1, u mapped from its 0.1-99.9 % range to [-1, 1]."""
        c = self.s["Lx"] / self.g.score
        U = 2 * (u - self.s["lo"]) / (self.s["hi"] - self.s["lo"]) - 1
        return torch.stack([c * X, c * Y, 2 * T - 1, U], -1)

    def factor_rms(self):
        """Rms of the four characteristic factors over the interior snapshots: the networks' output units."""
        f = self.factors(self.u[:, 2:-2], self.ux[:, 2:-2], self.uy[:, 2:-2], self.ut[:, 2:-2])
        return f.pow(2).mean(dim=(0, 1, 3, 4)).sqrt()


def true_fields(X, Y, T, U):
    """The algebra in scaled components: [*S, 6, 4, h, w]."""
    O, Z = torch.ones_like(X), torch.zeros_like(X)
    gens = [(O, Z, Z, Z), (Z, O, Z, Z), (Z, Z, O, Z), (-Y, X, Z, Z), (X, Y, 2 * T, Z), (Z, Z, T, -U)]
    return torch.stack([torch.stack(c, -3) for c in gens], -4)


def monomials(X, Y, T, U):
    return torch.stack([X ** a * Y ** b * T ** c * U ** e for a, b, c, e in MONS], -3)      # [*S, 15, h, w]


def true_coefficients():
    """The algebra as coefficient vectors of the 60 basis fields (component-major, MONS order)."""
    C = np.zeros((4 * len(MONS), 6))
    idx = {m: i for i, m in enumerate(MONS)}
    put = [[(0, (0, 0, 0, 0), 1)], [(1, (0, 0, 0, 0), 1)], [(2, (0, 0, 0, 0), 1)],
           [(0, (0, 1, 0, 0), -1), (1, (1, 0, 0, 0), 1)],
           [(0, (1, 0, 0, 0), 1), (1, (0, 1, 0, 0), 1), (2, (0, 0, 1, 0), 2)],
           [(2, (0, 0, 1, 0), 1), (3, (0, 0, 0, 1), -1)]]
    for j, terms in enumerate(put):
        for c, mon, v in terms:
            C[c * len(MONS) + idx[mon], j] = v
    return C


class GenNet(nn.Module):
    """m point fields R^4 -> R^4 (scaled components): shared swish trunk, one swish hidden layer per generator.
    Output c is divided by the rms of its characteristic factor (out_rms), so that unit outputs of the four
    components move the characteristic equally; a change of units, not of the class."""

    def __init__(self, m, width=64, head=16, out_rms=None):
        super().__init__()
        self.m = m
        self.trunk = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.hid = nn.Linear(width, m * head)
        self.w = nn.Parameter(torch.randn(m, head, 4) / math.sqrt(head))
        self.b = nn.Parameter(torch.zeros(m, 4))
        self.register_buffer("out_rms", torch.ones(4) if out_rms is None else torch.as_tensor(out_rms).float())

    def forward(self, z):
        h = tf.silu(self.hid(self.trunk(z)))
        h = h.view(*h.shape[:-1], self.m, -1)
        return (torch.einsum("...mh,mhc->...mc", h, self.w) + self.b) / self.out_rms.to(z.dtype)  # [..., m, 4]


def net_fields(net, d, r, j, reg):
    u, ux, uy, ut, X, Y, T = d.at(r, j, reg)
    return net(d.net_input(u, X, Y, T)).movedim(-2, -4).movedim(-1, -3), d.factors(u, ux, uy, ut)


def chars_net(net):
    def f(d, r, j, reg):
        o, fac = net_fields(net, d, r, j, reg)
        return (o * fac.unsqueeze(-4)).sum(-3)                                           # [*S, m, h, w]
    return f


def chars_poly(d, r, j, reg):
    u, ux, uy, ut, X, Y, T = d.at(r, j, reg)
    mono, fac = monomials(X, Y, T, u / d.s["sigma"]), d.factors(u, ux, uy, ut)
    Q = fac.unsqueeze(-3) * mono.unsqueeze(-4)                                           # [*S, 4, 15, h, w]
    return Q.reshape(*Q.shape[:-4], 4 * len(MONS), *Q.shape[-2:])


def chars_true(d, r, j, reg):
    u, ux, uy, ut, X, Y, T = d.at(r, j, reg)
    return (true_fields(X, Y, T, u / d.s["sigma"]) * d.factors(u, ux, uy, ut).unsqueeze(-4)).sum(-3)


# ---------------------------------------------------------------------------------------------------------------
# Transport pencil
# ---------------------------------------------------------------------------------------------------------------
def pencil_terms(op, d, charfn, r, j0, K):
    """Relative pencil of exp39 on pairs (r, j0): A = J_k Q0 - Q0, B = Q(t0 + k) - Q0 on the scored block, k = 1..K.
    Returns N = (A - B)'(A - B), A'A + B'B, K Q0'Q0 (float64) and tr B'B."""
    sl = d.g.sl
    r, j0 = r.to(d.u.device), j0.to(d.u.device)
    Q0 = charfn(d, r, j0, slice(None))                                                   # [P, m, n, n]
    dl = op.transport(d.u[r, j0], Q0, K)[..., sl, sl]                                    # [P, K, m, s, s]
    jk = j0[:, None] + torch.arange(1, K + 1, device=j0.device)
    Qk = charfn(d, r[:, None].expand_as(jk), jk, sl)                                     # [P, K, m, s, s]
    Q0s = Q0[..., sl, sl][:, None]
    m = Q0.shape[1]
    A = (dl - Q0s).movedim(2, 0).reshape(m, -1).double()
    B = (Qk - Q0s).movedim(2, 0).reshape(m, -1).double()
    Q = Q0s[:, 0].movedim(1, 0).reshape(m, -1).double()
    R = A - B
    return R @ R.T, A @ A.T + B @ B.T, K * (Q @ Q.T), float((B ** 2).sum())


def pencil(op, d, charfn, pairs, K, chunk):
    tot = None
    with torch.no_grad():
        for c in range(0, len(pairs), chunk):
            p = pairs[c:c + chunk]
            terms = pencil_terms(op, d, charfn, p[:, 0], p[:, 1], K)
            tot = list(terms) if tot is None else [a + b for a, b in zip(tot, terms)]
    N, DA, G0, trBB = tot
    return N.cpu().numpy(), DA.cpu().numpy(), G0.cpu().numpy(), trBB


def gsolve(N, D):
    """exp34.solve: generalized eigenproblem after diagonal scaling."""
    s = np.sqrt(np.clip(np.diag(D), 1e-60, None))
    lam, V = eigh(N / np.outer(s, s), D / np.outer(s, s) + 1e-10 * np.eye(len(s)))
    return lam, V / s[:, None]


def kyfan(N, D, eps=1e-6):
    """Tr(L^-1 N L^-T), L L' = D + eps after scaling D to unit diagonal: the sum of the generalized eigenvalues in
    the span (the scaling leaves it unchanged and keeps eps relative for every generator)."""
    m = N.shape[0]
    s = torch.sqrt(torch.diagonal(D))
    N, D = N / torch.outer(s, s), D / torch.outer(s, s)
    L = torch.linalg.cholesky(D + eps * torch.eye(m, dtype=D.dtype, device=D.device))
    M = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, N, upper=False).T, upper=False)
    return torch.trace(M), torch.linalg.eigvalsh(0.5 * (M + M.T)).detach()


def pairs_of(n_traj, nt, K, every, first=2):
    """(trajectory, start) with u_t of fourth order at every used snapshot: first <= j0, j0 + K <= nt - 3."""
    return torch.tensor([(r, j) for r in range(n_traj) for j in range(first, nt - 2 - K, every)])


# ---------------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------------
def orth(V):
    U, s, _ = np.linalg.svd(V.T, full_matrices=False)
    return U[:, s > TOL * s[0]] if s.size and s[0] > 0 else U[:, :0]


def missing(Fa, Fb):
    Qa, Qb = orth(Fa), orth(Fb)
    return float(Fb.shape[0] - ((Qa.T @ Qb) ** 2).sum())


def novel(Fa, Fb):
    """Share of each b-vector's part orthogonal to the other b-vectors inside span(Fa) (exp42 novel containment)."""
    Qa, out = orth(Fa), []
    for i in range(len(Fb)):
        Qo = orth(np.delete(Fb, i, 0))
        p = Fb[i] - Qo @ (Qo.T @ Fb[i])
        out.append(float(np.linalg.norm(Qa.T @ p) / np.linalg.norm(p)))
    return np.array(out)


class Probe:
    """Sample points of a held-out set: snapshots every 4 inside [2, nt - 3], every second point of the scored block;
    true fields / characteristics and the polynomial basis there (float64 numpy)."""

    def __init__(self, d, every_t=4, every_x=2):
        g = d.g
        sl = slice(g.sl.start, g.sl.stop, every_x)
        js = torch.arange(2, d.u.shape[1] - 2, every_t)
        R, J = torch.meshgrid(torch.arange(d.u.shape[0]), js, indexing="ij")
        self.d, self.r, self.j, self.reg = d, R, J, sl
        u, ux, uy, ut, X, Y, T = d.at(R, J, sl)
        U = u / d.s["sigma"]
        self.fac = d.factors(u, ux, uy, ut)                                              # [B, nj, 4, h, w]
        self.true_f = self.flat_fields(true_fields(X, Y, T, U))
        self.true_q = self.flat_chars((true_fields(X, Y, T, U) * self.fac.unsqueeze(-4)).sum(-3))
        mono = monomials(X, Y, T, U)                                                     # [B, nj, 15, h, w]
        self.mono = mono.movedim(2, 0).reshape(len(MONS), -1).numpy()
        self.facf = self.fac.movedim(2, 0).reshape(4, -1).numpy()

    @staticmethod
    def flat_fields(o):                                                                  # [*S, k, 4, h, w] -> [k, 4 M]
        return o.movedim(-4, 0).movedim(-3, 1).reshape(o.shape[-4], -1).numpy()

    @staticmethod
    def flat_chars(q):                                                                   # [*S, k, h, w] -> [k, M]
        return q.movedim(-3, 0).reshape(q.shape[-3], -1).numpy()

    def poly(self, C):
        """Fields and characteristics of coefficient vectors C [60, k]."""
        nm = len(MONS)
        comp = np.stack([C[c * nm:(c + 1) * nm].T @ self.mono for c in range(4)], 1)    # [k, 4, M]
        return comp.reshape(C.shape[1], -1), np.einsum("kcm,cm->km", comp, self.facf)

    def net(self, net):
        net = copy.deepcopy(net).to(dtype=self.d.u.dtype, device="cpu")
        with torch.no_grad():
            o, _ = net_fields(net, self.d, self.r, self.j, self.reg)
        return self.flat_fields(o), self.flat_chars((o * self.fac.unsqueeze(-4)).sum(-3))

    def scores(self, F, Qc):
        return dict(missing_q=missing(Qc, self.true_q), missing_f=missing(F, self.true_f),
                    novel_q=novel(Qc, self.true_q).tolist(), rank_q=int(orth(Qc).shape[1]))


def random_null(probe, k, n, rng):
    res = [probe.scores(*probe.poly(rng.standard_normal((4 * len(MONS), k)))) for _ in range(n)]
    out = {key: float(np.mean([r[key] for r in res])) for key in ("missing_q", "missing_f")}
    out.update({f"std_{key}": float(np.std([r[key] for r in res])) for key in ("missing_q", "missing_f")})
    return out


# ---------------------------------------------------------------------------------------------------------------
# FNO2d stepper with an explicit tangent
# ---------------------------------------------------------------------------------------------------------------
def gelu_prime(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.cout, self.modes = cout, modes
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.randn(cin, cout, modes, modes, 2))
        self.w2 = nn.Parameter(s * torch.randn(cin, cout, modes, modes, 2))

    def forward(self, x):
        m = self.modes
        xf = torch.fft.rfft2(x)
        w1, w2 = (torch.view_as_complex(w.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
                                        .contiguous()) for w in (self.w1, self.w2))
        out = torch.zeros(x.shape[0], self.cout, x.shape[-2], x.shape[-1] // 2 + 1, dtype=xf.dtype, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], w2)
        return torch.fft.irfft2(out, s=x.shape[-2:])


class FNO2d(nn.Module):
    """u_{j+1} = u_j + inc * P(layers(lift((u_j - c) / s))): periodic, no positional input; with grid=True the lift also
    takes the coordinates x / (L / 2), y / (L / 2) as two channels (the FNO paper's default), which breaks the
    translations and the rotation."""

    def __init__(self, width=32, modes=16, layers=4, proj=128, grid=False):
        super().__init__()
        self.grid = grid
        self.lift = nn.Conv2d(3 if grid else 1, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.p1, self.p2 = nn.Conv2d(width, proj, 1), nn.Conv2d(proj, 1, 1)
        self.register_buffer("norm", torch.tensor([0.0, 1.0, 1.0]))                      # c, s, inc

    def lifted(self, u):
        c, s, inc = self.norm
        x = ((u - c) / s)[:, None]
        if self.grid:
            n = u.shape[-1]
            r = (torch.arange(n, dtype=u.dtype, device=u.device) - n // 2) / (n // 2)
            X, Y = torch.meshgrid(r, r, indexing="ij")
            x = torch.cat([x, X.expand(u.shape[0], 1, n, n), Y.expand(u.shape[0], 1, n, n)], 1)
        return self.lift(x)

    def forward(self, u):
        c, s, inc = self.norm
        h = self.lifted(u)
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            h = k(h) + w(h)
            h = tf.gelu(h) if i < len(self.spec) - 1 else h
        return u + inc * self.p2(tf.gelu(self.p1(h)))[:, 0]

    def tangent(self, u, d):
        """S(u) and J_S(u) d: u [P, n, n] (no grad), d [P, m, n, n]; linear and differentiable in d."""
        c, s, inc = self.norm
        P, m = d.shape[:2]
        with torch.no_grad():
            h = self.lifted(u)
        dh = tf.conv2d(d.reshape(P * m, 1, *d.shape[-2:]) / s, self.lift.weight[:, :1])
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            with torch.no_grad():
                pre = k(h) + w(h)
            dpre = k(dh) + tf.conv2d(dh, w.weight)
            if i < len(self.spec) - 1:
                with torch.no_grad():
                    gp = gelu_prime(pre)
                    h = tf.gelu(pre)
                dh = gp.repeat_interleave(m, 0) * dpre
            else:
                with torch.no_grad():
                    h = pre
                dh = dpre
        with torch.no_grad():
            q = self.p1(h)
            out = u + inc * self.p2(tf.gelu(q))[:, 0]
            gq = gelu_prime(q)
        dq = gq.repeat_interleave(m, 0) * tf.conv2d(dh, self.p1.weight)
        dout = tf.conv2d(dq, self.p2.weight)[:, 0].reshape(P, m, *d.shape[-2:])
        return out, d + inc * dout


class FNOOp:
    def __init__(self, model):
        self.model = model

    def cast(self, dtype, device=None):
        m = copy.deepcopy(self.model).to(dtype)
        return FNOOp(m if device is None else m.to(device))

    def transport(self, u0, d0, K):
        u, d, out = u0, d0, []
        for _ in range(K):
            u, d = self.model.tangent(u, d)
            out.append(d)
        return torch.stack(out, 1)


def fno_path(args):
    path = os.path.join(RESULTS, "neural2d", f"fno2d_pme_s{args.seed}{args.fno_tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def train_fno(args, g):
    """One-step training on n_fno trajectories of data seed 6000 + seed (two-step rollout term), Adam, cosine."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    u = trajectories(g, args.fno_n, 6000 + args.seed, args.nt, args.dt, args.nsub).float()
    model = FNO2d(args.fno_width, args.fno_modes, args.fno_layers)
    inc = float((u[:, 1:] - u[:, :-1]).std())
    model.norm.copy_(torch.tensor([float(u.mean()), float(u.std()), inc]))
    model = model.to(dev)
    x0, x1, x2 = (u[:, i:u.shape[1] - 2 + i].reshape(-1, g.n, g.n) for i in range(3))
    n = x0.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.fno_lr)
    steps = args.fno_epochs * (n // args.fno_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    print(f"training FNO2d on {args.fno_n} trajectories ({n} pairs), width {args.fno_width}, modes "
          f"{args.fno_modes}, {args.fno_layers} layers, {args.fno_epochs} epochs, {dev}", flush=True)
    t0 = time.time()
    for ep in range(args.fno_epochs):
        perm, tot = torch.randperm(n), 0.0
        for b in range(n // args.fno_batch):
            i = perm[b * args.fno_batch:(b + 1) * args.fno_batch]
            a, y1, y2 = x0[i].to(dev), x1[i].to(dev), x2[i].to(dev)
            p1 = model(a)
            p2 = model(p1)
            loss = (((p1 - y1) ** 2).mean() + ((p2 - y2) ** 2).mean()) / inc ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % max(1, args.fno_epochs // 20) == 0 or ep == args.fno_epochs - 1:
            print(f"  epoch {ep}: loss {tot / (n // args.fno_batch):.3e} ({time.time() - t0:.0f} s)", flush=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=sd, cfg=dict(width=args.fno_width, modes=args.fno_modes, layers=args.fno_layers)),
               fno_path(args))
    print(f"saved {fno_path(args)}", flush=True)


def load_fno(args, g):
    path = fno_path(args)
    if not os.path.exists(path):
        if not args.train_fno_if_missing:
            raise SystemExit(f"no FNO at {path}; pass --train-fno-if-missing")
        train_fno(args, g)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FNO2d(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def operator_errors(model, d, K):
    """Equation-free reliability of a learned stepper on a data set: one-step and K-step rollout error relative to
    the data increment on the scored block, overall and per trajectory (one trajectory at a time)."""
    sl, u = d.g.sl, d.u
    n = u.shape[0]
    num, den, numK, denK = (np.zeros(n) for _ in range(4))
    with torch.no_grad():
        for r in range(n):
            one = model(u[r, :-1])
            num[r] = float(((one - u[r, 1:])[..., sl, sl] ** 2).sum())
            den[r] = float(((u[r, 1:] - u[r, :-1])[..., sl, sl] ** 2).sum())
            s = u[r, :1]
            for _ in range(K):
                s = model(s)
            numK[r] = float(((s[0] - u[r, K])[sl, sl] ** 2).sum())
            denK[r] = float(((u[r, K] - u[r, 0])[sl, sl] ** 2).sum())
    return float(np.sqrt(num.sum() / den.sum())), float(np.sqrt(numK.sum() / denK.sum())), np.sqrt(num / den)


# ---------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------
def check(args):
    g = Grid(args.n, args.L, args.score)
    t0 = time.time()
    d = Data(g, trajectories(g, 4, 100, args.nt, args.dt, args.nsub), args.dt)
    print(f"4 trajectories in {time.time() - t0:.1f} s; u in [{float(d.u.min()):.3f}, {float(d.u.max()):.3f}], "
          f"RK4 |z|max = {args.dt / args.nsub * float(d.u.max()) * g.kmax2:.2f} (stability 2.78)")
    mass = d.u.mean((-1, -2))
    print(f"mass drift {float((mass - mass[:, :1]).abs().max()):.1e}; rms of u - b0 within 1 of the seam / overall: "
          f"{float(((d.u - d.u[..., :1, :1])[..., :4, :] ** 2).mean().sqrt() / ((d.u - d.u[..., :1, :1]) ** 2).mean().sqrt()):.1e}")
    op = ExactOp(g, args.dt, args.nsub)
    r, j = torch.tensor([0, 1]), torch.tensor([5, 20])
    q = chars_poly(d, r, j, slice(None))[:, [1, 17, 40]]
    jv = op.transport(d.u[r, j], q, 3)[:, -1]
    eps = 1e-5 * float(d.u.abs().max()) / float(q.abs().max())
    up = solve(g, (d.u[r, j][:, None] + eps * q).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    um = solve(g, (d.u[r, j][:, None] - eps * q).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[:, -1]
    fd = ((up - um) / (2 * eps)).reshape(jv.shape)
    print(f"tangent vs central differences of the solver: rel err {float((jv - fd).norm() / fd.norm()):.1e}")
    pairs = pairs_of(4, args.nt, args.K, 8)
    t0 = time.time()
    N, DA, G0, trBB = pencil(op, d, chars_poly, pairs, args.K, 2)
    print(f"polynomial pencil on {len(pairs)} pairs in {time.time() - t0:.1f} s")
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = gsolve(N, D)
    Ct = true_coefficients()
    rq = [float(c @ N @ c / (c @ D @ c)) for c in Ct.T]
    print(f"relative change of the basis characteristics over K: {np.sqrt(cmu):.2e}")
    print("algebra Rayleigh quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(NAMES, rq)))
    print("lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:12]))
    rng = np.random.default_rng(0)
    rnd = [float(c @ N @ c / (c @ D @ c)) for c in rng.standard_normal((20, len(Ct)))]
    print(f"random fields: median {np.median(rnd):.1e}, min {np.min(rnd):.1e}")
    probe = Probe(Data(g, trajectories(g, 4, 101, args.nt, args.dt, args.nsub), args.dt, d.s))
    print(f"polynomial lowest 6 on held-out: {probe.scores(*probe.poly(C[:, :6]))}")
    print(f"random null: {random_null(probe, 6, 20, rng)}")


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def span_eigs(op, d, net, pairs, K, cmu, mu, chunk):
    N, DA, G0, _ = pencil(op, d, chars_net(net), pairs, K, chunk)
    return gsolve(N, DA + mu * cmu * G0)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--op", default="exact", choices=["exact", "fno"])
    ap.add_argument("--seed", type=int, default=42, help="network seed; data seeds default to 4000 + s, 5000 + s")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsub", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--every", type=int, default=4, help="start spacing of the polynomial pencil")
    ap.add_argument("--eval-every", type=int, default=8, help="start spacing on held-out data")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=4, help="pairs per step")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--decay", type=float, default=0.4, help="share of the steps with cosine decay at the end")
    ap.add_argument("--lr-min", type=float, default=0.02, help="final learning rate / lr")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--head", type=int, default=16)
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--save", default=None, help="stem under FINS_RESULTS_DIR (npz, json, pt)")
    ap.add_argument("--fno-tag", default="")
    ap.add_argument("--train-fno-if-missing", action="store_true")
    ap.add_argument("--fno-n", type=int, default=512)
    ap.add_argument("--fno-width", type=int, default=32)
    ap.add_argument("--fno-modes", type=int, default=16)
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-epochs", type=int, default=60)
    ap.add_argument("--fno-batch", type=int, default=32)
    ap.add_argument("--fno-lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.check:
        return check(args)
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = Grid(args.n, args.L, args.score)
    T0 = time.time()
    dtr = Data(g, trajectories(g, args.n_train, dseed, args.nt, args.dt, args.nsub), args.dt)
    dev = Data(g, trajectories(g, args.n_eval, eseed, args.nt, args.dt, args.nsub), args.dt, dtr.s)
    print(f"[{args.op}] seed {s}: {args.n_train} training trajectories (data seed {dseed}), {args.n_eval} held-out "
          f"(seed {eseed}); grid {args.n}^2 on [-{args.L / 2:g}, {args.L / 2:g})^2, scored |x|, |y| < {args.score:g}; "
          f"{args.nt} snapshots, dt {args.dt}, K {args.K}; u in [{float(dtr.u.min()):.3f}, {float(dtr.u.max()):.3f}]",
          flush=True)
    out = dict(config=vars(args), data_seed=dseed, eval_seed=eseed, scale=dtr.s)
    if args.op == "exact":
        op64 = ExactOp(g, args.dt, args.nsub)
    else:
        model = load_fno(args, g.cast(torch.float32))
        op64 = FNOOp(copy.deepcopy(model).double())
        for name, dd in (("training", dtr), ("held-out", dev)):
            e1, eK, per = operator_errors(op64.model, dd, args.K)
            out[f"fno_error_{name}"] = [e1, eK]
            out[f"fno_error_{name}_per_trajectory"] = per.tolist()
            print(f"  FNO rollout error relative to the data increment ({name}): 1 step {e1:.4f} (per trajectory "
                  f"{per.min():.4f}-{per.max():.4f}), K steps {eK:.4f}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    op32 = op64.cast(torch.float32, device)
    # evaluation pencils: float64 on the CPU, float32 on a GPU
    op_ev, dtr_ev, dev_ev = ((op64, dtr, dev) if device == "cpu" else
                             (op32, dtr.cast(torch.float32, device), dev.cast(torch.float32, device)))
    ev_dtype = dtr_ev.u.dtype

    # polynomial FINS on the training trajectories; its trace ratio fixes the normalisation constant for the networks
    t0 = time.time()
    ptr = pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = pencil(op_ev, dtr_ev, chars_poly, ptr, args.K, 2)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = gsolve(N, D)
    Ct = true_coefficients()
    rq = [float(c @ N @ c / (c @ D @ c)) for c in Ct.T]
    probe = Probe(dev)
    poly_sc = probe.scores(*probe.poly(C[:, :6]))
    out.update(poly_lam=lam[:16].tolist(), poly_true_rq=rq, poly_scores=poly_sc, cmu=cmu,
               poly_scores_m=probe.scores(*probe.poly(C[:, :args.m])))
    print(f"  polynomial class ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change over K {np.sqrt(cmu):.2e}")
    print("    lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:10]))
    print("    algebra Rayleigh quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(NAMES, rq)))
    print(f"    lowest 6 on held-out: missing (characteristics) {poly_sc['missing_q']:.3f}, (fields) "
          f"{poly_sc['missing_f']:.3f}", flush=True)
    rng = np.random.default_rng(s)
    null = random_null(probe, args.m, args.null_draws, rng)
    out["random_null"] = null
    print(f"    random {args.m}-dim subspaces of the class ({args.null_draws}): missing (characteristics) "
          f"{null['missing_q']:.3f} +- {null['std_missing_q']:.3f}, (fields) {null['missing_f']:.3f} +- "
          f"{null['std_missing_f']:.3f}", flush=True)

    # neural generators
    torch.manual_seed(s)
    net = GenNet(args.m, args.width, args.head, dtr.factor_rms()).to(device)
    net0 = copy.deepcopy(net)
    d32 = dtr.cast(torch.float32, device)
    pe = pairs_of(args.n_eval, args.nt, args.K, args.eval_every)
    Nt_, DAt, G0t, _ = pencil(op_ev, dev_ev, chars_true, pe, args.K, 4)
    true_eval = gsolve(Nt_, DAt + args.mu * cmu * G0t)[0]
    out["true_span_eigs_heldout"] = true_eval.tolist()
    print("  algebra's own pencil on held-out pairs: eigenvalues " + " ".join(f"{v:.1e}" for v in true_eval))
    sc0 = probe.scores(*probe.net(net0))
    out["untrained"] = sc0
    print(f"  untrained network: missing (characteristics) {sc0['missing_q']:.3f}, (fields) {sc0['missing_f']:.3f}",
          flush=True)
    ptr_all = pairs_of(args.n_train, args.nt, args.K, 1)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_decay = int(args.decay * args.steps)

    def lr_factor(i):
        if i < args.warmup:
            return (i + 1) / args.warmup
        if i < args.steps - n_decay:
            return 1.0
        x = (i - args.steps + n_decay) / max(n_decay, 1)
        return args.lr_min + (1 - args.lr_min) * 0.5 * (1 + math.cos(math.pi * x))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    gen = torch.Generator().manual_seed(s)
    curve, t0 = [], time.time()
    for it in range(args.steps + 1):
        if it % args.log_every == 0 or it == args.steps:
            pr = probe.scores(*probe.net(net))
            curve.append(dict(step=it, missing_q=pr["missing_q"], missing_f=pr["missing_f"], novel_q=pr["novel_q"]))
            print(f"    step {it}: held-out missing (characteristics) {pr['missing_q']:.3f}, (fields) "
                  f"{pr['missing_f']:.3f}; novel " + " ".join(f"{v:.2f}" for v in pr["novel_q"])
                  + f" ({time.time() - t0:.0f} s)", flush=True)
            if it == args.steps:
                break
        p = ptr_all[torch.randperm(len(ptr_all), generator=gen)[:args.batch]]
        Nb, DAb, G0b, _ = pencil_terms(op32, d32, chars_net(net), p[:, 0], p[:, 1], args.K)
        loss, ev = kyfan(Nb, DAb + args.mu * cmu * G0b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % args.log_every == 0:
            curve[-1].update(loss=float(loss), batch_eigs=ev.tolist())
            print(f"      loss {float(loss):.3e}; batch eigenvalues " + " ".join(f"{v:.1e}" for v in ev), flush=True)
    train_time = time.time() - t0
    net64 = copy.deepcopy(net).to(ev_dtype)
    F, Qc = probe.net(net64)
    sc = probe.scores(F, Qc)
    # pencil inside the learned span: training pairs (readout of the 6 lowest directions) and held-out pairs
    ptr8 = pairs_of(args.n_train, args.nt, args.K, args.eval_every)
    Nn, DAn, G0n, _ = pencil(op_ev, dtr_ev, chars_net(net64), ptr8, args.K, 4)
    ev_fit, W = gsolve(Nn, DAn + args.mu * cmu * G0n)
    sc6 = probe.scores(W[:, :6].T @ F.reshape(args.m, -1), W[:, :6].T @ Qc) if args.m > 6 else sc
    ev_tr = span_eigs(op_ev, dev_ev, net64, pe, args.K, cmu, args.mu, 4)
    ev_un = span_eigs(op_ev, dev_ev, copy.deepcopy(net0).to(ev_dtype), pe, args.K, cmu, args.mu, 4)
    out.update(trained=sc, trained_lowest6=sc6, train_time=train_time, curve=curve, span_eigs_trained=ev_tr.tolist(),
               span_eigs_trained_fit=ev_fit.tolist(), span_eigs_untrained=ev_un.tolist(), total_time=time.time() - T0)
    print(f"  trained network ({train_time:.0f} s): missing (characteristics) {sc['missing_q']:.3f}, (fields) "
          f"{sc['missing_f']:.3f}; novel containment " + " ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, sc["novel_q"])))
    if args.m > 6:
        print(f"    6 lowest directions of the span (training pairs): missing (characteristics) {sc6['missing_q']:.3f}, "
              f"(fields) {sc6['missing_f']:.3f}; novel " + " ".join(f"{v:.2f}" for v in sc6["novel_q"]))
    print("    eigenvalues in the span, training pairs: " + " ".join(f"{v:.1e}" for v in ev_fit))
    print("    held-out pairs, trained: " + " ".join(f"{v:.1e}" for v in ev_tr))
    print("    held-out pairs, untrained: " + " ".join(f"{v:.1e}" for v in ev_un), flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()},
                        state0={k: v.cpu() for k, v in net0.state_dict().items()}, m=args.m, width=args.width,
                        head=args.head, scale=dtr.s, W=W), stem + ".pt")
        np.savez(stem + ".npz", poly_C=C[:, :16], poly_lam=lam, poly_N=N, poly_D=D, span_N=Nn, span_D=DAn + args.mu * cmu * G0n)
        print(f"  saved {stem}.json/.pt/.npz")


if __name__ == "__main__":
    main()
