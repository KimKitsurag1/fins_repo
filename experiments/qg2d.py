"""Barotropic quasi-geostrophic flow with pyqg's BTModel numerics (pyqg 0.7.3.dev, github.com/pyqg/pyqg: kernel.pyx,
bt_model.py, model.py) in torch, with the exact tangent, on exp45's box [-L/2, L/2)^2.

q_t + d_x((u + U) q) + d_y(v q) + beta psi_x = -rek Lap(psi), u = -psi_y, v = psi_x, psi_hat = a q_hat with
a = -1 / |k|^2 (a = 0 at k = 0: the mean of q does not enter psi) or -1 / (|k|^2 + 1 / rd^2). The exponential filter
of Arbic and Flierl (2003), exp(-filterfac (|k dx| - 0.65 pi)^4) above 0.65 pi, multiplies q after every step. pyqg
steps with AB3; here RK4, so the snapshot map is a function of the snapshot and its tangent is exact (exp61 --check
compares the two). Arrays are [x, y] (exp45), pyqg's are [y, x].

Point symmetries of the flow of localised data (shielded vortices of zero circulation on a quiet background), in q:
inviscid f-plane: d_x, d_y, d_t, the shift d_q, the rotation -y d_x + x d_y, D_x = x d_x + y d_y, D_t = t d_t - q d_q;
beta breaks the rotation and leaves one scaling x d_x + y d_y - t d_t + q d_q; rd breaks D_x; rek breaks the
polynomial D_t (the damped flow stays point-equivalent to the inviscid one by q = e^(-rek t) w with the time
(1 - e^(-rek t)) / rek, and its time scaling becomes ((e^(rek t) - 1) / rek) d_t - e^(rek t) q d_q, outside the
polynomial class). The shift d_q is given by the inversion (its pencil row is zero): a convention of the periodic
inversion without rd, a symmetry with rd (a uniform psi carries no flow). The filter breaks every scaling at the
grid scale. In psi (the streamfunction): the same with psi d_psi weights 2 (D_x), -1 (D_t), 3 (beta), and the shift
d_psi only with rd.
"""

import math

import numpy as np
import torch

import exp45_neural_2d as e45

REGIMES = {"f": dict(beta=0.0, rd=None, rek=0.0), "beta": dict(beta=0.5, rd=None, rek=0.0),
           "rd": dict(beta=0.0, rd=1.0, rek=0.0), "drag": dict(beta=0.0, rd=None, rek=0.3)}
FILTERFAC = 23.6
CPHI = 0.65 * math.pi


class Grid(e45.Grid):
    """exp45's grid with pyqg's inversion, beta, drag and exponential filter."""

    def __init__(self, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu", beta=0.0, rd=None, rek=0.0,
                 U=0.0, filterfac=FILTERFAC):
        super().__init__(n, L, score, dtype, device)
        self.beta, self.rd, self.rek, self.U, self.filterfac = beta, rd, rek, U, filterfac
        kx = 2 * math.pi * torch.fft.fftfreq(n, d=self.dx, dtype=dtype, device=device)
        ky = 2 * math.pi * torch.fft.rfftfreq(n, d=self.dx, dtype=dtype, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        self.K2 = K2
        if rd:
            self.a = -1.0 / (K2 + rd ** -2)
        else:
            self.a = torch.where(K2 > 0, -1.0 / torch.where(K2 > 0, K2, torch.ones_like(K2)), torch.zeros_like(K2))
        wvx = torch.sqrt(KX ** 2 + KY ** 2) * self.dx
        self.filtr = torch.where(wvx > CPHI, torch.exp(-filterfac * (wvx - CPHI) ** 4), torch.ones_like(wvx))

    def cast(self, dtype, device=None):
        return Grid(self.n, self.L, self.score, dtype, self.device if device is None else device, self.beta, self.rd,
                    self.rek, self.U, self.filterfac)


def regime_grid(regime, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu", **over):
    p = dict(REGIMES[regime], **over)
    return Grid(n, L, score, dtype, device, p["beta"], p["rd"], p["rek"])


def irfft(g, h):
    return torch.fft.irfft2(h, s=(g.n, g.n))


def tendency_hat(g, q, ph):
    """Spectral tendency of pyqg's kernel for q [.., n, n] and psi_hat ph."""
    u, v = irfft(g, -g.iky * ph), irfft(g, g.ikx * ph)
    return (-(g.ikx * torch.fft.rfft2((u + g.U) * q) + g.iky * torch.fft.rfft2(v * q) + g.beta * g.ikx * ph)
            + g.rek * g.K2 * ph)


def rhs(g, q):
    return irfft(g, tendency_hat(g, q, g.a * torch.fft.rfft2(q)))


def rhs_tan(g, q, d):
    ph, pd = g.a * torch.fft.rfft2(q), g.a * torch.fft.rfft2(d)
    u, v = irfft(g, -g.iky * ph), irfft(g, g.ikx * ph)
    ud, vd = irfft(g, -g.iky * pd), irfft(g, g.ikx * pd)
    h = (-(g.ikx * torch.fft.rfft2((u + g.U) * d + ud * q) + g.iky * torch.fft.rfft2(v * d + vd * q)
           + g.beta * g.ikx * pd) + g.rek * g.K2 * pd)
    return irfft(g, h)


def filt(g, q):
    return irfft(g, g.filtr * torch.fft.rfft2(q))


def step(g, q, h):
    """One RK4 step, then pyqg's filter."""
    k1 = rhs(g, q)
    k2 = rhs(g, q + 0.5 * h * k1)
    k3 = rhs(g, q + 0.5 * h * k2)
    k4 = rhs(g, q + h * k3)
    return filt(g, q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def step_tan(g, q, d, h):
    """One step of q and its derivative along d (the exact Jacobian of the step)."""
    k1, l1 = rhs(g, q), rhs_tan(g, q, d)
    q2, d2 = q + 0.5 * h * k1, d + 0.5 * h * l1
    k2, l2 = rhs(g, q2), rhs_tan(g, q2, d2)
    q3, d3 = q + 0.5 * h * k2, d + 0.5 * h * l2
    k3, l3 = rhs(g, q3), rhs_tan(g, q3, d3)
    q4, d4 = q + h * k3, d + h * l3
    k4, l4 = rhs(g, q4), rhs_tan(g, q4, d4)
    return (filt(g, q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)), filt(g, d + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4)))


def solve(g, q0, nt, dt, nsub):
    """q0 [B, n, n] -> [B, nt, n, n], snapshots every dt, nsub steps each."""
    out, q = [q0], q0
    for _ in range(nt - 1):
        for _ in range(nsub):
            q = step(g, q, dt / nsub)
        out.append(q)
    return torch.stack(out, 1)


def to_psi(g, q, gauge="mean"):
    """psi of q: pyqg's gauge (zero mean without rd) or the far-field gauge (minus its mean on the seam frame of
    width 1, so that psi is the pressure anomaly against the quiet far field)."""
    psi = irfft(g, g.a * torch.fft.rfft2(q))
    if gauge == "far":
        w = max(1, int(round(1.0 / g.dx)))
        frame = torch.ones(g.n, g.n, dtype=torch.bool, device=psi.device)
        frame[w:g.n - w, w:g.n - w] = False
        psi = psi - psi[..., frame].mean(-1)[..., None, None]
    return psi


def to_q(g, psi):
    """The inverse of to_psi on its range (q's mean is not seen by psi without rd and is set to zero)."""
    inv = torch.where(g.a != 0, 1.0 / torch.where(g.a != 0, g.a, torch.ones_like(g.a)), torch.zeros_like(g.a))
    return irfft(g, inv * torch.fft.rfft2(psi))


class ExactOp:
    """The snapshot map in q (var = "q") or in psi (var = "psi": conjugated by the inversion) and its exact tangent."""

    def __init__(self, g, dt, nsub, var="q"):
        self.g, self.dt, self.nsub, self.var = g, dt, nsub, var

    def cast(self, dtype, device=None):
        return ExactOp(self.g.cast(dtype, device), self.dt, self.nsub, self.var)

    def out(self, d):
        return d if self.var == "q" else to_psi(self.g, d, "far" if self.var == "psi_far" else "mean")

    def transport(self, u0, d0, K):
        """J_k d0 for k = 1..K: u0 [P, n, n], d0 [P, m, n, n] -> [P, K, m, n, n]."""
        g = self.g
        if self.var != "q":
            u0, d0 = to_q(g, u0), to_q(g, d0)
        q, d, out = u0[:, None], d0, []
        for _ in range(K):
            for _ in range(self.nsub):
                q, d = step_tan(g, q, d, self.dt / self.nsub)
            out.append(self.out(d))
        return torch.stack(out, 1)


def initial(g, n_traj, rng, centre=2.0, radius=(0.7, 1.2), aspect=(1.0, 1.6), amp=(0.6, 1.2)):
    """Two or three shielded elliptic vortices A (1 - rho^2) exp(-rho^2) (zero circulation each) of random sign,
    then pyqg's filter."""
    X, Y = g.X.numpy(), g.Y.numpy()
    q0 = np.zeros((n_traj, g.n, g.n))
    for r in range(n_traj):
        for _ in range(rng.integers(2, 4)):
            cx, cy = rng.uniform(-centre, centre, 2)
            R, e = rng.uniform(*radius), rng.uniform(*aspect)
            th = rng.uniform(0, np.pi)
            xr = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
            yr = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
            rho2 = (xr / (R * np.sqrt(e))) ** 2 + (yr * np.sqrt(e) / R) ** 2
            q0[r] += rng.choice([-1.0, 1.0]) * rng.uniform(*amp) * (1 - rho2) * np.exp(-rho2)
    return filt(g, torch.tensor(q0, dtype=g.dtype))


def observe(g, q, var):
    return q if var == "q" else to_psi(g, q, "far" if var == "psi_far" else "mean")


def trajectories(g, n_traj, seed, nt, dt, nsub, var="q"):
    return observe(g, solve(g, initial(g, n_traj, np.random.default_rng(seed)), nt, dt, nsub), var)


GENERATORS = {
    "d_x": [(0, (0, 0, 0, 0), 1.0)], "d_y": [(1, (0, 0, 0, 0), 1.0)], "d_t": [(2, (0, 0, 0, 0), 1.0)],
    "shift": [(3, (0, 0, 0, 0), 1.0)],
    "rotation": [(0, (0, 1, 0, 0), -1.0), (1, (1, 0, 0, 0), 1.0)],
}


def scaling(wx, wt, wu):
    """wx (x d_x + y d_y) + wt t d_t + wu u d_u."""
    out = [(0, (1, 0, 0, 0), wx), (1, (0, 1, 0, 0), wx), (2, (0, 0, 1, 0), wt), (3, (0, 0, 0, 1), wu)]
    return [e for e in out if e[2] != 0]


def algebra(regime, var="q"):
    """(names, entries) of the point algebra of a regime in q or psi (either gauge; the far-field gauge fixes the
    level of psi, so no shift)."""
    p = REGIMES[regime]
    names = ["d_x", "d_y", "d_t"]
    ents = [GENERATORS[k] for k in names]
    if var == "q" or (var == "psi" and p["rd"]):
        names.append("shift")
        ents.append(GENERATORS["shift"])
    if not p["beta"]:
        names.append("rotation")
        ents.append(GENERATORS["rotation"])
    sx = 0.0 if var == "q" else 2.0                                     # psi d_psi weight of x d_x + y d_y
    if p["beta"]:
        if not (p["rd"] or p["rek"]):
            names.append("D_beta")
            ents.append(scaling(1.0, -1.0, 1.0 if var == "q" else 3.0))
    else:
        if not p["rd"]:
            names.append("D_x")
            ents.append(scaling(1.0, 0.0, sx))
        if not p["rek"]:
            names.append("D_t")
            ents.append(scaling(0.0, 1.0, -1.0))
    return names, ents


def coefficients(entries_list):
    """Coefficient vectors over exp45's 60 basis fields (scaled components, MONS order): [60, k]."""
    nm = len(e45.MONS)
    idx = {m: i for i, m in enumerate(e45.MONS)}
    C = np.zeros((4 * nm, len(entries_list)))
    for j, ents in enumerate(entries_list):
        for c, mon, v in ents:
            C[c * nm + idx[mon], j] = v
    return C
