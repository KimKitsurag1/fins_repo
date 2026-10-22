"""Experiment 69 -- symmetries of a variable that is not closed: a passive tracer in a hidden quasi-geostrophic flow
(README_EXPERIMENTS.md; development and pre-registration 10.10).

Full system on exp45's box with qg2d's f-plane numerics (RK4, pyqg's exponential filter on both fields):
  q_t + J(psi, q) = 0,   c_t + J(psi, c) = kappa Lap(c),   psi = Lap^-1 q.
Only c is observed. Point symmetries of the full system on localised data: d_x, d_y, d_t, the rotation, d_c, c d_c, the
shift of q (a convention of the inversion), S = x d_x + y d_y + 2t d_t - 2q d_q (kappa ties time to length; t d_t - q d_q
alone is broken). Those with a point lift form the observed algebra in c: d_x, d_y, d_t, rotation, d_c, c d_c and
S = x d_x + y d_y + 2t d_t (its lift scales the hidden q). There is no closed equation in c. A closed model of heat type
c_t = D Lap c has in the class these and seven generators the flow breaks: the boosts 2Dt d_x - x c d_c,
2Dt d_y - y c d_c and the superpositions x, y, xy, x^2 - y^2, x^2 + y^2 + 4Dt (H: their span over every D, ten fields).
  oracle        -- FINS through the full system's exact tangent (a simulator), c observed: exp45's 60 fields in c lifted
                   to q as scalars (xi, eta, tau act on q too, phi on c only) and q d_q, on the port's (q, c).
  regress       -- two-stage: weak SINDy on c alone (exp61's reference library), then the identified equation's
                   determining equations in FINS's class (exact rational arithmetic, hard null space).
  train         -- a window emulator c_{j+1} = F(c_{j-w+1}, ..., c_j) on c alone (WindowFNO2d; AdvectFNO2d, a learned
                   streamfunction advecting c_j, was a development variant).
  fins          -- arm self: the emulator's window pencil on its own rollouts from the data's first w frames; arm data:
                   on the data; arm oracle: as above.
  observability -- the best hidden lift: per candidate, min over a hidden perturbation dq(t0) of the c-part's K-step
                   mismatch under the exact dynamics (CGLS), against the scalar lift.
Defaults are the registered settings (128^2, emulators with 40 modes on 1024 trajectories, 10000 steps of 32).

  python exp69_hidden_tracer.py check --n 64
  python exp69_hidden_tracer.py fins --arm oracle --seed 42 --save <stem>
  python exp69_hidden_tracer.py regress --seed 42 --save <stem>
  python exp69_hidden_tracer.py run --runs <dir> --seeds 42,...,51 --threads 1      (oracle, regress, observability)
  python exp69_hidden_tracer.py train --seed 42;  python exp69_hidden_tracer.py fins --arm self --seed 42 --save <stem>
  python exp69_hidden_tracer.py summary --runs <dir> --seeds 42,...,51
Development (seed 7, 64^2 emulators): train --n 64 --modes 20 --n-traj 512 --steps 3000 --batch 16 [--w 1] [--emu adv].
"""

import argparse
import copy
import json
import math
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import exp61_wsindy as e61w                                # noqa: E402
import exp68_emulator_pairs as e68                         # noqa: E402
import method_wsindy2d as mw2                              # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
VORT = dict(count=(2, 4), disk=2.0, radius=(0.7, 1.2), aspect=(1.0, 1.6), amp=(0.6, 1.2))
TRACER = dict(count=(3, 6), disk=2.5, width=(0.6, 1.4), aspect=(1.0, 2.0), amp=(0.5, 1.0), scale=(0.5, 2.0))
NM = len(e45.MONS)


# ---------------------------------------------------------------------------------------------------------------------
# Full system: hidden q, observed c
# ---------------------------------------------------------------------------------------------------------------------
class Grid(qg2d.Grid):
    """qg2d's f-plane grid with the tracer's diffusivity."""

    def __init__(self, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu", kappa=0.05):
        super().__init__(n, L, score, dtype, device)
        self.kappa = kappa

    def cast(self, dtype, device=None):
        return Grid(self.n, self.L, self.score, dtype, self.device if device is None else device, self.kappa)


def velocity(g, q):
    ph = g.a * torch.fft.rfft2(q)
    return qg2d.irfft(g, -g.iky * ph), qg2d.irfft(g, g.ikx * ph)


def advect(g, u, v, f):
    return -(g.ikx * torch.fft.rfft2(u * f) + g.iky * torch.fft.rfft2(v * f))


def rhs(g, q, c):
    u, v = velocity(g, q)
    return (qg2d.irfft(g, advect(g, u, v, q)),
            qg2d.irfft(g, advect(g, u, v, c) - g.kappa * g.K2 * torch.fft.rfft2(c)))


def rhs_tan(g, q, c, dq, dc):
    u, v = velocity(g, q)
    ud, vd = velocity(g, dq)
    return (qg2d.irfft(g, advect(g, u, v, dq) + advect(g, ud, vd, q)),
            qg2d.irfft(g, advect(g, u, v, dc) + advect(g, ud, vd, c) - g.kappa * g.K2 * torch.fft.rfft2(dc)))


def step(g, q, c, h):
    """One RK4 step of (q, c), then pyqg's filter on both."""
    k1 = rhs(g, q, c)
    k2 = rhs(g, q + 0.5 * h * k1[0], c + 0.5 * h * k1[1])
    k3 = rhs(g, q + 0.5 * h * k2[0], c + 0.5 * h * k2[1])
    k4 = rhs(g, q + h * k3[0], c + h * k3[1])
    return (qg2d.filt(g, q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])),
            qg2d.filt(g, c + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])))


def step_tan(g, q, c, dq, dc, h):
    """One step of (q, c) and its derivative along (dq, dc): the exact Jacobian of the step."""
    s, ds = (q, c), (dq, dc)
    ks, ls = [], []
    for a in (0.0, 0.5, 0.5, 1.0):
        qs = s if not ks else tuple(x + a * h * k for x, k in zip(s, ks[-1]))
        dss = ds if not ls else tuple(x + a * h * k for x, k in zip(ds, ls[-1]))
        ks.append(rhs(g, *qs))
        ls.append(rhs_tan(g, *qs, *dss))
    w = (1.0, 2.0, 2.0, 1.0)
    new = tuple(qg2d.filt(g, x + h / 6 * sum(wi * k[i] for wi, k in zip(w, ks))) for i, x in enumerate(s))
    dnew = tuple(qg2d.filt(g, x + h / 6 * sum(wi * k[i] for wi, k in zip(w, ls))) for i, x in enumerate(ds))
    return new + dnew


def solve(g, q0, c0, nt, dt, nsub, h_scale=1.0):
    """(q0, c0) [B, n, n] -> (q, c) [B, nt, n, n], snapshots every dt * h_scale, nsub steps each."""
    qs, cs, q, c = [q0], [c0], q0, c0
    for _ in range(nt - 1):
        for _ in range(nsub):
            q, c = step(g, q, c, h_scale * dt / nsub)
        qs.append(q)
        cs.append(c)
    return torch.stack(qs, 1), torch.stack(cs, 1)


def disk(rng, R):
    r, th = R * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    return r * np.cos(th), r * np.sin(th)


def initial(g, n_traj, rng):
    """Hidden q: two or three shielded elliptic vortices (qg2d.initial's shapes and strengths) centred uniformly in the
    disk of radius 2 (an isotropic prior). Observed c: three to five positive elliptic Gaussian blobs (widths 0.6-1.4,
    centres uniform in the disk of radius 2.5) times a log-uniform factor in [0.5, 2]. pyqg's filter on both."""
    X, Y = g.X.cpu().numpy(), g.Y.cpu().numpy()
    q0, c0 = np.zeros((n_traj, g.n, g.n)), np.zeros((n_traj, g.n, g.n))

    def rot(cx, cy, th):
        return (X - cx) * np.cos(th) + (Y - cy) * np.sin(th), -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)

    for r in range(n_traj):
        for _ in range(rng.integers(*VORT["count"])):
            cx, cy = disk(rng, VORT["disk"])
            R, e = rng.uniform(*VORT["radius"]), rng.uniform(*VORT["aspect"])
            xr, yr = rot(cx, cy, rng.uniform(0, np.pi))
            rho2 = (xr / (R * np.sqrt(e))) ** 2 + (yr * np.sqrt(e) / R) ** 2
            q0[r] += rng.choice([-1.0, 1.0]) * rng.uniform(*VORT["amp"]) * (1 - rho2) * np.exp(-rho2)
        for _ in range(rng.integers(*TRACER["count"])):
            cx, cy = disk(rng, TRACER["disk"])
            wd, e = rng.uniform(*TRACER["width"]), rng.uniform(*TRACER["aspect"])
            xr, yr = rot(cx, cy, rng.uniform(0, np.pi))
            c0[r] += rng.uniform(*TRACER["amp"]) * np.exp(-(xr / (wd * np.sqrt(e))) ** 2 - (yr * np.sqrt(e) / wd) ** 2)
        c0[r] *= np.exp(rng.uniform(*np.log(TRACER["scale"])))
    cast = (lambda a: qg2d.filt(g, torch.tensor(a, dtype=g.dtype, device=g.device)))
    return cast(q0), cast(c0)


def trajectories(g, n_traj, seed, nt, dt, nsub, batch=64):
    """Port trajectories (q, c) [n_traj, nt, n, n] on the CPU, integrated in batches on g's device and dtype."""
    q0, c0 = initial(g, n_traj, np.random.default_rng(seed))
    qs, cs = [], []
    with torch.no_grad():
        for i in range(0, n_traj, batch):
            q, c = solve(g, q0[i:i + batch], c0[i:i + batch], nt, dt, nsub)
            qs.append(q.cpu())
            cs.append(c.cpu())
    return torch.cat(qs), torch.cat(cs)


def tracer_only(g, n_traj, seed, nt, dt, nsub, batch=64, dtype=np.float32):
    """c alone [n_traj, nt, n, n] in one preallocated array (an emulator's training set; q is not kept): the steps of
    solve, batch by batch on g's device."""
    q0, c0 = initial(g, n_traj, np.random.default_rng(seed))
    out = np.empty((n_traj, nt, g.n, g.n), dtype=dtype)
    with torch.no_grad():
        for i in range(0, n_traj, batch):
            q, c = q0[i:i + batch], c0[i:i + batch]
            out[i:i + len(c), 0] = c.cpu().numpy()
            for j in range(1, nt):
                for _ in range(nsub):
                    q, c = step(g, q, c, dt / nsub)
                out[i:i + len(c), j] = c.cpu().numpy()
    return out


def moments(c, dev, chunk=64):
    """Mean and standard deviation of c and of its one-snapshot increments, in float64 over chunks of trajectories."""
    acc = np.zeros(6)
    for i in range(0, len(c), chunk):
        x = c[i:i + chunk].to(dev).double()
        d = x[:, 1:] - x[:, :-1]
        acc += [float(x.sum()), float((x * x).sum()), x.numel(), float(d.sum()), float((d * d).sum()), d.numel()]
    m, md = acc[0] / acc[2], acc[3] / acc[5]
    return m, math.sqrt(acc[1] / acc[2] - m * m), math.sqrt(acc[4] / acc[5] - md * md)


# ---------------------------------------------------------------------------------------------------------------------
# Window emulator
# ---------------------------------------------------------------------------------------------------------------------
class WindowFNO2d(nn.Module):
    """c_{j+1} = c_j + inc * P(layers(lift(channels))): exp45's FNO2d with the window as the lift's input channels,
    (c_j - c0) / s and the increments (c_i - c_{i-1}) / inc (a fixed invertible linear map of the window, so that the
    history is not lost in the conditioning); no positional input (translation-equivariant on the grid)."""

    def __init__(self, w=4, width=32, modes=20, layers=4, proj=128):
        super().__init__()
        self.w = w
        self.lift = nn.Conv2d(w, width, 1)
        self.spec = nn.ModuleList([e45.SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.p1, self.p2 = nn.Conv2d(width, proj, 1), nn.Conv2d(proj, 1, 1)
        self.register_buffer("norm", torch.tensor([0.0, 1.0, 1.0]))                      # c0, s, inc

    def channels(self, u, tangent=False):
        c, s, inc = self.norm
        last = (u[:, -1:] - (0.0 if tangent else c)) / s
        return torch.cat([last, (u[:, 1:] - u[:, :-1]) / inc], 1) if self.w > 1 else last

    def body(self, u):
        """The network's output P(layers(lift(channels(u)))) [B, n, n]."""
        h = self.lift(self.channels(u))
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            h = k(h) + w(h)
            h = tf.gelu(h) if i < len(self.spec) - 1 else h
        return self.p2(tf.gelu(self.p1(h)))[:, 0]

    def body_tangent(self, u, d):
        """body(u) and its derivative along d: u [P, w, n, n] (no grad), d [P, m, w, n, n] -> [P, n, n], [P, m, n, n]."""
        P, m = d.shape[:2]
        with torch.no_grad():
            h = self.lift(self.channels(u))
        dh = tf.conv2d(self.channels(d.reshape(P * m, self.w, *d.shape[-2:]), True), self.lift.weight)
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            with torch.no_grad():
                pre = k(h) + w(h)
            dpre = k(dh) + tf.conv2d(dh, w.weight)
            if i < len(self.spec) - 1:
                with torch.no_grad():
                    gp = e45.gelu_prime(pre)
                    h = tf.gelu(pre)
                dh = gp.repeat_interleave(m, 0) * dpre
            else:
                with torch.no_grad():
                    h = pre
                dh = dpre
        with torch.no_grad():
            z = self.p1(h)
            out = self.p2(tf.gelu(z))[:, 0]
            gz = e45.gelu_prime(z)
        dz = gz.repeat_interleave(m, 0) * tf.conv2d(dh, self.p1.weight)
        return out, tf.conv2d(dz, self.p2.weight)[:, 0].reshape(P, m, *d.shape[-2:])

    def forward(self, u):
        """u [B, w, n, n] -> c_{j+1} [B, n, n]."""
        return u[:, -1] + self.norm[2] * self.body(u)

    def tangent(self, u, d):
        """F(u) and J_F(u) d: u [P, w, n, n] (no grad), d [P, m, w, n, n] -> [P, n, n], [P, m, n, n]."""
        b, db = self.body_tangent(u, d)
        return u[:, -1] + self.norm[2] * b, d[:, :, -1] + self.norm[2] * db


class AdvectFNO2d(WindowFNO2d):
    """c_{j+1} = nsub Heun steps of c_t = -J(psi, c) + k Lap(c) from c_j over the snapshot interval, with a frozen
    streamfunction psi = 0.1 body(window) inferred by the window network and a learned diffusivity k (log-parametrised):
    a passive scalar's transport structure, the flow left to the network (spectral derivatives on the grid)."""

    def __init__(self, w=4, width=32, modes=20, layers=4, proj=128, L=16.0, dt=0.05, nsub=2, kappa0=0.02):
        super().__init__(w, width, modes, layers, proj)
        self.L, self.dt, self.nsub = L, dt, nsub
        self.logk = nn.Parameter(torch.tensor(math.log(kappa0)))
        self.register_buffer("spsi", torch.tensor(0.1))

    def symbols(self, x):
        n = x.shape[-1]
        k = 2 * math.pi * torch.fft.fftfreq(n, d=self.L / n, dtype=x.dtype, device=x.device)
        kr = 2 * math.pi * torch.fft.rfftfreq(n, d=self.L / n, dtype=x.dtype, device=x.device)
        KX, KY = torch.meshgrid(k, kr, indexing="ij")
        return 1j * KX, 1j * KY, KX ** 2 + KY ** 2

    @staticmethod
    def irf(h, n):
        return torch.fft.irfft2(h, s=(n, n))

    def vel(self, psi, sym):
        ph = torch.fft.rfft2(psi)
        return self.irf(-sym[1] * ph, psi.shape[-1]), self.irf(sym[0] * ph, psi.shape[-1])

    def adv(self, c, u, v, sym):
        return self.irf(-(sym[0] * torch.fft.rfft2(u * c) + sym[1] * torch.fft.rfft2(v * c)), c.shape[-1])

    def f(self, c, u, v, sym):
        return self.adv(c, u, v, sym) - self.irf(self.logk.exp() * sym[2] * torch.fft.rfft2(c), c.shape[-1])

    def forward(self, u):
        sym = self.symbols(u)
        U, V = self.vel(self.spsi * self.body(u), sym)
        c, h = u[:, -1], self.dt / self.nsub
        for _ in range(self.nsub):
            k1 = self.f(c, U, V, sym)
            c = c + 0.5 * h * (k1 + self.f(c + h * k1, U, V, sym))
        return c

    def tangent(self, u, d):
        b, db = self.body_tangent(u, d)
        sym = self.symbols(u)
        with torch.no_grad():
            U, V = self.vel(self.spsi * b, sym)
        dU, dV = self.vel(self.spsi * db, sym)
        U, V, c, dc, h = U[:, None], V[:, None], u[:, -1:], d[:, :, -1], self.dt / self.nsub
        for _ in range(self.nsub):
            with torch.no_grad():
                k1 = self.f(c, U, V, sym)
                c2 = c + h * k1
                k2 = self.f(c2, U, V, sym)
            l1 = self.f(dc, U, V, sym) + self.adv(c, dU, dV, sym)
            l2 = self.f(dc + h * l1, U, V, sym) + self.adv(c2, dU, dV, sym)
            c, dc = c + 0.5 * h * (k1 + k2), dc + 0.5 * h * (l1 + l2)
        return c[:, 0], dc


class WindowOp:
    def __init__(self, model):
        self.model = model

    def cast(self, dtype, device=None):
        m = copy.deepcopy(self.model).to(dtype)
        return WindowOp(m if device is None else m.to(device))

    def transport(self, uw, dw, K):
        """J_k dw for k = 1..K along the emulator's own rollout: uw [P, w, n, n], dw [P, m, w, n, n] -> [P, K, m, n, n]."""
        out = []
        for _ in range(K):
            un, dn = self.model.tangent(uw, dw)
            uw = torch.cat([uw[:, 1:], un[:, None]], 1)
            dw = torch.cat([dw[:, :, 1:], dn[:, :, None]], 2)
            out.append(dn)
        return torch.stack(out, 1)


class Filtered(nn.Module):
    """An emulator followed by pyqg's filter (g's) on its output and on its tangent's output (post hoc rollout control)."""

    def __init__(self, model, g):
        super().__init__()
        self.model, self.g, self.w = model, g, model.w

    def forward(self, u):
        return qg2d.filt(self.g, self.model(u))

    def tangent(self, u, d):
        out, dout = self.model.tangent(u, d)
        return qg2d.filt(self.g, out), qg2d.filt(self.g, dout)


def rollout(model, c, nt):
    """The emulator's trajectories from the first w snapshots of c [B, >= w, n, n]: [B, nt, n, n]."""
    w = model.w
    win, out = c[:, :w], list(c[:, :w].unbind(1))
    with torch.no_grad():
        for _ in range(nt - w):
            nxt = model(win)
            out.append(nxt)
            win = torch.cat([win[:, 1:], nxt[:, None]], 1)
    return torch.stack(out, 1)


def window_errors(model, c, K, sl):
    """One-step error over all windows and K-step error from the first window, relative to the data increments on the
    scored block (exp45.operator_errors), one trajectory at a time."""
    w = model.w
    num, den, numK, denK = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for r in range(c.shape[0]):
            wins = torch.stack([c[r, j - w + 1:j + 1] for j in range(w - 1, c.shape[1] - 1)])
            one = model(wins)
            num += float(((one - c[r, w:])[..., sl, sl] ** 2).sum())
            den += float(((c[r, w:] - c[r, w - 1:-1])[..., sl, sl] ** 2).sum())
            ro = rollout(model, c[r:r + 1], w + K)[0, -1]
            numK += float(((ro - c[r, w - 1 + K])[sl, sl] ** 2).sum())
            denK += float(((c[r, w - 1 + K] - c[r, w - 1])[sl, sl] ** 2).sum())
    return math.sqrt(num / den), math.sqrt(numK / denK)


def commute(model, c, g, js=(10, 20, 30, 40)):
    """||F(R w) - R F(w)|| / ||F(w) - c_j|| over held-out windows ending at snapshots js, for rotations of every frame
    about the box centre by 90, 30 and 2 degrees (exp68.rotate)."""
    w = model.w
    win = torch.cat([c[:, j - w + 1:j + 1] for j in js])
    out = {}
    with torch.no_grad():
        F = model(win)
        den = ((F - win[:, -1]) ** 2).sum()
        for ang in (90.0, 30.0, 2.0):
            th = torch.full((win.shape[0],), math.radians(ang), dtype=win.dtype, device=win.device)
            Rw = torch.stack([e68.rotate(win[:, i], th, g.dx) for i in range(w)], 1)
            out[f"rotation {ang:g}"] = float((((model(Rw) - e68.rotate(F, th, g.dx)) ** 2).sum() / den).sqrt())
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Pencils
# ---------------------------------------------------------------------------------------------------------------------
def sums(dl, Q0s, Qk, K):
    """exp45.pencil_terms's sums from J_k Q0 [P, K, m, s, s], Q0 [P, 1, m, s, s], Q(t0 + k) [P, K, m, s, s]."""
    m = dl.shape[2]
    A = (dl - Q0s).movedim(2, 0).reshape(m, -1).double()
    B = (Qk - Q0s).movedim(2, 0).reshape(m, -1).double()
    Q = Q0s[:, 0].movedim(1, 0).reshape(m, -1).double()
    R = A - B
    return [R @ R.T, A @ A.T + B @ B.T, K * (Q @ Q.T), float((B ** 2).sum())]


def window_terms(op, d, r, j0, K, w):
    """Window pencil: the window's perturbation is Q at its w snapshots, J_k transports it, Q(t0 + k) is the data term."""
    sl, dv = d.g.sl, d.u.device
    r, j0 = r.to(dv), j0.to(dv)
    jw = j0[:, None] + torch.arange(1 - w, 1, device=dv)
    rw = r[:, None].expand_as(jw)
    Qw = e45.chars_poly(d, rw, jw, slice(None))                                         # [P, w, m, n, n]
    dl = op.transport(d.u[rw, jw], Qw.movedim(1, 2), K)[..., sl, sl]
    jk = j0[:, None] + torch.arange(1, K + 1, device=dv)
    Qk = e45.chars_poly(d, r[:, None].expand_as(jk), jk, sl)
    return sums(dl, Qw[:, -1][..., sl, sl][:, None], Qk, K)


def chars_obs(dc, r, j, reg):
    """c's characteristics of the 60 fields and a zero for q d_q: [*S, 61, h, w]."""
    Q = e45.chars_poly(dc, r, j, reg)
    return torch.cat([Q, torch.zeros_like(Q[..., :1, :, :])], -3)


def chars_lift(dc, dq, r, j, reg):
    """q's characteristics of the 60 fields lifted as scalars (xi, eta, tau act on q, phi does not) and of q d_q, in c's
    units (both divided by c's sigma): [*S, 61, h, w]."""
    u, _, _, _, X, Y, T = dc.at(r, j, reg)
    q, qx, qy, qt = (a[r, j][..., reg, reg] for a in (dq.u, dq.ux, dq.uy, dq.ut))
    s = dc.s
    mono = e45.monomials(X, Y, T, u / s["sigma"])
    sx, st = s["Lx"] / s["sigma"], s["T"] / s["sigma"]
    fac = torch.stack([-sx * qx, -sx * qy, -st * qt, torch.zeros_like(q)], -3)
    Q = fac.unsqueeze(-3) * mono.unsqueeze(-4)
    Q = Q.reshape(*Q.shape[:-4], 4 * NM, *Q.shape[-2:])
    return torch.cat([Q, (q / s["sigma"]).unsqueeze(-3)], -3)


def oracle_terms(g, dc, dq, r, j0, K, dt, nsub):
    sl = g.sl
    Qc, Qq = chars_obs(dc, r, j0, slice(None)), chars_lift(dc, dq, r, j0, slice(None))
    q, c = dq.u[r, j0][:, None], dc.u[r, j0][:, None]
    out = []
    for _ in range(K):
        for _ in range(nsub):
            q, c, Qq, Qc = step_tan(g, q, c, Qq, Qc, dt / nsub)
        out.append(Qc)
    dl = torch.stack(out, 1)[..., sl, sl]
    jk = j0[:, None] + torch.arange(1, K + 1)
    Qk = chars_obs(dc, r[:, None].expand_as(jk), jk, sl)
    return sums(dl, chars_obs(dc, r, j0, sl)[:, None], Qk, K)


def c_response(g, q0, c0, dq, dc, K, dt, nsub, sl):
    """c's part of the full tangent over K snapshots from (q0, c0) [1, n, n] along (dq, dc) [m, n, n]: [m, K, s, s]."""
    q, c, a, b = q0[:, None], c0[:, None], dq[None], dc[None]
    out = []
    for _ in range(K):
        for _ in range(nsub):
            q, c, a, b = step_tan(g, q, c, a, b, dt / nsub)
        out.append(b[0])
    return torch.stack(out, 1)[..., sl, sl]


HIST = (10, 25, 50, 100, 200, 400)


def observability(args):
    """Best hidden lift (development diagnostic): for each candidate and pair, min over a hidden perturbation dq0 of
    sum_k |c-response of (dq0, Q_c(t0)) - Q_c(t0 + k)|^2 (CGLS on the full system's tangent, warm start at the scalar
    lift), against the scalar lift's value; both over the scalar lift's pencil denominator (the oracle's quotient)."""
    s = args.seed
    g = grid_of(args)
    c = load(args, args.n_train, 4000 + s, "c", np.float64)
    q = load(args, args.n_train, 4000 + s, "q", np.float64)
    dc, dq = e45.Data(g, c, args.dt), e45.Data(g, q, args.dt)
    cand = candidates(dc.s, args.kappa)
    names = list(cand)
    V = torch.tensor(vecs([cand[k] for k in names], [(-2.0 if k == "S" else 0.0) for k in names]))   # [61, m]
    pairs = [(r, 10 + 16 * (r % 2)) for r in range(args.obs_pairs)]
    sl, K = g.sl, args.K
    tot = {k: np.zeros(4) for k in names}
    hist = {k: np.zeros(len(HIST)) for k in names}
    trBB = trG0 = 0.0
    t0 = time.time()
    for r, j in pairs:
        rr, jj = torch.tensor([r]), torch.tensor([j])
        Qc0 = torch.einsum("fm,fhw->mhw", V, chars_obs(dc, rr, jj, slice(None))[0])
        Qq0 = torch.einsum("fm,fhw->mhw", V, chars_lift(dc, dq, rr, jj, slice(None))[0])
        jk = j + torch.arange(1, K + 1)
        Qk = torch.einsum("fm,kfhw->mkhw", V, chars_obs(dc, torch.full((K,), r), jk, sl))
        base = chars_obs(dc, rr, jj, sl)[0, :4 * NM]
        trBB += float(((chars_obs(dc, torch.full((K,), r), jk, sl)[:, :4 * NM] - base) ** 2).sum())
        trG0 += K * float((base ** 2).sum())
        q0, c0 = dq.u[r, j][None], dc.u[r, j][None]
        zero = torch.zeros_like(Qc0)
        Op = (lambda x: c_response(g, q0, c0, x, zero, K, args.dt, args.nsub, sl))

        def OpT(y):
            x = torch.zeros_like(Qq0, requires_grad=True)
            return torch.autograd.grad((Op(x) * y).sum(), x)[0]

        with torch.no_grad():
            b = c_response(g, q0, c0, zero, Qc0, K, args.dt, args.nsub, sl) - Qk
        x = Qq0.clone()
        with torch.no_grad():
            res = -(b + Op(x))
        R0 = (res ** 2).sum((1, 2, 3))
        sv = OpT(res)
        p_, gam = sv.clone(), (sv ** 2).sum((1, 2))
        for it in range(1, args.cg_iters + 1):
            with torch.no_grad():
                qv = Op(p_)
                al = gam / (qv ** 2).sum((1, 2, 3)).clamp_min(1e-300)
                x = x + al[:, None, None] * p_
                res = res - al[:, None, None, None] * qv
            sv = OpT(res)
            gn = (sv ** 2).sum((1, 2))
            p_ = sv + (gn / gam.clamp_min(1e-300))[:, None, None] * p_
            gam = gn
            if it in HIST:
                for i, k in enumerate(names):
                    hist[k][HIST.index(it)] += float((res[i] ** 2).sum())
        R1 = (res ** 2).sum((1, 2, 3))
        with torch.no_grad():
            A = c_response(g, q0, c0, Qq0, Qc0, K, args.dt, args.nsub, sl) - Qc0[:, None][..., sl, sl]
            B = Qk - Qc0[:, None][..., sl, sl]
        den = (A ** 2).sum((1, 2, 3)) + (B ** 2).sum((1, 2, 3))
        Qn = K * (Qc0[..., sl, sl] ** 2).sum((1, 2))
        for i, k in enumerate(names):
            tot[k] += np.array([float(R0[i]), float(R1[i]), float(den[i]), float(Qn[i])])
        print(f"  pair ({r}, {j}) done ({time.time() - t0:.0f} s)", flush=True)
    cmu = trBB / trG0
    out = {}
    for k in names:
        R0, R1, den, Qn = tot[k]
        D = den + args.mu * cmu * Qn
        out[k] = dict(scalar=R0 / D, best=R1 / D, ratio=R1 / R0 if R0 > 0 else None,
                      history={str(i): float(h / D) for i, h in zip(HIST, hist[k]) if i <= args.cg_iters})
    print(f"[observability s{s}] {len(pairs)} pairs, {args.cg_iters} CGLS iterations; quotient with the scalar lift -> "
          f"with the best hidden lift:")
    for k in names:
        print(f"  {k:10s} {out[k]['scalar']:.2e} -> {out[k]['best']:.2e} (" + ", ".join(
            f"{i}: {v:.2e}" for i, v in out[k]["history"].items()) + ")")
    if args.save:
        with open(args.save + ".json", "w") as f:
            json.dump(dict(config=vars(args), cmu=cmu, pairs=pairs, quotients=out), f, indent=1)


def accumulate(fn, pairs, chunk):
    tot = None
    with torch.no_grad():
        for i in range(0, len(pairs), chunk):
            p = pairs[i:i + chunk]
            t = fn(p[:, 0], p[:, 1])
            tot = t if tot is None else [a + b for a, b in zip(tot, t)]
    N, DA, G0, trBB = tot
    return N.cpu().numpy(), DA.cpu().numpy(), G0.cpu().numpy(), trBB


# ---------------------------------------------------------------------------------------------------------------------
# Generators and readouts
# ---------------------------------------------------------------------------------------------------------------------
OBS = {"d_x": [(0, (0, 0, 0, 0), 1.0)], "d_y": [(1, (0, 0, 0, 0), 1.0)], "d_t": [(2, (0, 0, 0, 0), 1.0)],
       "rotation": [(0, (0, 1, 0, 0), -1.0), (1, (1, 0, 0, 0), 1.0)], "d_c": [(3, (0, 0, 0, 0), 1.0)],
       "c_dc": [(3, (0, 0, 0, 1), 1.0)]}
H_FIELDS = [(0, (0, 0, 1, 0)), (1, (0, 0, 1, 0))] + [(3, m) for m in ((1, 0, 0, 1), (0, 1, 0, 1), (1, 0, 0, 0),
                                                                         (0, 1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0),
                                                                         (0, 2, 0, 0), (0, 0, 1, 0))]


def candidates(s, D):
    """{name: entries} in scaled components for the scale s (Lx, T, sigma) and a heat model's diffusivity D."""
    Lx, Tm = s["Lx"], s["T"]
    out = dict(OBS)
    out["S"] = qg2d.scaling(1.0, 2.0, 0.0)
    out["boost_x"] = [(0, (0, 0, 1, 0), 2 * D * Tm / Lx), (3, (1, 0, 0, 1), -Lx)]
    out["boost_y"] = [(1, (0, 0, 1, 0), 2 * D * Tm / Lx), (3, (0, 1, 0, 1), -Lx)]
    out["sup_x"] = [(3, (1, 0, 0, 0), 1.0)]
    out["sup_y"] = [(3, (0, 1, 0, 0), 1.0)]
    out["sup_xy"] = [(3, (1, 1, 0, 0), 1.0)]
    out["sup_x2-y2"] = [(3, (2, 0, 0, 0), 1.0), (3, (0, 2, 0, 0), -1.0)]
    out["sup_r2"] = [(3, (2, 0, 0, 0), Lx ** 2), (3, (0, 2, 0, 0), Lx ** 2), (3, (0, 0, 1, 0), 4 * D * Tm)]
    out["c2_dc"] = [(3, (0, 0, 0, 2), 1.0)]
    out["D_x"] = qg2d.scaling(1.0, 0.0, 0.0)
    out["t_dt"] = qg2d.scaling(0.0, 1.0, 0.0)
    return out


TRUE6 = tuple(OBS)
HEAT_FALSE = ("boost_x", "boost_y", "sup_x", "sup_y", "sup_xy", "sup_x2-y2", "sup_r2")


def vecs(ents_list, extra=None):
    """Coefficient vectors [60 (+1), k]; extra: the q d_q coefficient of each (oracle)."""
    C = qg2d.coefficients(ents_list)
    if extra is None:
        return C
    return np.vstack([C, np.asarray(extra, dtype=float)[None]])


def basis(fields, extra=False):
    C = np.zeros((4 * NM + (1 if extra else 0), len(fields)))
    idx = {m: i for i, m in enumerate(e45.MONS)}
    for j, (c, m) in enumerate(fields):
        C[c * NM + idx[m], j] = 1.0
    return C


def readouts(N, D, s, kappa, probe, oracle=False):
    """Eigenvalues, candidate quotients (S lifted with -2 q d_q in the oracle), the rotation plane, the scaling space
    (x d_x + y d_y, t d_t, c d_c; and q d_q in the oracle), the pencil on H, the gap count and spans."""
    lam, C = e45.gsolve(N, D)
    cand = candidates(s, kappa)
    names = list(cand)
    V = vecs([cand[k] for k in names], [(-2.0 if k == "S" else 0.0) for k in names] if oracle else None)
    rq = {k: e61.quotient(N, D, V[:, i]) for i, k in enumerate(names)}
    rot = vecs([[(0, (0, 1, 0, 0), 1.0)], [(1, (1, 0, 0, 0), 1.0)]], [0.0, 0.0] if oracle else None)
    lr, Wr = e61.sub_eig(N, D, rot)
    sc = vecs([qg2d.scaling(1.0, 0.0, 0.0), qg2d.scaling(0.0, 1.0, 0.0), qg2d.scaling(0.0, 0.0, 1.0)],
              [0.0, 0.0, 0.0] if oracle else None)
    if oracle:
        sc = np.hstack([sc, np.eye(sc.shape[0])[:, -1:]])
    ls, Ws = e61.sub_eig(N, D, sc)
    lh, _ = e61.sub_eig(N, D, basis(H_FIELDS, oracle))
    lf6 = np.maximum(lam[:17], 1e-6)
    own = int(np.argmax(lf6[1:] / lf6[:-1])) + 1
    true_q = probe.poly(vecs([OBS[k] for k in TRUE6]))[1]
    true7 = probe.poly(vecs([OBS[k] for k in TRUE6] + [cand["S"]]))[1]
    spans = {}
    for lab, k in (("span6", 6), ("span7", 7), ("span_own", own)):
        Qc = probe.poly(C[:4 * NM, :k])[1]
        Qc = np.where(np.isfinite(Qc), Qc, 0.0)
        spans[lab] = dict(m=int(k), missing6=e45.missing(Qc, true_q), missing7=e45.missing(Qc, true7),
                          novel6=e45.novel(Qc, true_q).tolist())
    out = dict(lam=lam[:24].tolist(), quotients=rq, own=own, rot_lam=lr.tolist(),
               rot_e=float(np.sqrt(-Wr[1, 0] / Wr[0, 0])) if Wr[0, 0] * Wr[1, 0] < 0 else None,
               scal_lam=ls.tolist(), scal_vec=[(w / w[np.argmax(np.abs(w))]).tolist() for w in Ws.T],
               H_lam=lh.tolist(), **spans)
    if oracle:
        # the hidden lift of S read off the plane (S in c, q d_q): the q d_q coefficient of its deepest direction
        P = np.hstack([vecs([cand["S"]], [0.0]), np.eye(len(N))[:, -1:]])
        lp, Wp = e61.sub_eig(N, D, P)
        out.update(S_lift=float(Wp[1, 0] / Wp[0, 0]), S_plane=lp.tolist(), S_unlifted=e61.quotient(N, D, P[:, 0]))
    return out, C


def show(r, tag):
    q = r["quotients"]
    print(f"  [{tag}] lowest " + " ".join(f"{v:.1e}" for v in r["lam"][:14]) + f"; own count {r['own']}")
    print("    observed: " + "  ".join(f"{k} {q[k]:.1e}" for k in TRUE6) + f";  S {q['S']:.1e}")
    print("    heat-false: " + "  ".join(f"{k} {q[k]:.1e}" for k in HEAT_FALSE) + ";  other: " + "  ".join(
        f"{k} {q[k]:.1e}" for k in ("c2_dc", "D_x", "t_dt")))
    print(f"    rotation plane {r['rot_lam'][0]:.1e}, {r['rot_lam'][1]:.1e} (stretch {r['rot_e']}); scaling space "
          + "; ".join(f"{l:.1e} (" + ", ".join(f"{x:+.2f}" for x in v) + ")" for l, v in zip(r["scal_lam"],
                                                                                          r["scal_vec"])))
    print("    H (10 fields): " + " ".join(f"{v:.1e}" for v in r["H_lam"]))
    print("    " + "; ".join(f"{k} (m {r[k]['m']}) missing6 {r[k]['missing6']:.3f}, missing7 {r[k]['missing7']:.3f}"
                             for k in ("span6", "span7", "span_own")), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Two-stage: weak SINDy on c alone, determining equations in FINS's class
# ---------------------------------------------------------------------------------------------------------------------
def determining2d(terms, s):
    """Rows [rows, 60] of the determining system of c_t = sum coef d_x^p d_y^q (c^j) (terms [((j, p, q), coef)]) in
    exp45's class: in X = x / Lx, Y, T = t / T, U = c / sigma, Q = phi - xi U_X - eta U_Y - tau F and
    D_T Q - F_U Q - sum F_{U_ab} D_X^a D_Y^b Q = 0 identically in X, Y, T, U and the jets (exact rationals, exp50)."""
    X, Y, T, U = sp.symbols("X Y T U")
    m = max([p + q for (j, p, q), _ in terms if j] + [1])
    top = 2 * m + 1
    J = {(a, k - a): sp.Symbol(f"U{a}_{k - a}") for k in range(1, top + 1) for a in range(k + 1)}
    inv = {v: ab for ab, v in J.items()}

    def total(g, ax):
        out = sp.diff(g, (X, Y)[ax]) + sp.diff(g, U) * J[(1, 0) if ax == 0 else (0, 1)]
        for v in g.free_symbols & set(inv):
            a, b = inv[v]
            if a + b >= top:
                raise ValueError("jet order")
            out += sp.diff(g, v) * J[(a + 1, b) if ax == 0 else (a, b + 1)]
        return sp.expand(out)

    sig, Lx, Tm = (sp.Rational(float(s[k])) for k in ("sigma", "Lx", "T"))
    F = sp.Integer(0)
    for (j, p, q), coef in terms:
        g = U ** j if j else sp.Integer(1)
        for _ in range(p):
            g = total(g, 0)
        for _ in range(q):
            g = total(g, 1)
        F += sp.Rational(float(coef)) * Tm / sig * sig ** j / Lx ** (p + q) * g
    F = sp.expand(F)
    DF = {(0, 0): F}
    for k in range(1, m + 1):
        for a in range(k + 1):
            DF[(a, k - a)] = total(DF[(a - 1, k - a)], 0) if a else total(DF[(0, k - 1)], 1)
    lin = [(ab, sp.diff(F, J[ab])) for ab in J if J[ab] in F.free_symbols]
    FU = sp.diff(F, U)

    def dt_total(g):
        out = sp.diff(g, T) + sp.diff(g, U) * F
        for v in g.free_symbols & set(inv):
            out += sp.diff(g, v) * DF[inv[v]]
        return sp.expand(out)

    gens = (X, Y, T, U) + tuple(J.values())
    cols = []
    for comp in range(4):
        for a, b, c, e in e45.MONS:
            f = X ** a * Y ** b * T ** c * U ** e
            Q = sp.expand((-f * J[(1, 0)], -f * J[(0, 1)], -f * F, f)[comp])
            cache = {(0, 0): Q}

            def D(p, q):
                if (p, q) not in cache:
                    cache[(p, q)] = total(D(p - 1, q), 0) if p else total(D(p, q - 1), 1)
                return cache[(p, q)]

            cond = dt_total(Q) - FU * Q
            for (p, q), coef in lin:
                cond -= coef * D(p, q)
            cols.append(sp.Poly(sp.expand(cond), *gens).as_dict())
    keys = sorted({k for col in cols for k in col})
    index = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(cols)))
    for jc, col in enumerate(cols):
        for k, v in col.items():
            M[index[k], jc] = float(v)
    nrm = np.linalg.norm(M, axis=1)
    return M[nrm > 0] / nrm[nrm > 0, None]


def algebra_of(M, tol):
    """Hard null space of the determining rows: (singular values / largest ascending, basis [60, hard])."""
    _, sv, Vh = np.linalg.svd(M, full_matrices=True)
    s = np.zeros(Vh.shape[0])
    s[:len(sv)] = sv
    o = np.argsort(s)
    s, V = s[o] / s.max(), Vh[o].T
    hard = int((s < tol).sum())
    return s, V[:, :hard]


def inter_dim(A, B, tol=1e-6):
    """dim(span A & span B) for column bases (cosines of the principal angles within tol of 1)."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0
    Qa, Qb = e45.orth(A.T), e45.orth(B.T)
    return int((np.linalg.svd(Qa.T @ Qb, compute_uv=False) > 1 - tol).sum())


def algebra_report(V, s, kappa):
    """Dimensions of an algebra V [60, k] against the observed algebra, S and H."""
    cand = candidates(s, kappa)
    obs = vecs([OBS[k] for k in TRUE6])
    obs7 = vecs([OBS[k] for k in TRUE6] + [cand["S"]])
    Hb = basis(H_FIELDS)
    return dict(dim=int(V.shape[1]), observed_in=inter_dim(V, obs), S_in=bool(inter_dim(V, obs7) > inter_dim(V, obs)),
                outside_obs7=int(V.shape[1] - inter_dim(V, obs7)), in_H=inter_dim(V, Hb),
                contains={k: bool(inter_dim(V, vecs([OBS[k]])) == 1) for k in TRUE6})


# ---------------------------------------------------------------------------------------------------------------------
# Data, training
# ---------------------------------------------------------------------------------------------------------------------
def grid_of(args, dtype=torch.float64, device="cpu"):
    return Grid(args.n, args.L, args.score, dtype, device, args.kappa)


def cache_path(args, n_traj, seed, var):
    return os.path.join(args.cache, f"tracer_n{args.n}_k{args.kappa:g}_traj{n_traj}_seed{seed}_{var}.npy")


def load(args, n_traj, seed, var="c", dtype=np.float32):
    """Port trajectories of one data seed ([n_traj, nt, n, n]) cached as .npy. Sets of at most 64 trajectories (the FINS
    and two-stage data) are integrated on the CPU in float64 and keep q; larger ones (an emulator's training set) on the
    GPU when there is one, in float32, c only."""
    path = cache_path(args, n_traj, seed, var)
    if not os.path.exists(path):
        os.makedirs(args.cache, exist_ok=True)
        dev = "cpu" if n_traj <= 64 else (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        gdt = torch.float64 if (dev == "cpu" or dtype == np.float64) else torch.float32
        t0 = time.time()
        g = grid_of(args, gdt, dev)
        if n_traj <= 64:
            q, c = trajectories(g, n_traj, seed, args.nt, args.dt, args.nsub)
            sets = (("q", q.numpy().astype(dtype)), ("c", c.numpy().astype(dtype)))
        else:
            sets = (("c", tracer_only(g, n_traj, seed, args.nt, args.dt, args.nsub, dtype=dtype)),)
        for v, a in sets:
            p = cache_path(args, n_traj, seed, v)
            tmp = f"{p}.{os.getpid()}.tmp.npy"
            np.save(tmp, a)
            os.replace(tmp, p)
        del sets
        print(f"  generated {n_traj} trajectories (seed {seed}, {gdt}, {dev}) in {time.time() - t0:.0f} s -> {path}",
              flush=True)
    a = np.load(path)
    return torch.from_numpy(a if a.dtype == dtype else a.astype(dtype))


def emu_path(args, w=None):
    path = os.path.join(RESULTS, "tracer", f"w{args.emu}_n{args.n}_k{args.kappa:g}_w{args.w if w is None else w}_"
                        f"s{args.seed}{args.tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def train(args):
    """One-step plus two-step loss on random windows of the training trajectories (data seed 6000 + s), Adam, cosine;
    resumable from <checkpoint>.partial."""
    path = emu_path(args)
    if os.path.exists(path):
        print(f"exists: {path}")
        return
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    c = load(args, args.n_traj, 6000 + args.seed)
    torch.manual_seed(args.seed)
    w = args.w
    model = (AdvectFNO2d(w, args.width, args.modes, args.layers, L=args.L, dt=args.dt) if args.emu == "adv" else
             WindowFNO2d(w, args.width, args.modes, args.layers))
    mean, std, inc = moments(c, dev)
    model.norm.copy_(torch.tensor([mean, std, inc]))
    model = model.to(dev)
    cd = c.to(dev)
    del c
    B, T = cd.shape[:2]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + args.seed)
    start, log = 0, []
    part = path + ".partial"
    ck = e68.read_partial(part, dev)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        start, log = ck["step"], ck["log"]
        print(f"resumed from {part} at step {start}", flush=True)
    print(f"training {type(model).__name__} (w {w}) on {B} trajectories, width {args.width}, modes {args.modes}, {args.layers} "
          f"layers, {args.steps} steps of {args.batch}, lr {args.lr}, {dev}", flush=True)
    ar = torch.arange(1 - w, 1, device=dev)
    t0, tot, cnt = time.time(), 0.0, 0
    for st in range(start, args.steps):
        r = torch.randint(0, B, (args.batch,), device=dev, generator=gen)
        j = torch.randint(w - 1, T - 2, (args.batch,), device=dev, generator=gen)
        win = cd[r[:, None], j[:, None] + ar]
        if args.aug == "rot":
            th = 2 * math.pi * torch.rand(args.batch, device=dev, generator=gen)
            n = win.shape[-1]
            win = e68.rotate(win.reshape(-1, n, n), th.repeat_interleave(w), args.L / args.n).reshape(win.shape)
            y1, y2 = (e68.rotate(cd[r, j + i], th, args.L / args.n) for i in (1, 2))
        else:
            y1, y2 = cd[r, j + 1], cd[r, j + 2]
        p1 = model(win)
        p2 = model(torch.cat([win[:, 1:], p1[:, None]], 1))
        loss = (((p1 - y1) ** 2).mean() + ((p2 - y2) ** 2).mean()) / inc ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot, cnt = tot + loss.item(), cnt + 1
        if (st + 1) % args.log_every == 0 or st + 1 == args.steps:
            log.append((st + 1, tot / cnt))
            print(f"  step {st + 1}: loss {tot / cnt:.3e} ({time.time() - t0:.0f} s)", flush=True)
            tot, cnt = 0.0, 0
        if (st + 1) % args.ckpt_every == 0 and st + 1 < args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=gen.get_state(), step=st + 1, log=log), part + ".tmp")
            os.replace(part + ".tmp", part)
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    cfg=dict(w=w, width=args.width, modes=args.modes, layers=args.layers, emu=args.emu, L=args.L,
                             dt=args.dt), log=log, steps=args.steps,
                    n_traj=B, aug=args.aug), path)
    if os.path.exists(part):
        os.remove(part)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


def load_emulator(args):
    ck = torch.load(emu_path(args), map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    emu, L, dt = cfg.pop("emu", "fno"), cfg.pop("L", 16.0), cfg.pop("dt", 0.05)
    model = AdvectFNO2d(**cfg, L=L, dt=dt) if emu == "adv" else WindowFNO2d(**cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------------------------------------------------
# FINS arms
# ---------------------------------------------------------------------------------------------------------------------
def rollcheck(args):
    """Rollout stability of a trained emulator from the FINS trajectories' first w frames (data seed 4000 + s): float32
    and float64 on the device, without and with pyqg's filter after each step; distance to the data, max |c| and the first
    step where max |c| exceeds twice the data's."""
    s = args.seed
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    c = load(args, args.n_train, 4000 + s, "c", np.float64)
    model = load_emulator(args)
    cmax = float(c.abs().max())
    out = dict(config=vars(args), data_max=cmax)
    for dtype in (torch.float32, torch.float64):
        for flt in (False, True):
            m = copy.deepcopy(model).to(dtype).to(dev)
            if flt:
                m = Filtered(m, grid_of(args, dtype, dev))
            ro = rollout(m, c.to(dtype).to(dev), args.nt).double().cpu()
            mx = ro.abs().amax((0, 2, 3))
            over = [int(j) for j in range(len(mx)) if float(mx[j]) > 2 * cmax]
            lab = f"{str(dtype).split('.')[-1]}{' filter' if flt else ''}"
            out[lab] = dict(drift=float((ro - c).norm() / c.norm()), max=float(mx.max()), first_over=over[0] if over else None,
                            max_per_step=mx.tolist())
            print(f"[rollcheck s{s}] {lab:15s} distance to the data {out[lab]['drift']:.4g}; max |c| {out[lab]['max']:.4g} "
                  f"(data {cmax:.3f}); first step above twice the data's max: {out[lab]['first_over']}", flush=True)
    save_json(args, out)


def save_json(args, out):
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"  saved {stem}.json")


def fins(args):
    s = args.seed
    g = grid_of(args)
    t0 = time.time()
    ctr = load(args, args.n_train, 4000 + s, "c", np.float64)
    cev = load(args, args.n_eval, 5000 + s, "c", np.float64)
    out = dict(config=vars(args))
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.arm == "oracle":
        qtr = load(args, args.n_train, 4000 + s, "q", np.float64)
        dtr, dq = e45.Data(g, ctr, args.dt), e45.Data(g, qtr, args.dt)
        dev_d = e45.Data(g, cev, args.dt, dtr.s)
        pairs = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
        fn = (lambda r, j: oracle_terms(g, dtr, dq, r, j, args.K, args.dt, args.nsub))
        w = 1
    else:
        model = load_emulator(args)
        w = model.w
        dtype = torch.float64 if (dev == "cpu" or args.fp64) else torch.float32
        m = copy.deepcopy(model).to(dtype).to(dev)
        if args.roll_filter:
            m = Filtered(m, grid_of(args, dtype, dev))
        out.update(dtype=str(dtype), roll_filter=bool(args.roll_filter))
        e1, eK = window_errors(m, cev.to(dtype).to(dev), args.K, g.sl)
        out.update(error_one=e1, error_K=eK, commute=commute(m, cev.to(dtype).to(dev), g))
        print(f"[{args.emu} w {w}, emulator s{s}] held-out error relative to the increment: one step {e1:.4f}, {args.K} "
              f"steps {eK:.4f}; rotation commutator / increment " + "  ".join(f"{k} {v:.1e}" for k, v in
                                                                             out["commute"].items()), flush=True)
        if args.arm == "self":
            rtr = rollout(m, ctr.to(dtype).to(dev), args.nt).double().cpu()
            rev = rollout(m, cev.to(dtype).to(dev), args.nt).double().cpu()
            out["drift"] = float((rtr - ctr).norm() / ctr.norm())
            print(f"  own rollouts from the data's first {w} snapshots: relative distance to the data {out['drift']:.3f}; "
                  f"c in [{float(rtr.min()):.3f}, {float(rtr.max()):.3f}] (data [{float(ctr.min()):.3f}, "
                  f"{float(ctr.max()):.3f}])", flush=True)
            ctr, cev = rtr, rev
        dtr = e45.Data(g, ctr, args.dt)
        dev_d = e45.Data(g, cev, args.dt, dtr.s)
        d_ev = dtr if dev == "cpu" else dtr.cast(dtype, dev)
        op = WindowOp(m)
        # self: every frame and time-derivative stencil of a window from the emulator's own steps (j0 - w + 1 >= w + 2)
        pairs = e45.pairs_of(args.n_train, args.nt, args.K, args.every, first=2 * w + 1 if args.arm == "self" else w + 1)
        fn = (lambda r, j: window_terms(op, d_ev, r, j, args.K, w))
    print(f"  data {time.time() - t0:.0f} s; {len(pairs)} pairs, window {w}", flush=True)
    t0 = time.time()
    N, DA, G0, trBB = accumulate(fn, pairs, args.chunk)
    cmu = trBB / np.trace(G0[:4 * NM, :4 * NM])
    D = DA + args.mu * cmu * G0
    probe = e45.Probe(dev_d)
    r, C = readouts(N, D, dtr.s, args.kappa, probe, args.arm == "oracle")
    out.update(r, cmu=cmu, scale=dtr.s, pencil_time=time.time() - t0)
    print(f"  pencil {time.time() - t0:.0f} s; relative change over K {np.sqrt(cmu):.2e}", flush=True)
    show(r, args.arm)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", C=C[:, :24], N=N, D=D)
        print(f"  saved {stem}.json/.npz")


def regress(args):
    """Weak SINDy on c alone (the reference library of exp61_wsindy) on data seed 4000 + s; the identified equation's
    algebra in FINS's class."""
    s = args.seed
    g = grid_of(args)
    c = load(args, args.n_train, 4000 + s, "c", np.float64)
    d = e45.Data(g, c, args.dt)
    U = c.numpy().transpose(0, 2, 3, 1)
    x, t = g.x.numpy(), args.dt * np.arange(args.nt)
    t0 = time.time()
    w = mw2.WSINDy2D(e61w.POLYS, e61w.MAX_D)
    b, Mlhs = w.prepare(U, x, x, t)
    local = list(w.terms)
    G, M = w.columns(U, [(j, p, q, -1) for j, p, q in local])
    W, lam, resid = w.mstls(G, b, M / Mlhs)
    terms = [((j, p, q), float(cc)) for (j, p, q), cc in zip(local, W) if cc != 0]
    name = " ".join(e61w.qg_term_name(dict(kind="loc", key=["loc", j, p, q], c=cc)) for (j, p, q), cc in terms)
    print(f"[regress s{s}] {len(U)} trajectories, {w.info['rows']} rows; c_t = {name} (resid {resid:.3g}, "
          f"{time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    Mdet = determining2d(terms, d.s)
    sv, V = algebra_of(Mdet, args.tol)
    rep = algebra_report(V, d.s, args.kappa)
    print(f"  determining system {Mdet.shape[0]} x {Mdet.shape[1]} ({time.time() - t0:.0f} s): smallest singular values "
          + " ".join(f"{v:.1e}" for v in sv[:18]) + f"; algebra {rep}", flush=True)
    out = dict(config=vars(args), terms=[[list(k), v] for k, v in terms], names=name, resid=float(resid),
               lam=float(lam), sv=sv[:30].tolist(), algebra=rep, scale=d.s,
               wsindy={k: float(v) for k, v in w.info.items()})
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", V=V, sv=sv)
        print(f"  saved {stem}.json/.npz")


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------
def rescale(g, f, lam):
    """f(x / lam, y / lam) of band-limited periodic fields f [B, n, n] about the box centre (Fourier series evaluated
    at the scaled points)."""
    k = 2 * math.pi * torch.fft.fftfreq(g.n, d=g.dx, dtype=g.dtype)
    k[g.n // 2] = 0.0
    E = torch.exp(1j * k[None, :] * (g.x[:, None] / lam - g.x[0]))
    return torch.real(E @ torch.fft.fft2(f) @ E.T) / g.n ** 2


def check(args):
    g = grid_of(args)
    T = args.dt * (args.nt - 1)
    t0 = time.time()
    q0, c0 = initial(g, 4, np.random.default_rng(100))
    q, c = solve(g, q0, c0, args.nt, args.dt, args.nsub)
    u, v = velocity(g, q)
    print(f"[kappa {args.kappa}, n {args.n}] 4 trajectories, T {T:g}: {time.time() - t0:.1f} s; q in "
          f"[{float(q.min()):.2f}, {float(q.max()):.2f}], c in [{float(c.min()):.3f}, {float(c.max()):.3f}], max |u| "
          f"{float((u ** 2 + v ** 2).sqrt().max()):.2f}")
    mass = c.sum((-1, -2))
    print(f"  tracer mass drift {float(((mass - mass[:, :1]) / mass[:, :1]).abs().max()):.1e}; rms of c within 1 of "
          f"the seam / overall {float((c[..., :4, :] ** 2).mean().sqrt() / (c ** 2).mean().sqrt()):.1e}")
    ch = (c[:, -1] - c[:, 0]).norm()
    _, c2 = solve(g, q0, c0, args.nt, args.dt, 2 * args.nsub)
    print(f"  half step: |c - c(h/2)| / |c(T) - c(0)| {float((c2[:, -1] - c[:, -1]).norm() / ch):.1e}")
    rel = float(((c[:, args.K:] - c[:, :-args.K]) ** 2).mean().sqrt() / (c ** 2).mean().sqrt())
    print(f"  relative change of c over K = {args.K} snapshots {rel:.3f}")
    # finite transformations of the full system
    th = torch.full((4,), math.radians(30.0), dtype=g.dtype)
    rq0, rc0 = (e68.rotate(f, th, g.dx) for f in (q0, c0))
    rq, rc = solve(g, rq0, rc0, args.nt, args.dt, args.nsub)
    print(f"  rotation by 30 deg commutes with the flow: |c' - R c| / |c(T) - c(0)| "
          f"{float((rc[:, -1] - e68.rotate(c[:, -1], th, g.dx)).norm() / ch):.1e}")
    lam = 1.1
    sq0, sc0 = rescale(g, q0, lam) / lam ** 2, rescale(g, c0, lam)
    _, cS = solve(g, sq0, sc0, args.nt, args.dt, args.nsub, lam ** 2)
    _, cT = solve(g, q0 / lam, c0, args.nt, args.dt, args.nsub, lam)
    _, cX = solve(g, rescale(g, q0, lam), sc0, args.nt, args.dt, args.nsub)
    print(f"  finite flows, |c' - g c| / |c(T) - c(0)|: S (x -> 1.1 x, t -> 1.21 t, q -> q / 1.21) "
          f"{float((cS[:, -1] - rescale(g, c[:, -1], lam)).norm() / ch):.1e}; t d_t - q d_q (t -> 1.1 t, q -> q / 1.1) "
          f"{float((cT[:, -1] - c[:, -1]).norm() / ch):.1e}; x d_x + y d_y (x -> 1.1 x) "
          f"{float((cX[:, -1] - rescale(g, c[:, -1], lam)).norm() / ch):.1e}")
    # tangents
    dc, dq = e45.Data(g, c, args.dt), e45.Data(g, q, args.dt)
    r, j = torch.tensor([0, 1]), torch.tensor([5, 20])
    Qc, Qq = chars_obs(dc, r, j, slice(None))[:, [1, 17, 40, 60]], chars_lift(dc, dq, r, j, slice(None))[:, [1, 17, 40, 60]]
    qs, cs, a, b = dq.u[r, j][:, None], dc.u[r, j][:, None], Qq, Qc
    for _ in range(3 * args.nsub):
        qs, cs, a, b = step_tan(g, qs, cs, a, b, args.dt / args.nsub)
    eps = 1e-6
    sol = [solve(g, (dq.u[r, j][:, None] + sg * eps * Qq).reshape(-1, g.n, g.n),
                 (dc.u[r, j][:, None] + sg * eps * Qc).reshape(-1, g.n, g.n), 4, args.dt, args.nsub)[1][:, -1]
           for sg in (1, -1)]
    fd = ((sol[0] - sol[1]) / (2 * eps)).reshape(b.shape)
    print(f"  full tangent vs central differences: rel err {float((b - fd).norm() / fd.norm()):.1e}")
    torch.manual_seed(0)
    for w in (1, 4):
        mdl = WindowFNO2d(w, 16, 8, 2).double()
        mdl.norm.copy_(torch.tensor([float(c.mean()), float(c.std()), 0.05], dtype=torch.float64))
        jj = torch.tensor([10, 30])
        uw = torch.stack([dc.u[r[i], jj[i] - w + 1:jj[i] + 1] for i in range(2)])
        dw = torch.randn(2, 3, w, g.n, g.n, dtype=torch.float64)
        tr = WindowOp(mdl).transport(uw, dw, 3)[:, -1]

        def roll(x):
            for _ in range(3):
                x = torch.cat([x[:, 1:], mdl(x)[:, None]], 1)
            return x[:, -1]

        fd = torch.stack([(roll(uw + 1e-6 * dw[:, i]) - roll(uw - 1e-6 * dw[:, i])) / 2e-6 for i in range(3)], 1)
        print(f"  window emulator (w {w}, untrained) 3-step tangent vs central differences: rel err "
              f"{float((tr - fd).norm() / fd.norm()):.1e}")
    s = dict(Lx=g.L / 2, T=T, sigma=float(c.std()))
    t0 = time.time()
    M = determining2d([((1, 2, 0), args.kappa), ((1, 0, 2), args.kappa)], s)
    sv, V = algebra_of(M, args.tol)
    rep = algebra_report(V, s, args.kappa)
    cand = candidates(s, args.kappa)
    inside = {k: inter_dim(V, vecs([cand[k]])) for k in ("S",) + HEAT_FALSE}
    print(f"  heat equation c_t = kappa Lap c in the class ({time.time() - t0:.0f} s): singular values "
          + " ".join(f"{v:.0e}" for v in sv[:16]) + f"\n    {rep}\n    contains {inside}")


# ---------------------------------------------------------------------------------------------------------------------
# Jobs, summary
# ---------------------------------------------------------------------------------------------------------------------
EMUS = (("fno", 4), ("fno", 1), ("adv", 4))
OBS_SEEDS = ("42", "43", "44")


def stem_of(runs, what, s):
    return os.path.join(runs, f"ex69_{what}_s{s}")


def jobs(args):
    """CPU runs per seed: the oracle and the two-stage arm on the FINS data (data seeds 4000 + s, 5000 + s); the best
    hidden lift on seeds 42-44."""
    out = []
    for s in args.seeds.split(","):
        base = ["--seed", s, "--n", str(args.n), "--threads", "1"]
        st = stem_of(args.runs, "oracle", s)
        out.append((st + ".json", ["exp69_hidden_tracer.py", "fins", "--arm", "oracle", *base, "--save", st]))
        st = stem_of(args.runs, "regress", s)
        out.append((st + ".json", ["exp69_hidden_tracer.py", "regress", *base, "--save", st]))
    for s in args.seeds.split(","):
        if s in OBS_SEEDS:
            st = stem_of(args.runs, "observability", s)
            out.append((st + ".json", ["exp69_hidden_tracer.py", "observability", "--seed", s, "--n", str(args.n),
                                       "--obs-pairs", "4", "--cg-iters", "100", "--threads", "1", "--save", st]))
    return out


def max_obs(r):
    return max(r["quotients"][k] for k in TRUE6)


def h_accepted(r):
    """Directions of H with pencil eigenvalue at most the largest quotient of the observed algebra."""
    return int(sum(v <= max_obs(r) for v in r["H_lam"]))


def row_stats(cell):
    med = (lambda f: float(np.median([f(r) for r in cell.values()])))
    out = dict(n=len(cell), own=sorted(r["own"] for r in cell.values()), missing6=med(lambda r: r["span6"]["missing6"]),
               missing7=med(lambda r: r["span7"]["missing7"]), max_obs=med(max_obs),
               S=med(lambda r: r["quotients"]["S"]), heat_min=med(lambda r: min(r["quotients"][k] for k in HEAT_FALSE)),
               H0=med(lambda r: r["H_lam"][0]), H_acc=sorted(h_accepted(r) for r in cell.values()),
               c2=med(lambda r: r["quotients"]["c2_dc"]), rot=med(lambda r: r["rot_lam"][0]))
    if all("error_one" in r for r in cell.values()):
        out.update(error=med(lambda r: r["error_one"]), commute30=med(lambda r: r["commute"]["rotation 30"]))
    return out


def summary(args):
    """results/tracer/summary.{json,md}: per arm and seed, the readouts; medians over seeds."""
    res = {}
    labels = [("oracle", "oracle")] + [(f"{a}_{e}_w{w}", f"{a} {e} w{w}") for e, w in EMUS
                                       for a in ("self", "self64", "selffilt", "data")]
    for what, lab in labels + [("regress", "regress"), ("observability", "observability")]:
        for s in args.seeds.split(","):
            p = stem_of(args.runs, what, s) + ".json"
            if os.path.exists(p):
                j = json.load(open(p))
                j.pop("config", None)
                res.setdefault(lab, {})[s] = j
    lines = ["# Tracer in a hidden flow (exp69): medians over seeds\n",
             "| arm | n | one-step error | rotation commutator 30 deg | own counts | span6 missing (6) | span7 missing "
             "(6 + S) | max observed quotient | S | min heat-false quotient | H lowest | H accepted | c^2 d_c |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    stats = {}
    for _, lab in labels:
        if lab not in res:
            continue
        st = stats[lab] = row_stats(res[lab])
        f = (lambda k: f"{st[k]:.2e}" if k in st else "-")
        lines.append(f"| {lab} | {st['n']} | {f('error')} | {f('commute30')} | {st['own']} | {st['missing6']:.3f} | "
                     f"{st['missing7']:.3f} | {st['max_obs']:.1e} | {st['S']:.1e} | {st['heat_min']:.1e} | "
                     f"{st['H0']:.1e} | {st['H_acc']} | {st['c2']:.1e} |")
    if "regress" in res:
        lines += ["", "| two-stage seed | identified c_t | algebra dim | observed in | S in | outside (6 + S) | in H |",
                  "|---|---|---|---|---|---|---|"]
        for s, r in res["regress"].items():
            a = r["algebra"]
            lines.append(f"| {s} | {r['names']} | {a['dim']} | {a['observed_in']} | {a['S_in']} | {a['outside_obs7']} | "
                         f"{a['in_H']} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tracer")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(dict(runs=res, stats=stats), fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "data", "train", "fins", "regress", "observability", "rollcheck", "run",
                                     "summary"])
    ap.add_argument("--arm", default="self", choices=["oracle", "self", "data"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(RESULTS, "tracer_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=os.path.join(RESULTS, "tracer_cache"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--aug", default="none", choices=["none", "rot"])
    ap.add_argument("--emu", default="fno", choices=["fno", "adv"], help="window FNO or the advective emulator")
    ap.add_argument("--n-traj", type=int, default=1024)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=40)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--nsub", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--cg-iters", type=int, default=100)
    ap.add_argument("--fp64", action="store_true", help="emulator arms in float64 on the GPU (always on the CPU)")
    ap.add_argument("--roll-filter", action="store_true", help="pyqg's filter after every emulator step (post hoc)")
    ap.add_argument("--obs-pairs", type=int, default=4, help="observability: pairs (trajectory r, start 10 or 26)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    elif args.what == "data":
        for seed, n_traj, dtype in ((6000 + args.seed, args.n_traj, np.float32), (4000 + args.seed, args.n_train,
                                                                                   np.float64),
                                    (5000 + args.seed, args.n_eval, np.float64)):
            load(args, n_traj, seed, "c", dtype)
    else:
        {"check": check, "train": train, "fins": fins, "regress": regress, "summary": summary,
         "observability": observability, "rollcheck": rollcheck}[args.what](args)


if __name__ == "__main__":
    main()
