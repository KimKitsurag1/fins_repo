"""Experiment 70 -- can a learned operator replace the simulator for a variable without a closed equation? exp69's tracer
c in a hidden f-plane flow q (README_EXPERIMENTS.md, development and pre-registration 11.10).

  observe -- the best hidden lift with a uniform flow: exp69's best hidden lift (CGLS over a hidden perturbation dq0 under
             the exact dynamics) with a uniform background flow (dU, dV) added to the hidden perturbation (projected out
             exactly). The Galilean t d_x is exact once the flow may change; the question is how much of the heat
             closure's boosts and superpositions the change explains.
  train   -- an emulator of the full state x = (q, c) on port trajectories of data seed 6000 + s (StateFNO2d: exp69's
             window recipe with the state as channels; q is observed in training only).
  fins    -- FINS through the emulator's tangent with exp69's oracle lift (60 fields in c lifted to q as scalars, and
             q d_q; c compared): arm data on the data's (q, c), arm self on the emulator's own rollouts from the data's
             first state.
  lift    -- the best hidden lift through the emulator's tangent (exp69's observability with the emulator in place of
             the exact dynamics).
Defaults are the registered settings (128^2, 40 modes, 1024 trajectories, 10000 steps of 32).

  python exp70_learned_hidden.py check --n 64
  python exp70_learned_hidden.py observe --seed 42 --save <stem>
  python exp70_learned_hidden.py train --seed 42;  python exp70_learned_hidden.py fins --arm data --seed 42 --save <stem>
Development (seed 7, 64^2): train --n 64 --modes 20 --n-traj 512 --steps 3000 --batch 16.
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
import torch
import torch.nn as nn
import torch.nn.functional as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp45_neural_2d as e45                              # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import exp68_emulator_pairs as e68                         # noqa: E402
import exp69_hidden_tracer as e69                          # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = e69.RESULTS
NM = e69.NM
GAL = {"gal_x": [(0, (0, 0, 1, 0), 1.0)], "gal_y": [(1, (0, 0, 1, 0), 1.0)]}


# ---------------------------------------------------------------------------------------------------------------------
# Exact dynamics with a uniform background flow (U, V): q_t + (U + u) q_x + (V + v) q_y = 0, the same for c with kappa
# ---------------------------------------------------------------------------------------------------------------------
def grad(g, f):
    h = torch.fft.rfft2(f)
    return qg2d.irfft(g, g.ikx * h), qg2d.irfft(g, g.iky * h)


def rhs_u(g, q, c, a, b):
    tq, tc = e69.rhs(g, q, c)
    qx, qy = grad(g, q)
    cx, cy = grad(g, c)
    return tq - a * qx - b * qy, tc - a * cx - b * cy


def rhs_tan_u(g, q, c, dq, dc, a, b):
    """exp69.rhs_tan at zero background flow plus a uniform flow perturbation (a, b) advecting q and c."""
    tq, tc = e69.rhs_tan(g, q, c, dq, dc)
    qx, qy = grad(g, q)
    cx, cy = grad(g, c)
    return tq - a * qx - b * qy, tc - a * cx - b * cy


def step_u(g, q, c, a, b, h):
    """One RK4 step with a uniform flow (a, b), then pyqg's filter (check only)."""
    k1 = rhs_u(g, q, c, a, b)
    k2 = rhs_u(g, q + 0.5 * h * k1[0], c + 0.5 * h * k1[1], a, b)
    k3 = rhs_u(g, q + 0.5 * h * k2[0], c + 0.5 * h * k2[1], a, b)
    k4 = rhs_u(g, q + h * k3[0], c + h * k3[1], a, b)
    return (qg2d.filt(g, q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])),
            qg2d.filt(g, c + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])))


def step_tan_u(g, q, c, dq, dc, a, b, h):
    """exp69.step_tan with the uniform flow perturbation (a, b) in the tangent (the base flow stays zero)."""
    s, ds = (q, c), (dq, dc)
    ks, ls = [], []
    for al in (0.0, 0.5, 0.5, 1.0):
        qs = s if not ks else tuple(x + al * h * k for x, k in zip(s, ks[-1]))
        dss = ds if not ls else tuple(x + al * h * k for x, k in zip(ds, ls[-1]))
        ks.append(e69.rhs(g, *qs))
        ls.append(rhs_tan_u(g, *qs, *dss, a, b))
    w = (1.0, 2.0, 2.0, 1.0)
    new = tuple(qg2d.filt(g, x + h / 6 * sum(wi * k[i] for wi, k in zip(w, ks))) for i, x in enumerate(s))
    dnew = tuple(qg2d.filt(g, x + h / 6 * sum(wi * k[i] for wi, k in zip(w, ls))) for i, x in enumerate(ds))
    return new + dnew


def u_response(g, q0, c0, K, dt, nsub, sl):
    """c's part of the tangent along a unit uniform flow in x and in y (no field perturbation): [2, K, s, s]."""
    q, c = q0[:, None], c0[:, None]
    a = torch.tensor([1.0, 0.0], dtype=q0.dtype)[None, :, None, None]
    b = torch.tensor([0.0, 1.0], dtype=q0.dtype)[None, :, None, None]
    dq = torch.zeros(1, 2, g.n, g.n, dtype=q0.dtype)
    dc = torch.zeros_like(dq)
    out = []
    with torch.no_grad():
        for _ in range(K):
            for _ in range(nsub):
                q, c, dq, dc = step_tan_u(g, q, c, dq, dc, a, b, dt / nsub)
            out.append(dc[0])
    return torch.stack(out, 1)[..., sl, sl]


def observe(args):
    """exp69.observability with a uniform flow in the hidden perturbation: per candidate and pair, the scalar lift (as
    exp69), the scalar lift with the best uniform flow, and min over (dq0, dU, dV) by CGLS on dq0 with the flow
    projected out exactly; all over exp69's denominator (the oracle's quotient)."""
    s = args.seed
    g = e69.grid_of(args)
    c = e69.load(args, args.n_train, 4000 + s, "c", np.float64)
    q = e69.load(args, args.n_train, 4000 + s, "q", np.float64)
    dc, dq = e45.Data(g, c, args.dt), e45.Data(g, q, args.dt)
    cand = dict(e69.candidates(dc.s, args.kappa), **GAL)
    names = list(cand)
    V = torch.tensor(e69.vecs([cand[k] for k in names], [(-2.0 if k == "S" else 0.0) for k in names]))
    pairs = [(r, 10 + 16 * (r % 2)) for r in range(args.obs_pairs)]
    sl, K = g.sl, args.K
    unit = dc.s["Lx"] / (dc.s["T"] * dc.s["sigma"])       # gal_x's uniform flow in the characteristics' units
    tot = {k: np.zeros(5) for k in names}
    hist = {k: np.zeros(len(e69.HIST)) for k in names}
    flows = {k: [] for k in names}
    trBB = trG0 = 0.0
    t0 = time.time()
    for r, j in pairs:
        rr, jj = torch.tensor([r]), torch.tensor([j])
        Qc0 = torch.einsum("fm,fhw->mhw", V, e69.chars_obs(dc, rr, jj, slice(None))[0])
        Qq0 = torch.einsum("fm,fhw->mhw", V, e69.chars_lift(dc, dq, rr, jj, slice(None))[0])
        jk = j + torch.arange(1, K + 1)
        Qk = torch.einsum("fm,kfhw->mkhw", V, e69.chars_obs(dc, torch.full((K,), r), jk, sl))
        base = e69.chars_obs(dc, rr, jj, sl)[0, :4 * NM]
        trBB += float(((e69.chars_obs(dc, torch.full((K,), r), jk, sl)[:, :4 * NM] - base) ** 2).sum())
        trG0 += K * float((base ** 2).sum())
        q0, c0 = dq.u[r, j][None], dc.u[r, j][None]
        zero = torch.zeros_like(Qc0)
        Ru = u_response(g, q0, c0, K, args.dt, args.nsub, sl).reshape(2, -1)
        E = torch.linalg.qr(Ru.T)[0].T                                                      # [2, K s s]

        def proj(y):
            f = y.reshape(len(y), -1)
            return (f - (f @ E.T) @ E).reshape(y.shape)

        Op = (lambda x: e69.c_response(g, q0, c0, x, zero, K, args.dt, args.nsub, sl))

        def OpT(y):
            x = torch.zeros_like(Qq0, requires_grad=True)
            return torch.autograd.grad((Op(x) * y).sum(), x)[0]

        with torch.no_grad():
            b = e69.c_response(g, q0, c0, zero, Qc0, K, args.dt, args.nsub, sl) - Qk
            x = Qq0.clone()
            r0 = b + Op(x)
            R0 = (r0 ** 2).sum((1, 2, 3))
            ab = torch.linalg.lstsq(Ru.T, -r0.reshape(len(r0), -1).T).solution.T                 # [m, 2]
            res = -proj(r0)
        R0u = (res ** 2).sum((1, 2, 3))
        sv = OpT(res)
        p_, gam = sv.clone(), (sv ** 2).sum((1, 2))
        for it in range(1, args.cg_iters + 1):
            with torch.no_grad():
                qv = proj(Op(p_))
                al = gam / (qv ** 2).sum((1, 2, 3)).clamp_min(1e-300)
                x = x + al[:, None, None] * p_
                res = res - al[:, None, None, None] * qv
            sv = OpT(res)
            gn = (sv ** 2).sum((1, 2))
            p_ = sv + (gn / gam.clamp_min(1e-300))[:, None, None] * p_
            gam = gn
            if it in e69.HIST:
                for i, k in enumerate(names):
                    hist[k][e69.HIST.index(it)] += float((res[i] ** 2).sum())
        R1u = (res ** 2).sum((1, 2, 3))
        with torch.no_grad():
            A = e69.c_response(g, q0, c0, Qq0, Qc0, K, args.dt, args.nsub, sl) - Qc0[:, None][..., sl, sl]
            B = Qk - Qc0[:, None][..., sl, sl]
        den = (A ** 2).sum((1, 2, 3)) + (B ** 2).sum((1, 2, 3))
        Qn = K * (Qc0[..., sl, sl] ** 2).sum((1, 2))
        for i, k in enumerate(names):
            tot[k] += np.array([float(R0[i]), float(R0u[i]), float(R1u[i]), float(den[i]), float(Qn[i])])
            flows[k].append([float(ab[i, 0]) / unit, float(ab[i, 1]) / unit])
        print(f"  pair ({r}, {j}) done ({time.time() - t0:.0f} s); uniform flow at the scalar lift / gal_x's unit: "
              f"gal_x ({flows['gal_x'][-1][0]:+.4f}, {flows['gal_x'][-1][1]:+.4f}), boost_x "
              f"({flows['boost_x'][-1][0]:+.4f}, {flows['boost_x'][-1][1]:+.4f})", flush=True)
    cmu = trBB / trG0
    out = {}
    for k in names:
        R0, R0u, R1u, den, Qn = tot[k]
        D = den + args.mu * cmu * Qn
        out[k] = dict(scalar=R0 / D, scalar_u=R0u / D, best_u=R1u / D, ratio_u=R1u / R0 if R0 > 0 else None,
                      flow=flows[k], history={str(i): float(h / D) for i, h in zip(e69.HIST, hist[k])
                                              if i <= args.cg_iters})
    print(f"[observe s{s}] {len(pairs)} pairs, {args.cg_iters} CGLS iterations; quotient: scalar lift -> with the best "
          f"uniform flow -> with the best (dq0, uniform flow):")
    for k in names:
        print(f"  {k:10s} {out[k]['scalar']:.2e} -> {out[k]['scalar_u']:.2e} -> {out[k]['best_u']:.2e} (" + ", ".join(
            f"{i}: {v:.2e}" for i, v in out[k]["history"].items()) + ")", flush=True)
    save(args, dict(config=vars(args), cmu=cmu, pairs=pairs, unit=unit, quotients=out))


# ---------------------------------------------------------------------------------------------------------------------
# Emulator of the full state
# ---------------------------------------------------------------------------------------------------------------------
class StateFNO2d(nn.Module):
    """x_{j+1} = x_j + inc * P(layers(lift((x_j - mean) / std))) for a state of f fields [B, f, n, n]: exp45's FNO2d
    with the fields as channels (per-field normalisation and increment scale), no positional input."""

    def __init__(self, fields=2, width=32, modes=20, layers=4, proj=128):
        super().__init__()
        self.f = fields
        self.lift = nn.Conv2d(fields, width, 1)
        self.spec = nn.ModuleList([e45.SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.p1, self.p2 = nn.Conv2d(width, proj, 1), nn.Conv2d(proj, fields, 1)
        self.register_buffer("mean", torch.zeros(fields))
        self.register_buffer("std", torch.ones(fields))
        self.register_buffer("inc", torch.ones(fields))

    @staticmethod
    def per(v):
        return v[None, :, None, None]

    def body(self, x):
        h = self.lift((x - self.per(self.mean)) / self.per(self.std))
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            h = k(h) + w(h)
            h = tf.gelu(h) if i < len(self.spec) - 1 else h
        return self.p2(tf.gelu(self.p1(h)))

    def forward(self, x):
        return x + self.per(self.inc) * self.body(x)

    def tangent(self, x, d):
        """F(x) and J_F(x) d: x [P, f, n, n] (no grad), d [P, m, f, n, n] -> [P, f, n, n], [P, m, f, n, n]."""
        P, m = d.shape[:2]
        mul = (lambda a, b: (a[:, None] * b.view(P, m, *b.shape[1:])).view(P * m, *b.shape[1:]))
        with torch.no_grad():
            h = self.lift((x - self.per(self.mean)) / self.per(self.std))
        dh = tf.conv2d(d.reshape(P * m, self.f, *d.shape[-2:]) / self.per(self.std), self.lift.weight)
        for i, (k, w) in enumerate(zip(self.spec, self.pw)):
            with torch.no_grad():
                pre = k(h) + w(h)
            dpre = k(dh) + tf.conv2d(dh, w.weight)
            if i < len(self.spec) - 1:
                with torch.no_grad():
                    gp = e45.gelu_prime(pre)
                    h = tf.gelu(pre)
                dh = mul(gp, dpre)
            else:
                with torch.no_grad():
                    h = pre
                dh = dpre
        with torch.no_grad():
            z = self.p1(h)
            out = x + self.per(self.inc) * self.p2(tf.gelu(z))
            gz = e45.gelu_prime(z)
        dz = mul(gz, tf.conv2d(dh, self.p1.weight))
        db = tf.conv2d(dz, self.p2.weight).reshape(P, m, self.f, *d.shape[-2:])
        return out, d + self.per(self.inc)[:, None] * db


class ExactStep:
    """The full system's snapshot step with its exact tangent in StateFNO2d's interface (checks)."""

    def __init__(self, g, dt, nsub):
        self.g, self.dt, self.nsub = g, dt, nsub

    def tangent(self, x, d):
        a, b, da, db = x[:, 0][:, None], x[:, 1][:, None], d[:, :, 0], d[:, :, 1]
        for _ in range(self.nsub):
            a, b, da, db = e69.step_tan(self.g, a, b, da, db, self.dt / self.nsub)
        return torch.cat([a, b], 1), torch.stack([da, db], 2)


class StateOp:
    def __init__(self, model):
        self.model = model

    def transport(self, x, d, K):
        """J_k d for k = 1..K along the emulator's rollout from x [P, f, n, n]: d [P, m, f, n, n] -> [P, K, m, f, n, n]."""
        out = []
        for _ in range(K):
            x, d = self.model.tangent(x, d)
            out.append(d)
        return torch.stack(out, 1)


def state_set(g, n_traj, seed, nt, dt, nsub, device, batch=64):
    """Port trajectories (q, c) [n_traj, nt, 2, n, n] in one preallocated float32 tensor on device: exp69.solve's steps
    batch by batch on g's device and dtype (exp69.trajectories' data for the same seed)."""
    q0, c0 = e69.initial(g, n_traj, np.random.default_rng(seed))
    out = torch.empty((n_traj, nt, 2, g.n, g.n), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, n_traj, batch):
            q, c = q0[i:i + batch], c0[i:i + batch]
            out[i:i + len(c), 0, 0], out[i:i + len(c), 0, 1] = q.to(out), c.to(out)
            for j in range(1, nt):
                for _ in range(nsub):
                    q, c = e69.step(g, q, c, dt / nsub)
                out[i:i + len(c), j, 0], out[i:i + len(c), j, 1] = q.to(out), c.to(out)
    return out


def state_data(args, n_traj, seed, dev):
    """An emulator's training set on dev: generated on the GPU (float32) when there is one, else on the CPU (float64) and
    cached as .npy (float32)."""
    path = os.path.join(args.cache, f"state_n{args.n}_k{args.kappa:g}_traj{n_traj}_seed{seed}.npy")
    if os.path.exists(path):
        a = np.load(path, mmap_mode="r")
        out = torch.empty(a.shape, dtype=torch.float32, device=dev)
        for i in range(0, len(a), 64):
            out[i:i + 64] = torch.from_numpy(np.ascontiguousarray(a[i:i + 64])).to(dev)
        return out
    t0 = time.time()
    gdt = torch.float32 if dev != "cpu" else torch.float64
    out = state_set(e69.grid_of(args, gdt, dev), n_traj, seed, args.nt, args.dt, args.nsub, dev)
    print(f"  generated {n_traj} state trajectories (seed {seed}, {gdt}, {dev}) in {time.time() - t0:.0f} s", flush=True)
    if dev == "cpu":
        os.makedirs(args.cache, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npy"
        np.save(tmp, out.numpy())
        os.replace(tmp, path)
    return out


def state_moments(x, chunk=64):
    """Per field: mean and standard deviation of the state and of its one-snapshot increments (float64 over chunks)."""
    f = x.shape[2]
    acc = np.zeros((f, 6))
    for i in range(0, len(x), chunk):
        a = x[i:i + chunk].double()
        d = a[:, 1:] - a[:, :-1]
        for k in range(f):
            acc[k] += [float(a[:, :, k].sum()), float((a[:, :, k] ** 2).sum()), a[:, :, k].numel(),
                       float(d[:, :, k].sum()), float((d[:, :, k] ** 2).sum()), d[:, :, k].numel()]
    m, md = acc[:, 0] / acc[:, 2], acc[:, 3] / acc[:, 5]
    return m, np.sqrt(acc[:, 1] / acc[:, 2] - m * m), np.sqrt(acc[:, 4] / acc[:, 5] - md * md)


class LatentModel(nn.Module):
    """A model of c alone with a learned hidden field z: z_j = E(c_{j-w+1}, ..., c_j) (exp69's WindowFNO2d body, one
    output channel), then (z, c)_{k+1} = G((z, c)_k) (StateFNO2d with two fields: z in q's place)."""

    def __init__(self, w=4, width=32, modes=20, layers=4):
        super().__init__()
        self.w = w
        self.enc = e69.WindowFNO2d(w, width, modes, layers)
        self.G = StateFNO2d(2, width, modes, layers)

    def start(self, win):
        """(z, c) at the window's last snapshot: win [B, w, n, n] -> [B, 2, n, n]."""
        return torch.cat([self.enc.body(win)[:, None], win[:, -1:]], 1)


def emu_path(args):
    name = (f"state_n{args.n}_k{args.kappa:g}_m{args.modes}" + (f"_r{args.roll}" if args.roll != 2 else "")
            if args.emu == "state" else f"latent_n{args.n}_k{args.kappa:g}_m{args.modes}_w{args.w}_r{args.roll}")
    path = os.path.join(RESULTS, "tracer", f"{name}_s{args.seed}{args.tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def train(args):
    """state: exp69's recipe on (q, c), one-step plus two-step loss per field over its increment scale. latent: c alone
    (exp69's training set), z from the first w snapshots of a window, loss on c over roll steps of G. Adam, cosine;
    resumable from <checkpoint>.partial."""
    path = emu_path(args)
    if os.path.exists(path):
        print(f"exists: {path}")
        return
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.emu == "state":
        X = state_data(args, args.n_traj, 6000 + args.seed, dev)
        model = StateFNO2d(2, args.width, args.modes, args.layers)
        mean, std, inc = state_moments(X)
        for k, v in (("mean", mean), ("std", std), ("inc", inc)):
            getattr(model, k).copy_(torch.tensor(v))
        what = f"StateFNO2d on (q, c); increments q {inc[0]:.3e}, c {inc[1]:.3e}"
    else:
        c = e69.load(args, args.n_traj, 6000 + args.seed)
        mean, std, inc = e69.moments(c, dev)
        X = c.to(dev)
        del c
        model = LatentModel(args.w, args.width, args.modes, args.layers)
        model.enc.norm.copy_(torch.tensor([mean, std, inc]))
        for k, v in (("mean", [0.0, mean]), ("std", [1.0, std]), ("inc", [args.z_inc, inc])):
            getattr(model.G, k).copy_(torch.tensor(v))
        what = f"LatentModel (w {args.w}, roll {args.roll}) on c alone; increment {inc:.3e}"
    model = model.to(dev)
    B, T = X.shape[:2]
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
    print(f"training {what} on {B} trajectories, width {args.width}, modes {args.modes}, {args.layers} layers, "
          f"{args.steps} steps of {args.batch}, lr {args.lr}, {dev}", flush=True)
    if args.emu == "state":
        w2 = (1.0 / torch.tensor(inc, dtype=torch.float32, device=dev) ** 2)[None, :]
    else:
        ar = torch.arange(1 - args.w, 1, device=dev)
    t0, tot, cnt = time.time(), 0.0, 0
    for st in range(start, args.steps):
        r = torch.randint(0, B, (args.batch,), device=dev, generator=gen)
        if args.emu == "state":
            j = torch.randint(0, T - args.roll, (args.batch,), device=dev, generator=gen)
            x, loss = X[r, j], 0.0
            for k in range(1, args.roll + 1):
                x = model(x)
                loss = loss + (((x - X[r, j + k]) ** 2).mean((2, 3)) * w2).mean(0).sum()
        else:
            j = torch.randint(args.w - 1, T - args.roll, (args.batch,), device=dev, generator=gen)
            x = model.start(X[r[:, None], j[:, None] + ar])
            loss = 0.0
            for k in range(1, args.roll + 1):
                x = model.G(x)
                loss = loss + ((x[:, 1] - X[r, j + k]) ** 2).mean()
            loss = loss / (args.roll * inc ** 2)
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
    cfg = (dict(fields=2, width=args.width, modes=args.modes, layers=args.layers) if args.emu == "state" else
           dict(w=args.w, width=args.width, modes=args.modes, layers=args.layers))
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, cfg=cfg, emu=args.emu,
                    roll=args.roll, log=log, steps=args.steps, n_traj=B), path)
    if os.path.exists(part):
        os.remove(part)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


def load_emulator(args):
    ck = torch.load(emu_path(args), map_location="cpu", weights_only=False)
    model = StateFNO2d(**ck["cfg"]) if ck.get("emu", "state") == "state" else LatentModel(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rollout(model, x0, nt):
    """The emulator's trajectories from x0 [B, f, n, n]: [B, nt, f, n, n]."""
    out, x = [x0], x0
    with torch.no_grad():
        for _ in range(nt - 1):
            x = model(x)
            out.append(x)
    return torch.stack(out, 1)


def state_errors(model, X, K, sl):
    """Per field: one-step error over all snapshots and K-step error from the first, relative to the data increments on
    the scored block (exp69.window_errors with w = 1), one trajectory at a time."""
    f = X.shape[2]
    num, den, numK, denK = (np.zeros(f) for _ in range(4))
    with torch.no_grad():
        for r in range(X.shape[0]):
            one = model(X[r, :-1])
            num += ((one - X[r, 1:])[..., sl, sl] ** 2).sum((0, 2, 3)).cpu().numpy()
            den += ((X[r, 1:] - X[r, :-1])[..., sl, sl] ** 2).sum((0, 2, 3)).cpu().numpy()
            ro = rollout(model, X[r:r + 1, 0], K + 1)[0, -1]
            numK += ((ro - X[r, K])[..., sl, sl] ** 2).sum((1, 2)).cpu().numpy()
            denK += ((X[r, K] - X[r, 0])[..., sl, sl] ** 2).sum((1, 2)).cpu().numpy()
    return np.sqrt(num / den).tolist(), np.sqrt(numK / denK).tolist()


def state_commute(model, X, g, js=(10, 20, 30, 40)):
    """||F(R x) - R F(x)|| / ||F(x) - x|| per field over held-out states at snapshots js, for rotations of both fields
    about the box centre by 90, 30 and 2 degrees."""
    x = torch.cat([X[:, j] for j in js])
    out = {}
    with torch.no_grad():
        F = model(x)
        den = ((F - x) ** 2).sum((0, 2, 3))
        for ang in (90.0, 30.0, 2.0):
            th = torch.full((x.shape[0],), math.radians(ang), dtype=x.dtype, device=x.device)
            rot = (lambda y: torch.stack([e68.rotate(y[:, k], th, g.dx) for k in range(y.shape[1])], 1))
            num = ((model(rot(x)) - rot(F)) ** 2).sum((0, 2, 3))
            out[f"rotation {ang:g}"] = (num / den).sqrt().tolist()
    return out


def state_terms(op, dc, dq, r, j0, K):
    """exp69.oracle_terms through the emulator: (Qq, Qc) of the lifted fields transported K snapshots by its tangent,
    c's part compared with Q(t0 + k)."""
    sl, dv = dc.g.sl, dc.u.device
    r, j0 = r.to(dv), j0.to(dv)
    Qc, Qq = e69.chars_obs(dc, r, j0, slice(None)), e69.chars_lift(dc, dq, r, j0, slice(None))
    x = torch.stack([dq.u[r, j0], dc.u[r, j0]], 1)
    dl = op.transport(x, torch.stack([Qq, Qc], 2), K)[:, :, :, 1][..., sl, sl]
    jk = j0[:, None] + torch.arange(1, K + 1, device=dv)
    Qk = e69.chars_obs(dc, r[:, None].expand_as(jk), jk, sl)
    return e69.sums(dl, e69.chars_obs(dc, r, j0, sl)[:, None], Qk, K)


class OneStep(nn.Module):
    """A latent model as a window emulator of c (its first step from a window), for exp69.commute."""

    def __init__(self, model):
        super().__init__()
        self.model, self.w = model, model.w

    def forward(self, win):
        return self.model.G(self.model.start(win))[:, 1]


def latent_paths(model, c, nt, own):
    """(z, c) [B, nt, n, n] along c: own=False, z_j = E(window ending at j) on the data; own=True, the model's rollout
    from the data's first w snapshots (c's first w - 1 snapshots from the data). The first w - 1 z's repeat z_{w-1}."""
    w = model.w
    with torch.no_grad():
        if own:
            x = model.start(c[:, :w])
            zs, cs = [x[:, 0]] * w, list(c[:, :w - 1].unbind(1)) + [x[:, 1]]
            for _ in range(nt - w):
                x = model.G(x)
                zs.append(x[:, 0])
                cs.append(x[:, 1])
            return torch.stack(zs, 1), torch.stack(cs, 1)
        zs = [torch.stack([model.enc.body(c[r:r + 1, j - w + 1:j + 1])[0] for j in range(w - 1, nt)])
              for r in range(c.shape[0])]
        z = torch.stack(zs)
        return torch.cat([z[:, :1].expand(-1, w - 1, -1, -1), z], 1), c


def latent_errors(model, c, K, sl):
    """c's one-step error over all windows and K-step error of G's rollout from the first window, relative to the data
    increments on the scored block, one trajectory at a time."""
    w = model.w
    num = den = numK = denK = 0.0
    with torch.no_grad():
        for r in range(c.shape[0]):
            wins = torch.stack([c[r, j - w + 1:j + 1] for j in range(w - 1, c.shape[1] - 1)])
            one = model.G(model.start(wins))[:, 1]
            num += float(((one - c[r, w:])[..., sl, sl] ** 2).sum())
            den += float(((c[r, w:] - c[r, w - 1:-1])[..., sl, sl] ** 2).sum())
            x = model.start(c[r:r + 1, :w])
            for _ in range(K):
                x = model.G(x)
            numK += float(((x[0, 1] - c[r, w - 1 + K])[sl, sl] ** 2).sum())
            denK += float(((c[r, w - 1 + K] - c[r, w - 1])[sl, sl] ** 2).sum())
    return math.sqrt(num / den), math.sqrt(numK / denK)


def lift_arm(args, G, dc, dh, out):
    """exp69.observability through G's tangent: per candidate, the hidden perturbation dh0 that minimises c's K-step
    mismatch (CGLS from the scalar lift, adjoint by autograd), against the scalar lift and no hidden perturbation; over
    the scalar lift's denominator."""
    cand = e69.candidates(dc.s, args.kappa)
    names = list(cand)
    dv, dt_ = dc.u.device, dc.u.dtype
    V = torch.tensor(e69.vecs([cand[k] for k in names], [(-2.0 if k == "S" else 0.0) for k in names]), dtype=dt_,
                     device=dv)
    first = args.w + 1 if args.emu == "latent" else 2
    pairs = [(r, max(first, 10) + 16 * (r % 2)) for r in range(args.obs_pairs)]
    sl, K = dc.g.sl, args.K
    op = StateOp(G)
    tot = {k: np.zeros(5) for k in names}
    hist = {k: np.zeros(len(e69.HIST)) for k in names}
    trBB = trG0 = 0.0
    t0 = time.time()
    for r, j in pairs:
        rr, jj = torch.tensor([r], device=dv), torch.tensor([j], device=dv)
        Qc0 = torch.einsum("fm,fhw->mhw", V, e69.chars_obs(dc, rr, jj, slice(None))[0])
        Qh0 = torch.einsum("fm,fhw->mhw", V, e69.chars_lift(dc, dh, rr, jj, slice(None))[0])
        jk = j + torch.arange(1, K + 1, device=dv)
        Ck = e69.chars_obs(dc, torch.full((K,), r, device=dv), jk, sl)
        Qk = torch.einsum("fm,kfhw->mkhw", V, Ck)
        base = e69.chars_obs(dc, rr, jj, sl)[0, :4 * NM]
        trBB += float(((Ck[:, :4 * NM] - base) ** 2).sum())
        trG0 += K * float((base ** 2).sum())
        x0 = torch.stack([dh.u[r, j], dc.u[r, j]], 0)[None]

        def resp(a, b):
            return op.transport(x0, torch.stack([a, b], 1)[None], K)[0, :, :, 1].movedim(0, 1)[..., sl, sl]

        zero = torch.zeros_like(Qc0)

        def OpT(y):
            x = torch.zeros_like(Qh0, requires_grad=True)
            return torch.autograd.grad((resp(x, zero) * y).sum(), x)[0]

        with torch.no_grad():
            b = resp(zero, Qc0) - Qk
            R00 = (b ** 2).sum((1, 2, 3))
            x = Qh0.clone()
            res = -(b + resp(x, zero))
        R0 = (res ** 2).sum((1, 2, 3))
        sv = OpT(res)
        p_, gam = sv.clone(), (sv ** 2).sum((1, 2))
        for it in range(1, args.cg_iters + 1):
            with torch.no_grad():
                qv = resp(p_, zero)
                al = gam / (qv ** 2).sum((1, 2, 3)).clamp_min(1e-300)
                x = x + al[:, None, None] * p_
                res = res - al[:, None, None, None] * qv
            sv = OpT(res)
            gn = (sv ** 2).sum((1, 2))
            p_ = sv + (gn / gam.clamp_min(1e-300))[:, None, None] * p_
            gam = gn
            if it in e69.HIST:
                for i, k in enumerate(names):
                    hist[k][e69.HIST.index(it)] += float((res[i] ** 2).sum())
        R1 = (res ** 2).sum((1, 2, 3))
        with torch.no_grad():
            A = resp(Qh0, Qc0) - Qc0[:, None][..., sl, sl]
            B = Qk - Qc0[:, None][..., sl, sl]
        den = (A ** 2).sum((1, 2, 3)) + (B ** 2).sum((1, 2, 3))
        Qn = K * (Qc0[..., sl, sl] ** 2).sum((1, 2))
        for i, k in enumerate(names):
            tot[k] += np.array([float(R00[i]), float(R0[i]), float(R1[i]), float(den[i]), float(Qn[i])])
        print(f"  pair ({r}, {j}) done ({time.time() - t0:.0f} s)", flush=True)
    cmu = trBB / trG0
    q = {}
    for k in names:
        R00, R0, R1, den, Qn = tot[k]
        D = den + args.mu * cmu * Qn
        q[k] = dict(none=R00 / D, scalar=R0 / D, best=R1 / D, ratio=R1 / R0 if R0 > 0 else None,
                    history={str(i): float(h / D) for i, h in zip(e69.HIST, hist[k]) if i <= args.cg_iters})
    print(f"[{args.emu} lift s{args.seed}] {len(pairs)} pairs, {args.cg_iters} CGLS iterations; quotient with no hidden "
          f"perturbation / the scalar lift -> the best hidden lift:")
    for k in names:
        print(f"  {k:10s} {q[k]['none']:.2e} / {q[k]['scalar']:.2e} -> {q[k]['best']:.2e}", flush=True)
    out.update(cmu=cmu, pairs=pairs, quotients=q)
    save(args, out)


def plane_readouts(N, D, s, kappa):
    """Per candidate, the pencil on the plane (candidate with the scalar lift, hidden h d_h): the lowest eigenvalue and
    the h d_h weight of its direction (exp69's S_lift for every candidate; a learned hidden field's weights are not known
    in advance)."""
    cand = e69.candidates(s, kappa)
    out = {}
    for k, ents in cand.items():
        P = np.hstack([e69.vecs([ents], [0.0]), np.eye(len(N))[:, -1:]])
        lp, Wp = e61.sub_eig(N, D, P)
        out[k] = dict(lam=float(lp[0]), weight=float(Wp[1, 0] / Wp[0, 0]) if Wp[0, 0] != 0 else None)
    return out


def fins(args):
    """Arms data and self: the pencil with the scalar lift to the hidden field (q, or the latent z); arm lift: the best
    hidden lift on the data arm's states."""
    s = args.seed
    g = e69.grid_of(args)
    t0 = time.time()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if (dev == "cpu" or args.fp64) else torch.float32
    data = {k: (e69.load(args, args.n_train, 4000 + s, v, np.float64) if k[1] == "t" else
                e69.load(args, args.n_eval, 5000 + s, v, np.float64))
            for k, v in (("qtr", "q"), ("ctr", "c"), ("qev", "q"), ("cev", "c"))}
    m = copy.deepcopy(load_emulator(args)).to(dtype).to(dev)
    on = (lambda a: a.to(dtype).to(dev))
    own = args.arm == "self"
    out = dict(config=vars(args), dtype=str(dtype))
    if args.emu == "state":
        G = m
        Xtr = torch.stack([data["qtr"], data["ctr"]], 2)
        Xev = torch.stack([data["qev"], data["cev"]], 2)
        e1, eK = state_errors(m, on(Xev), args.K, g.sl)
        out.update(error_one=e1, error_K=eK, commute=state_commute(m, on(Xev), g))
        print(f"[state emulator s{s}] held-out error relative to the increment (q, c): one step ({e1[0]:.4f}, "
              f"{e1[1]:.4f}), {args.K} steps ({eK[0]:.4f}, {eK[1]:.4f}); rotation commutator / increment (q, c) "
              + "  ".join(f"{k} ({v[0]:.1e}, {v[1]:.1e})" for k, v in out["commute"].items()), flush=True)
        if own:
            Xtr = rollout(m, on(Xtr[:, 0]), args.nt).double().cpu()
            Xev = rollout(m, on(Xev[:, 0]), args.nt).double().cpu()
        Htr, Ctr, Cev = Xtr[:, :, 0].contiguous(), Xtr[:, :, 1].contiguous(), Xev[:, :, 1].contiguous()
        first = 3 if own else 2
    else:
        G = m.G
        e1, eK = latent_errors(m, on(data["cev"]), args.K, g.sl)
        out.update(error_one=e1, error_K=eK, commute=e69.commute(OneStep(m), on(data["cev"]), g))
        print(f"[latent model s{s}] held-out error of c relative to the increment: one step {e1:.4f}, {args.K} steps "
              f"{eK:.4f}; rotation commutator / increment " + "  ".join(f"{k} {v:.1e}" for k, v in
                                                                     out["commute"].items()), flush=True)
        Htr, Ctr = (a.double().cpu() for a in latent_paths(m, on(data["ctr"]), args.nt, own))
        Cev = latent_paths(m, on(data["cev"]), args.nt, own)[1].double().cpu()
        first = args.w + 2 if own else args.w + 1
    if own:
        out["drift"] = float((Ctr - data["ctr"]).norm() / data["ctr"].norm())
        print(f"  own rollouts from the data's first snapshots: relative distance of c to the data {out['drift']:.3f}; "
              f"c in [{float(Ctr.min()):.3f}, {float(Ctr.max()):.3f}] (data [{float(data['ctr'].min()):.3f}, "
              f"{float(data['ctr'].max()):.3f}])", flush=True)
    dtr, dhtr = e45.Data(g, Ctr, args.dt), e45.Data(g, Htr, args.dt)
    dev_d = e45.Data(g, Cev, args.dt, dtr.s)
    d_ev, dh_ev = (dtr, dhtr) if dev == "cpu" and dtype == torch.float64 else (dtr.cast(dtype, dev),
                                                                                 dhtr.cast(dtype, dev))
    if args.arm == "lift":
        lift_arm(args, G, d_ev, dh_ev, out)
        return
    op = StateOp(G)
    pairs = e45.pairs_of(args.n_train, args.nt, args.K, args.every, first=first)
    fn = (lambda r, j: state_terms(op, d_ev, dh_ev, r, j, args.K))
    print(f"  data {time.time() - t0:.0f} s; {len(pairs)} pairs", flush=True)
    t0 = time.time()
    N, DA, G0, trBB = e69.accumulate(fn, pairs, args.chunk)
    cmu = trBB / np.trace(G0[:4 * NM, :4 * NM])
    D = DA + args.mu * cmu * G0
    probe = e45.Probe(dev_d)
    r, C = e69.readouts(N, D, dtr.s, args.kappa, probe, True)
    out.update(r, cmu=cmu, scale=dtr.s, pencil_time=time.time() - t0, plane=plane_readouts(N, D, dtr.s, args.kappa))
    print(f"  pencil {time.time() - t0:.0f} s; relative change over K {np.sqrt(cmu):.2e}; S's hidden lift "
          f"{r['S_lift']:+.4f}", flush=True)
    e69.show(r, f"{args.emu} {args.arm}")
    print("    with the best h d_h weight: " + "  ".join(f"{k} {v['lam']:.1e}" for k, v in out["plane"].items()),
          flush=True)
    save(args, out, dict(C=C[:, :24], N=N, D=D))


def save(args, out, arrays=None):
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        if arrays is not None:
            np.savez(stem + ".npz", **arrays)
        print(f"  saved {stem}.json", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------------------
ARMS = [(f"{e}_{a}", f"{e} {a}") for e in ("state", "latent") for a in ("data", "self", "lift")]


def separation(r):
    """min over the heat closure's seven / max over the observed algebra and S (a pencil arm's quotients)."""
    q = r["quotients"]
    return min(q[k] for k in e69.HEAT_FALSE) / max(max(q[k] for k in e69.TRUE6), q["S"])


def summary(args):
    """results/tracer/exp70_summary.{json,md}: exp70's arms per seed with exp69's references (the simulator oracle, its
    best hidden lift, the window emulators' data arm) on the same seeds."""
    res = {}
    for s in args.seeds.split(","):
        for what, lab in [("observe", "observe")] + ARMS:
            p = os.path.join(args.runs, f"ex70_{what}_s{s}.json")
            if os.path.exists(p):
                j = json.load(open(p))
                j.pop("config", None)
                res.setdefault(lab, {})[s] = j
        for what, lab in (("oracle", "oracle"), ("observability", "observability"), ("data_fno_w4", "window data")):
            p = e69.stem_of(args.runs69, what, s) + ".json"
            if os.path.exists(p):
                j = json.load(open(p))
                j.pop("config", None)
                res.setdefault(lab, {})[s] = j
    lines = ["# exp70: learned operators for a tracer in a hidden flow\n",
             "| arm | seed | one-step error (c; q) | own | max observed (6) | S | S's hidden lift | min heat-false | "
             "boosts | sup x, y | sup r^2 | H lowest | separation |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ["oracle", "window data"] + [lab for _, lab in ARMS if not lab.endswith("lift")]:
        for s, r in sorted(res.get(lab, {}).items()):
            q = r["quotients"]
            e = r.get("error_one")
            err = "-" if e is None else (f"{e[1]:.4f}; {e[0]:.4f}" if isinstance(e, list) else f"{e:.4f}")
            lines.append(f"| {lab} | {s} | {err} | {r['own']} | {max(q[k] for k in e69.TRUE6):.1e} | {q['S']:.1e} | "
                         f"{r.get('S_lift', float('nan')):+.3f} | {min(q[k] for k in e69.HEAT_FALSE):.1e} | "
                         f"{q['boost_x']:.1e}, {q['boost_y']:.1e} | {q['sup_x']:.1e}, {q['sup_y']:.1e} | "
                         f"{q['sup_r2']:.1e} | {r['H_lam'][0]:.1e} | {separation(r):.2g} |")
    lines += ["", "Rows above: the scalar lift. The latent model's registered readout (LT1-LT4) is the plane readout, each candidate "
              "with its best z d_z weight:", "",
              "| arm (plane readout) | seed | max observed (6) and S | c d_c (z weight) | boosts | sup x, y, xy, x^2 - y^2 | "
              "sup r^2 | separation |", "|---|---|---|---|---|---|---|---|"]
    for lab in ("latent data", "latent self"):
        for s, r in sorted(res.get(lab, {}).items()):
            if "plane" not in r:
                continue
            p = {k: v["lam"] for k, v in r["plane"].items()}
            top = max(p[k] for k in e69.TRUE6 + ("S",))
            lines.append(f"| {lab} | {s} | {top:.1e} | {p['c_dc']:.1e} ({r['plane']['c_dc']['weight']:+.2f}) | "
                         f"{p['boost_x']:.1e}, {p['boost_y']:.1e} | " + ", ".join(f"{p[k]:.1e}" for k in e69.HEAT_FALSE[2:6])
                         + f" | {p['sup_r2']:.1e} | {min(p[k] for k in e69.HEAT_FALSE) / top:.2g} |")
    lines += ["", "| best hidden lift | seed | max observed (6) and S | boosts | superpositions x, y, xy, x^2 - y^2, r^2 |",
              "|---|---|---|---|---|"]
    for lab in ("observability", "state lift", "latent lift"):
        for s, r in sorted(res.get(lab, {}).items()):
            b = {k: v["best"] for k, v in r["quotients"].items()}
            lines.append(f"| {lab} | {s} | {max(b[k] for k in e69.TRUE6 + ('S',)):.1e} | {b['boost_x']:.1e}, "
                         f"{b['boost_y']:.1e} | " + ", ".join(f"{b[k]:.1e}" for k in e69.HEAT_FALSE[2:]) + " |")
    lines += ["", "| uniform flow | seed | quantity | gal_x, gal_y | boosts | superpositions x, y, xy, x^2 - y^2, r^2 | "
              "observed (6) and S, max |", "|---|---|---|---|---|---|---|"]
    for s, r in sorted(res.get("observe", {}).items()):
        q = r["quotients"]
        for key in ("scalar", "scalar_u", "best_u"):
            lines.append(f"| observe | {s} | {key} | {q['gal_x'][key]:.1e}, {q['gal_y'][key]:.1e} | "
                         f"{q['boost_x'][key]:.1e}, {q['boost_y'][key]:.1e} | "
                         + ", ".join(f"{q[k][key]:.1e}" for k in e69.HEAT_FALSE[2:])
                         + f" | {max(q[k][key] for k in e69.TRUE6 + ('S',)):.1e} |")
        lines.append(f"| observe | {s} | gal_x's flow / its unit | " + "; ".join(
            f"({f[0]:+.4f}, {f[1]:+.4f})" for f in q["gal_x"]["flow"]) + " | | | |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tracer")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp70_summary.json"), "w") as fh:
        json.dump(dict(runs=res), fh, indent=1)
    with open(os.path.join(out, "exp70_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------
def check(args):
    g = e69.grid_of(args)
    q0, c0 = e69.initial(g, 2, np.random.default_rng(100))
    q, c = e69.solve(g, q0, c0, 12, args.dt, args.nsub)
    dq, dc = e45.Data(g, q, args.dt), e45.Data(g, c, args.dt)
    # the uniform flow's tangent against central differences of the solver with a uniform flow
    h, eps = args.dt / args.nsub, 1e-6
    sol = []
    for sg in (1.0, -1.0):
        qq, cc = q[:, 5], c[:, 5]
        for _ in range(3 * args.nsub):
            qq, cc = step_u(g, qq, cc, sg * eps * 0.3, -sg * eps * 0.7, h)
        sol.append(cc)
    fd = (sol[0] - sol[1]) / (2 * eps)
    st = (q[:, 5][:, None], c[:, 5][:, None], torch.zeros_like(q[:, 5][:, None]), torch.zeros_like(c[:, 5][:, None]))
    for _ in range(3 * args.nsub):
        st = step_tan_u(g, *st, 0.3, -0.7, h)
    print(f"[check n {args.n}] uniform-flow tangent vs central differences: rel err "
          f"{float((st[3][:, 0] - fd).norm() / fd.norm()):.1e}")
    # the Galilean boost is exact with the flow: Q = -t (q_x, c_x), dU = 1 maps to Q(t + k dt)
    T5 = 5 * args.dt
    qx, qy = grad(g, q[:, 5])
    cx, cy = grad(g, c[:, 5])
    a1 = torch.tensor(1.0, dtype=g.dtype)
    s0 = (q[:, 5][:, None], c[:, 5][:, None], (-T5 * qx)[:, None], (-T5 * cx)[:, None])
    st = s0
    for _ in range(3 * args.nsub):
        st = step_tan_u(g, *st, a1, 0 * a1, h)
    T8 = 8 * args.dt
    tq, tc = grad(g, q[:, 8])[0], grad(g, c[:, 8])[0]
    print(f"  Galilean boost with a unit uniform flow after 3 snapshots: |dq - (-t q_x)| / |t q_x| "
          f"{float((st[2][:, 0] + T8 * tq).norm() / (T8 * tq).norm()):.1e}, c {float((st[3][:, 0] + T8 * tc).norm() / (T8 * tc).norm()):.1e}")
    # the state emulator's tangent against central differences (untrained, float64)
    torch.manual_seed(0)
    mdl = StateFNO2d(2, 16, 8, 2).double()
    mdl.std.copy_(torch.tensor([float(q.std()), float(c.std())], dtype=torch.float64))
    mdl.inc.copy_(torch.tensor([0.05, 0.05], dtype=torch.float64))
    x = torch.stack([q[:, 5], c[:, 5]], 1)
    d = torch.randn(2, 3, 2, g.n, g.n, dtype=torch.float64)
    with torch.no_grad():
        tr = StateOp(mdl).transport(x, d, 3)[:, -1]

    def roll(y):
        for _ in range(3):
            y = mdl(y)
        return y

    with torch.no_grad():
        fd = torch.stack([(roll(x + 1e-6 * d[:, i]) - roll(x - 1e-6 * d[:, i])) / 2e-6 for i in range(3)], 1)
    print(f"  state emulator (untrained) 3-step tangent vs central differences: rel err "
          f"{float((tr - fd).norm() / fd.norm()):.1e}")
    # the emulator pencil with the exact solver's increments in place of a network reproduces the oracle's sums
    r_, j_ = torch.tensor([0, 1]), torch.tensor([2, 3])
    N1 = e69.oracle_terms(g, dc, dq, r_, j_, 2, args.dt, args.nsub)
    N2 = state_terms(StateOp(ExactStep(g, args.dt, args.nsub)), dc, dq, r_, j_, 2)
    print(f"  state pencil with the exact step vs exp69's oracle terms: rel err "
          f"{max(float((a_ - b_).abs().max() / b_.abs().max()) for a_, b_ in zip(N2[:3], N1[:3])):.1e}")
    # the latent model's paths and one step
    lm = LatentModel(4, 16, 8, 2).double()
    with torch.no_grad():
        z, cc = latent_paths(lm, c, 12, False)
        zo, co = latent_paths(lm, c, 12, True)
    print(f"  latent paths: data z {tuple(z.shape)}, own (z, c) {tuple(zo.shape)}, {tuple(co.shape)}; own c equals the "
          f"data on the first 4 snapshots: {bool(torch.equal(co[:, :4], c[:, :4]))}")
    # the best hidden lift through the exact step reproduces exp69's observability (one pair, ten iterations)
    if os.path.exists(e69.cache_path(args, args.n_train, 4007, "q")):
        tmp = os.path.join(args.cache, f"exp70_check_{os.getpid()}")
        a1 = argparse.Namespace(**{**vars(args), "seed": 7, "obs_pairs": 1, "cg_iters": 10, "save": tmp + "_e69"})
        e69.observability(a1)
        cq = e69.load(a1, a1.n_train, 4007, "c", np.float64)
        qq = e69.load(a1, a1.n_train, 4007, "q", np.float64)
        d1, dq1 = e45.Data(g, cq, args.dt), e45.Data(g, qq, args.dt)
        a2 = argparse.Namespace(**{**vars(a1), "save": tmp + "_e70", "emu": "state"})
        lift_arm(a2, ExactStep(g, args.dt, args.nsub), d1, dq1, {})
        r1 = json.load(open(tmp + "_e69.json"))["quotients"]
        r2 = json.load(open(tmp + "_e70.json"))["quotients"]
        err = max(abs(r2[k][v] - r1[k][v]) / max(abs(r1[k][v]), 1e-12) for k in r1 for v in ("scalar", "best"))
        print(f"  best hidden lift through the exact step vs exp69.observability: max rel diff {err:.1e} (values below "
              f"1e-12 compared absolutely)")
        for p in (tmp + "_e69.json", tmp + "_e70.json"):
            os.remove(p)


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "observe", "train", "fins", "summary"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(RESULTS, "tracer_runs"))
    ap.add_argument("--runs69", default=os.path.join(RESULTS, "tracer_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--emu", default="state", choices=["state", "latent"])
    ap.add_argument("--arm", default="data", choices=["data", "self", "lift"])
    ap.add_argument("--w", type=int, default=4, help="latent: the encoder's window")
    ap.add_argument("--roll", type=int, default=None, help="training rollout length (default: state 2, latent 4)")
    ap.add_argument("--z-inc", type=float, default=0.01, help="latent: z's increment scale in G")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=os.path.join(RESULTS, "tracer_cache"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--n-traj", type=int, default=1024)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=40)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=5000)
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
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--cg-iters", type=int, default=100)
    ap.add_argument("--obs-pairs", type=int, default=4)
    ap.add_argument("--fp64", action="store_true", help="emulator arms in float64 on the GPU (always on the CPU)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.roll is None:
        args.roll = 2 if args.emu == "state" else 4
    if args.threads:
        torch.set_num_threads(args.threads)
    {"check": check, "observe": observe, "train": train, "fins": fins, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
