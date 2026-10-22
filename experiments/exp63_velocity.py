"""Experiment 63 -- FINS for systems: point symmetries of pyqg's flow in the velocity (u, v) (README_EXPERIMENTS.md,
entry of 09.10).

State (u, v) = grad^perp psi of pyqg's data (u = -psi_y, v = psi_x). Operator: (u, v) -> q (curl, and with rd the
screening q = zeta (1 + kd2 / |k|^2)), the port's step (qg2d), back to (u, v); its exact tangent. Two operators:
  pyqg     -- pyqg's state space: the mean velocity is annihilated (the curl ignores it), so constant boosts break;
  meanflow -- a conserved mean flow (U0, V0) that advects q, with rd the PV gradient kd2 (U0 y - V0 x) it implies;
              uniform q tendencies dropped (pyqg's inversion ignores the mean of q).
Class: monomials of degree <= 2 in (X, Y, T, U, V) in each of (xi, eta, tau, phi_u, phi_v): 105 fields, one sigma for
u and v. Readouts: own count (floor 1e-6); spans against the regime's algebra; restricted pencils on the rotation
space (y d_x, x d_y, v d_u, u d_v), the scaling space (x d_x + y d_y, t d_t, u d_u, v d_v) and the boost space
(t d_x, d_u, t d_y, d_v).

Run:  python exp63_velocity.py fins --regime f --op pyqg --seed 7 --data-seed 100 --eval-seed 101 --save <stem>
"""

import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pyqg
import torch
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp61_pyqg as e61                                   # noqa: E402
import qg2d                                                # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
MONS5 = [m for m in ((a, b, c, d, e) for a in range(3) for b in range(3) for c in range(3) for d in range(3)
                     for e in range(3)) if sum(m) <= 2]
NM = len(MONS5)
NF = 5 * NM
XI, ETA, TAU, PU, PV = range(5)


# ---------------------------------------------------------------------------------------------------------------
# Velocity operator
# ---------------------------------------------------------------------------------------------------------------
def kd2_of(g):
    return g.rd ** -2 if g.rd else 0.0


def to_q(g, uv):
    """(u, v) [.., 2, n, n] -> q [.., n, n]: q_hat = zeta_hat (1 + kd2 / |k|^2), zero mean."""
    uh, vh = torch.fft.rfft2(uv[..., 0, :, :]), torch.fft.rfft2(uv[..., 1, :, :])
    fac = torch.where(g.K2 > 0, 1 + kd2_of(g) / torch.where(g.K2 > 0, g.K2, torch.ones_like(g.K2)),
                      torch.zeros_like(g.K2))
    return qg2d.irfft(g, fac * (g.ikx * vh - g.iky * uh))


def to_uv(g, q):
    ph = g.a * torch.fft.rfft2(q)
    return torch.stack([qg2d.irfft(g, -g.iky * ph), qg2d.irfft(g, g.ikx * ph)], -3)


def rhs_tan_mf(g, q, d, dm):
    """The port's tangent plus the mean-flow terms: -dU0 q_x - dV0 q_y - kd2 (dU0 v - dV0 u); dm [P, m, 2]."""
    out = qg2d.rhs_tan(g, q, d)
    if dm is None:
        return out
    qh = torch.fft.rfft2(q)
    qx, qy = qg2d.irfft(g, g.ikx * qh), qg2d.irfft(g, g.iky * qh)
    du, dv = dm[..., 0, None, None], dm[..., 1, None, None]
    out = out - du * qx - dv * qy
    if g.rd:
        uv = to_uv(g, q)
        out = out - kd2_of(g) * (du * uv[..., 1, :, :] - dv * uv[..., 0, :, :])
    return out


def step_tan_mf(g, q, d, dm, h):
    k1, l1 = qg2d.rhs(g, q), rhs_tan_mf(g, q, d, dm)
    q2, d2 = q + 0.5 * h * k1, d + 0.5 * h * l1
    k2, l2 = qg2d.rhs(g, q2), rhs_tan_mf(g, q2, d2, dm)
    q3, d3 = q + 0.5 * h * k2, d + 0.5 * h * l2
    k3, l3 = qg2d.rhs(g, q3), rhs_tan_mf(g, q3, d3, dm)
    q4, d4 = q + h * k3, d + h * l3
    k4, l4 = qg2d.rhs(g, q4), rhs_tan_mf(g, q4, d4, dm)
    return (qg2d.filt(g, q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)),
            qg2d.filt(g, d + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4)))


class VelocityOp:
    """The snapshot map in (u, v) and its exact tangent; meanflow: the mean perturbation is conserved and advects q."""

    def __init__(self, g, dt, nsub, meanflow=False):
        self.g, self.dt, self.nsub, self.meanflow = g, dt, nsub, meanflow

    def transport(self, uv0, d0, K):
        """uv0 [P, 2, n, n], d0 [P, m, 2, n, n] -> J_k d0 [P, K, m, 2, n, n], k = 1..K."""
        g = self.g
        q, d = to_q(g, uv0)[:, None], to_q(g, d0)
        dm = d0.mean((-2, -1)) if self.meanflow else None
        out = []
        for _ in range(K):
            for _ in range(self.nsub):
                q, d = step_tan_mf(g, q, d, dm, self.dt / self.nsub)
            o = to_uv(g, d)
            out.append(o if dm is None else o + dm[..., None, None])
        return torch.stack(out, 1)


# ---------------------------------------------------------------------------------------------------------------
# Data, class, generators
# ---------------------------------------------------------------------------------------------------------------
class DataUV:
    """Velocity trajectories [B, nt, 2, n, n] with spectral x, y and fourth-order t derivatives; one sigma."""

    def __init__(self, g, uv, dt, scale=None):
        self.g, self.u, self.dt = g, uv, dt
        self.ux, self.uy, self.ut = e45.derivatives(g, uv, dt)
        self.t = dt * torch.arange(uv.shape[1], dtype=uv.dtype)
        self.s = scale or dict(Lx=g.L / 2, T=float(self.t[-1]), sigma=float(uv.std()))

    def at(self, r, j, reg):
        w, wx, wy, wt = (a[r, j][..., reg, reg] for a in (self.u, self.ux, self.uy, self.ut))
        sh = w.shape[:-3] + w.shape[-2:]
        X = (self.g.X[reg, reg] / self.s["Lx"]).expand(sh)
        Y = (self.g.Y[reg, reg] / self.s["Lx"]).expand(sh)
        T = (self.t[j] / self.s["T"])[..., None, None].expand(sh)
        return w, wx, wy, wt, X, Y, T


def chars_uv(d, r, j, reg):
    """Q / sigma of the 105 fields: [*S, 105, 2, h, w] (component-major, MONS5 order)."""
    w, wx, wy, wt, X, Y, T = d.at(r, j, reg)
    U, V = w[..., 0, :, :] / d.s["sigma"], w[..., 1, :, :] / d.s["sigma"]
    mono = torch.stack([X ** a * Y ** b * T ** c * U ** e * V ** f for a, b, c, e, f in MONS5], -3)
    sx, st = d.s["Lx"] / d.s["sigma"], d.s["T"] / d.s["sigma"]
    eu, ev = torch.zeros_like(w), torch.zeros_like(w)
    eu[..., 0, :, :], ev[..., 1, :, :] = 1.0, 1.0
    fac = torch.stack([-sx * wx, -sx * wy, -st * wt, eu, ev], -4)                  # [*S, 5, 2, h, w]
    Q = fac[..., :, None, :, :, :] * mono[..., None, :, None, :, :]                  # [*S, 5, 21, 2, h, w]
    return Q.reshape(*Q.shape[:-5], NF, *Q.shape[-3:])


def index(c, a=0, b=0, t=0, u=0, v=0):
    return c * NM + MONS5.index((a, b, t, u, v))


def vec(entries):
    out = np.zeros(NF)
    for c, mon, val in entries:
        out[index(c, *mon)] += val
    return out


def scaling(wx, wt, wu, wv):
    return vec([(XI, (1, 0, 0, 0, 0), wx), (ETA, (0, 1, 0, 0, 0), wx), (TAU, (0, 0, 1, 0, 0), wt),
                (PU, (0, 0, 0, 1, 0), wu), (PV, (0, 0, 0, 0, 1), wv)])


def generators(s):
    """Named fields in scaled components; s: the data scale (the boosts mix t with a velocity shift)."""
    orb = [(XI, (0, 1, 0, 0, 0), -1.0), (ETA, (1, 0, 0, 0, 0), 1.0)]
    spin = [(PU, (0, 0, 0, 0, 1), -1.0), (PV, (0, 0, 0, 1, 0), 1.0)]
    bt, bs = s["T"] / s["Lx"], 1.0 / s["sigma"]
    return {"d_x": vec([(XI, (0,) * 5, 1.0)]), "d_y": vec([(ETA, (0,) * 5, 1.0)]), "d_t": vec([(TAU, (0,) * 5, 1.0)]),
            "rotation": vec(orb + spin), "D_x": scaling(1.0, 0.0, 1.0, 1.0), "D_t": scaling(0.0, 1.0, -1.0, -1.0),
            "D_beta": scaling(1.0, -1.0, 2.0, 2.0),
            "boost_x": vec([(XI, (0, 0, 1, 0, 0), bt), (PU, (0,) * 5, bs)]),
            "boost_y": vec([(ETA, (0, 0, 1, 0, 0), bt), (PV, (0,) * 5, bs)]),
            "scalar rotation": vec(orb), "spin": vec(spin),
            "opposite spin": vec(orb + [(c, m, -x) for c, m, x in spin]),
            "u d_u - v d_v": scaling(0.0, 0.0, 1.0, -1.0), "u d_u + v d_v": scaling(0.0, 0.0, 1.0, 1.0),
            "d_u": vec([(PU, (0,) * 5, 1.0)]), "d_v": vec([(PV, (0,) * 5, 1.0)])}


def algebra(regime, op):
    """The point algebra of pyqg's operator in (u, v) for localised data; meanflow adds the constant boosts at f and
    beta (the meridional one because uniform q tendencies are dropped)."""
    p = qg2d.REGIMES[regime]
    names = ["d_x", "d_y", "d_t"]
    if not p["beta"]:
        names.append("rotation")
    if p["beta"]:
        names.append("D_beta")
    else:
        if not p["rd"]:
            names.append("D_x")
        if not p["rek"]:
            names.append("D_t")
    if op == "meanflow" and not p["rd"] and not p["rek"]:
        names += ["boost_x", "boost_y"]
    return names


SPACES = {"rotation": [(XI, (0, 1, 0, 0, 0)), (ETA, (1, 0, 0, 0, 0)), (PU, (0, 0, 0, 0, 1)), (PV, (0, 0, 0, 1, 0))],
          "scaling": None,
          "boost": [(XI, (0, 0, 1, 0, 0)), (PU, (0,) * 5), (ETA, (0, 0, 1, 0, 0)), (PV, (0,) * 5)]}


def space_basis(name):
    if name == "scaling":
        return np.stack([scaling(1, 0, 0, 0), scaling(0, 1, 0, 0), scaling(0, 0, 1, 0), scaling(0, 0, 0, 1)], 1)
    return np.stack([vec([(c, m, 1.0)]) for c, m in SPACES[name]], 1)


def restricted(N, D, name):
    V = space_basis(name)
    lam, W = eigh(V.T @ N @ V, V.T @ D @ V)
    return lam, W


def rotation_readout(N, D):
    """Deepest direction of (y d_x, x d_y, v d_u, u d_v) normalised to x d_y = 1: stretch sqrt(-c_yx / c_xy) and the
    spin-to-orbital ratio (c_ud_v - c_vd_u) / (c_xy - c_yx) (1: the vector rotation)."""
    lam, W = restricted(N, D, "rotation")
    w = W[:, 0] / W[1, 0]
    e = float(np.sqrt(-w[0] / w[1])) if w[0] * w[1] < 0 else None
    return dict(lam=lam.tolist(), vec=w.tolist(), stretch=e, spin=float((w[3] - w[2]) / (w[1] - w[0])))


def scaling_readout(N, D):
    lam, W = restricted(N, D, "scaling")
    return dict(lam=lam.tolist(), vecs=[(w / w[np.argmax(np.abs(w))]).tolist() for w in W.T])


def boost_readout(N, D, s):
    """Eigenvalues of (t d_x, d_u, t d_y, d_v); the deepest two directions' speeds: the d_u (d_v) coefficient over the
    t d_x (t d_y) one, in units of the boost (1: t d_x + d_u)."""
    lam, W = restricted(N, D, "boost")
    bt, bs = s["T"] / s["Lx"], 1.0 / s["sigma"]
    sp = []
    for w in W.T[:2]:
        x = float((w[1] / bs) / (w[0] / bt)) if abs(w[0]) > 1e-12 else None
        y = float((w[3] / bs) / (w[2] / bt)) if abs(w[2]) > 1e-12 else None
        sp.append(dict(x=x, y=y, share_x=float((w[0] ** 2 + w[1] ** 2) / (w ** 2).sum())))
    return dict(lam=lam.tolist(), deep=sp)


class Points:
    """exp45.Probe's sample points on held-out data: class characteristics [105, M]."""

    def __init__(self, d, every_t=4, every_x=2):
        g = d.g
        reg = slice(g.sl.start, g.sl.stop, every_x)
        js = torch.arange(2, d.u.shape[1] - 2, every_t)
        parts = []
        with torch.no_grad():
            for b in range(d.u.shape[0]):
                r = torch.full_like(js, b)
                parts.append(chars_uv(d, r, js, reg).movedim(1, 0).reshape(NF, -1).numpy())
        self.Q = np.concatenate(parts, 1)


def pyqg_q(g, n_traj, seed, nt, dt, nsub, aspect):
    """exp61's pyqg integration (AB3, step dt / nsub) from qg2d.initial with the given vortex aspect range."""
    q0 = qg2d.initial(g, n_traj, np.random.default_rng(seed), aspect=aspect)
    out = np.zeros((n_traj, nt, g.n, g.n))
    for r in range(n_traj):
        m = pyqg.BTModel(nx=g.n, L=g.L, beta=g.beta, rd=g.rd, rek=g.rek, U=g.U, filterfac=g.filterfac, dt=dt / nsub,
                         tmax=dt * nt, twrite=10 ** 9, tavestart=10 * dt * nt, log_level=0)
        m.set_q(q0[r].double().numpy().T[None].copy())
        out[r, 0] = np.asarray(m.q)[0].T
        for j in range(1, nt):
            for _ in range(nsub):
                m._step_forward()
            out[r, j] = np.asarray(m.q)[0].T
    return torch.tensor(out, dtype=g.dtype)


def velocity_data(args, g, n, seed):
    """pyqg's q trajectories mapped to (u, v); circular shielded vortices by default (aspect 1: elliptic vortices'
    quadrupole velocity tails reach the seam, development entry)."""
    if args.data == "pyqg":
        q = pyqg_q(g, n, seed, args.nt, args.dt, args.nsub, (args.aspect_min, args.aspect_max))
    else:
        q = qg2d.solve(g, qg2d.initial(g, n, np.random.default_rng(seed), aspect=(args.aspect_min, args.aspect_max)),
                       args.nt, args.dt, args.nsub)
    return to_uv(g, q)


def fins(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    t0 = time.time()
    dtr = DataUV(g, velocity_data(args, g, args.n_train, dseed), args.dt)
    pts = Points(DataUV(g, velocity_data(args, g, args.n_eval, eseed), args.dt, dtr.s))
    mean = float(dtr.u.mean((-2, -1)).abs().max())
    print(f"[{args.regime} (beta {g.beta}, rd {g.rd}, rek {g.rek}), (u, v), operator {args.op}] seed {s}: "
          f"{args.n_train} training trajectories (data seed {dseed}), {args.n_eval} held-out ({eseed}); data "
          f"{time.time() - t0:.0f} s; sigma {dtr.s['sigma']:.4f}; max |mean velocity| {mean:.1e}", flush=True)
    op = VelocityOp(g, args.dt, args.nsub, args.op == "meanflow")
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = e45.pencil(op, dtr, chars_uv, ptr, args.K, args.chunk)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N, D)
    gens = generators(dtr.s)
    names = algebra(args.regime, args.op)
    k = len(names)
    rq = {n: float(c @ N @ c / (c @ D @ c)) for n, c in gens.items()}
    lf6 = np.maximum(lam[:17], 1e-6)
    own = int(np.argmax(lf6[1:] / lf6[:-1])) + 1
    rot, sca, boo = rotation_readout(N, D), scaling_readout(N, D), boost_readout(N, D, dtr.s)
    print(f"  class of {NF} fields ({len(ptr)} pairs, {time.time() - t0:.0f} s): relative change over K "
          f"{np.sqrt(cmu):.2e}")
    print("    lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:12]))
    print(f"    algebra ({k}: {', '.join(names)}); own count (floor 1e-6) {own}")
    print("    quotients: " + "  ".join(f"{n} {v:.1e}" + ("*" if n in names else "") for n, v in rq.items()))
    print(f"    rotation space: eigenvalues " + " ".join(f"{v:.1e}" for v in rot["lam"])
          + f"; stretch {rot['stretch']}; spin / orbital {rot['spin']:.4f}")
    print("    scaling space: " + "; ".join(f"{l:.1e} (" + ", ".join(f"{x:+.3f}" for x in v) + ")"
                                         for l, v in zip(sca["lam"], sca["vecs"])))
    print("    boost space: eigenvalues " + " ".join(f"{v:.1e}" for v in boo["lam"]) + "; deepest two: "
          + "; ".join(f"speed x {b['x']}, y {b['y']}, x share {b['share_x']:.3f}" for b in boo["deep"]), flush=True)
    true_q = np.stack([gens[n] for n in names]) @ pts.Q
    spans = {}
    for lab, m in (("span_k", k), ("span_own", own)):
        Qc = C[:, :m].T @ pts.Q
        spans[lab] = dict(m=int(m), missing=e45.missing(Qc, true_q), novel=e45.novel(Qc, true_q).tolist())
        print(f"    span of the {m} lowest ({lab}): missing {spans[lab]['missing']:.3f}; novel "
              + " ".join(f"{n} {v:.2f}" for n, v in zip(names, spans[lab]["novel"])), flush=True)
    out = dict(config=vars(args), data_seed=dseed, eval_seed=eseed, scale=dtr.s, mean_velocity=mean,
               lam=lam[:20].tolist(), names=names, quotients=rq, own=own, cmu=cmu, rotation=rot, scaling=sca,
               boost=boo, **spans)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        np.savez(stem + ".npz", C=C[:, :20], lam=lam, N=N, D=D)
        print(f"  saved {stem}.json/.npz")


def check(args):
    """Tangent of both operators against central differences; the velocity map against the q map."""
    g = qg2d.regime_grid(args.regime, args.n, args.L, args.score)
    uv = velocity_data(args, g, 2, 100)
    d = DataUV(g, uv, args.dt)
    r, j = torch.tensor([0, 1]), torch.tensor([5, 20])
    qc = chars_uv(d, r, j, slice(None))[:, [3, 40, 70, 95]]
    for mf in (False, True):
        op = VelocityOp(g, args.dt, args.nsub, mf)
        jv = op.transport(d.u[r, j], qc, 3)[:, -1]
        eps = 1e-5 * float(d.u.abs().max()) / float(qc.abs().max())
        out = []
        for sgn in (1, -1):
            x = (d.u[r, j][:, None] + sgn * eps * qc).reshape(-1, 2, g.n, g.n)
            q = to_q(g, x)
            mean = x.mean((-2, -1))
            for _ in range(3 * args.nsub):
                if mf:
                    q = step_mf(g, q, mean, args.dt / args.nsub)
                else:
                    q = qg2d.step(g, q, args.dt / args.nsub)
            o = to_uv(g, q)
            out.append(o + mean[..., None, None] if mf else o)
        fd = ((out[0] - out[1]) / (2 * eps)).reshape(jv.shape)
        print(f"  {'meanflow' if mf else 'pyqg'}: tangent vs central differences {float((jv - fd).norm() / fd.norm()):.1e}")


def rhs_mf(g, q, mean):
    """The meanflow operator's right-hand side for a uniform flow mean [B, 2] (with rd its PV gradient term)."""
    out = qg2d.rhs(g, q)
    qh = torch.fft.rfft2(q)
    U, V = mean[..., 0, None, None], mean[..., 1, None, None]
    out = out - U * qg2d.irfft(g, g.ikx * qh) - V * qg2d.irfft(g, g.iky * qh)
    if g.rd:
        uv = to_uv(g, q)
        out = out - kd2_of(g) * (U * uv[..., 1, :, :] - V * uv[..., 0, :, :])
    return out


def step_mf(g, q, mean, h):
    k1 = rhs_mf(g, q, mean)
    k2 = rhs_mf(g, q + 0.5 * h * k1, mean)
    k3 = rhs_mf(g, q + 0.5 * h * k2, mean)
    k4 = rhs_mf(g, q + h * k3, mean)
    return qg2d.filt(g, q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


ROWS = [(r, "pyqg") for r in ("f", "beta", "rd", "drag")] + [(r, "meanflow") for r in ("f", "beta", "rd")]
MF_SEEDS = "42,43,44,45,46"


def stem_of(runs, regime, op, s):
    return os.path.join(runs, f"uv_{op}_{regime}_s{s}")


def jobs(args):
    out = []
    for regime, op in ROWS:
        for s in (args.seeds if op == "pyqg" else MF_SEEDS).split(","):
            stem = stem_of(args.runs, regime, op, s)
            out.append((stem + ".json", ["exp63_velocity.py", "fins", "--regime", regime, "--op", op, "--seed", s,
                                         "--threads", "1", "--save", stem]))
    return out


def summary(args):
    """results/velocity/summary.{json,md}."""
    res, lines = {}, ["# FINS for systems: pyqg's flow in (u, v) (exp63)\n",
                      "| row | n | own = k | missing <= 0.05 | max kept (median) | rotation spin / orbital | "
                      "boost space lowest (median) |", "|---|---|---|---|---|---|---|"]
    for regime, op in ROWS:
        cell = {}
        for s in (args.seeds if op == "pyqg" else MF_SEEDS).split(","):
            p = stem_of(args.runs, regime, op, s) + ".json"
            if os.path.exists(p):
                j = json.load(open(p))
                cell[s] = {k: j[k] for k in ("names", "own", "span_k", "span_own", "quotients", "rotation",
                                             "scaling", "boost", "lam")}
        if not cell:
            continue
        lab = f"{op} {regime}"
        res[lab] = cell
        mk = [max(c["quotients"][n] for n in c["names"]) for c in cell.values()]
        sp = [c["rotation"]["spin"] for c in cell.values()]
        bl = [c["boost"]["lam"][0] for c in cell.values()]
        lines.append(f"| {lab} | {len(cell)} | {sum(c['own'] == len(c['names']) for c in cell.values())} | "
                     f"{sum(c['span_k']['missing'] <= 0.05 for c in cell.values())} | {np.median(mk):.1e} | "
                     f"{min(sp):.3f} to {max(sp):.3f} | {np.median(bl):.1e} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "velocity")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "fins", "run", "summary"])
    ap.add_argument("--regime", default="f", choices=list(qg2d.REGIMES))
    ap.add_argument("--op", default="pyqg", choices=["pyqg", "meanflow"])
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "velocity_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--var", default="q")
    ap.add_argument("--aspect-min", type=float, default=1.0)
    ap.add_argument("--aspect-max", type=float, default=1.0)
    ap.add_argument("--data", default="pyqg", choices=["port", "pyqg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
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
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"check": check, "fins": fins, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
