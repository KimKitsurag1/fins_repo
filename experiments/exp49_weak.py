"""Experiment 49 -- the transport criterion in weak form: no derivative of the data is taken.

Pointwise FINS compares J_{n,k} Q(n) - Q(n) with Q(n+k) - Q(n) at every grid point and every start n, with
Q = phi - xi u_x - tau u_t built from finite differences of the snapshots, so noise in u reaches the criterion
multiplied by 1/dt (u_t) and by the grid wavenumber (u_x). Here both residual terms are tested against smooth
functions: psi_j(x) on the scored region (a bump times low Fourier modes) and w_l(s) over dense start times
(overlapping compact bumps).

Class: tau = tau(t), xi = xi(x, t), phi = phi(x, t, u), the form every point symmetry of a quasilinear second-order
or any higher-order evolution equation takes. Then
  * tau(t) u_t: along the data J_{n,k} u_t(n) = d/ds S^k(u(s)), so the transported and the data terms are both
    s-derivatives and are integrated by parts against w_l: only S^k applied to the snapshots is needed;
  * xi(x, t) u_x: transported by forward JVP with the spectral u_x and then tested against psi_j, which equals
    -<D(xi J^T psi_j), u> exactly (D skew-adjoint); the noise of u_x reaches the criterion only through that
    smooth adjoint field;
  * phi(x, t, u): no derivative.
The pencil, normaliser and readout are exp34's; the eigenvectors are saved in the full degree-d class (zeros
outside the restricted one) so exp42 scores them unchanged.

Operators: heat on four-period windows (exact heat kernel, or a CNO), Burgers on windows (CNO), KdV and Benjamin-Ono
(bo.py) on the torus (exact solver by central-difference JVP, or the FNO stepper). Data, corruption and scoring as
exp48. --wx auto: the spatial modes from the data's bandwidth (bandwidth_modes), no ground truth.

Run:  python exp49_weak.py --pde heat --op exact --data-seed 1007 --noise 0.01 --save weak_heat.npz
"""

import argparse
import itertools
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp34_normalization as e34                          # noqa: E402
import exp39_localised as e39                              # noqa: E402
import exp40_window as e40                                 # noqa: E402
import corrupt as cr                                       # noqa: E402
import fburgers as fb                                      # noqa: E402
from window_stepper import GlobalStepper, to_float64      # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # learned operators; numpy elsewhere
MAX_OWN = 16          # own count: at most this many generators
OWN_FLOOR = 1e-6      # own count: squared relative residuals below this (0.1%) count as exact


# ---------------------------------------------------------------------------------------------------------------------
# data and operators

def window_setup(args, cfg):
    """Heat / Burgers on args.periods-period windows: noisy and clean data on the window, the operator, its margin."""
    smooth, only_L = (True, 1.0) if args.pde == "heat" else (False, None)
    xs, ts, u, _ = e40.trajectories(args.pde, cfg, args.n, args.dseed, smooth, only_L)
    dt0 = ts[0, 1] - ts[0, 0]
    u_clean, u = cr.corrupt(u, args.noise, args.thin, args.dseed, cr.EVAL)
    ts = ts[:, ::args.thin]
    N, Nx, Nt = u.shape
    W = args.periods * Nx
    idx = np.arange(W) % Nx
    dx = e31.lengths(xs) / Nx
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)[None, :, None] / dx[:, None, None]
    ux = np.real(np.fft.ifft(1j * k * np.fft.fft(u, axis=1), axis=1))
    if args.op == "exact":
        if args.pde != "heat":
            raise SystemExit("exact window operator only for heat")
        sig2 = cfg.alpha_heat * dt0 / dx[0] ** 2
        base = ker = e40.moment_kernel(sig2, 6)
        for _ in range(args.thin - 1):                  # one step between kept snapshots = thin original steps
            ker = np.convolve(ker, base)
        stp = e40.HeatKernelWindow(ker)
    else:
        garch = dict(cno_layers=3, cno_res=2, cno_mult=16, fno_modes=32, fno_width=48, fno_layers=4, fno_pad=0.25)
        name = args.stepper or f"S_theta_{args.pde}_s{args.seed}_g{cfg.grid_size}x{cfg.time_steps}_cno{args.tag}.pt"
        path = name if os.path.isabs(name) else os.path.join(os.environ.get("FINS_RESULTS_DIR", "."), name)
        stp = GlobalStepper("cno", W, **garch)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        stp.load_state_dict({kk: v for kk, v in sd.items() if torch.is_tensor(v)})
    stp = to_float64(stp).eval().to(DEVICE)
    for p in stp.parameters():
        p.requires_grad_(False)
    r = getattr(stp, "r", 0)
    shrink = r * args.K if r > 0 else args.score_margin
    xw = dx[:, None] * np.arange(W)[None]
    X = (xw - xw[:, W // 2:W // 2 + 1]) / (W * dx).max()
    return dict(kind="window", u=u[:, idx], ux=ux[:, idx], u_clean=u_clean, xs=xs, ts=ts, X=X, stp=stp, r=r,
                shrink=shrink, reg=slice(shrink, W - shrink), Lw=(W * dx).max(), x_offset=dx * ((W // 2) % Nx),
                xgt=xw - xw[:, W // 2:W // 2 + 1])


def torus_setup(args, cfg):
    """KdV / Benjamin-Ono on the torus: periodised two-soliton data (exp39 --ic periodic), the band [seam, 1 - seam] L
    scored."""
    T = e39.horizon(args.pde)
    L, x, t = e39.grid(args.pde, cfg, args.nx, args.nt, T)
    u, _, ts = e39.evaluation_data(args.pde, "periodic", args.n, args.dseed, x, t, 0.002)
    u_clean, u = cr.corrupt(u, args.noise, args.thin, args.dseed, cr.EVAL)
    ts = ts[:, ::args.thin]
    kk = 2 * np.pi * np.fft.fftfreq(args.nx, d=L / args.nx)[None, :, None]
    ux = np.real(np.fft.ifft(1j * kk * np.fft.fft(u, axis=1), axis=1))
    frac = np.arange(args.nx) / args.nx
    keep = np.flatnonzero((frac >= args.seam) & (frac <= 1 - args.seam))
    stp = None
    if args.op == "learned":
        a = argparse.Namespace(pde=args.pde, train_ic=args.train_ic, stepper=args.stepper, seed=args.seed, tag=args.tag,
                               modes=64, train_if_missing=False)
        stp = e39.load_or_train(a, x, T, args.nt).to(DEVICE)
    xs = np.tile(x, (args.n, 1))
    return dict(kind="torus", u=u, ux=ux, u_clean=u_clean, xs=xs, ts=ts, X=xs / L, stp=stp, r=0, shrink=0,
                reg=keep, Lw=L, x_offset=np.zeros(args.n), xgt=xs, T_horizon=T)


class Operator:
    """S^k on the scored region: rollouts and forward JVPs, for torch steppers and for the exact KdV solver."""

    def __init__(self, args, S):
        self.args, self.S = args, S

    def _cut(self, v, k):
        """Field on the operator's output grid after k steps -> scored region."""
        S = self.S
        if S["kind"] == "torus":
            return v[:, S["reg"]]
        off = S["shrink"] - k * S["r"]
        width = S["u"].shape[1] - 2 * S["shrink"]
        return v[:, off:off + width]

    def rollout(self, u0):
        """u0 [B, D] -> [K, B, R]: S^k(u0) for k = 1..K on the scored region."""
        if self.S["stp"] is None:
            sol = e39.solve(self.args.pde, u0, self._horizon(), self.args.K * self.args.thin + 1, 0.002)[:, :, ::self.args.thin]
            return np.stack([self._cut(sol[:, :, k], k) for k in range(1, self.args.K + 1)])
        dev = self._device()
        out, s = [], torch.tensor(u0, device=dev)[:, None]
        with torch.no_grad():
            for k in range(1, self.args.K + 1):
                s = self.S["stp"](s)
                out.append(self._cut(s[:, 0].cpu().numpy(), k))
        return np.stack(out)

    def jvp_chain(self, u0, q0):
        """Tangents q0 [B, D] at states u0 -> [K, B, R]: J^k q0 on the scored region."""
        if self.S["stp"] is None:
            qn = np.linalg.norm(q0, axis=1, keepdims=True)
            eps = np.where(qn > 1e-12, 1e-4 * np.linalg.norm(u0, axis=1, keepdims=True) / np.maximum(qn, 1e-30), 0.0)
            sol = e39.solve(self.args.pde, np.concatenate([u0 + eps * q0, u0 - eps * q0]), self._horizon(),
                            self.args.K * self.args.thin + 1, 0.002)[:, :, ::self.args.thin]
            B = len(u0)
            e = np.where(eps > 0, eps, 1.0)[:, :, None]
            d = np.where(eps[:, :, None] > 0, (sol[:B] - sol[B:]) / (2 * e), 0.0)
            return np.stack([self._cut(d[:, :, k], k) for k in range(1, self.args.K + 1)])
        dev = self._device()
        out, s, t = [], torch.tensor(u0, device=dev)[:, None], torch.tensor(q0, device=dev)[:, None]
        for k in range(1, self.args.K + 1):
            s, t = jvp(self.S["stp"], (s,), (t,))
            out.append(self._cut(t[:, 0].cpu().numpy(), k))
        return np.stack(out)

    def _device(self):
        stp = self.S["stp"]
        return next(itertools.chain(stp.parameters(), stp.buffers())).device     # the heat kernel has buffers only

    def _horizon(self):
        dt = self.S["ts"][0, 1] - self.S["ts"][0, 0]
        return self.args.K * dt


# ---------------------------------------------------------------------------------------------------------------------
# test functions

def bandwidth_modes(u, R, factor):
    """Spatial test modes from the data alone: M = round(factor n95 R / D), n95 the smallest Fourier index (over the
    D-point domain) below which 95% of the non-mean energy of the (noisy) snapshots lies, R the scored points."""
    E = (np.abs(np.fft.rfft(u, axis=1)) ** 2).sum(axis=(0, 2))[1:]
    n95 = int(np.searchsorted(np.cumsum(E) / E.sum(), 0.95)) + 1
    return max(1, int(round(factor * n95 * R / u.shape[1]))), n95


def own_count(lam):
    """Generators without the ground truth: the largest ratio of consecutive eigenvalues among the lowest MAX_OWN + 1,
    eigenvalues (squared relative residuals) floored at OWN_FLOOR."""
    top = min(MAX_OWN, len(lam) - 1)
    lf = np.maximum(np.asarray(lam)[:top + 1], OWN_FLOOR)
    return int(np.argmax(lf[1:] / lf[:-1])) + 1


def spatial_tests(R, M, p):
    """[2 M + 1, R]: bump (1 - z^2)^p times 1, cos(pi m z), sin(pi m z), z in (-1, 1) across the scored region."""
    z = np.linspace(-1, 1, R + 2)[1:-1]
    b = (1 - z ** 2) ** p
    rows = [b] + [b * f(np.pi * m * z) for m in range(1, M + 1) for f in (np.cos, np.sin)]
    P = np.stack(rows)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def temporal_weights(s, s0, s1, J, p):
    """w [J, len(s)] and dw/ds: J bumps (1 - z^2)^p, half-width h, centres s0 + (l + 1) h, l = 0..J-1, h = (s1 - s0)
    / (J + 1): each supported inside [s0, s1], neighbours overlapping by half."""
    h = (s1 - s0) / (J + 1)
    c = s0 + h * (np.arange(J) + 1)
    z = (s[None, :] - c[:, None]) / h
    inside = np.abs(z) < 1
    w = np.where(inside, (1 - z ** 2) ** p, 0.0)
    dw = np.where(inside, -2 * p * z * (1 - z ** 2) ** (p - 1) / h, 0.0)
    return w, dw


# ---------------------------------------------------------------------------------------------------------------------
# the weak pencil

def restricted(mons):
    """Indices (comp, mono) of the class tau = tau(t), xi = xi(x, t), phi = phi(x, t, u); mons (a, b, c) ~ X^a T^b U^c."""
    out = []
    for comp in range(3):
        for j, (a, b, c) in enumerate(mons):
            if comp == 0 and c:
                continue
            if comp == 1 and (a or c):
                continue
            out.append((comp, j))
    return out


def weak_blocks(args, S, op, mons, basis):
    """A, B [rows, P] (rows = trajectory, psi, w, k) and Q0 [rows0, P] of the weak criterion."""
    u, ux, ts = S["u"], S["ux"], S["ts"]
    N, D, Nt = u.shape
    K = args.K
    Tmax = ts[:, -1].max()
    U = u / u.std()
    Tn = ts / Tmax
    reg = S["reg"]
    R = len(range(*reg.indices(D))) if isinstance(reg, slice) else len(reg)
    if args.wx == "auto":
        args.wx, n95 = bandwidth_modes(u, R, args.wx_factor)
        print(f"  spatial modes from the data: n95 {n95} of a {D}-point domain, {R} scored points -> M = {args.wx}")
    args.wx = int(args.wx)
    psi = spatial_tests(R, args.wx, args.wp)                              # [J, R]
    starts = np.arange(args.n0, Nt - K, args.stride)
    s = ts[0, starts]
    ds = args.stride * (ts[0, 1] - ts[0, 0])
    w, dw = temporal_weights(s, s[0], s[-1], args.wt, args.wp)         # [L, S]
    dt_k = (ts[0, 1] - ts[0, 0]) * np.arange(K + 1)
    Jn, Ln, Sn = psi.shape[0], w.shape[0], len(starts)

    def proj(v):                                                       # [..., R] -> [..., J]
        return v @ psi.T

    def region(f):                                                     # [N, D, ...] -> [N, R, ...]
        return f[:, reg]

    t0 = time.time()
    # S^k of every start snapshot (scored region) and the data on the scored region
    u0 = np.concatenate([u[:, :, n] for n in starts])                  # [S N, D] start-major
    Fk = op.rollout(u0).reshape(K, Sn, N, -1)                          # [K, S, N, R]
    uR = region(u)                                                     # [N, R, Nt]
    gF = proj(Fk - np.moveaxis(uR[:, :, starts], 2, 0)[None])         # <psi, S^k u(n) - u(n)>  [K, S, N, J]
    gU = proj(np.moveaxis(uR, 2, 0))                                   # <psi, u(n)>             [Nt, N, J]
    print(f"  rollouts of {len(u0)} start states: {time.time() - t0:.0f} s", flush=True)

    A, B, Q0 = [], [], []
    for comp, j in basis:
        a_, b_, c_ = mons[j]
        if comp == 1:
            # tau = T^b: every u_t term is an s-derivative, integrated by parts against w_l
            f = lambda tt, b_=b_: (tt / Tmax) ** b_                    # noqa: E731
            fp = lambda tt, b_=b_: b_ * (tt / Tmax) ** max(b_ - 1, 0) / Tmax if b_ else 0.0 * tt  # noqa: E731
            wf_d = dw * f(s)[None] + w * fp(s)[None]                   # d/ds (w f)            [L, S]
            # A: sum_n w_l A_i(n, k) = + sum_n (w_l f)'(s_n) <psi, S^k u(n) - u(n)>
            Ai = np.einsum("ls,ksnj->nlkj", wf_d, gF) * ds
            # B: + sum_n d/ds[w_l(s) f(s + k dt)] <psi, u(n + k)> - sum_n (w_l f)'(s_n) <psi, u(n)>
            Bi = np.zeros_like(Ai)
            for k in range(1, K + 1):
                shift = dw * f(s + dt_k[k])[None] + w * fp(s + dt_k[k])[None]
                Bi[:, :, k - 1] = (np.einsum("ls,snj->nlj", shift, gU[starts + k])
                                   - np.einsum("ls,snj->nlj", wf_d, gU[starts]))
            Bi *= ds
            Qi = np.einsum("ls,snj->nlj", wf_d, gU[starts]) * ds       # sum_n w_l q(n) = + sum (w f)' <psi, u>
        else:
            mono = np.broadcast_to(S["X"][:, :, None] ** a_ * Tn[:, None, :] ** b_, u.shape) * U ** c_
            q = mono if comp == 2 else -mono * ux                      # [N, D, Nt]
            q0 = np.concatenate([q[:, :, n] for n in starts])          # [S N, D]
            Jq = op.jvp_chain(u0, q0).reshape(K, Sn, N, -1)            # [K, S, N, R]
            qR = region(q)                                             # [N, R, Nt]
            a_k = proj(Jq - np.moveaxis(qR[:, :, starts], 2, 0)[None])   # [K, S, N, J]
            Ai = np.einsum("ls,ksnj->nlkj", w, a_k) * ds
            gq = proj(np.moveaxis(qR, 2, 0))                           # [Nt, N, J]
            Bi = np.stack([np.einsum("ls,snj->nlj", w, gq[starts + k] - gq[starts]) for k in range(1, K + 1)], 2) * ds
            Qi = np.einsum("ls,snj->nlj", w, gq[starts]) * ds
        A.append(Ai.ravel())
        B.append(Bi.ravel())
        Q0.append(Qi.ravel())
    print(f"  weak blocks for {len(basis)} basis fields, {Sn} starts, {Jn} x {Ln} tests: {time.time() - t0:.0f} s",
          flush=True)
    return np.stack(A, 1), np.stack(B, 1), np.stack(Q0, 1)


def class_fields(S, mons, basis, d):
    """Fields [P, 3, N, R, Nt] of the restricted basis on the scored region, in exp40's coordinates."""
    u = S["u"]
    Tn = S["ts"] / S["ts"][:, -1].max()
    U = u / u.std()
    reg = S["reg"]
    out = []
    for comp, j in basis:
        a, b, c = mons[j]
        f = np.broadcast_to(S["X"][:, :, None] ** a * Tn[:, None, :] ** b, u.shape) * U ** c
        fld = np.zeros((3,) + f[:, reg].shape)
        fld[comp] = f[:, reg]
        out.append(fld)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["heat", "burgers", "kdv", "bo", "ks", *fb.EQUATIONS, *e39.ccf.EQUATIONS])
    ap.add_argument("--op", default="exact", choices=["exact", "learned"])
    ap.add_argument("--seed", type=int, default=42, help="operator seed (learned)")
    ap.add_argument("--data-seed", type=int, default=None, help="evaluation data seed (default 1000 + seed)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1, help="start stride (snapshots)")
    ap.add_argument("--n0", type=int, default=0, help="first start")
    ap.add_argument("--wx", default="auto", help="spatial tests: bump times Fourier modes 1..wx; 'auto': from the "
                    "data's bandwidth (bandwidth_modes)")
    ap.add_argument("--wx-factor", type=float, default=2.0, help="'auto': modes per 95%%-energy Fourier index, "
                    "scaled to the scored region")
    ap.add_argument("--wt", type=int, default=8, help="temporal bumps")
    ap.add_argument("--wp", type=int, default=4, help="bump exponent")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--score-margin", type=int, default=96)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--stepper", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-ic", default="mixed", choices=["mixed", "attractor"],
                    help="torus steppers: their training data (KS 'attractor': zero mean)")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    args.dseed = 1000 + args.seed if args.data_seed is None else args.data_seed
    cfg = e31.ExperimentConfig()
    d = args.degree or (3 if args.pde == "heat" else 2)
    S = torus_setup(args, cfg) if args.pde in e39.TORUS else window_setup(args, cfg)
    op = Operator(args, S)
    mons = [mm for mm in itertools.product(range(d + 1), repeat=3) if sum(mm) <= d]
    basis = restricted(mons)
    t0 = time.time()
    A, B, Q0 = weak_blocks(args, S, op, mons, basis)
    Nr = (A - B).T @ (A - B)
    G0 = args.K * Q0.T @ Q0
    Dr = A.T @ A + B.T @ B + args.mu * np.trace(B.T @ B) / np.trace(G0) * G0
    lam, C = e34.solve(Nr, Dr)

    names, gt = e31.algebra(args.pde, S["xgt"], S["ts"], S["u"], cfg.alpha_heat, d)
    Fr = class_fields(S, mons, basis, d)                               # [P, 3, N, R, Nt]
    tm = max(1, 8 // args.thin)
    reg = S["reg"]
    Fv = Fr[..., tm:-tm].reshape(len(basis), -1)
    gtv = np.stack([g[:, :, reg][..., tm:-tm].ravel() for g in gt])
    cg, res, *_ = np.linalg.lstsq(Fv.T, gtv.T, rcond=None)
    fit = np.linalg.norm(Fv.T @ cg - gtv.T, axis=0) / np.linalg.norm(gtv, axis=1)
    rq = [float(c @ Nr @ c / (c @ Dr @ c)) for c in cg.T]
    k = len(names)
    Qe, _ = np.linalg.qr((C[:, :k].T @ Fv).T)
    cont = [float(np.linalg.norm(Qe.T @ g) / np.linalg.norm(g)) for g in gtv]
    print(f"[{args.pde}, {args.op}, weak] data seed {args.dseed}, {S['u'].shape[0]} trajectories, degree {d}, "
          f"restricted class {len(basis)} of {3 * len(mons)}, K = {args.K}, starts every {args.stride}, tests "
          f"{2 * args.wx + 1} x {args.wt} (p = {args.wp}), mu {args.mu}, noise {args.noise:g}, thin {args.thin}; "
          f"{time.time() - t0:.0f} s")
    print(f"  ground truth in the restricted class: max relative fit residual {fit.max():.1e}")
    print("  lowest eigenvalues: " + " ".join(f"{v:.1e}" for v in lam[:k + 3]))
    print("  algebra losses: " + "  ".join(f"{n} {v:.1e}" for n, v in zip(names, rq)) + f"   (lambda{k + 1} = {lam[k]:.1e})")
    print(f"  algebra inside the {k} lowest: " + "  ".join(f"{n} {c:.2f}" for n, c in zip(names, cont)))
    # own count, without the ground truth: the largest ratio of consecutive eigenvalues among the lowest MAX_OWN + 1
    # (eigenvalues are squared relative residuals; those below OWN_FLOOR count as equal, i.e. exact)
    nown = own_count(lam)
    print(f"  own count (largest eigenvalue ratio among the lowest {MAX_OWN + 1}, floor {OWN_FLOOR:g}): {nown} "
          f"(k = {k})")
    if args.save:
        nm = len(mons)
        Cfull = np.zeros((3 * nm, C.shape[1]))
        for row, (comp, j) in enumerate(basis):
            Cfull[comp * nm + j] = C[row]
        u = S["u"]
        sigma = u.std()
        np.savez(args.save, pde=args.pde, method=f"FINS-weak-{args.op}", seed=args.seed, data_seed=args.dseed,
                 C=Cfull, lam=lam, nown=nown, mons=np.array(mons), degree=d, Lw=S["Lw"],
                 T_max=S["ts"][:, -1].max(), sigma=sigma, x_offset=S["x_offset"], u=S["u_clean"], xs=S["xs"],
                 ts=S["ts"], names=np.array(names), losses=np.array(rq), containment=np.array(cont), K=args.K, k=k,
                 noise=args.noise, thin=args.thin, form="weak", wx=args.wx, wt=args.wt, wp=args.wp,
                 stride=args.stride, wx_factor=args.wx_factor)
        print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
