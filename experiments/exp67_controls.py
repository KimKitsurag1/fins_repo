"""Experiment 67 -- exp60's controls and the linear-response baseline (README_EXPERIMENTS.md, entry of 09.10).

Equations (aniso2d's solver, exp60's data and settings, a = 1.5):
  aniso    -- exp60: u_t = (1/2) A(u^2) - nu (-A)^(a/2) u, A = d_xx + eps^2 d_yy, eps 0.6;
  mismatch -- the dissipation's anisotropy differs: eps 0.6 in the nonlinearity, 0.8 in the dissipation (no stretched
              rotation; the scaling x d_x + y d_y + a t d_t + (2 - a) u d_u kept);
  cubic    -- u_t = A(u^2 / 2 + gamma u^3) - nu (-A)^(a/2) u, gamma 0.2 (the scaling broken; the stretched rotation
              kept).
FINS: exp60's pencil (60 fields, exact tangent, 160 pairs) and readouts (the rotation and scaling scans, sl(2), the
quotients of the generators each equation keeps or breaks).
Linear response (no data): the stepper's Jacobian at constant states b in {0.25, 0.35, 0.45} is a Fourier multiplier
m(k; b) = exp(dt sigma(k; b)); sigma = sigma0(k) - b S(k) - b^2 S2(k) through the three states (|b^2 S2| / |b S| at the
mean b measures a b-dependence that is not linear); from sigma0 the dissipation's eps (argmin over e of its variation
on ellipses) and order (slope of log |sigma0| against log |k|_e); from S the nonlinearity's eps.

Run:  python exp67_controls.py fins --case mismatch --seed 7 --data-seed 100 --eval-seed 101 --save <stem>
      python exp67_controls.py response
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aniso2d as an                                       # noqa: E402
import exp45_neural_2d as e45                              # noqa: E402
import exp51_noise_weak as e51                             # noqa: E402
import exp60_aniso2d as e60                                # noqa: E402
import exp60_readouts as e60r                              # noqa: E402

RESULTS = os.environ.get("FINS_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
CASES = {"aniso": dict(eps_d=0.6, gamma=0.0), "mismatch": dict(eps_d=0.8, gamma=0.0), "cubic": dict(eps_d=0.6, gamma=0.2)}
BACKGROUNDS = (0.25, 0.35, 0.45)


class Grid(an.Grid):
    """aniso2d's grid with the dissipation's own anisotropy eps_d and a cubic term gamma A(u^3)."""

    def __init__(self, n=64, L=16.0, score=4.5, dtype=torch.float64, device="cpu", eps=an.EPS, a=1.5, nu=an.NU,
                 eps_d=an.EPS, gamma=0.0):
        super().__init__(n, L, score, dtype, device, eps, a, nu)
        self.eps_d, self.gamma = eps_d, gamma
        kx = 2 * math.pi * torch.fft.fftfreq(n, d=self.dx, dtype=dtype, device=device)
        ky = 2 * math.pi * torch.fft.rfftfreq(n, d=self.dx, dtype=dtype, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.frac = -nu * (KX ** 2 + eps_d ** 2 * KY ** 2) ** (a / 2) * self.mask


def rhs(g, u):
    return torch.fft.irfft2(g.lapA * torch.fft.rfft2(0.5 * u * u + g.gamma * u ** 3) + g.frac * torch.fft.rfft2(u),
                            s=(g.n, g.n))


def rhs_tan(g, u, d):
    return torch.fft.irfft2(g.lapA * torch.fft.rfft2((u + 3 * g.gamma * u * u) * d) + g.frac * torch.fft.rfft2(d),
                            s=(g.n, g.n))


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
    out, u = [u0], u0
    for _ in range(nt - 1):
        for _ in range(nsub):
            u = rk4(g, u, dt / nsub)
        out.append(u)
    return torch.stack(out, 1)


class ExactOp(e45.ExactOp):
    def transport(self, u0, d0, K):
        u, d, out = u0[:, None], d0, []
        for _ in range(K):
            for _ in range(self.nsub):
                u, d = rk4_tan(self.g, u, d, self.dt / self.nsub)
            out.append(d)
        return torch.stack(out, 1)


def grid_of(args, case):
    kw = dict(CASES[case])
    if getattr(args, "eps_d", None) is not None:
        kw["eps_d"] = args.eps_d
    if getattr(args, "gamma", None) is not None:
        kw["gamma"] = args.gamma
    return Grid(args.n, args.L, args.score, eps=an.EPS, a=args.a, nu=an.NU, **kw)


def fins(args):
    s = args.seed
    dseed = 4000 + s if args.data_seed is None else args.data_seed
    eseed = 5000 + s if args.eval_seed is None else args.eval_seed
    g = grid_of(args, args.case)
    t0 = time.time()
    dtr = e45.Data(g, solve(g, e45.initial(g, args.n_train, np.random.default_rng(dseed)), args.nt, args.dt,
                            args.nsub), args.dt)
    print(f"[{args.case}: eps {g.eps} (nonlinearity), {g.eps_d} (dissipation), gamma {g.gamma}, a {g.a}] seed {s}: "
          f"{args.n_train} trajectories (data seed {dseed}); u in [{float(dtr.u.min()):.3f}, {float(dtr.u.max()):.3f}]; "
          f"{time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    ptr = e45.pairs_of(args.n_train, args.nt, args.K, args.every)
    N, DA, G0, trBB = e45.pencil(ExactOp(g, args.dt, args.nsub), dtr, e45.chars_poly, ptr, args.K, 2)
    cmu = trBB / np.trace(G0)
    D = DA + args.mu * cmu * G0
    lam, C = e45.gsolve(N, D)
    Ct = an.true_coefficients(an.EPS, args.a)
    gens = dict(zip(an.NAMES, Ct.T))
    gens["stretched rotation at eps_d"] = an.true_coefficients(g.eps_d, args.a)[:, 3]
    rq = {n: float(c @ N @ c / (c @ D @ c)) for n, c in gens.items()}
    sa, sc, sq = e60.scan_scaling(N, D)
    se, seq = e60.scan_rotation(N, D)
    lf6 = np.maximum(lam[:17], 1e-6)
    own = int(np.argmax(lf6[1:] / lf6[:-1])) + 1
    sl = e60r.sl2(N, D)
    print(f"  pencil ({len(ptr)} pairs, {time.time() - t0:.0f} s): lowest " + " ".join(f"{v:.1e}" for v in lam[:8])
          + f"; own count {own}")
    print("    quotients: " + "  ".join(f"{n} {v:.1e}" for n, v in rq.items()))
    print(f"    scans: scaling a {sa:.3f}, c {sc:.3f} ({sq:.1e}); stretched rotation eps {se:.3f} ({seq:.1e}); sl(2) "
          f"eps {sl['eps']}, angle {sl['angle']}, eigenvalues {sl['lam'][0]:.1e} {sl['lam'][1]:.1e}", flush=True)
    out = dict(config=vars(args), case=args.case, eps_d=g.eps_d, gamma=g.gamma, lam=lam[:16].tolist(), own=own,
               cmu=cmu, quotients=rq, scan=dict(a=sa, c=sc, q=sq, eps=se, q_eps=seq), sl2=sl)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
        np.savez(stem + ".npz", C=C[:, :16], lam=lam, N=N, D=D)
        print(f"  saved {stem}.json/.npz")


def multiplier(g, b, dt, nsub):
    """sigma(k) = log m(k) / dt of the one-snapshot stepper's Jacobian at the constant state b, on the full k grid
    (fftshifted) and the mask."""
    d = torch.zeros(1, 1, g.n, g.n, dtype=g.dtype)
    d[..., 0, 0] = 1.0
    u = torch.full((1, g.n, g.n), b, dtype=g.dtype)
    op = ExactOp(g, dt, nsub)
    m = torch.fft.fft2(op.transport(u, d, 1)[0, 0, 0]).numpy()
    sig = np.log(np.abs(m)) / dt
    return np.fft.fftshift(sig.real)


def soft_eps(field, k, rmax, radii, es=np.arange(0.2, 2.0 + 1e-12, 0.001), n=96):
    """argmin over e of the largest relative variation of a symbol on the ellipses kx^2 + e^2 ky^2 = r^2."""
    f = RegularGridInterpolator((k, k), field, method="cubic")
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best = None
    for e in es:
        dev = 0.0
        for r in radii:
            pts = np.stack([r * np.cos(th), r * np.sin(th) / e], -1)
            if np.abs(pts).max() >= rmax:
                dev = np.inf
                break
            v = f(pts)
            dev = max(dev, np.abs(v - v.mean()).max() / max(np.abs(v).max(), 1e-300))
        if best is None or dev < best[1]:
            best = (float(e), float(dev))
    return best


def order(field, KX, KY, e, band, kmax):
    rho = np.sqrt(KX ** 2 + e ** 2 * KY ** 2)
    sel = (rho >= band[0]) & (rho <= band[1]) & (KX ** 2 + KY ** 2 < kmax ** 2)
    A = np.stack([np.ones(sel.sum()), np.log(rho[sel])], 1)
    c = np.linalg.lstsq(A, np.log(np.abs(field[sel])), rcond=None)[0]
    return float(c[1])


def response(args):
    out = {}
    for case in CASES:
        g = grid_of(args, case)
        k = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(g.n, d=g.dx))
        KX, KY = np.meshgrid(k, k, indexing="ij")
        sig = np.stack([multiplier(g, b, args.dt, args.nsub) for b in BACKGROUNDS])         # [nb, n, n]
        bb = np.array(BACKGROUNDS)
        B = np.stack([np.ones(len(bb)), -bb, -bb ** 2], 1)
        coef = np.linalg.solve(B, sig.reshape(len(bb), -1))
        s0, S, S2 = (coef[i].reshape(g.n, g.n) for i in range(3))
        resid = float(np.linalg.norm((bb.mean() ** 2 * S2).ravel()) / np.linalg.norm((bb.mean() * S).ravel()))
        rmax = (2.0 / 3.0) * math.pi / g.dx * 0.95
        radii = (1.0, 2.0, 3.0)
        ed, dd = soft_eps(-s0, k, rmax, radii)
        en, dn = soft_eps(S, k, rmax, radii)
        a_hat = order(s0, KX, KY, ed, (0.5, 6.0), rmax)
        out[case] = dict(eps_d=g.eps_d, gamma=g.gamma, eps_d_hat=ed, defect_d=dd, eps_n_hat=en, defect_n=dn,
                         a_hat=a_hat, b_nonlinearity=resid)
        print(f"  {case:9s} linear response: dissipation eps {ed:.3f} (variation {dd:.1e}), order {a_hat:.3f}; "
              f"nonlinearity eps {en:.3f} (variation {dn:.1e}); departure from linear in b {resid:.1e}", flush=True)
    if args.save:
        stem = args.save if os.path.isabs(args.save) else os.path.join(RESULTS, args.save)
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        with open(stem + ".json", "w") as fh:
            json.dump(dict(config=vars(args), cases=out), fh, indent=1)
        print(f"  saved {stem}.json")


ROWS = ("mismatch", "cubic")


def jobs(args):
    out = []
    for case in ROWS:
        for s in args.seeds.split(","):
            stem = os.path.join(args.runs, f"ctl_{case}_s{s}")
            out.append((stem + ".json", ["exp67_controls.py", "fins", "--case", case, "--seed", s, "--threads", "1",
                                         "--save", stem]))
    return out


def summary(args):
    res, lines = {}, ["# exp60's controls (exp67): FINS through the exact solver\n",
                      "| case | n | rotation scan: eps, best quotient | scaling scan: a, best quotient | "
                      "largest kept quotient |", "|---|---|---|---|---|"]
    keep = {"mismatch": ("d_x", "d_y", "d_t", "scaling"), "cubic": ("d_x", "d_y", "d_t", "stretched rotation")}
    for case in ROWS:
        cell = {}
        for s in args.seeds.split(","):
            p = os.path.join(args.runs, f"ctl_{case}_s{s}.json")
            if os.path.exists(p):
                j = json.load(open(p))
                cell[s] = {k: j[k] for k in ("quotients", "scan", "sl2", "own", "lam")}
        if not cell:
            continue
        res[case] = cell
        kq = [max(c["quotients"][n] for n in keep[case]) for c in cell.values()]
        lines.append(f"| {case} | {len(cell)} | {min(c['scan']['eps'] for c in cell.values()):.3f} to "
                     f"{max(c['scan']['eps'] for c in cell.values()):.3f}, {min(c['scan']['q_eps'] for c in cell.values()):.1e} "
                     f"to {max(c['scan']['q_eps'] for c in cell.values()):.1e} | {min(c['scan']['a'] for c in cell.values()):.3f} "
                     f"to {max(c['scan']['a'] for c in cell.values()):.3f}, {min(c['scan']['q'] for c in cell.values()):.1e} to "
                     f"{max(c['scan']['q'] for c in cell.values()):.1e} | {min(kq):.1e} to {max(kq):.1e} |")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "aniso2d")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "controls_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, "controls_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["fins", "response", "run", "summary"])
    ap.add_argument("--case", default="mismatch", choices=list(CASES))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--runs", default=os.path.join(RESULTS, "controls_runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", type=float, default=1.5)
    ap.add_argument("--eps-d", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--score", type=float, default=4.5)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsub", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.what == "run":
        e51.run(args, jobs(args))
    else:
        {"fins": fins, "response": response, "summary": summary}[args.what](args)


if __name__ == "__main__":
    main()
