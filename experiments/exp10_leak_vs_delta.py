"""Experiment 10 — does the seam-windowed test survive on a surrogate?

Crosses exp5 (surrogate-quality sweep) with exp8 (seam-windowed residual)
and exp9 (Jacobian locality): one loop over training budgets, three numbers
per budget measured on the SAME S_theta.

  delta   test relative L2 error of the surrogate
  leak    fraction of |dS_theta/du0| mass outside the exact 4-sigma cone
          of the heat kernel at an early output time (exp9)
  R(m)    equivariance residual of the x-weighted heat generators with the
          residual restricted to x in [m, L - m] (exp8)

The question this answers: the oracle reaches R(0.05 L) ~ 1e-7 on the
conference version's grid, but the window removes the seam only from the residual,
not from the input. If R(m) on the surrogate falls with delta, the windowed
test is usable and the rank-6 regime is reachable in the pipeline. If R(m)
plateaus while delta keeps falling, the limit is architectural: the FNO's
Jacobian has no domain of dependence, and the surrogate needs an explicit
locality constraint before the windowed test means anything.

Data is single-domain (L = 2 pi, the exp8 setting) rather than the
repository's multi-scale heat mixture, because a local test needs the
diffusion length to be small against the domain; at L = 0.5 it is 40
percent of it.

Run:  python exp10_leak_vs_delta.py --smoke
      python exp10_leak_vs_delta.py --budgets 128:150 512:400 1024:800
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.func import jvp

import exp_common as C
from config import ExperimentConfig
from method_fins import train_solution_operator

L, ALPHA, T = 2 * np.pi, 0.01, 1.0
XW = ["Galilean", "scaling", "projective"]


def make_dataset(n, Nx, Nt, rng):
    x = np.linspace(0, L, Nx, endpoint=False)
    t = np.linspace(0, T, Nt)
    u0 = np.concatenate([C.ic_heat_gauss(n // 2, x, L, rng),
                         C.ic_heat_sine(n - n // 2, x, L, rng)])
    u = C.heat_solve(u0, ALPHA, L, t)
    to = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to(u0).view(n, 1, Nx), to(u).view(n, 1, Nx, Nt),
            to(np.tile(x, (n, 1))).view(n, 1, Nx),
            to(np.tile(t, (n, 1))).view(n, 1, Nt), x, t)


def rel_l2(S, ic, sol, xs, ts, device, bs=16):
    e = []
    with torch.no_grad():
        for i in range(0, ic.shape[0], bs):
            p = S(ic[i:i + bs].to(device), xs[i:i + bs].to(device),
                  ts[i:i + bs].to(device))
            u = sol[i:i + bs].to(device)
            e += (torch.linalg.vector_norm(p - u, dim=(1, 2, 3))
                  / torch.linalg.vector_norm(u, dim=(1, 2, 3))).tolist()
    return float(np.mean(e))


def leak_at(S, ic, xs, ts, x_np, t_np, device, t_frac=0.125):
    """exp9 read-out on one sample at an early time."""
    Nx, Nt = x_np.size, t_np.size
    xc, ti = Nx // 2, max(1, int(t_frac * Nt))
    u0 = ic[:1].clone().to(device).requires_grad_(True)
    out = S(u0, xs[:1].to(device), ts[:1].to(device))
    g = torch.autograd.grad(out[0, 0, xc, ti], u0)[0][0, 0].cpu().numpy()
    i = np.arange(Nx)
    d = np.minimum((i - xc) % Nx, (xc - i) % Nx) * (x_np[1] - x_np[0])
    sigma = np.sqrt(2 * ALPHA * float(t_np[ti]))
    return float(np.abs(g)[d > 4 * sigma].sum() / (np.abs(g).sum() + 1e-30))


def windowed_resid(S, ic, sol, xs, ts, x_np, t_np, device, margins):
    """R(m) for the x-weighted heat generators, JVP through S_theta."""
    u = sol.numpy()[:, 0]
    gens = C.gt_characteristics("heat", x_np, t_np, u, alpha=ALPHA)
    xd, td = xs.to(device), ts.to(device)
    f = lambda z: S(z, xd, td)
    out = {}
    for n in XW:
        xi, eta, phi, Q = gens[n]
        q0 = torch.tensor(Q[:, :, 0], dtype=torch.float32,
                          device=device).view(*ic.shape)
        _, dSQ = jvp(f, (ic.to(device),), (q0,))
        dSQ = dSQ[:, 0].detach().cpu().numpy()
        out[n] = [C_windowed(dSQ, Q, xi, eta, phi, x_np, m) for m in margins]
    return out


def C_windowed(dSQ, Q, xi, eta, phi, x, m, t_margin=2):
    Nt = Q.shape[-1]
    ts = slice(t_margin, Nt - t_margin)
    xsel = (x >= m) & (x <= L - m)
    r = dSQ[:, xsel][..., ts] - Q[:, xsel][..., ts]
    v = (xi ** 2 + eta ** 2 + phi ** 2)[:, xsel][..., ts]
    return float(((r ** 2).mean(axis=(1, 2))
                  / (v.mean(axis=(1, 2)) + 1e-6)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budgets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.smoke:
        cfg.grid_size, cfg.time_steps = 64, 24
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 12, 20, 2
        budgets = [(48, 30), (96, 60)]
    else:
        cfg.grid_size, cfg.time_steps = 64, 128
        budgets = [tuple(map(int, b.split(":")))
                   for b in (args.budgets or ["128:150", "512:400",
                                              "1024:800"])]

    Nx, Nt = cfg.grid_size, cfg.time_steps
    rng = np.random.default_rng(args.seed)
    n_max = max(b[0] for b in budgets)
    n_test = 16 if args.smoke else 64
    ic, sol, xs, ts, x_np, t_np = make_dataset(n_max + n_test, Nx, Nt, rng)
    te = slice(n_max, n_max + n_test)
    margins = [0.0, 0.05 * L, 0.10 * L]

    print("=" * 86)
    print(f"EXP 10 - windowed residual and Jacobian leak vs surrogate "
          f"quality (heat, L = {L:.3f})")
    print(f"device = {device}, Nx = {Nx}, Nt = {Nt}, smoke = {args.smoke}")
    print("Oracle reference (exp8, exact operator): "
          "R(0.05L) = 1.7e-07 / 7.0e-05 / 2.7e-07")
    print("=" * 86)

    path = os.path.join(C.RESULTS_DIR, "exp10_leak_vs_delta.csv")

    def dump(rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_train", "epochs", "delta", "leak", "m_over_L",
                        "generator", "windowed_residual"])
            w.writerows(rows)

    rows, tab = [], []
    for n_train, epochs in budgets:
        t0 = time.time()
        torch.manual_seed(args.seed)
        S = train_solution_operator(ic[:n_train], sol[:n_train],
                                    xs[:n_train], ts[:n_train], cfg,
                                    device, epochs=epochs, verbose=False)
        delta = rel_l2(S, ic[te], sol[te], xs[te], ts[te], device)
        leak = leak_at(S, ic[te], xs[te], ts[te], x_np, t_np, device)
        nb = min(8, n_test)
        R = windowed_resid(S, ic[te][:nb], sol[te][:nb], xs[te][:nb],
                           ts[te][:nb], x_np, t_np, device, margins)
        print(f"\n  n_train = {n_train}, epochs = {epochs}: "
              f"delta = {delta:.4f}, leak = {leak:.3f}  "
              f"[{time.time() - t0:.0f} s]")
        sub = [[f"{m / L:.2f}"] + [f"{R[n][j]:.2e}" for n in XW]
               for j, m in enumerate(margins)]
        print("    " + C.fmt_table(sub, ["m/L"] + XW).replace("\n", "\n    "))
        for j, m in enumerate(margins):
            for n in XW:
                rows.append([n_train, epochs, f"{delta:.6f}",
                             f"{leak:.6f}", f"{m / L:.3f}", n,
                             f"{R[n][j]:.6e}"])
        tab.append([str(n_train), f"{delta:.4f}", f"{leak:.3f}"]
                   + [f"{R[n][1]:.2e}" for n in XW])
        dump(rows)                     # crash-safe: rewrite after each budget

    print("\nsummary at m = 0.05 L:")
    print(C.fmt_table(tab, ["n_train", "delta", "leak"] + XW))
    print("\nFalling R with falling delta -> the windowed test transfers to "
          "the pipeline.\nR plateauing while delta falls -> the limit is "
          "the surrogate's missing\ndomain of dependence, and S_theta needs "
          "an explicit locality constraint.")

    dump(rows)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
