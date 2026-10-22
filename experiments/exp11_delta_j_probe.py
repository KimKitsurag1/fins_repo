"""Experiment 11 — delta_J: the variable the stability lemma actually needs.

exp5/exp10 plot residuals against the surrogate's TEST L2 error delta. But
the FINS residual is built from the Fréchet derivative dS_theta, and the L2
error of a network does not bound the error of its Jacobian. The relevant
lemma is R(v) <= C * delta_J^2 with

    delta_J = || dS_theta[u0](q) - dS[u0](q) || / || dS[u0](q) ||

measured over random smooth probes q. For heat the exact reference is one
call to the exact operator (linearity: dS[u0](q) = S(q), independent of u0).

Per budget this script reports, on the SAME surrogate:
    delta      test relative L2 error
    delta_J    Jacobian error over probes (mean +/- std)
    k          amplification delta_J / delta
    leak       Jacobian mass outside the exact 4-sigma cone, AVERAGED over
               samples and early times (the single-sample read-out of
               exp9/exp10 was too noisy to trend)
    R(0.05 L)  seam-windowed residual of the x-weighted generators
    R/delta^2 and R/delta_J^2 — whichever is more stable across budgets is
               the right normaliser for the lemma.

Run:  python exp11_delta_j_probe.py --budgets 96:30 96:60 [--tag a]
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.func import jvp

import exp_common as C
import exp10_leak_vs_delta as E10
from config import ExperimentConfig
from method_fins import train_solution_operator

L, ALPHA = E10.L, E10.ALPHA
XW = E10.XW


def delta_j_probe(S, ic, xs, ts, x_np, t_np, device, n_probe=8,
                  n_samples=4, seed=0, t_margin=2, window_m=0.05):
    """Jacobian error over random band-limited probes, exact heat reference.

    Returns global delta_J and a windowed variant restricted to
    x in [m, L - m]: the seam-windowed residual only sees the error inside
    the window, so the windowed delta_J is the tighter normaliser for the
    lemma applied to windowed tests.
    """
    rng = np.random.default_rng(seed)
    Nx, Nt = x_np.size, t_np.size
    q_np = C.superposition_directions(n_probe, x_np, t_np, L, ALPHA,
                                      rng)[:, :, 0]         # [P, Nx] slices
    sl = slice(t_margin, Nt - t_margin)
    xw = (x_np >= window_m * L) & (x_np <= (1 - window_m) * L)
    u0 = ic[:n_samples].to(device)
    xd, td = (xs[:n_samples].to(device), ts[:n_samples].to(device))
    f = lambda z: S(z, xd, td)
    errs, errs_w = [], []
    for p in range(n_probe):
        q = torch.tensor(q_np[p], dtype=torch.float32, device=device) \
                 .view(1, 1, Nx).expand(n_samples, 1, Nx).contiguous()
        _, dSq = jvp(f, (u0,), (q,))
        dSq = dSq[:, 0].detach().cpu().numpy()               # [B, Nx, Nt]
        ex = C.heat_solve(q_np[p:p + 1], ALPHA, L, t_np)[0]  # [Nx, Nt]
        d = dSq - ex[None]
        num = np.linalg.norm(d[..., sl], axis=(1, 2))
        den = np.linalg.norm(ex[..., sl]) + 1e-30
        errs += list(num / den)
        num_w = np.linalg.norm(d[:, xw][..., sl], axis=(1, 2))
        den_w = np.linalg.norm(ex[xw][..., sl]) + 1e-30
        errs_w += list(num_w / den_w)
    return (float(np.mean(errs)), float(np.std(errs)),
            float(np.mean(errs_w)), float(np.std(errs_w)))


def leak_avg(S, ic, xs, ts, x_np, t_np, device, n_samples=6):
    Nx, Nt = x_np.size, t_np.size
    xc = Nx // 2
    i = np.arange(Nx)
    d = np.minimum((i - xc) % Nx, (xc - i) % Nx) * (x_np[1] - x_np[0])
    t_ids = sorted({max(1, Nt // 8), Nt // 4, Nt // 2})
    vals = []
    for b in range(n_samples):
        u0 = ic[b:b + 1].clone().to(device).requires_grad_(True)
        out = S(u0, xs[b:b + 1].to(device), ts[b:b + 1].to(device))
        for ti in t_ids:
            g = torch.autograd.grad(out[0, 0, xc, ti], u0,
                                    retain_graph=True)[0][0, 0]
            g = g.detach().cpu().numpy()
            sig = np.sqrt(2 * ALPHA * float(t_np[ti]))
            vals.append(float(np.abs(g)[d > 4 * sig].sum()
                              / (np.abs(g).sum() + 1e-30)))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="*", default=["96:30", "96:60"])
    ap.add_argument("--full-grid", action="store_true",
                    help="paper grid 64x128 and default FNO config "
                         "(neuralop if installed) instead of the small "
                         "64x24 probe model")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    budgets = [tuple(map(int, b.split(":"))) for b in args.budgets]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.full_grid:
        cfg.grid_size, cfg.time_steps = 64, 128
    else:
        cfg.grid_size, cfg.time_steps = 64, 24
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 12, 20, 2

    Nx, Nt = cfg.grid_size, cfg.time_steps
    rng = np.random.default_rng(args.seed)
    n_max = max(b[0] for b in budgets)
    n_test = 16
    ic, sol, xs, ts, x_np, t_np = E10.make_dataset(n_max + n_test, Nx, Nt,
                                                   rng)
    te = slice(n_max, n_max + n_test)
    m = 0.05 * L

    print("=" * 96)
    print(f"EXP 11 - delta_J probe (heat, L = {L:.3f}, Nx = {Nx}, "
          f"device = {device})")
    print("=" * 96)
    path = os.path.join(
        C.RESULTS_DIR,
        f"exp11_delta_j{('_' + args.tag) if args.tag else ''}.csv")

    def dump(rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_train", "epochs", "delta", "delta_J",
                        "delta_J_std", "delta_J_win", "delta_J_win_std",
                        "leak", "leak_std"]
                       + [f"R_{n}" for n in XW])
            w.writerows(rows)

    rows, tab = [], []
    for n_train, epochs in budgets:
        torch.manual_seed(args.seed)
        S = train_solution_operator(ic[:n_train], sol[:n_train],
                                    xs[:n_train], ts[:n_train], cfg,
                                    device, epochs=epochs, verbose=False)
        delta = E10.rel_l2(S, ic[te], sol[te], xs[te], ts[te], device)
        dj, dj_s, djw, djw_s = delta_j_probe(S, ic[te], xs[te], ts[te],
                                             x_np, t_np, device)
        lk, lk_s = leak_avg(S, ic[te], xs[te], ts[te], x_np, t_np, device)
        R = E10.windowed_resid(S, ic[te][:8], sol[te][:8], xs[te][:8],
                               ts[te][:8], x_np, t_np, device, [m])
        Rg = {n: R[n][0] for n in XW}
        Rm = float(np.mean([Rg[n] for n in XW]))
        tab.append([f"{n_train}:{epochs}", f"{delta:.4f}",
                    f"{dj:.4f}", f"{djw:.4f}", f"{dj / delta:.1f}x",
                    f"{lk:.3f}"]
                   + [f"{Rg[n]:.2e}" for n in XW]
                   + [f"{Rm / delta ** 2:.2e}", f"{Rm / dj ** 2:.2e}",
                      f"{Rm / djw ** 2:.2e}"])
        rows.append([n_train, epochs, f"{delta:.6f}", f"{dj:.6f}",
                     f"{dj_s:.6f}", f"{djw:.6f}", f"{djw_s:.6f}",
                     f"{lk:.6f}", f"{lk_s:.6f}"]
                    + [f"{Rg[n]:.6e}" for n in XW])
        dump(rows)                     # crash-safe: rewrite after each budget
        print(f"  done {n_train}:{epochs}  delta={delta:.4f}  "
              f"delta_J={dj:.4f} (win {djw:.4f})  k={dj / delta:.1f}x  "
              f"leak={lk:.3f}")

    print()
    print(C.fmt_table(tab, ["budget", "delta", "dJ", "dJ_win", "k",
                            "leak", *XW, "R/d^2", "R/dJ^2", "R/dJw^2"]))
    print("\nRead-out: k is the Jacobian amplification factor; the more "
          "stable of the last\ntwo columns across budgets is the correct "
          "normaliser for the stability lemma.")

    dump(rows)
    print(f"CSV -> {path}")


if __name__ == "__main__":
    main()
