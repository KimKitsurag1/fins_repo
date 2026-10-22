"""Experiment 9 — does the trained surrogate's Jacobian respect the
equation's domain of dependence?

This is the linchpin for the seam-windowed test of exp8. Windowing removes
the seam from the RESIDUAL, but not from the INPUT: dS_theta[u0](Q0) inside
the window still reads Q0 at the seam. For the exact heat operator the
contamination is bounded by the Green kernel, exp(-d^2 / 4 alpha t), which
is what makes exp8 work. An FNO has no domain of dependence by
construction — one spectral layer mixes the whole domain — so the question
is empirical: does training induce the correct locality, or does the
learned Jacobian stay global?

Measurement
-----------
J(x', t) = d S_theta[u0](x_c, t) / d u0(x') by autograd (one row of the
Fréchet derivative), profiled against distance d = |x' - x_c| on the torus,
at several output times. Compared against:
  - the exact operator's own row, which for heat is the Gaussian kernel
    exp(-d^2 / 4 alpha t) / sqrt(4 pi alpha t) (periodised);
  - the diffusion length sqrt(4 alpha t).
Reported per time slice: the exact and learned kernel widths (radius
holding 95 percent of |J| mass), and the fraction of learned |J| mass
sitting beyond the exact kernel's 4-sigma radius — the leakage that a
seam-windowed test would let through.

Smoke defaults train a small FNO on CPU; --full uses the paper config.

Run:  python exp9_surrogate_locality.py [--full] [--epochs N]
"""

import argparse
import csv
import os

import numpy as np
import torch

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
from method_fins import train_solution_operator


def jacobian_row(S, u0, xc_idx, xs, ts, device, t_ids):
    """d S[u0](x_c, t) / d u0(x') for each t in t_ids. Returns [n_t, Nx]."""
    u0 = u0.clone().to(device).requires_grad_(True)
    out = S(u0, xs.to(device), ts.to(device))          # [1, 1, Nx, Nt]
    rows = []
    for ti in t_ids:
        g = torch.autograd.grad(out[0, 0, xc_idx, ti], u0,
                                retain_graph=True)[0]
        rows.append(g[0, 0].detach().cpu().numpy())
    return np.stack(rows)


def circ_dist(Nx, xc_idx, dx):
    i = np.arange(Nx)
    d = np.minimum((i - xc_idx) % Nx, (xc_idx - i) % Nx)
    return d * dx


def mass_radius(profile, dist, frac=0.95):
    """Smallest radius holding `frac` of the absolute mass."""
    a = np.abs(profile)
    order = np.argsort(dist)
    cs = np.cumsum(a[order])
    if cs[-1] <= 0:
        return np.nan
    cs = cs / cs[-1]
    k = int(np.searchsorted(cs, frac))
    return float(dist[order][min(k, len(order) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if not args.full:
        cfg.grid_size, cfg.time_steps = 64, 24
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 12, 20, 2
        n_train, epochs = 96, 60
    else:
        n_train, epochs = 1024, cfg.epochs_surrogate
    if args.epochs:
        epochs = args.epochs

    alpha = cfg.alpha_heat
    torch.manual_seed(args.seed)
    ics, sols, xs, ts = generate_pde_data("heat", cfg, n_train + 32,
                                          seed=args.seed)
    S = train_solution_operator(ics[:n_train], sols[:n_train], xs[:n_train],
                                ts[:n_train], cfg, device, epochs=epochs,
                                verbose=False)

    # test error, for context
    with torch.no_grad():
        p = S(ics[n_train:].to(device), xs[n_train:].to(device),
              ts[n_train:].to(device))
        u = sols[n_train:].to(device)
        delta = float((torch.linalg.vector_norm(p - u, dim=(1, 2, 3))
                       / torch.linalg.vector_norm(u, dim=(1, 2, 3))).mean())

    # pick a test sample and read its physical grid
    b = n_train
    x_np = xs[b, 0].numpy()
    t_np = ts[b, 0].numpy()
    Nx, Nt = x_np.size, t_np.size
    L = x_np[-1] + (x_np[1] - x_np[0])
    dx = x_np[1] - x_np[0]
    xc = Nx // 2
    dist = circ_dist(Nx, xc, dx)

    t_ids = [max(1, Nt // 8), Nt // 4, Nt // 2, Nt - 1]
    J = jacobian_row(S, ics[b:b + 1], xc, xs[b:b + 1], ts[b:b + 1],
                     device, t_ids)

    print("=" * 80)
    print(f"EXP 9 - locality of dS_theta/du0  (heat, L = {L:.3f}, "
          f"Nx = {Nx}, alpha = {alpha}, full={args.full})")
    print(f"surrogate test rel-L2 delta = {delta:.4f}, "
          f"n_train = {n_train}, epochs = {epochs}, device = {device}")
    print("=" * 80)

    rows_csv, tab = [], []
    for j, ti in enumerate(t_ids):
        tt = float(t_np[ti])
        sigma = np.sqrt(2 * alpha * tt)                  # Gaussian sigma
        # exact periodised heat kernel row
        k = C.wavenumbers(Nx, L)
        ker = np.real(np.fft.ifft(np.exp(-alpha * k ** 2 * tt)
                                  * np.exp(-1j * k * x_np[xc])))
        ker = np.roll(ker, 0)
        ker_row = np.real(np.fft.ifft(np.fft.fft(
            np.eye(Nx)[xc]) * np.exp(-alpha * k ** 2 * tt)))
        r_exact = mass_radius(ker_row, dist)
        r_learn = mass_radius(J[j], dist)
        cut = 4 * sigma
        leak = float(np.abs(J[j])[dist > cut].sum()
                     / (np.abs(J[j]).sum() + 1e-30))
        leak_ex = float(np.abs(ker_row)[dist > cut].sum()
                        / (np.abs(ker_row).sum() + 1e-30))
        tab.append([f"{tt:.3f}", f"{sigma:.4f}", f"{r_exact:.3f}",
                    f"{r_learn:.3f}", f"{r_learn / (r_exact + 1e-12):.1f}x",
                    f"{leak_ex:.1e}", f"{leak:.3f}"])
        rows_csv.append([f"{tt:.6f}", f"{sigma:.6f}", f"{r_exact:.6f}",
                         f"{r_learn:.6f}", f"{leak_ex:.6e}", f"{leak:.6e}"])

    print(C.fmt_table(tab, ["t", "sigma", "r95 exact", "r95 learned",
                            "ratio", "leak exact", "leak learned"]))

    # ASCII profiles at the final time
    print(f"\nnormalised |J| along x at t = {t_np[t_ids[-1]]:.3f} "
          f"(centre = column {xc}):")
    for name, prof in [("exact ", ker_row), ("learned", J[-1])]:
        a = np.abs(prof) / (np.abs(prof).max() + 1e-30)
        marks = "".join("#" if v >= 0.1 else ("+" if v >= 0.01 else
                        ("-" if v >= 1e-3 else ".")) for v in a)
        print(f"  {name}: {marks}")
    print("  legend: # >=0.1   + >=0.01   - >=1e-3   . below")

    print("\nRead-out: leak learned is the fraction of the surrogate "
          "Jacobian mass\nsitting outside the exact 4-sigma cone. The "
          "seam-windowed test of exp8\nis only as valid as this number is "
          "small.")

    path = os.path.join(C.RESULTS_DIR, "exp9_surrogate_locality.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "sigma", "r95_exact", "r95_learned",
                    "leak_exact", "leak_learned"])
        w.writerows(rows_csv)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
