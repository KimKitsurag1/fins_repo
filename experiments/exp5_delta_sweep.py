"""Experiment 5 — surrogate quality sweep (stability-lemma validation).

Trains the repository surrogate S_theta at several (n_train, epochs)
budgets, records the test relative L2 error delta, and evaluates the
per-generator equivariance residual with the PER-SAMPLE sanity check from
exp_common (the repository version broadcasts sample-0 ground-truth fields
across the batch; both variants are reported here so the size of that bias
is visible on a real surrogate, not only on the oracle).

The stability lemma to be validated on the full run: residual(v) <=
C(v) * delta^2 for exact symmetries v, i.e. log residual vs log delta has
slope 2 at fixed architecture. The smoke run (--smoke) uses a tiny FNO and
two budgets to validate mechanics on CPU; the paper-scale run needs a GPU.

Run:  python exp5_delta_sweep.py --smoke
      python exp5_delta_sweep.py --budgets 64:100 256:200 1024:500
"""

import argparse
import time

import numpy as np
import torch

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
from method_fins import train_solution_operator, sanity_check_equivariance
import csv, os


def rel_l2(S, ic, sol, xc, tc, device, bs=16):
    errs = []
    with torch.no_grad():
        for i in range(0, ic.shape[0], bs):
            p = S(ic[i:i + bs].to(device), xc[i:i + bs].to(device),
                  tc[i:i + bs].to(device))
            u = sol[i:i + bs].to(device)
            errs += (torch.linalg.vector_norm(p - u, dim=(1, 2, 3))
                     / torch.linalg.vector_norm(u, dim=(1, 2, 3))).tolist()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budgets", nargs="*", default=None,
                    help="list of n_train:epochs")
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent (re-seeded) trainings per budget; "
                         "median reported")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.smoke:
        cfg.grid_size, cfg.time_steps = 32, 32
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
        budgets = [(32, 25), (96, 50)]
    else:
        budgets = [tuple(map(int, b.split(":"))) for b in args.budgets]

    n_max = max(b[0] for b in budgets)
    n_test = 32 if args.smoke else 128
    torch.manual_seed(args.seed)
    ics, sols, xs, ts = generate_pde_data(args.pde, cfg,
                                          n_max + n_test, seed=args.seed)
    te = slice(n_max, n_max + n_test)

    print("=" * 78)
    print(f"EXP 5 - delta sweep ({args.pde}, device={device}, "
          f"smoke={args.smoke})")
    print("=" * 78)
    path = os.path.join(C.RESULTS_DIR, "exp5_delta_sweep.csv")

    def dump(rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_train", "epochs", "repeat", "delta", "generator",
                        "resid_fixed"])
            w.writerows(rows)

    rows = []
    for n_train, epochs in budgets:
        deltas, fixed_runs = [], []
        for rep in range(args.repeats):
            # fresh, reproducible RNG state per training: without this the
            # budgets share one evolving stream and the delta curve is
            # non-monotone noise (last-epoch model, no early stopping)
            torch.manual_seed(args.seed + rep)
            t0 = time.time()
            S = train_solution_operator(ics[:n_train], sols[:n_train],
                                        xs[:n_train], ts[:n_train],
                                        cfg, device, epochs=epochs,
                                        verbose=False)
            delta = rel_l2(S, ics[te], sols[te], xs[te], ts[te], device)
            for mode in ("jvp", "finite_tau"):
                try:
                    fixed = C.sanity_check_equivariance_fixed(
                        S, args.pde, sols[te], xs[te], ts[te], cfg, device,
                        alpha=cfg.alpha_heat, loss_mode=mode, n_samples=4,
                        verbose=False)
                    break
                except Exception as e:                    # jvp unsupported
                    print(f"  [mode={mode} failed: "
                          f"{type(e).__name__}: {e}]")
                    fixed = None
            if fixed is None:
                continue
            deltas.append(delta)
            fixed_runs.append(fixed)
            print(f"\nbudget n_train={n_train}, epochs={epochs}, "
                  f"rep={rep}: test rel-L2 delta = {delta:.4f}  "
                  f"[{time.time()-t0:.0f} s]  (mode={mode})")
            for name, vf in fixed:
                print(f"    {name:>12}: resid (per-sample GT) = {vf:.4e}")
                rows.append([n_train, epochs, rep, f"{delta:.6f}", name,
                             f"{vf:.6e}"])
            if rep == 0:
                orig = sanity_check_equivariance(
                    S, args.pde, sols[te], xs[te], ts[te], cfg, device,
                    alpha=cfg.alpha_heat, loss_mode=mode, n_samples=4)
                dev = max(abs(vf - vo) / (abs(vo) + 1e-30)
                          for (_, vf), (_, _, vo) in zip(fixed, orig))
                if dev < 1e-6:
                    print("    [repo sanity identical to per-sample]")
                else:
                    print("    repo (sample-0 broadcast) for comparison:")
                    for (name, vf), (_, _, vo) in zip(fixed, orig):
                        print(f"      {name:>12}: repo = {vo:.4e}   "
                              f"ratio fixed/repo = "
                              f"{vf / (vo + 1e-30):7.2f}")
        dump(rows)                     # crash-safe: rewrite after each budget
        if len(deltas) > 1:
            med = float(np.median(deltas))
            print(f"  median delta over {len(deltas)} repeats: {med:.4f}")

    dump(rows)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
