"""Experiment 6 — factorising out the trivial (superposition) subalgebra.

For linear PDEs, v_beta = beta(x,t) d_u with beta any solution satisfies the
FINS equivariance condition exactly, so an unconstrained generator network
can fill discovery slots with superposition directions. The proposed fix is
a structural ansatz for the phi-channel:

    phi(x, t, u) = a(x, t) * u + b(x, t),

with a penalty on the energy share of the u-independent part b.  All six
finite heat generators have phi in {0, u, -x u, -(x^2+2 alpha t) u} — i.e.
they live in the a-part — while superposition directions live purely in b.
The penalty therefore removes the degenerate directions without touching
the target algebra.

This script runs the sequential deflation of `method_fins` with the
constrained generator, reusing `equivariance_loss` and `deflate_generator`
unchanged, for lambda_b in {0, 30}, and reports per-slot deflated norms,
the b-share of phi, and the spectral rank of the norm sequence.

Smoke on CPU validates mechanics (tiny FNO, few epochs); the paper-scale
run needs the full config on GPU.

Run:  python exp6_trivial_subalgebra.py --smoke
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
import exp10_leak_vs_delta as E10
from ground_truth import Metrics
from method_fins import train_solution_operator, equivariance_loss
from method_fins_components import deflate_generator, create_grids
from shared_modules import make_fno


class ConstrainedGeneratorFNO(nn.Module):
    """FNO generator with phi = a(x,t) * u + b(x,t)."""

    def __init__(self, cfg):
        super().__init__()
        self.fno = make_fno(3, 4, cfg)

    def forward(self, x_grid, t_grid, u):
        out = self.fno(torch.cat([x_grid, t_grid, u], dim=1))
        xi, eta, a, b = (out[:, i:i + 1] for i in range(4))
        return xi, eta, a * u + b, b


def train_constrained_deflation(train_sol, train_x, train_t, S_theta, cfg,
                                device, n_sym, epochs_per_gen, lr,
                                lambda_b, lambda_norm=1.0, loss_mode="jvp",
                                slot_dir=None, seed=42, n_starts=3,
                                screen_epochs=4, n_eval=32):
    loader = DataLoader(TensorDataset(train_sol, train_x, train_t),
                        batch_size=cfg.batch_size, shuffle=True)
    dx = (train_x[0, 0, 1] - train_x[0, 0, 0]).item()
    dt = (train_t[0, 0, 1] - train_t[0, 0, 0]).item()
    # The anchor holds the BATCH-AVERAGE norm at 1, so reading the norm off
    # a single sample is inconsistent with what is optimised — and on the
    # repo's multi-scale heat data (L in 0.5..2.0) sample 0 is the SMALLEST
    # domain. Average over n_eval samples instead; use --single-scale to
    # remove the geometry mismatch entirely.
    n_ev = min(n_eval, train_sol.shape[0])
    u_ref = train_sol[:n_ev].to(device)
    xg_ref, tg_ref = create_grids(train_x[:n_ev].to(device),
                                  train_t[:n_ev].to(device), device)

    frozen, norms, b_shares, fields = [], [], [], []
    start_k = 0
    if slot_dir is not None:
        os.makedirs(slot_dir, exist_ok=True)
        # resume: reload every completed slot as a frozen basis element
        for k in range(n_sym):
            p = os.path.join(slot_dir, f"slot_{k}.npz")
            if not os.path.exists(p):
                break
            z = np.load(p)
            f = z["field"]
            frozen.append(tuple(
                torch.tensor(f[c], dtype=torch.float32,
                             device=device)[None, None] for c in range(3)))
            norms.append(float(z["norm"]))
            b_shares.append(float(z["b_share"]))
            fields.append(f)
            print(f"    slot {k + 1}: resumed from {p} "
                  f"(norm = {norms[-1]:.3e}, b-share = {b_shares[-1]:.3f})")
            start_k = k + 1
    def train_epochs(gen, opt, n_ep):
        last = 0.0
        for _ in range(n_ep):
            tot, nb = 0.0, 0
            for u_b, x_b, t_b in loader:
                u = u_b.to(device)
                xg, tg = create_grids(x_b.to(device), t_b.to(device), device)
                xi, eta, phi, b = gen(xg, tg, u)
                xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
                l_eq = equivariance_loss(
                    S_theta, xi_d, eta_d, phi_d,
                    u[:, :, :, 0].contiguous(), u,
                    x_b.to(device), t_b.to(device), dx, dt, device,
                    mode=loss_mode)
                v_d = torch.cat([xi_d, eta_d, phi_d], dim=1)
                l_norm = (v_d.pow(2).mean().sqrt() + 1e-8 - 1.0).pow(2)
                share_b = b.pow(2).mean() / (phi.pow(2).mean() + 1e-12)
                loss = l_eq + lambda_norm * l_norm + lambda_b * share_b
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            last = tot / max(nb, 1)
        return last

    for k in range(start_k, n_sym):
        t_slot = time.time()
        # multi-init screening: short-training several inits and keeping the best fixes
        # the initialisation lottery while staying resume-reproducible
        # through slot- and start-indexed seeds
        best = None
        for s_i in range(max(n_starts, 1)):
            torch.manual_seed(seed * 1000 + k * 10 + s_i)
            cand = ConstrainedGeneratorFNO(cfg).to(device)
            copt = optim.AdamW(cand.parameters(), lr=lr, weight_decay=1e-4)
            l_scr = train_epochs(cand, copt,
                                 min(screen_epochs, epochs_per_gen))
            if best is None or l_scr < best[0]:
                best = (l_scr, cand, copt)
        _, gen, opt = best
        if n_starts > 1:
            print(f"      slot {k + 1}: kept best of {n_starts} starts "
                  f"(screen loss {best[0]:.3e})")
        train_epochs(gen, opt,
                     epochs_per_gen - min(screen_epochs, epochs_per_gen))

        with torch.no_grad():
            xi, eta, phi, b = gen(xg_ref, tg_ref, u_ref)
            xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
            nrm = float(torch.cat([xi_d, eta_d, phi_d], 1).pow(2)
                        .mean().sqrt())
            nrm_s0 = float(torch.cat([xi_d[:1], eta_d[:1], phi_d[:1]], 1)
                           .pow(2).mean().sqrt())
            share = float(b.pow(2).mean() / (phi.pow(2).mean() + 1e-12))
        frozen.append((xi_d[:1].detach(), eta_d[:1].detach(),
                       phi_d[:1].detach()))
        norms.append(nrm)
        b_shares.append(share)
        fields.append(np.stack([f[0, 0].cpu().numpy()
                                for f in (xi_d, eta_d, phi_d)]))
        norms[-1] = nrm
        if slot_dir is not None:
            np.savez(os.path.join(slot_dir, f"slot_{k}.npz"),
                     field=fields[-1], norm=nrm, b_share=share)
        dt_s = time.time() - t_slot
        print(f"    slot {k + 1}: deflated norm = {nrm:.3e} "
              f"(sample-0 read-out {nrm_s0:.3e}), "
              f"b-share of phi = {share:.3f}  [{dt_s:.0f} s; "
              f"~{dt_s * (n_sym - k - 1) / 60:.0f} min left in this run]")
    return norms, b_shares, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--single-scale", action="store_true",
                    help="train on one domain (L = 2 pi) instead of the "
                         "repo multi-scale heat mixture; required for the "
                         "deflation basis and the norm anchor to be "
                         "comparable across samples")
    ap.add_argument("--n-eval", type=int, default=32,
                    help="samples the deflated norm and b-share are "
                         "averaged over (the anchor is a batch average, "
                         "so a single-sample read-out is inconsistent)")
    ap.add_argument("--n-starts", type=int, default=3,
                    help="random restarts screened per slot, against "
                         "optimisation-limited deflation (0 or 1 disables)")
    ap.add_argument("--screen-epochs", type=int, default=4,
                    help="epochs each restart is trained before selection")
    ap.add_argument("--outdir", default="results",
                    help="artefact directory; point to a mounted Drive "
                         "path so checkpoints and slots survive a VM restart")
    ap.add_argument("--lam", type=float, default=None,
                    help="run a single lambda_b (default: both 0 and 30); "
                         "use to split the run across sessions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.smoke:
        cfg.grid_size, cfg.time_steps = 32, 32
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
        n_train, ep_sur, n_sym, ep_gen = 32, 25, 3, 8
    else:
        # ep_gen = 50 matches the repository's own epochs_symmetry budget
        n_train, ep_sur, n_sym, ep_gen = 1024, 500, 6, 50

    torch.manual_seed(args.seed)
    if args.single_scale:
        rng = np.random.default_rng(args.seed)
        ics, sols, xs, ts, _, _ = E10.make_dataset(
            n_train, cfg.grid_size, cfg.time_steps, rng)
        print(f"  data: single scale L = {E10.L:.3f} "
              f"(exp8/exp10/exp11 setting)")
    else:
        ics, sols, xs, ts = generate_pde_data("heat", cfg, n_train,
                                              seed=args.seed)
        print("  data: repo multi-scale mixture (L in 0.5..2.0)")
    print("=" * 78)
    print(f"EXP 6 - constrained deflation phi = a*u + b "
          f"(device={device}, smoke={args.smoke})")
    print("=" * 78)
    ck = os.path.join(args.outdir,
                      f"exp6_surrogate_s{args.seed}"
                      f"{'_ss' if args.single_scale else ''}"
                      f"{'_smoke' if args.smoke else ''}.pt")
    os.makedirs(args.outdir, exist_ok=True)
    from shared_modules_v2 import FNOSurrogateEquivariant
    if os.path.exists(ck):
        S = FNOSurrogateEquivariant(cfg).to(device)
        state = torch.load(ck, map_location=device, weights_only=False)
        # torch.save preserves the OrderedDict's _metadata, which comes back
        # as a KEY on reload and trips load_state_dict's strict check
        state.pop("_metadata", None)
        S.load_state_dict(state)
        S.eval()
        print(f"  surrogate checkpoint reused: {ck}")
    else:
        t0 = time.time()
        S = train_solution_operator(ics, sols, xs, ts, cfg, device,
                                    epochs=ep_sur, verbose=False)
        sd = S.state_dict()
        sd.pop("_metadata", None)
        torch.save(sd, ck)
        print(f"  surrogate trained in {time.time() - t0:.0f} s, "
              f"checkpoint -> {ck}")

    lams = (args.lam,) if args.lam is not None else (0.0, 30.0)
    for lam in lams:
        print(f"\n  lambda_b = {lam:g}")
        t0 = time.time()
        for mode in ("jvp", "finite_tau"):
            try:
                tag0 = (f"lam{lam:g}_s{args.seed}"
                        f"{'_ss' if args.single_scale else ''}"
                        f"{'_smoke' if args.smoke else ''}")
                norms, shares, fields = train_constrained_deflation(
                    sols, xs, ts, S, cfg, device, n_sym, ep_gen,
                    lr=cfg.lr_symmetry, lambda_b=lam, loss_mode=mode,
                    slot_dir=os.path.join(args.outdir,
                                          f"exp6_slots_{tag0}"),
                    seed=args.seed, n_starts=args.n_starts,
                    screen_epochs=args.screen_epochs, n_eval=args.n_eval)
                break
            except Exception as e:
                print(f"    [mode={mode} failed: {type(e).__name__}: {e}]")
        r = Metrics.spectral_rank(np.array(norms))
        print(f"    spectral rank of deflated norms: {r}   "
              f"[{time.time()-t0:.0f} s, mode={mode}]")
        tag = (f"lam{lam:g}_s{args.seed}"
               f"{'_ss' if args.single_scale else ''}"
               f"{'_smoke' if args.smoke else ''}")
        np.savez(os.path.join(args.outdir, f"exp6_generators_{tag}.npz"),
                 fields=np.stack(fields), norms=np.array(norms),
                 b_shares=np.array(shares), lam=lam, seed=args.seed)
        with open(os.path.join(args.outdir, f"exp6_{tag}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["lam", "slot", "deflated_norm", "b_share",
                        "spectral_rank"])
            for k, (n_, s_) in enumerate(zip(norms, shares)):
                w.writerow([lam, k + 1, f"{n_:.6e}", f"{s_:.6f}", r])
        print(f"    artefacts -> {args.outdir}/exp6_generators_{tag}.npz, "
              f"{args.outdir}/exp6_{tag}.csv")


if __name__ == "__main__":
    main()
