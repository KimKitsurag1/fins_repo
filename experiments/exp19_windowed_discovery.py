"""Experiment 19 — windowed FINS discovery: does the loop FIND the full algebra?

exp10 and exp18 established DETECTABILITY: the ground-truth generators pass
the seam-windowed equivariance test, through a trained surrogate, with three
orders of margin. What was never run is the discovery loop itself under the
same window. This script closes that gap.

Design
------
* Windowed equivariance loss. The residual and the generator norm are both
  restricted to x in [m, L - m]. Restricting the residual alone would let
  the network park arbitrary content at the seam for free.
* Shift averaging. A fixed window makes discovery depend on where the
  coordinate origin sits. Each step draws a random cyclic shift of the
  field and its coordinates, so the loss is an average over window
  placements and the origin dependence is integrated out. `--n-shifts 1`
  disables this for ablation.
* Deflation with the anchor on the RAW generator. exp15 showed that
  anchoring the deflated field pins ||v_perp|| = 1 at every slot and
  destroys the rank signal. Here the anchor holds ||v|| = 1 and the
  reported statistic is the deflated norm ||v_perp||, which is free to
  collapse when directions run out. The failure mode the repository
  comment warns about (the net copying a frozen generator, sending
  v_perp -> 0 with the loss also -> 0) is handled by the deflated-norm
  floor penalty: a slot is scored only while ||v_perp|| stays above
  `--min-perp`.
* Constrained ansatz phi = a(x,t) u + b(x,t) with the b-share penalty, per
  exp6: without it the superposition subalgebra fills the slots.

Reported per slot: deflated norm, windowed Rayleigh residual, b-share, and
the best-matching ground-truth generator with its cosine, so the run states
plainly WHICH generators were recovered rather than only how many.

Run:  python exp19_windowed_discovery.py --smoke
      python exp19_windowed_discovery.py --n-train 1024 --ep-sur 600 \
             --ep-gen 60 --n-sym 8
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import exp_common as C
import exp10_leak_vs_delta as E10
from config import ExperimentConfig
from exp6_trivial_subalgebra import ConstrainedGeneratorFNO
from exp15_rank_statistic import OracleHeatOperator
from ground_truth import GroundTruthAlgebra, Metrics, get_gt_names
from method_fins import train_solution_operator
from method_fins_components import (SingleGeneratorFNO, create_grids,
                                    deflate_generator, fd_t_high_order)
from shared_modules_v2 import FNOSurrogateEquivariant

L, ALPHA = E10.L, 0.01


def windowed_equivariance(S, xi, eta, phi, u, xb, tb, dx, m_frac,
                          t_margin=2):
    """Rayleigh residual of dS[u0](Q0) - Q, restricted in x to the window.

    Both numerator and denominator use the same window, so the quotient
    stays scale-invariant and the network gains nothing by hiding energy
    outside it.
    """
    B, _, Nx, Nt = u.shape
    ux = torch.fft.irfft(
        1j * (2 * np.pi / L) * torch.fft.rfftfreq(
            Nx, d=1.0 / Nx, device=u.device)[None, None, :, None]
        * torch.fft.rfft(u, dim=2), n=Nx, dim=2)
    ut = fd_t_high_order(u, (tb[0, 0, 1] - tb[0, 0, 0]).item())
    Q = phi - xi * ux - eta * ut
    f = lambda z: S(z, xb, tb)
    _, dSQ = torch.func.jvp(f, (u[:, :, :, 0].contiguous(),),
                            (Q[:, :, :, 0].contiguous(),))
    i0, i1 = int(m_frac * Nx), Nx - int(m_frac * Nx)
    sl = slice(t_margin, Nt - t_margin)
    r = (dSQ[:, :, i0:i1, sl] - Q[:, :, i0:i1, sl]).pow(2).mean()
    v = (xi[:, :, i0:i1, sl].pow(2) + eta[:, :, i0:i1, sl].pow(2)
         + phi[:, :, i0:i1, sl].pow(2)).mean()
    return r / (v + 1e-8), v.sqrt()


def random_shift(u, xb, batch_rng, device):
    """Cyclic shift of the field; x stays the fixed lab grid.

    Shifting the data rather than the window is equivalent by periodicity
    and keeps the coordinate channels untouched, which is what the
    x-weighted generators are defined against.
    """
    s = int(batch_rng.integers(0, u.shape[2]))
    return torch.roll(u, shifts=s, dims=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="exact operator instead of a surrogate "
                         "(delta_J = 0): isolates the discovery loop")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--n-shifts", type=int, default=4,
                    help="random window placements averaged per step; "
                         "1 disables shift averaging")
    ap.add_argument("--n-sym", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--ep-sur", type=int, default=600)
    ap.add_argument("--ep-gen", type=int, default=60)
    ap.add_argument("--lam-b", type=float, default=30.0)
    ap.add_argument("--min-perp", type=float, default=0.05)
    ap.add_argument("--net", default="constrained",
                    choices=["constrained", "single"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    if args.smoke:
        cfg.grid_size, cfg.time_steps = 32, 32
        cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
        args.n_train, args.ep_sur, args.ep_gen = 48, 40, 10
        args.n_sym = min(args.n_sym, 4)
    outdir = args.outdir or C.RESULTS_DIR
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ics, sols, xs, ts, x_np, t_np = E10.make_dataset(
        args.n_train, cfg.grid_size, cfg.time_steps, rng)

    print("=" * 84)
    print(f"EXP 19 - windowed FINS discovery (L = {L:.3f}, "
          f"m/L = {args.margin}, shifts = {args.n_shifts}, "
          f"net = {args.net}, device = {device})")
    print("=" * 84)

    ck = os.path.join(outdir, f"exp19_surrogate_s{args.seed}"
                              f"{'_smoke' if args.smoke else ''}.pt")
    if args.oracle:
        S = OracleHeatOperator(ALPHA, L).to(device)
        print("  operator: EXACT (delta_J = 0)")
    elif os.path.exists(ck):
        S = FNOSurrogateEquivariant(cfg).to(device)
        st = torch.load(ck, map_location=device, weights_only=False)
        st.pop("_metadata", None)
        S.load_state_dict(st)
        S.eval()
        print(f"  surrogate checkpoint reused: {ck}")
    else:
        t0 = time.time()
        S = train_solution_operator(ics, sols, xs, ts, cfg, device,
                                    epochs=args.ep_sur, verbose=False)
        sd = S.state_dict()
        sd.pop("_metadata", None)
        torch.save(sd, ck)
        print(f"  surrogate trained in {time.time() - t0:.0f} s -> {ck}")
    if not args.oracle:
        print(f"  surrogate delta = "
              f"{E10.rel_l2(S, ics, sols, xs, ts, device):.4f}")

    loader = DataLoader(TensorDataset(sols, xs, ts),
                        batch_size=cfg.batch_size, shuffle=True)
    dx = (xs[0, 0, 1] - xs[0, 0, 0]).item()

    # ground truth on a reference sample, for identification
    gt = GroundTruthAlgebra.heat_generators(x_np, t_np,
                                           sols[0, 0].numpy(), alpha=ALPHA)
    gt_names = get_gt_names("heat")
    n_ev = min(16, args.n_train)
    u_ev = sols[:n_ev].to(device)
    xg_ev, tg_ev = create_grids(xs[:n_ev].to(device), ts[:n_ev].to(device),
                                device)

    frozen, rows, disc = [], [], []
    brng = np.random.default_rng(args.seed)
    for k in range(args.n_sym):
        t0 = time.time()
        torch.manual_seed(args.seed * 1000 + k)
        gen = (ConstrainedGeneratorFNO(cfg) if args.net == "constrained"
               else SingleGeneratorFNO(cfg)).to(device)
        opt = optim.AdamW(gen.parameters(), lr=cfg.lr_symmetry,
                          weight_decay=1e-4)
        for _ in range(args.ep_gen):
            for u_b, x_b, t_b in loader:
                u0 = u_b.to(device)
                xb, tb = x_b.to(device), t_b.to(device)
                xg, tg = create_grids(xb, tb, device)
                loss = 0.0
                for _s in range(max(args.n_shifts, 1)):
                    u = (random_shift(u0, xb, brng, device)
                         if args.n_shifts > 1 else u0)
                    out = gen(xg, tg, u)
                    xi, eta, phi = out[0], out[1], out[2]
                    b = out[3] if args.net == "constrained" else None
                    xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi,
                                                           frozen)
                    l_eq, _ = windowed_equivariance(
                        S, xi_d, eta_d, phi_d, u, xb, tb, dx, args.margin)
                    # anchor on the RAW generator (exp15): leaves the
                    # deflated norm free to collapse when directions run out
                    v_raw = torch.cat([xi, eta, phi], dim=1)
                    l_norm = (v_raw.pow(2).mean().sqrt() - 1.0).pow(2)
                    v_perp = torch.cat([xi_d, eta_d, phi_d],
                                       dim=1).pow(2).mean().sqrt()
                    # keep the slot from collapsing onto a frozen direction
                    l_floor = torch.relu(args.min_perp - v_perp).pow(2)
                    l = l_eq + l_norm + 10.0 * l_floor
                    if b is not None:
                        l = l + args.lam_b * (b.pow(2).mean()
                                              / (phi.pow(2).mean() + 1e-12))
                    loss = loss + l / max(args.n_shifts, 1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
                opt.step()

        with torch.no_grad():
            out = gen(xg_ev, tg_ev, u_ev)
            xi, eta, phi = out[0], out[1], out[2]
            xi_d, eta_d, phi_d = deflate_generator(xi, eta, phi, frozen)
            nrm = float(torch.cat([xi_d, eta_d, phi_d], 1).pow(2)
                        .mean().sqrt())
            share = (float(out[3].pow(2).mean() / (phi.pow(2).mean() + 1e-12))
                     if args.net == "constrained" else float("nan"))
        res, _ = windowed_equivariance(S, xi_d, eta_d, phi_d, u_ev,
                                       xs[:n_ev].to(device),
                                       ts[:n_ev].to(device), dx, args.margin)
        fld = tuple(f[0, 0].detach().cpu().numpy()
                    for f in (xi_d, eta_d, phi_d))
        disc.append(fld)
        sim = Metrics.cosine_similarity_matrix([fld], gt)[0]
        j = int(np.argmax(sim))
        frozen.append((xi_d[:1].detach(), eta_d[:1].detach(),
                       phi_d[:1].detach()))
        rows.append([k + 1, nrm, float(res), share, gt_names[j],
                     float(sim[j])])
        print(f"  slot {k + 1}: |v_perp| = {nrm:.3f}  R_win = {float(res):.2e}"
              f"  b-share = {share:.3f}  -> {gt_names[j]} "
              f"(cos {sim[j]:.3f})  [{time.time() - t0:.0f} s]")

    print()
    print(C.fmt_table([[str(r[0]), f"{r[1]:.3f}", f"{r[2]:.2e}",
                        f"{r[3]:.3f}", r[4], f"{r[5]:.3f}"] for r in rows],
                      ["slot", "|v_perp|", "R_win", "b-share",
                       "best GT", "cos"]))
    found = {r[4] for r in rows if r[5] >= 0.7}
    print(f"\nGT generators recovered at cos >= 0.7: "
          f"{sorted(found) if found else 'none'}  ({len(found)}/6)")
    print(f"Hungarian mean |cos| over the discovered set: "
          f"{Metrics.best_cosine_similarity(disc, gt):.3f}   "
          f"(null floor 0.263 +/- 0.048)")
    A = Metrics.generators_to_matrix(disc)
    B = Metrics.generators_to_matrix(gt)
    print(f"Grassmann distance vs the full 6-dim GT: "
          f"{Metrics.grassmann_distance(A, B):.3f}   (null floor 3.011)")

    tag = (f"m{args.margin}_sh{args.n_shifts}_{args.net}"
           f"{'_oracle' if args.oracle else ''}_s{args.seed}")
    np.savez(os.path.join(outdir, f"exp19_generators_{tag}.npz"),
             fields=np.stack([np.stack(d) for d in disc]),
             norms=np.array([r[1] for r in rows]),
             residuals=np.array([r[2] for r in rows]),
             # self-contained analysis: the u-dependent GT triples must be
             # built on the run's OWN reference sample (sample 0 depends on
             # n_train through make_dataset's gauss/sine split)
             u0_ref=sols[0, 0].numpy(), x=x_np, t=t_np,
             n_train=args.n_train)
    with open(os.path.join(outdir, f"exp19_{tag}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "perp_norm", "windowed_residual", "b_share",
                    "best_gt", "cos"])
        w.writerows(rows)
    print(f"\nartefacts -> {outdir}/exp19_{tag}.csv")


if __name__ == "__main__":
    main()
