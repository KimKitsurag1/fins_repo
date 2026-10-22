"""probe_surrogate — is a trained S_theta spuriously LINEAR?

FINS on KdV scores cos ~0.03, BELOW a random null (0.57). That is not
noise: it means the discovered directions are structurally orthogonal to
the ground truth, and the simplest way to be orthogonal to d_x, d_t and a
xi-dominated Galilean boost is to live entirely in the phi channel. A
phi-only family beta(x,t) d_u is an EXACT symmetry of any LINEAR operator
(superposition), so a surrogate that has not learned the nonlinearity
carries a whole spurious abelian algebra that the discovery loop will
find -- the exp2/exp6 story of heat, now as a defect.

This probe loads the cached S_theta checkpoint of a campaign run and
measures, on the training data:

  linearity   ||S(2u0) - 2 S(u0)|| / ||S(2u0)||  -- 0 for a linear
              operator, O(1) for Burgers/KdV whose nonlinear terms scale
              as u^2
  GT trio     equivariance residual of d_x, d_t, Galilean -- the essential
              trio that exp25 showed the DATA admits to machine precision
  u d_u       NOT a symmetry of Burgers/KdV; a small residual means the
              surrogate is linear enough to fake it
  beta d_u    random superposition directions; same reading, whole family

If u d_u and beta d_u score as low as the GT trio, the surrogate is the
blocker and no loss change will fix it: the remedy is upstream (more
surrogate training, or a nonlinearity-aware fidelity check as a gate).

Run:  python probe_surrogate.py --pde kdv --seed 42
"""

import argparse
import os

import numpy as np
import torch

from config import ExperimentConfig
from data_generation import generate_pde_data
from ground_truth import GroundTruthAlgebra, get_gt_names
from method_fins import equivariance_loss, train_solution_operator
from shared_modules_v2 import FNOSurrogateEquivariant


def load_surrogate_state(path, device):
    """Load a surrogate checkpoint across torch versions.

    torch >= 2.6 defaults weights_only=True, and these checkpoints carry a
    reference to the activation (neuralop stores it in the state dict), so
    the strict loader refuses them. The files are produced by this
    pipeline, so weights_only=False is safe here. `_metadata` is dropped
    because load_state_dict rejects it as an unexpected key.
    """
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict):
        sd = dict(sd)
        sd.pop('_metadata', None)
    return sd


def residual(S, xi, eta, phi, u, xc, tc, device, horizon=None, seam=0.2):
    dx = xc[:, 0, 1] - xc[:, 0, 0]
    dt = tc[:, 0, 1] - tc[:, 0, 0]
    with torch.enable_grad():
        return float(equivariance_loss(
            S, xi, eta, phi, u[:, :, :, 0].contiguous(), u, xc, tc,
            dx, dt, device, mode='jvp', normalize=True,
            horizon=horizon, seam=seam))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="kdv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--horizon", type=int, default=None,
                    help="score with the short-horizon increment loss over "
                         "this many slices (the local criterion)")
    ap.add_argument("--seam", type=float, default=0.2)
    ap.add_argument("--train-if-missing", action="store_true",
                    help="train the surrogate when no checkpoint exists, "
                         "reproducing run_single_pde exactly (same seed, "
                         "same config, same call) and saving it to the "
                         "shared checkpoint path for reuse")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ExperimentConfig()
    cfg.device = device
    ckpt = args.ckpt or os.path.join(
        os.environ.get("FINS_RESULTS_DIR", "."),
        f"S_theta_{args.pde}_s{args.seed}_g{cfg.grid_size}x{cfg.time_steps}.pt")
    have_ckpt = os.path.exists(ckpt)
    if not have_ckpt and not args.train_if_missing:
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            f"Re-run with --train-if-missing to train one the "
            f"same way run_single_pde does (same seed, same config); it is "
            f"saved to the path above and\nreused by later runs. This "
            f"is a surrogate trained by the SAME procedure, not the exact "
            f"weights that produced the campaign numbers.")

    # Data exactly as run_experiment builds it: seed first, full n_train.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ics, sols, xs, ts = generate_pde_data(args.pde, cfg, cfg.n_train,
                                          seed=args.seed)
    S = FNOSurrogateEquivariant(cfg).to(device)
    if have_ckpt:
        S.load_state_dict(load_surrogate_state(ckpt, device))
        print(f"  loaded {ckpt}")
    else:
        print(f"  training S_theta (same call as run_single_pde)...")
        S = train_solution_operator(ics, sols, xs, ts, cfg, device,
                                    equivariant=True)
        torch.save(S.state_dict(), ckpt)
        print(f"  saved -> {ckpt}")
    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)

    n = min(args.n, sols.shape[0])
    u, xc, tc = (sols[:n].to(device), xs[:n].to(device), ts[:n].to(device))
    args.n = n

    print("=" * 76)
    print(f"surrogate probe: {os.path.basename(ckpt)}  ({args.pde}, "
          f"{args.n} training samples)")
    print("=" * 76)

    u0 = u[:, :, :, 0].contiguous()
    with torch.no_grad():
        s1 = S(u0, xc, tc)
        s2 = S(2.0 * u0, xc, tc)
        lin = float((s2 - 2 * s1).norm() / (s2.norm() + 1e-12))
        fit = float((s1 - u).norm() / (u.norm() + 1e-12))
    print(f"  fit error delta        = {fit:.4f}")
    print(f"  nonlinearity ratio     = {lin:.4f}   "
          f"(0 = linear operator; Burgers/KdV data: O(1))")

    x_np, t_np = xs[0, 0].numpy(), ts[0, 0].numpy()
    build = {"heat": lambda uu: GroundTruthAlgebra.heat_generators(x_np, t_np, uu, alpha=0.01),
             "burgers": lambda uu: GroundTruthAlgebra.burgers_generators(x_np, t_np, uu),
             "kdv": lambda uu: GroundTruthAlgebra.kdv_generators(x_np, t_np, uu)}[args.pde]
    # per-sample ground truth: phi depends on u for scaling and projective,
    # so broadcasting sample 0's fields misstates their residual
    per = [build(sols[r, 0].numpy()) for r in range(args.n)]
    names = get_gt_names(args.pde)
    gt = [tuple(np.stack([np.asarray(per[r][j][c]) for r in range(args.n)])
                for c in range(3)) for j in range(len(names))]
    T = lambda a: (torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
                   .view(args.n, 1, *np.asarray(a).shape[-2:])
                   if np.asarray(a).ndim == 3 else
                   torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
                   .view(1, 1, *np.asarray(a).shape).expand(args.n, 1, -1, -1))
    lab = ("sqrt of the short-horizon increment loss" if args.horizon
           else "full-horizon loss")
    print(f"\n  GT generators, per-sample fields ({lab}):")
    for n, (xi, eta, phi) in zip(names, gt):
        v = residual(S, T(xi), T(eta), T(phi), u, xc, tc, device, args.horizon, args.seam)
        print(f"    {n:>12}: {np.sqrt(v) if args.horizon else v:.3e}")

    Z = torch.zeros_like(u)
    print("\n  spurious candidates (NOT symmetries of a nonlinear PDE):")
    v = residual(S, Z, Z, u, u, xc, tc, device, args.horizon, args.seam)
    print(f"    {'u d_u':>12}: {np.sqrt(v) if args.horizon else v:.3e}")
    rng = np.random.default_rng(0)
    X, Tm = np.meshgrid(x_np, t_np, indexing="ij")
    for k in range(3):
        beta = sum(rng.standard_normal() / m * np.cos(m * X + rng.uniform(0, 6.28))
                   * np.exp(-0.05 * m ** 2 * Tm) for m in range(1, 4))
        print(f"    {'beta_' + str(k) + ' d_u':>12}: "
              f"{(lambda v: np.sqrt(v) if args.horizon else v)(residual(S, Z, Z, T(beta), u, xc, tc, device, args.horizon, args.seam)):.3e}")
    print("\n  reading: if u d_u / beta d_u sit near the GT trio, the surrogate "
          "is linear enough\n  to carry a spurious superposition algebra -- "
          "the discovery loop will find it.")


if __name__ == "__main__":
    main()
