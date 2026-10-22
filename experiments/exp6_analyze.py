"""Analyse the generator fields saved by exp6 — what did each slot find?

Loads `exp6_generators_*.npz`, regenerates the reference sample (sample 0
of the training set, deterministic from the seed), builds the alpha-correct
ground-truth algebra on it, and reports per slot:

  - deflated norm and b-share (from the run);
  - channel energy split xi/eta/phi (translations are pure xi or eta,
    u d_u is pure phi — the split alone often identifies the slot);
  - best-matching GT generator and the cosine;
plus the Hungarian-matched mean cosine and the rank-aware Grassmann
distance of the whole set, via the repository Metrics.

Run:  python exp6_analyze.py --npz results/exp6_generators_lam30_s42.npz
      python exp6_analyze.py            # analyses every exp6 npz in results/
"""

import argparse
import glob
import os

import numpy as np
import torch

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
from ground_truth import GroundTruthAlgebra, Metrics, get_gt_names


def analyse(path):
    z = np.load(path)
    fields = z["fields"]                       # [n_sym, 3, Nx, Nt]
    norms = z["norms"]
    shares = z["b_shares"]
    seed = int(z["seed"]) if "seed" in z else 42
    n_sym, _, Nx, Nt = fields.shape

    cfg = ExperimentConfig()
    cfg.grid_size, cfg.time_steps = Nx, Nt     # matches smoke or full run
    torch.manual_seed(seed)
    ics, sols, xs, ts = generate_pde_data("heat", cfg, 1, seed=seed)
    x = xs[0, 0].numpy()
    t = ts[0, 0].numpy()
    u = sols[0, 0].numpy()

    gt = GroundTruthAlgebra.heat_generators(x, t, u, alpha=cfg.alpha_heat)
    names = get_gt_names("heat")
    disc = [tuple(fields[k]) for k in range(n_sym)]

    sim = Metrics.cosine_similarity_matrix(disc, gt)      # [n_sym, 6]
    print("=" * 86)
    print(os.path.basename(path),
          f"(lam = {float(z['lam']):g}, seed = {seed}, "
          f"ref sample: L = {x[-1] + x[1] - x[0]:.2f})")
    print("=" * 86)
    rows = []
    for k in range(n_sym):
        e = fields[k] ** 2
        tot = e.sum() + 1e-30
        split = "/".join(f"{e[c].sum() / tot:.2f}" for c in range(3))
        j = int(np.argmax(sim[k]))
        rows.append([f"{k + 1}", f"{norms[k]:.3e}", f"{shares[k]:.3f}",
                     split, names[j], f"{sim[k, j]:.3f}"])
    print(C.fmt_table(rows, ["slot", "defl. norm", "b-share",
                             "energy xi/eta/phi", "best GT match", "|cos|"]))

    A = Metrics.generators_to_matrix(disc)
    B = Metrics.generators_to_matrix(gt)
    print(f"\nHungarian-matched mean |cos| : "
          f"{Metrics.best_cosine_similarity(disc, gt):.3f}")
    print(f"Grassmann distance (rank-aware, vs full 6-dim GT): "
          f"{Metrics.grassmann_distance(A, B):.3f}")
    print("Null-floor reference (exp2, heat): mean cos 0.263 +/- 0.048, "
          "d_G 3.011 +/- 0.056\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()
    # search the same env-aware directory the runner writes to
    paths = ([args.npz] if args.npz else
             sorted(glob.glob(os.path.join(C.RESULTS_DIR,
                                           "exp6_generators_*.npz")))
             or sorted(glob.glob(os.path.join("results",
                                              "exp6_generators_*.npz"))))
    if not paths:
        print("no exp6_generators_*.npz found")
        return
    for p in paths:
        analyse(p)


if __name__ == "__main__":
    main()
