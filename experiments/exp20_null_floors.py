"""Experiment 20 — null floors for Table 1, all three equations.

exp2's null (the superposition subalgebra) consists of EXACT spurious
symmetries and exists only for linear PDEs. For Burgers and KdV the null
answers a different question: how much do the metrics award to random
fields with the ground truth's statistics? Two nulls bracket the floor
from two sides:

  band-limited   each component (xi, eta, phi) is coloured noise with the
                 amplitude spectrum and energy matched to the mean of the
                 GT's same component — matches one-point and spectral
                 statistics, destroys everything else;
  shuffled-GT    components permuted across the GT generators with random
                 signs — additionally inherits the GT's own correlation
                 structure, which exp19 showed inflates cosines for free
                 on x in [0, L).

Floors are reported in triple space and in Q-space (characteristics with
spectral u_x and fd6 u_t on the repository's own data), pooled over a few
reference samples, mean +/- sigma over trials, plus the per-GT-generator
inflation profile (which columns of Table 2 a null can fake).

Run:  python exp20_null_floors.py [--pde all] [--trials 21]
"""

import argparse
import csv
import os

import numpy as np
import torch

import exp_common as C
from config import ExperimentConfig
from data_generation import generate_pde_data
from ground_truth import Metrics

PDES = ("heat", "burgers", "kdv")
GT_NAMES = {"heat": ["d_x", "d_t", "u d_u", "Galilean", "scaling",
                     "projective"],
            "burgers": ["d_x", "d_t", "Galilean", "scaling", "projective"],
            "kdv": ["d_x", "d_t", "Galilean", "scaling"]}


def gt_triples(pde, x, t, u, cfg):
    ut = C.fd_t6(u[None], t[1] - t[0])[0]
    gens = C.gt_characteristics(
        pde, x, t, u[None],
        alpha=getattr(cfg, "alpha_heat", 0.01),
        nu=getattr(cfg, "nu_burgers", 0.01), ut=ut[None])
    trip = [tuple(np.asarray(gens[n][c][0]) for c in range(3))
            for n in GT_NAMES[pde]]
    Qs = [np.asarray(gens[n][3][0]) for n in GT_NAMES[pde]]
    return trip, Qs


def null_bandlimited(gt, rng):
    """Coloured noise per component: GT-mean amplitude spectrum + energy."""
    r = len(gt)
    amp = [np.sqrt(np.mean([np.abs(np.fft.fft2(g[c])) ** 2 for g in gt],
                           axis=0)) for c in range(3)]
    en = [np.mean([np.mean(g[c] ** 2) for g in gt]) for c in range(3)]
    out = []
    for _ in range(r):
        comps = []
        for c in range(3):
            w = np.fft.fft2(rng.standard_normal(amp[c].shape))
            f = np.real(np.fft.ifft2(w / (np.abs(w) + 1e-30) * amp[c]))
            f *= np.sqrt(en[c] / (np.mean(f ** 2) + 1e-30))
            comps.append(f)
        out.append(tuple(comps))
    return out


def null_shuffled(gt, rng):
    """Independent per-component permutations of the GT set, random signs."""
    r = len(gt)
    out = []
    perms = [rng.permutation(r) for _ in range(3)]
    signs = rng.choice([-1.0, 1.0], size=(3, r))
    for j in range(r):
        out.append(tuple(signs[c, j] * gt[perms[c][j]][c]
                         for c in range(3)))
    return out


def q_of(triples, x, t, u, L):
    ux = C.dx_spectral(u[None], L)[0]
    ut = C.fd_t6(u[None], t[1] - t[0])[0]
    return [g[2] - g[0] * ux - g[1] * ut for g in triples]


def metrics_pair(disc, gt):
    A = Metrics.generators_to_matrix(disc)
    B = Metrics.generators_to_matrix(gt)
    return (Metrics.grassmann_distance(A, B),
            Metrics.best_cosine_similarity(disc, gt))


def vec_metrics(disc_vecs, gt_vecs):
    A = np.stack([v.ravel() for v in disc_vecs]).T
    B = np.stack([v.ravel() for v in gt_vecs]).T
    d = Metrics.grassmann_distance(A, B)
    dn = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-30)
    gn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
    sim = np.abs(dn.T @ gn)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-sim)
    return d, float(sim[ri, ci].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="all")
    ap.add_argument("--trials", type=int, default=21)
    ap.add_argument("--n-samples", type=int, default=3,
                    help="reference samples the trials are pooled over")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pdes = PDES if args.pde == "all" else (args.pde,)

    cfg = ExperimentConfig()
    rows = []
    print("=" * 84)
    print(f"EXP 20 - null floors for Table 1 "
          f"(trials = {args.trials}, samples = {args.n_samples})")
    print("=" * 84)
    for pde in pdes:
        torch.manual_seed(args.seed)
        ics, sols, xs, ts = generate_pde_data(pde, cfg,
                                              max(args.n_samples, 2),
                                              seed=args.seed)
        rng = np.random.default_rng(args.seed)
        acc = {("band", "triple"): [], ("band", "q"): [],
               ("shuf", "triple"): [], ("shuf", "q"): []}
        per_gen = {n: [] for n in GT_NAMES[pde]}
        per_trial = max(1, args.trials // args.n_samples)
        for b in range(args.n_samples):
            x = xs[b, 0].numpy()
            t = ts[b, 0].numpy()
            u = sols[b, 0].numpy()
            L = float(x[-1] + x[1] - x[0])
            gt, gtQ = gt_triples(pde, x, t, u, cfg)
            for _ in range(per_trial):
                for kind, maker in (("band", null_bandlimited),
                                    ("shuf", null_shuffled)):
                    nul = maker(gt, rng)
                    dg, cs = metrics_pair(nul, gt)
                    acc[(kind, "triple")].append((dg, cs))
                    dq, cq = vec_metrics(q_of(nul, x, t, u, L), gtQ)
                    acc[(kind, "q")].append((dq, cq))
                    if kind == "band":
                        sim = Metrics.cosine_similarity_matrix(nul, gt)
                        for j, n in enumerate(GT_NAMES[pde]):
                            per_gen[n].append(float(sim[:, j].max()))
        print(f"\n  {pde} (GT rank {len(GT_NAMES[pde])}):")
        for kind, label in (("band", "band-limited"), ("shuf",
                                                       "shuffled-GT")):
            for space in ("triple", "q"):
                a = np.array(acc[(kind, space)])
                print(f"    {label:<13} {space:>6}-space:  "
                      f"d_G = {a[:, 0].mean():.3f} +/- {a[:, 0].std():.3f}"
                      f"   cos = {a[:, 1].mean():.3f} +/- "
                      f"{a[:, 1].std():.3f}")
                rows.append([pde, label, space, f"{a[:, 0].mean():.4f}",
                             f"{a[:, 0].std():.4f}",
                             f"{a[:, 1].mean():.4f}",
                             f"{a[:, 1].std():.4f}"])
        print("    per-generator inflation (mean of best null |cos|, "
              "band-limited): "
              + ", ".join(f"{n}: {np.mean(v):.2f}"
                          for n, v in per_gen.items()))
        for n, v in per_gen.items():
            rows.append([pde, "per_gen_band", n, f"{np.mean(v):.4f}",
                         f"{np.std(v):.4f}", "", ""])

    path = os.path.join(C.RESULTS_DIR, "exp20_null_floors.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pde", "null", "space_or_gen", "dG_or_cos_mean",
                    "std", "cos_mean", "cos_std"])
        w.writerows(rows)
    print(f"\nCSV -> {path}")


if __name__ == "__main__":
    main()
