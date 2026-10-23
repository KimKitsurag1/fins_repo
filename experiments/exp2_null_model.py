"""Experiment 2 — null-model floor for Table 1, computed with the repository
metrics themselves.

For linear PDEs every field v_beta = beta(x,t) d_u with beta a solution of
the same PDE satisfies the FINS equivariance condition exactly (superposition
subalgebra). A method that returns SIX RANDOM DIRECTIONS from this
infinite-dimensional trivial subalgebra therefore passes the equivariance
test perfectly while carrying no information about the finite algebra.

This script asks: what (d_G, mean cosine) does that null model score under
`pde/ground_truth.Metrics` — the exact functions behind Table 1?  The floor
belongs in the table: methods statistically indistinguishable from it have
not discovered anything.

Two metric spaces are reported:
  triple space  — flattened (xi, eta, phi), the repository convention;
  Q space       — flattened characteristics Q = phi - xi u_x - eta u_t,
                  which identifies generators equivalent up to trivial parts.

Run:  python exp2_null_model.py [--trials 20] [--seed 0]
"""

import argparse

import numpy as np

import exp_common as C
from ground_truth import Metrics, GroundTruthAlgebra, get_gt_names
from scipy.optimize import linear_sum_assignment

NX, NT, T, L, ALPHA = 64, 128, 1.0, 1.0, 0.01


def flatten_triples(gens):
    return Metrics.generators_to_matrix(gens)


def q_of(xi, eta, phi, u, L):
    ux = C.dx_spectral(u[None], L)[0]
    ut = ALPHA * C.dx_spectral(u[None], L, 2)[0]
    return phi - xi * ux - eta * ut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    x = np.linspace(0, L, NX, endpoint=False)
    t = np.linspace(0, T, NT)
    u0 = C.ic_heat_gauss(1, x, L, rng)
    u = C.heat_solve(u0, ALPHA, L, t)[0]

    gt = GroundTruthAlgebra.heat_generators(x, t, u, alpha=ALPHA)
    names = get_gt_names("heat")
    A_gt = flatten_triples(gt)
    Q_gt = np.stack([q_of(*g, u, L).ravel() for g in gt], axis=1)

    d_tri, c_tri, d_q, c_q = [], [], [], []
    for _ in range(args.trials):
        betas = C.superposition_directions(6, x, t, L, ALPHA, rng)
        null = [(np.zeros_like(u), np.zeros_like(u), b) for b in betas]
        A_null = flatten_triples(null)
        d_tri.append(Metrics.grassmann_distance(A_null, A_gt))
        c_tri.append(Metrics.best_cosine_similarity(null, gt))
        Q_null = np.stack([b.ravel() for b in betas], axis=1)
        s = np.linalg.svd(np.linalg.qr(Q_null)[0].T @ np.linalg.qr(Q_gt)[0],
                          compute_uv=False)
        d_q.append(float(np.sqrt(np.sum(
            np.arccos(np.clip(s, -1, 1)) ** 2))))
        sim = np.abs(Q_null.T @ Q_gt) / (
            np.linalg.norm(Q_null, axis=0)[:, None]
            * np.linalg.norm(Q_gt, axis=0)[None, :] + 1e-12)
        ri, ci = linear_sum_assignment(-sim)
        c_q.append(float(sim[ri, ci].mean()))

    print("=" * 70)
    print("EXP 2 - null model: 6 random superposition directions, "
          f"{args.trials} trials")
    print("Metrics: pde/ground_truth.Metrics (identical to Table 1)")
    print("=" * 70)
    rows = [
        ("triple space (repo)", f"{np.mean(d_tri):.3f} +/- {np.std(d_tri):.3f}",
         f"{np.mean(c_tri):.3f} +/- {np.std(c_tri):.3f}"),
        ("Q space", f"{np.mean(d_q):.3f} +/- {np.std(d_q):.3f}",
         f"{np.mean(c_q):.3f} +/- {np.std(c_q):.3f}"),
    ]
    print(C.fmt_table(rows, ["metric space", "d_G (null)", "mean cos (null)"]))
    print("\nPaper Table 1, heat:  FINS d_G 2.17 cos 0.50 | LaLiGAN d_G 3.39")
    print("Reference maxima: d_G(6 dims fully orthogonal) = "
          f"{np.sqrt(6) * np.pi / 2:.3f}")

    # per-GT-generator best |cos| against the null set, averaged over trials
    per = np.zeros((args.trials, 6))
    rng2 = np.random.default_rng(args.seed)
    for tr in range(args.trials):
        betas = C.superposition_directions(6, x, t, L, ALPHA, rng2)
        null = [(np.zeros_like(u), np.zeros_like(u), b) for b in betas]
        sim = Metrics.cosine_similarity_matrix(null, gt)
        per[tr] = sim.max(axis=0)
    rows = [(n, f"{m:.3f}") for n, m in zip(names, per.mean(axis=0))]
    print("\nBest |cos| of the null set against each GT generator "
          "(triple space):")
    print(C.fmt_table(rows, ["generator", "mean best |cos|"]))


if __name__ == "__main__":
    main()
