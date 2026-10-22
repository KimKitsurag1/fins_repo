"""exp19_analyze — subspace view of the windowed-discovery output.

Per-slot best-match tables understate what exp19 found: slots recover
NOISY copies of a few true directions (cos 0.7-0.95), deflation removes
the noisy field rather than the true direction, and the leftover keeps
winning later slots. The statistics below are taken at the subspace level:

  * Gram matrix of the discovered fields — duplication made explicit;
  * singular spectrum of the stacked set — the EFFECTIVE dimension of what
    was discovered, used in place of "rank";
  * containment of every ground-truth generator in the discovered span,
    ||P_span g|| in [0, 1] — whether scaling and projective are truly
    absent or merely smeared across slots below the best-match radar;
  * junk fraction of every slot — how much of it lies outside the full
    6-dim GT span.

Run:  python exp19_analyze.py                      # all exp19 npz found
      python exp19_analyze.py --npz path/to/file.npz
"""

import argparse
import glob
import os

import numpy as np

import exp_common as C
import exp10_leak_vs_delta as E10
from ground_truth import GroundTruthAlgebra, Metrics, get_gt_names

ALPHA = 0.01


def rows_matrix(triples):
    M = np.stack([np.concatenate([c.ravel() for c in t]) for t in triples])
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)


def analyse(path, seed=42, n_train=None):
    z = np.load(path)
    F = z["fields"]                                   # [n_sym, 3, Nx, Nt]
    n_sym, _, Nx, Nt = F.shape
    if "u0_ref" in z:                       # npz with an embedded reference sample
        u0 = z["u0_ref"]
        x_np, t_np = z["x"], z["t"]
        src = "embedded reference sample"
    else:                                   # npz without one: must match the run
        if n_train is None:
            print("=" * 78)
            print(os.path.basename(path))
            print("  SKIPPED: npz without an embedded reference "
                  "sample.\n  Re-run with --n-train equal to the RUN's "
                  "--n-train (sample 0 depends\n  on it through "
                  "make_dataset's gauss/sine split).\n")
            return None
        rng = np.random.default_rng(seed)
        _, sols, _, _, x_np, t_np = E10.make_dataset(n_train, Nx, Nt, rng)
        u0 = sols[0, 0].numpy()
        src = f"regenerated with n_train = {n_train}"
    gt = GroundTruthAlgebra.heat_generators(x_np, t_np, u0, alpha=ALPHA)
    names = get_gt_names("heat")

    A = rows_matrix([tuple(F[k]) for k in range(n_sym)])
    G = rows_matrix(gt)

    print("=" * 78)
    print(os.path.basename(path))
    print(f"reference sample: {src}, "
          f"fingerprint {np.round(np.asarray(u0).ravel()[:4], 4)}")
    print("=" * 78)

    gram = A @ A.T
    off = [(abs(gram[i, j]), i + 1, j + 1)
           for i in range(n_sym) for j in range(i + 1, n_sym)]
    off.sort(reverse=True)
    print("duplication (largest |cos| between slots): "
          + ", ".join(f"{i}&{j}: {v:.2f}" for v, i, j in off[:4]))

    s = np.linalg.svd(A, compute_uv=False)
    pr = float((s.sum() ** 2) / (s ** 2).sum())
    print("singular values of the slot set: "
          + "  ".join(f"{v:.2f}" for v in s))
    print(f"effective dimension: participation ratio = {pr:.2f}, "
          f"sigma_k/sigma_1 > 0.3 -> {int((s / s[0] > 0.3).sum())}")

    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[sv / sv[0] > 1e-6]
    cont = np.linalg.norm(G @ V.T, axis=1)
    print("\ncontainment of each GT generator in the discovered span:")
    print("  " + C.fmt_table(
        [[n, f"{c:.3f}"] for n, c in zip(names, cont)],
        ["generator", "||P_span g||"]).replace("\n", "\n  "))

    _, svg, Gt_ = np.linalg.svd(G, full_matrices=False)
    W = Gt_[svg / svg[0] > 1e-6]
    junk = 1.0 - np.linalg.norm(A @ W.T, axis=1) ** 2
    print("junk fraction per slot (energy outside the 6-dim GT span): "
          + "  ".join(f"{v:.2f}" for v in junk))

    # Raw containments are inflated by GT correlations: on x in [0, L) the
    # fields u, x u, x^2 u are strongly mutually correlated, so a span that
    # holds the essential trio contains the x-weighted generators almost
    # for free. The per-generator statistic is the containment of
    # the NOVEL component — the part orthogonal to the essential trio.
    ess = [names.index(n) for n in ("∂_x", "∂_t", "u∂_u")]
    Qe, _ = np.linalg.qr(G[ess].T)
    print("\nnovel-component analysis (relative to the essential trio):")
    tab = []
    for j, n in enumerate(names):
        if j in ess:
            continue
        base = float(np.linalg.norm(Qe.T @ G[j]))
        g_nov = G[j] - Qe @ (Qe.T @ G[j])
        nf = float(np.linalg.norm(g_nov))
        c_nov = (float(np.linalg.norm(V @ (g_nov / nf)))
                 if nf > 1e-8 else float("nan"))
        tab.append([n, f"{base:.3f}", f"{nf:.3f}", f"{c_nov:.3f}"])
    print("  " + C.fmt_table(
        tab, ["generator", "free baseline", "novel fraction",
              "||P_span g_nov||"]).replace("\n", "\n  "))
    print("  (random-span reference for the last column: ~"
          f"{np.sqrt(V.shape[0] / V.shape[1]):.3f})\n")
    return dict(path=os.path.basename(path), sv=s, cont=cont, junk=junk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=None,
                    help="for npz without an embedded reference sample: the run's --n-train")
    args = ap.parse_args()
    paths = ([args.npz] if args.npz else
             sorted(glob.glob(os.path.join(C.RESULTS_DIR,
                                           "exp19_generators_*.npz")))
             or sorted(glob.glob(os.path.join("results",
                                              "exp19_generators_*.npz"))))
    if not paths:
        print("no exp19_generators_*.npz found")
        return
    for p in paths:
        analyse(p, seed=args.seed, n_train=args.n_train)


if __name__ == "__main__":
    main()
