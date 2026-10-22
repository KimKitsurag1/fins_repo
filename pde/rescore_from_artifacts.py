"""rescore_from_artifacts — recompute Table 1 metrics from saved fields.

Reads the `generators_{pde}_seed{N}.npz` written by
run_single_pde.py and recomputes every method's metrics WITHOUT re-running
discovery. This is the tool that turns any later change of ground truth,
rank convention or metric into seconds instead of GPU-hours.

Modes
-----
--verify results_....json   PASS/FAIL equivalence check: metrics recomputed
                            from the artifacts must match the live run's
                            json to a relative 1e-6.
--gt-rank {full,essential}  which ground-truth subspace to score against:
                            the classical algebra stored in the npz, or the
                            essential torus trio {d_x, d_t, u d_u /
                            Galilean} (rows are selected by name).
--crop M                    crop M boundary points from BOTH the fields and
                            the GT before scoring (0 disables), the same
                            for every method.

Run:  python rescore_from_artifacts.py generators_heat_seed42.npz
      python rescore_from_artifacts.py generators_heat_seed42.npz \
             --verify results_heat_s42.json
"""

import argparse
import json

import numpy as np

from closure_metrics import (align_refs, closure_report, describe,
                             fit_affine)
from ground_truth import GroundTruthAlgebra, Metrics

METHODS = ("FINS", "LieGAN", "LaLiGAN", "LIG", "LieNLSD", "Augerino")
# names as get_gt_names emits them (Unicode partials), not ASCII
ESSENTIAL = {"heat": ("\u2202_x", "\u2202_t", "u\u2202_u"),
             "burgers": ("\u2202_x", "\u2202_t", "Galilean"),
             "kdv": ("\u2202_x", "\u2202_t", "Galilean")}


def load_triples(z, prefix, crop=0):
    xi, eta, phi = (z[f"{prefix}_xi"], z[f"{prefix}_eta"],
                    z[f"{prefix}_phi"])
    if crop:
        sl = (slice(None), slice(crop, -crop), slice(crop, -crop))
        xi, eta, phi = xi[sl], eta[sl], phi[sl]
    return [tuple(np.asarray(c[k]) for c in (xi, eta, phi))
            for k in range(xi.shape[0])]


def score(disc, gt):
    A = Metrics.generators_to_matrix(disc)
    B = Metrics.generators_to_matrix(gt)
    return dict(grassmann=float(Metrics.grassmann_distance(A, B)),
                cos=float(Metrics.best_cosine_similarity(disc, gt)),
                rank=int(len(disc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--verify", default=None,
                    help="results json of the live run for the PASS/FAIL "
                         "equivalence check")
    ap.add_argument("--gt-rank", default="full",
                    choices=["full", "essential"])
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--ace", action="store_true",
                    help="recompute the closure column with closure_metrics.py "
                         "(needs fields on >= 2 samples)")
    ap.add_argument("--ace-raw", action="store_true",
                    help="skip the polynomial projection; reports the "
                         "noise-saturated variant for comparison")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    pde = str(z["pde"])
    gt_names = list(z["gt_names"])
    if pde == "burgers":
        # Burgers GT recomputed with the factor-2 pi algebra of ground_truth.py
        cr = ((slice(args.crop, -args.crop),) * 2 if args.crop
              else (slice(None),) * 2)
        gt = [tuple(np.asarray(q)[cr] for q in g) for g in
              GroundTruthAlgebra.burgers_generators(z["x"], z["t"],
                                                    z["u_sample"])]
    else:
        gt = load_triples(z, "gt", crop=args.crop)
    if args.gt_rank == "essential":
        keep = [i for i, n in enumerate(gt_names) if n in ESSENTIAL[pde]]
        if len(keep) != len(ESSENTIAL[pde]):
            raise SystemExit(
                f"essential filter matched {len(keep)} of "
                f"{len(ESSENTIAL[pde])} names; artifact has {gt_names}")
        gt = [gt[i] for i in keep]
        gt_names = [gt_names[i] for i in keep]
    print(f"{args.npz}: pde = {pde}, GT rank = {len(gt)} "
          f"({args.gt_rank}), crop = {args.crop}")
    print(f"{'method':>10} | {'d_G':>7} | {'cos':>6} | rank")
    print("-" * 38)
    out = {}
    for m in METHODS:
        if f"{m}_xi" not in z:
            continue
        disc = load_triples(z, m, crop=args.crop)
        out[m] = score(disc, gt)
        print(f"{m:>10} | {out[m]['grassmann']:7.3f} | "
              f"{out[m]['cos']:6.3f} | {out[m]['rank']}")

    if args.ace:
        x, t = z["x"], z["t"]
        dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
        basis = None if args.ace_raw else (x, t)
        print("\nclosure metric (closure_metrics.py)"
              + ("" if basis is None else " (polynomial projection, deg 2)"))
        u = z["u_ref"] if "u_ref" in z else z["u_sample"]
        if u.ndim < 3 or u.shape[0] < 2:
            print("  SKIPPED: the artifacts hold fields on a single sample, "
                  "and a\n  single sample cannot separate a*u from b. "
                  "Re-serialise with\n  several reference samples, or use "
                  "the exp23/exp24 npz.")
        else:
            for m in METHODS:
                if f"{m}_xi" not in z:
                    continue
                gens, spreads = [], []
                if f"{m}_xi_ref" not in z:
                    print(f"  {m:>10}: no multi-sample block")
                    continue
                axi, aeta, aphi = align_refs(z[f"{m}_xi_ref"],
                                             z[f"{m}_eta_ref"],
                                             z[f"{m}_phi_ref"])
                for k in range(axi.shape[0]):
                    g, sp = fit_affine(axi[k], aeta[k], aphi[k], u)
                    gens.append(g)
                    spreads.append(sp)
                rep = closure_report(gens, dx, dt, basis=basis)
                print(f"  {m:>10}: escaped {rep['escaped']:.3f}  "
                      f"structure {rep['structure']:.2e}  "
                      f"xi/eta sample-spread {np.mean(spreads):.2f}  "
                      f"| {describe(rep)}")

    if args.verify:
        ref = json.load(open(args.verify))
        print(f"\nverification against {args.verify} "
              f"(full GT, crop as in the live run):")
        ok = True
        for m, sc in out.items():
            if m not in ref:
                continue
            for key_art, key_json in (("grassmann", "grassmann"),
                                      ("cos", "cosine")):
                if key_json not in ref[m]:
                    for alt in ("cos", "cos_sim", "cosine_sim",
                                "best_cos"):
                        if alt in ref[m]:
                            key_json = alt
                            break
                if key_json not in ref[m]:
                    continue
                a, b = sc[key_art], float(ref[m][key_json])
                rel = abs(a - b) / (abs(b) + 1e-30)
                status = "PASS" if rel < args.tol else "FAIL"
                if status == "FAIL":
                    ok = False
                print(f"  {m:>10} {key_art:>9}: artifact {a:.6f} vs "
                      f"json {b:.6f}  rel {rel:.2e}  {status}")
        print("\nOVERALL:", "PASS" if ok else
              "FAIL — see mismatches above; the usual cause is the "
              "asymmetric M = 8 crop of evaluate_method's default "
              "(re-run with --crop 8 to reproduce FINS, --crop 0 for "
              "the rest)")


if __name__ == "__main__":
    main()
