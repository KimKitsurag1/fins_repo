"""Experiment 42 -- the protocol's tables (PROTOCOL.md, section 5).

Every method is scored on the same ten evaluation trajectories, the same region (torus band
[0.2, 0.8] L, times 8 .. Nt - 8) and the same component scaling (xi / L, eta / T, phi / sigma_u).
Ground truth is used only here. Inputs in --dir:
  fins_{pde}_s{s}.npz        FINS through the learned operator (exp40 / exp39 --save)
  exact_{pde}_s{s}.npz       FINS through the exact operator (reference; heat, KdV)
  baselines_{pde}_s{s}.npz   exp41
  untrained_{pde}_s{s}.npz   exp41 --untrained (own-class nulls)
Class nulls: random subspaces of FINS's polynomial class (--null-draws per seed).
--t-margin: scored times t_margin .. Nt - t_margin (thinned data: 8 // thin keeps the physical region); --json: the
per-seed scores of every row.

Run:  python exp42_tables.py --dir results
"""

import argparse
import json
import os
import sys

# progress bars off unless TQDM_DISABLE is set explicitly (must precede tqdm's import)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp31_polynomial_discovery as e31                   # noqa: E402

FINITE = {"burgers": 5, "kdv": 4, "heat": 6, "bo": 4, "ks": 3, "fb100": 4, "fb125": 4, "fb150": 4, "fb175": 4,
          "fb200": 5, "ccf125": 4, "ccf150": 4, "ccf175": 4}
# methods that are given the equation (determining equations or residual built from the PDE)
# LieNLSD: run_lienlsd_general (exact equation); LieNLSD_RHS: LieNLSD with the exact right-hand side; LieNLSD_F does not
EQUATION_AWARE = {"FINS, exact operator", "LIG", "LieNLSD", "LieNLSD_RHS", "LieNLSD_RHS_spec"}
DEGREE = {"burgers": 2, "kdv": 2, "heat": 3, "bo": 2, "ks": 2, "fb100": 2, "fb125": 2, "fb150": 2, "fb175": 2,
          "fb200": 2, "ccf125": 2, "ccf150": 2, "ccf175": 2}
TOL = 1e-6
# Augerino returns its fixed dictionary (six affine fields and u d_u), each scaled by its learned width;
# a span metric is blind to the widths, so its protocol row scores the dictionary. The post hoc row keeps
# the generators whose width grew past GROWN (initial width 0.018; in the v1 run the final widths are
# either < 0.025 or >= 0.15, so any threshold in between selects the same generators).
GROWN = 0.05


def fins_fields(z):
    """Fields [k, 3, N, Nx, Nt] of the k lowest eigen-generators saved by exp40 / exp39."""
    u, xs, ts = z["u"], z["xs"], z["ts"]
    X = ((xs - z["x_offset"][:, None]) / z["Lw"])[:, :, None]
    T = (ts / z["T_max"])[:, None, :]
    U = u / z["sigma"]
    mons = z["mons"]
    mono = np.stack([np.broadcast_to(X ** a * T ** b * U ** c, u.shape) for a, b, c in mons])
    nm, k = len(mons), int(z["k"])
    C = z["C"][:, :k]
    return np.stack([np.stack([np.tensordot(C[q * nm:(q + 1) * nm, j], mono, 1) for q in range(3)])
                     for j in range(k)])


def vectors(fields, keep, w, tm=8):
    """[k, 3, N, Nx, Nt] -> [k, dim]: scaled components on the scored region."""
    f = fields[:, :, :, keep][..., tm:-tm] * w[None, :, None, None, None]
    return f.reshape(f.shape[0], int(np.prod(f.shape[1:])))


def orth(V):
    U, s, _ = np.linalg.svd(V.T, full_matrices=False)
    return U[:, s > TOL * s[0]] if s.size and s[0] > 0 else U[:, :0]


def novel_parts(G):
    """Part of each ground-truth generator orthogonal to the others."""
    out = []
    for i in range(len(G)):
        others = orth(np.delete(G, i, 0)) if len(G) > 1 else np.zeros((G.shape[1], 0))
        out.append(G[i] - others @ (others.T @ G[i]))
    return out


def score(D, G, gps=None):
    """Subspace and per-generator scores of discovered vectors D [kA, dim] against ground truth G [kB, dim].
    missing = kB - ||Qa' Qb||_F^2 (ground-truth dimensions not captured), spurious = kA_rank - ||Qa' Qb||_F^2
    (discovered dimensions outside the ground truth); both basis-free. Novel containment of generator i:
    the part of g_i orthogonal to the other ground-truth generators, its share inside the discovered span
    -- robust to near-dependent ground truth (on a band of x, 1, x, x^2, x^3 are nearly dependent).
    Grassmann angle part and rank penalty as in LieNLSD and pde/ground_truth.Metrics, for comparability."""
    Qa, Qb = orth(D), orth(G)
    M = Qa.T @ Qb if Qa.shape[1] else np.zeros((0, Qb.shape[1]))
    overlap = float((M ** 2).sum())
    s = np.linalg.svd(M, compute_uv=False) if M.size else np.zeros(0)
    angle = float(np.sqrt((np.arccos(np.clip(s, -1, 1)) ** 2).sum()))
    gps = novel_parts(G) if gps is None else gps
    novel = np.array([float(np.linalg.norm(Qa.T @ gp) / np.linalg.norm(gp)) if Qa.shape[1] else 0.0 for gp in gps])
    contain = np.array([float(np.linalg.norm(Qa.T @ g) / np.linalg.norm(g)) if Qa.shape[1] else 0.0 for g in G])
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-30)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    cos = np.abs(Dn @ Gn.T)
    r, c = linear_sum_assignment(-cos)
    return dict(missing=len(G) - overlap, spurious=Qa.shape[1] - overlap, angle=angle,
                penalty=abs(len(D) - len(G)) * (np.pi / 2) ** 2, novel=novel, contain=contain,
                recovered=int((novel >= 0.9).sum()),
                cosine=float(cos[r, c].mean()) if len(D) else 0.0, k=len(D), rank=Qa.shape[1],
                floor=max(0, len(G) - Qa.shape[1]))


def class_fields(z):
    """Every monomial of FINS's class in every component: fields [3 nm, 3, N, Nx, Nt]."""
    z = dict(z)
    nm = len(z["mons"])
    out = []
    for q in range(3):
        for j in range(nm):
            C = np.zeros((3 * nm, 1))
            C[q * nm + j, 0] = 1.0
            out.append(fins_fields(dict(z, C=C, k=1))[0])
    return np.stack(out)


def null_scores(M, k, G, gps, n, rng, uniform):
    """Mean scores of n random k-dim subspaces of the class: uniform on its span (Haar) or Gaussian
    coefficients in the normalised monomial basis (FINS's parameterisation)."""
    Q = orth(M)
    res = [score((Q @ rng.standard_normal((Q.shape[1], k))).T if uniform else rng.standard_normal((k, len(M))) @ M,
                 G, gps) for _ in range(n)]
    out = {key: float(np.mean([r[key] for r in res])) for key in res[0] if key not in ("novel", "contain")}
    out["novel"] = np.mean([r["novel"] for r in res], 0)
    out["contain"] = np.mean([r["contain"] for r in res], 0)
    out["k"] = k
    out["std_missing"] = float(np.std([r["missing"] for r in res]))
    return out


def augerino_widths(f, u):
    """Learned widths from Augerino's fields [7, 3, N, Nx, Nt]: G1..G6 have unit maximum on the
    normalised grid, G7 is phi = w u."""
    w = np.abs(f[:6]).reshape(6, -1).max(1)
    g7 = np.abs(f[6, 2]).reshape(len(u), -1).max(1) / np.abs(u).reshape(len(u), -1).max(1)
    return np.append(w, g7.mean())


def ground_truth(pde, u, xs, ts, cfg):
    names, gt = e31.algebra(pde, xs, ts, u, cfg.alpha_heat, DEGREE[pde])
    return list(names), np.stack(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("FINS_RESULTS_DIR", "."))
    ap.add_argument("--pdes", default="burgers,kdv,heat")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--band", type=float, default=0.2)
    ap.add_argument("--null-draws", type=int, default=100, help="random subspaces of FINS's class per seed (0: off)")
    ap.add_argument("--t-margin", type=int, default=8, help="scored times t_margin .. Nt - t_margin")
    ap.add_argument("--json", default=None, help="write the per-seed scores of every row here")
    args = ap.parse_args()
    tm = args.t_margin
    dump = {}
    cfg = e31.ExperimentConfig()
    seeds = [int(s) for s in args.seeds.split(",")]
    for pde in args.pdes.split(","):
        rows = {}
        extra = {}
        widths = []
        for s in seeds:
            runs = {}
            for tag, label in (("fins", "FINS"), ("exact", "FINS, exact operator")):
                p = os.path.join(args.dir, f"{tag}_{pde}_s{s}.npz")
                if os.path.exists(p):
                    runs[label] = np.load(p, allow_pickle=True)
            zbs = {}
            for tag, suffix in (("baselines", ""), ("untrained", ", untrained")):
                pb = os.path.join(args.dir, f"{tag}_{pde}_s{s}.npz")
                if os.path.exists(pb):
                    zbs[suffix] = np.load(pb, allow_pickle=True)
            zb = zbs.get("", next(iter(zbs.values()), None))
            ref = next((z for z in (runs.get("FINS"), runs.get("FINS, exact operator"), zb) if z is not None), None)
            if ref is None:
                continue
            u, xs, ts = ref["u"], ref["xs"], ref["ts"]
            for other in list(runs.values()) + list(zbs.values()):
                if other["u"].shape != u.shape or not np.allclose(other["u"], u, atol=1e-6 * np.abs(u).max()):
                    raise SystemExit(f"{pde} s{s}: evaluation trajectories differ between inputs")
            # exp48 inputs store the clean data: the corruption they saw must agree too
            corr = {(float(z["noise"]), int(z["thin"])) for z in list(runs.values()) + list(zbs.values())
                    if "noise" in z.files and "thin" in z.files}
            if len(corr) > 1:
                raise SystemExit(f"{pde} s{s}: inputs saw different corruptions {sorted(corr)}")
            if corr and tm != max(1, 8 // next(iter(corr))[1]):
                print(f"NOTE {pde} s{s}: --t-margin {tm} differs from 8 // thin = {8 // next(iter(corr))[1]}")
            Nx = u.shape[1]
            frac = np.arange(Nx) / Nx
            keep = (frac >= args.band) & (frac <= 1 - args.band)
            L = (xs[0, 1] - xs[0, 0]) * Nx
            w = np.array([1 / L, 1 / ts[:, -1].max(), 1 / u.std()])
            names, gt = ground_truth(pde, u, xs, ts, cfg)
            G_full = vectors(gt, keep, w, tm)
            G_fin = G_full[:FINITE[pde]]
            gps_fin, gps_full = novel_parts(G_fin), novel_parts(G_full)
            fields = {label: fins_fields(z) for label, z in runs.items()}
            for suffix, zz in zbs.items():
                for m in zz["methods"]:
                    f = zz[f"fields_{m}"].astype(np.float64)          # [N, k, 3, Nx, Nt]
                    lab = str(m)
                    if lab.startswith("LIG"):
                        lab = f"LIG, {f.shape[1]} generators" + (f", {int(zz[f'epochs_{m}'])} epochs"
                                                                 if f"epochs_{m}" in zz and not suffix else "")
                    lab += suffix
                    fields[lab] = np.moveaxis(f, 0, 2)                # -> [k, 3, N, Nx, Nt]
                    if f"nown_{m}" in zz:                             # the method's own count: leading generators
                        fields[f"{lab}, own count"] = fields[lab][:int(zz[f"nown_{m}"])]
                    if str(m) == "Augerino" and not suffix:
                        wa = augerino_widths(fields[lab], u)
                        widths.append((s, wa))
                        fields["Augerino, grown widths only (post hoc)"] = fields[lab][wa > GROWN]
            for label, f in fields.items():
                D = vectors(f, keep, w, tm)
                bad = ~np.isfinite(D).all(1)
                if bad.any():
                    print(f"WARNING {pde} s{s} {label}: {bad.sum()} of {len(D)} generators non-finite, scored as zero")
                    D = np.where(np.isfinite(D), D, 0.0)
                res = {"finite": score(D, G_fin, gps_fin)}
                if len(G_full) > len(G_fin):
                    res["full"] = score(D, G_full, gps_full)
                rows.setdefault(label, []).append(res)
            if runs and args.null_draws:
                z0 = next(iter(runs.values()))
                M = vectors(class_fields(z0), keep, w, tm)
                kc = int(z0["k"])
                rng = np.random.default_rng(s)
                for uniform, lab in ((True, "null: uniform on the class"), (False, "null: Gaussian coefficients")):
                    lab = f"{lab} ({kc} of {np.linalg.matrix_rank(M, tol=TOL * np.linalg.norm(M, 2))})"
                    res = {"finite": null_scores(M, kc, G_fin, gps_fin, args.null_draws, rng, uniform)}
                    if len(G_full) > len(G_fin):
                        res["full"] = null_scores(M, kc, G_full, gps_full, args.null_draws, rng, uniform)
                    rows.setdefault(lab, []).append(res)
            for label, z in runs.items():
                lam, k = z["lam"], int(z["k"])
                ratios = lam[1:k + 5] / np.maximum(np.abs(lam[:k + 4]), 1e-30)
                extra.setdefault(label, []).append(dict(
                    below=bool((z["losses"] < lam[k]).all()), margin=float(lam[k] / z["losses"].max()),
                    gap=float(lam[k] / max(lam[k - 1], 1e-30)), rank=int(np.argmax(ratios)) + 1,
                    band=z["band"], rel=float(z["rel_change"])))
        if not rows:
            continue
        dump[pde] = {label: {cv: {key: [float(r[cv][key]) for r in rs] for key in ("missing", "spurious", "k")}
                             | {"novel": [np.asarray(r[cv]["novel"]).tolist() for r in rs],
                                "contain": [np.asarray(r[cv]["contain"]).tolist() for r in rs]}
                             for cv in rs[0]} for label, rs in rows.items()}
        dump[pde]["_names"] = list(names)
        print(f"\n## {pde}: {len(seeds)} seeds, ground truth finite {FINITE[pde]}"
              + (f", full in the class {len(names)}" if len(names) > FINITE[pde] else "")
              + f"; component scaling xi / L, eta / T, phi / sigma_u; region [{args.band}, {1 - args.band}] L\n")
        conv = ["finite"] + (["full"] if len(names) > FINITE[pde] else [])
        for cv in conv:
            gtn = names[:FINITE[pde]] if cv == "finite" else names
            print(f"### ground truth: {cv} ({len(gtn)})\n")
            print("| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | "
                  "Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | " + " | ".join(gtn) + " |")
            print("|---" * (9 + len(gtn)) + "|")
            for label, rs in rows.items():
                g = lambda key: np.array([r[cv][key] for r in rs])
                mi, sp_, a, pen, rec, cs = g("missing"), g("spurious"), g("angle"), g("penalty"), g("recovered"), g("cosine")
                nov = np.array([r[cv]["novel"] for r in rs]).mean(0)
                aware = label in EQUATION_AWARE or label.split(",")[0].split("@")[0] in EQUATION_AWARE
                shown = label + (" (uses the equation)" if aware else "")
                ks = g("k")
                kk = f"{ks.min()}" if ks.min() == ks.max() else f"{ks.min()}-{ks.max()}"
                print(f"| {shown} | {len(rs)} | {kk} ({g('rank').mean():.1f}) | {mi.mean():.2f} +/- {mi.std():.2f} | "
                      f"{g('floor').mean():.1f} | {sp_.mean():.2f} +/- {sp_.std():.2f} | {a.mean():.2f} + {pen.mean():.2f} | "
                      f"{rec.mean():.1f} / {len(gtn)} | {cs.mean():.2f} | " + " | ".join(f"{c:.2f}" for c in nov) + " |")
            print()
        for s_, wa in widths:
            print(f"Augerino s{s_}: learned widths " + ", ".join(f"{x:.3f}" for x in wa)
                  + f" (G1..G7: trans x, trans t, rotation, scale, anti-scale, shear, u-scale); grown past {GROWN}: "
                  + (", ".join(f"G{i + 1}" for i in np.flatnonzero(wa > GROWN)) or "none"))
        for label, es in extra.items():
            print(f"{label}: all below lambda_(k+1) on {sum(e['below'] for e in es)}/{len(es)} seeds; worst margin "
                  + ", ".join(f"{e['margin']:.1f}x" for e in es) + "; lambda_(k+1) / lambda_k "
                  + ", ".join(f"{e['gap']:.1f}" for e in es) + "; rank from the largest gap "
                  + ", ".join(str(e["rank"]) for e in es) + "; relative change "
                  + ", ".join(f"{e['rel']:.2f}" for e in es) + "; band check u_t share "
                  + ", ".join(f"{e['band'][1]:.1e}" for e in es))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=1)


if __name__ == "__main__":
    main()
