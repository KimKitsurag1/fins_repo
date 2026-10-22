"""table1 — build the benchmark table from campaign artifacts.

Reads every `generators_{pde}_seed{N}.npz` in the results directory and
produces the table the paper needs, aggregated over seeds. Nothing here
re-runs discovery: the artifacts hold the discovered fields, so a change
of convention costs seconds.

Every column carries the corrections this campaign established:

* d_G is SPLIT. The rank-aware distance is
  d_G^2 = sum(theta_i^2) + |k_A - k_B| (pi/2)^2, and k_A is the number of
  generators a method was configured to emit -- LIG emits 4, Augerino 7 --
  so the penalty measures configuration, not discovery. Both the total and
  the angle-only part are reported; on heat the two orderings differ.
* The crop is SYMMETRIC and explicit.
* The ground-truth rank is SELECTABLE: `full` uses the classical algebra,
  `essential` the torus subalgebra that a global periodic residual can
  actually see (exp1, exp17, exp18).
* CLOSURE replaces ACE. The published metric is not scale-invariant, uses
  a bracket missing the phi d/du term, and scores abelian or duplicated
  sets as perfect; `escaped` / `structure` / `spread` replace it, and the
  legacy value is kept beside them for comparability.
* NULL FLOORS are printed with the table and turned into z-scores, because
  none of these numbers is interpretable without them (exp20).

Run:  python table1.py
      python table1.py --crop 8 --gt-rank essential --latex
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

from closure_metrics import align_refs, closure_report, fit_affine
from ground_truth import GroundTruthAlgebra, Metrics

METHODS = ("FINS", "LieGAN", "LaLiGAN", "LIG", "LieNLSD", "Augerino")
EQ_FREE = ("FINS", "LieGAN", "LaLiGAN", "Augerino")
ESSENTIAL = {"heat": ("\u2202_x", "\u2202_t", "u\u2202_u"),
             "burgers": ("\u2202_x", "\u2202_t", "Galilean"),
             "kdv": ("\u2202_x", "\u2202_t", "Galilean")}
# exp20, 21 trials over 3 reference samples, triple space: (d_G, cos) +/- sd
NULLS = {
    "heat": {"band": ((1.92, 0.31), (0.557, 0.027)),
             "shuffled": ((1.81, 0.37), (0.755, 0.096))},
    "burgers": {"band": ((2.09, 0.15), (0.548, 0.044)),
                "shuffled": ((1.70, 0.23), (0.719, 0.069))},
    "kdv": {"band": ((2.01, 0.13), (0.566, 0.058)),
            "shuffled": ((1.57, 0.29), (0.687, 0.109))},
}


def load_triples(z, prefix, crop):
    xi, eta, phi = z[f"{prefix}_xi"], z[f"{prefix}_eta"], z[f"{prefix}_phi"]
    if crop:
        sl = (slice(None), slice(crop, -crop), slice(crop, -crop))
        xi, eta, phi = xi[sl], eta[sl], phi[sl]
    return [(xi[k], eta[k], phi[k]) for k in range(xi.shape[0])]


def gt_triples(z, crop):
    """Ground truth for an artifact. Burgers is RECOMPUTED from x, t,
    u_sample with the factor-2 pi algebra: the textbook Galilean (t, 0, 1)
    and projective (x t, t^2, x - t u) are not symmetries of the repo's
    Burgers data (factor-2 pi solver scaling)."""
    if str(z["pde"]) == "burgers":
        c = (slice(crop, -crop), slice(crop, -crop)) if crop else (slice(None),) * 2
        return [tuple(np.asarray(q)[c] for q in g) for g in
                GroundTruthAlgebra.burgers_generators(z["x"], z["t"],
                                                      z["u_sample"])]
    return load_triples(z, "gt", crop)


def score_one(path, crop, gt_rank, with_closure=True, eff_thresh=0.2):
    z = np.load(path, allow_pickle=True)
    pde = str(z["pde"])
    names = list(z["gt_names"])
    gt = gt_triples(z, crop)
    if gt_rank == "essential":
        keep = [i for i, n in enumerate(names) if n in ESSENTIAL[pde]]
        if len(keep) != len(ESSENTIAL[pde]):
            raise SystemExit(f"essential filter matched {len(keep)} names "
                             f"in {os.path.basename(path)}: {names}")
        gt = [gt[i] for i in keep]
    kB = len(gt)
    B = Metrics.generators_to_matrix(gt)
    out = {}
    for m in METHODS:
        if f"{m}_xi" not in z:
            continue
        disc = load_triples(z, m, crop)
        kA = len(disc)
        dG = float(Metrics.grassmann_distance(
            Metrics.generators_to_matrix(disc), B))
        pen = abs(kA - kB) * (np.pi / 2) ** 2
        rec = {"d_G": dG, "angle": float(np.sqrt(max(dG ** 2 - pen, 0.0))),
               "cos": float(Metrics.best_cosine_similarity(disc, gt)),
               "k": kA, "penalty": float(pen)}
        # channel energy split: a phi-dominated set is orthogonal to every
        # xi/eta-carried generator by construction, which scores BELOW a
        # random null on cosine -- the KdV signature
        e = np.array([[float(np.sum(np.asarray(c) ** 2)) for c in g]
                      for g in disc])
        e = e / (e.sum(axis=1, keepdims=True) + 1e-30)
        rec["frac_xi"], rec["frac_eta"], rec["frac_phi"] = \
            [float(v) for v in e.mean(axis=0)]
        if f"{m}_norms" in z:
            rec["mean_norm"] = float(np.mean(z[f"{m}_norms"]))
        # Span containment for CORRELATED algebras. Per-slot
        # Hungarian cosine under-credits a method that recovers a subspace
        # rather than separate generators; raw span containment
        # over-credits it, because on x, t >= 0 fields like xi = 1, t, x are
        # strongly correlated (KdV: d_x ~ Galilean 0.85) and a vector in the
        # dominant region "contains" several generators for free. So:
        #   eff span  -- slots with norm >= 0.2 x max only, so collapsed
        #                deflation slots (normalised noise) do not inflate it
        #   novel     -- containment of the part of each GT generator that is
        #                orthogonal to the OTHER GT generators (exp19)
        norms = np.asarray(z[f"{m}_norms"], dtype=float) if f"{m}_norms" in z \
            else np.ones(kA)
        keep = norms >= eff_thresh * norms.max()
        A = Metrics.generators_to_matrix([d for d, k in zip(disc, keep) if k])
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-30)
        U, sv, _ = np.linalg.svd(A, full_matrices=False)
        Us = U[:, sv / sv[0] > 1e-6]
        Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
        rec["eff_rank"] = int(Us.shape[1])
        rec["span_ref"] = float(np.sqrt(Us.shape[1] / Bn.shape[0]))
        for j in range(Bn.shape[1]):
            g = Bn[:, j]
            rec[f"cont_{j}"] = float(np.linalg.norm(Us.T @ g))
            others = np.delete(Bn, j, axis=1)
            Qo, _ = np.linalg.qr(others)
            g_nov = g - Qo @ (Qo.T @ g)
            nf = float(np.linalg.norm(g_nov))
            rec[f"novfrac_{j}"] = nf
            rec[f"nov_{j}"] = (float(np.linalg.norm(Us.T @ (g_nov / nf)))
                               if nf > 1e-6 else float("nan"))
        Nb, sn, Qa = nonaffine_basis(gt, z["x"], z["t"], z["u_sample"], crop)
        if Nb.shape[1]:
            Up = nonaffine_part(Us, Qa)
            rec["nonaff"] = weighted_containment(Up, Nb, sn)
        if with_closure and "u_ref" in z and f"{m}_xi_ref" in z:
            u = z["u_ref"]
            x, t = z["x"], z["t"]
            dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
            axi, aeta, aphi = align_refs(z[f"{m}_xi_ref"],
                                         z[f"{m}_eta_ref"],
                                         z[f"{m}_phi_ref"])
            gens, spreads = [], []
            for k in range(axi.shape[0]):
                g, sp = fit_affine(axi[k], aeta[k], aphi[k], u)
                gens.append(g)
                spreads.append(sp)
            r = closure_report(gens, dx, dt, basis=(x, t))
            rec.update(escaped=r["escaped"], structure=r["structure"],
                       spread=float(np.mean(spreads)),
                       abelian=r["abelian_pairs"])
        out[m] = rec
    return pde, int(z["seed"]), out


def band_null_novel(gt_triples, r, trials=100, seed=0, channels=(0, 1, 2)):
    """Chance level of NOVEL-component containment for a rank-r span.

    The random-direction reference sqrt(r / D) badly understates chance for
    SMOOTH spans: GT novel components are smooth fields, and any span of
    smooth fields contains them cheaply (KdV at r = 6: 0.29-0.76 by chance,
    against 0.019 for random directions). Null: r random triples, each
    component with the GT-mean amplitude spectrum and energy (the exp20
    band-limited construction). Returns (mean, std) per GT generator.
    """
    rng = np.random.default_rng(seed)
    B = np.stack([np.concatenate([np.asarray(c).ravel() for c in g])
                  for g in gt_triples]).T
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
    novel = []
    for j in range(B.shape[1]):
        Qo, _ = np.linalg.qr(np.delete(B, j, axis=1))
        g = B[:, j] - Qo @ (Qo.T @ B[:, j])
        novel.append(g / (np.linalg.norm(g) + 1e-30))
    amp = [np.sqrt(np.mean([np.abs(np.fft.fft2(np.asarray(g[c]))) ** 2
                            for g in gt_triples], axis=0)) for c in range(3)]
    en = [np.mean([np.mean(np.asarray(g[c]) ** 2) for g in gt_triples])
          for c in range(3)]

    def field():
        out = []
        for c in range(3):
            if c not in channels:
                # restricted class: a method that structurally never emits
                # this channel (LaLiGAN: xi = eta = 0) is compared against
                # fields that also never do
                out.append(np.zeros(amp[c].shape))
                continue
            w = np.fft.fft2(rng.standard_normal(amp[c].shape))
            f = np.real(np.fft.ifft2(w / (np.abs(w) + 1e-30) * amp[c]))
            out.append(f * np.sqrt(en[c] / (np.mean(f ** 2) + 1e-30)))
        return np.concatenate([c.ravel() for c in out])

    acc = np.zeros((trials, len(novel)))
    for k in range(trials):
        A = np.stack([field() for _ in range(r)]).T
        U, _, _ = np.linalg.svd(A / (np.linalg.norm(A, axis=0) + 1e-30),
                                full_matrices=False)
        for j, g in enumerate(novel):
            acc[k, j] = np.linalg.norm(U.T @ g)
    return acc.mean(axis=0), acc.std(axis=0)


AFFINE_CLASS = ("LieGAN", "Augerino")
PHI_ONLY_CLASS = ("LaLiGAN",)      # extractor sets xi = eta = 0 by construction


def affine_null_novel(gt_triples, x, t, u, crop, r, trials=300, seed=0):
    """Chance level of novel containment for a random r-dim subspace of the
    12-dim AFFINE vector-field class xi, eta, phi = c0 + c1 x + c2 t + c3 u.

    LieGAN outputs v = L (x, t, u, 1)^T and Augerino a fixed affine group,
    so their hypothesis class is this space. Every KdV generator lies inside
    it, and 4 of 5 Burgers and 4 of 6 heat generators do too, so a random
    element of the class contains them by construction: a band-limited null
    credits such a method with discovering what its parameterisation
    already encodes. The fair null is drawn from the method's own class.
    """
    rng = np.random.default_rng(seed)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    c = (slice(crop, -crop), slice(crop, -crop)) if crop else (slice(None),) * 2
    B = np.stack([np.concatenate([np.asarray(q).ravel() for q in g])
                  for g in gt_triples]).T
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
    novel = []
    for j in range(B.shape[1]):
        Qo, _ = np.linalg.qr(np.delete(B, j, axis=1))
        g = B[:, j] - Qo @ (Qo.T @ B[:, j])
        novel.append(g / (np.linalg.norm(g) + 1e-30))
    mono = [np.ones_like(X)[c], X[c], Tm[c], np.asarray(u)[c]]
    Z = np.zeros_like(mono[0])
    basis = []
    for comp in range(3):
        for m in mono:
            parts = [Z, Z, Z]
            parts[comp] = m
            basis.append(np.concatenate([p.ravel() for p in parts]))
    Aff = np.stack(basis).T
    acc = np.zeros((trials, len(novel)))
    for k in range(trials):
        S = Aff @ rng.standard_normal((Aff.shape[1], r))
        U, _, _ = np.linalg.svd(S / (np.linalg.norm(S, axis=0) + 1e-30),
                                full_matrices=False)
        for j, g in enumerate(novel):
            acc[k, j] = np.linalg.norm(U.T @ g)
    return acc.mean(axis=0), acc.std(axis=0)


def nonaffine_basis(gt_triples, x, t, u, crop, tol=0.1):
    """Orthonormal basis of the part of span(GT) OUTSIDE the affine class.

    Basis-free and invariant under x -> x - c: span(GT) and the affine
    class are both shift-invariant, and principal angles are computed from
    an orthonormal basis of span(GT). Directions whose sine to the affine
    class is below tol are dropped. On heat this subspace is 2-dim with
    sines 0.68-0.98 and 0.17-0.70 across samples -- structure that no
    affine-class method (LieGAN, Augerino) can represent at all.
    """
    X, Tm = np.meshgrid(x, t, indexing="ij")
    c = (slice(crop, -crop), slice(crop, -crop)) if crop else (slice(None),) * 2
    B = np.stack([np.concatenate([np.asarray(q).ravel() for q in g])
                  for g in gt_triples]).T
    Qg, _ = np.linalg.qr(B)
    Z = np.zeros_like(X[c])
    aff = []
    for comp in range(3):
        for m in (np.ones_like(X), X, Tm, np.asarray(u)):
            parts = [Z, Z, Z]
            parts[comp] = m[c]
            aff.append(np.concatenate([p.ravel() for p in parts]))
    Qa, _ = np.linalg.qr(np.stack(aff).T)
    R = Qg - Qa @ (Qa.T @ Qg)
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    keep = s > tol
    return U[:, keep], s[keep], Qa


def nonaffine_part(A, Qa, tol=0.05):
    """Orthonormal basis of the part of span(A) outside the affine class.
    Both the method's span and the algebra are compared THERE: the
    non-affine subspace of the algebra is a projection and does not lie in
    span(GT) itself, so containing it directly is the wrong question."""
    P = A - Qa @ (Qa.T @ A)
    U, s, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, s > tol]


def weighted_containment(U, targets, sines):
    """Containment of the algebra's non-affine directions in span(U),
    weighted by sin^2 -- each direction counts in proportion to how much of
    the algebra actually sticks out of the affine class there. Unweighted,
    a weak direction (sine 0.24 on heat) is drowned by ~1 percent field
    error, since unstructured error is almost entirely non-affine, and
    drags a perfect method below chance. Fixed on synthetic data with a
    known answer, before any real artifact was scored."""
    if U.shape[1] == 0:
        return 0.0
    w = np.asarray(sines) ** 2
    c2 = np.linalg.norm(U.T @ targets, axis=0) ** 2
    return float(np.sqrt(np.sum(w * c2) / np.sum(w)))


def band_null_targets(gt_triples, targets, r, trials=100, seed=1, Qa=None,
                      sines=None):
    """Chance containment of fixed unit target vectors in a rank-r span of
    smooth band-limited fields (exp20 construction)."""
    rng = np.random.default_rng(seed)
    amp = [np.sqrt(np.mean([np.abs(np.fft.fft2(np.asarray(g[c]))) ** 2
                            for g in gt_triples], axis=0)) for c in range(3)]
    en = [np.mean([np.mean(np.asarray(g[c]) ** 2) for g in gt_triples])
          for c in range(3)]

    def field():
        out = []
        for c in range(3):
            w = np.fft.fft2(rng.standard_normal(amp[c].shape))
            f = np.real(np.fft.ifft2(w / (np.abs(w) + 1e-30) * amp[c]))
            out.append(f * np.sqrt(en[c] / (np.mean(f ** 2) + 1e-30)))
        return np.concatenate([q.ravel() for q in out])

    acc = np.zeros(trials)
    for k in range(trials):
        A = np.stack([field() for _ in range(r)]).T
        A = A / (np.linalg.norm(A, axis=0) + 1e-30)
        U = (nonaffine_part(A, Qa) if Qa is not None else
             np.linalg.svd(A, full_matrices=False)[0])
        acc[k] = weighted_containment(
            U, targets, sines if sines is not None
            else np.ones(targets.shape[1]))
    return float(acc.mean()), float(acc.std())


def agg(vals):
    v = np.array([x for x in vals if x == x], dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(np.median(v)), float(v.max() - v.min())


class Emitter:
    """Print and buffer. Console output is truncated by some notebook
    wrappers, so the full report always has a file destination."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\nfull report -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("FINS_RESULTS_DIR", "."))
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--gt-rank", default="full",
                    choices=["full", "essential"])
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write the full report here (default: "
                         "table1_<crop>_<gtrank>.txt in --dir)")
    ap.add_argument("--fins-tag", default="",
                    help="score the tagged FINS variant campaign, e.g. _v2")
    ap.add_argument("--null-trials", type=int, default=100,
                    help="band-limited null trials for --span z-scores")
    ap.add_argument("--eff-thresh", type=float, default=0.2,
                    help="slots with norm >= this x max norm form the "
                         "effective span (0 = all slots); a robustness "
                         "knob, reported, never tuned")
    ap.add_argument("--nonaffine", action="store_true",
                    help="containment of the part of the GT algebra outside "
                         "the affine class -- the structure only a "
                         "general-class method can represent")
    ap.add_argument("--span", action="store_true",
                    help="print span / novel-component containment per GT "
                         "generator (use when the algebra is correlated)")
    ap.add_argument("--csv", default=None,
                    help="also write one row per (pde, method, seed)")
    args = ap.parse_args()
    say = Emitter()

    ap_tag = args.fins_tag
    paths = sorted(glob.glob(os.path.join(
        args.dir, f"generators_*{ap_tag}_seed*.npz")))
    if not ap_tag:
        # base campaign only: exactly generators_{pde}_seed{N}.npz.
        base = re.compile(r"^generators_(heat|burgers|kdv)_seed\d+\.npz$")
        paths = [p for p in paths if base.match(os.path.basename(p))]
    if not paths:
        say(f"no generators_*.npz in {args.dir}")
        return
    data = {}
    for p in paths:
        pde, seed, rec = score_one(p, args.crop, args.gt_rank,
                                   not args.no_closure, args.eff_thresh)
        data.setdefault(pde, {})[seed] = rec

    say(f"crop = {args.crop} (symmetric), GT rank = {args.gt_rank}, "
          f"{len(paths)} artifacts")
    say("d_G is split: total, and the angle part with the "
          "|k_A - k_B|(pi/2)^2 configuration penalty removed\n")
    for pde in ("heat", "burgers", "kdv"):
        if pde not in data:
            continue
        seeds = sorted(data[pde])
        kB = len(ESSENTIAL[pde]) if args.gt_rank == "essential" else \
            {"heat": 6, "burgers": 5, "kdv": 4}[pde]
        say("=" * 104)
        say(f"{pde.upper()}   GT rank {kB}   seeds {seeds}   "
              f"(median, spread = max - min)")
        say("=" * 104)
        hdr = (f"{'method':<10}{'eq-free':>8}{'k':>4}{'d_G':>14}"
               f"{'angle':>14}{'cos':>14}")
        if not args.no_closure:
            hdr += f"{'escaped':>10}{'struct':>10}{'spread':>8}"
        say(hdr)
        for m in METHODS:
            if m not in data[pde][seeds[0]]:
                continue
            g = lambda k: [data[pde][s][m].get(k, float("nan"))
                           for s in seeds]
            dm, dr = agg(g("d_G"))
            am, ar = agg(g("angle"))
            cm, cr = agg(g("cos"))
            line = (f"{m:<10}{'yes' if m in EQ_FREE else 'no':>8}"
                    f"{data[pde][seeds[0]][m]['k']:>4}"
                    f"{dm:>9.3f}±{dr:<4.2f}{am:>9.3f}±{ar:<4.2f}"
                    f"{cm:>9.3f}±{cr:<4.2f}")
            if not args.no_closure:
                em, _ = agg(g("escaped"))  # nan when no multi-sample block
                sm, _ = agg(g("structure"))
                pm, _ = agg(g("spread"))
                line += f"{em:>10.3f}{sm:>10.1e}{pm:>8.2f}"
            fx, _ = agg(g("frac_xi"))
            fe, _ = agg(g("frac_eta"))
            fp, _ = agg(g("frac_phi"))
            line += f"   xi/eta/phi {fx:.2f}/{fe:.2f}/{fp:.2f}"
            say(line)
        nb, ns = NULLS[pde]["band"], NULLS[pde]["shuffled"]
        if args.span:
            # names from an artifact of THIS equation (paths[0] may belong
            # to another PDE, which mislabels the heat block)
            p_pde = next(p for p in paths
                         if os.path.basename(p).startswith(f"generators_{pde}"))
            gt_n = list(np.load(p_pde, allow_pickle=True)["gt_names"])
            if args.gt_rank == "essential":
                gt_n = [n for n in gt_n if n in ESSENTIAL[pde]]
            zp = np.load(p_pde, allow_pickle=True)
            gt_tr = gt_triples(zp, args.crop)
            if args.gt_rank == "essential":
                gt_tr = [g for g, n in zip(gt_tr, list(zp["gt_names"]))
                         if n in ESSENTIAL[pde]]
            say("  span containment -- containment/novel, z of novel vs a "
                "null from the METHOD'S OWN class at the same effective rank "
                "(band = smooth fields; affine = affine class, LieGAN/Augerino; "
                "phi = smooth phi-only fields, LaLiGAN):")
            null_cache = {}
            for m in METHODS:
                if m not in data[pde][seeds[0]]:
                    continue
                r0 = data[pde][seeds[0]][m]
                r_eff = max(int(r0.get("eff_rank", 1)), 1)
                kind = ("affine" if m in AFFINE_CLASS else
                        "phi" if m in PHI_ONLY_CLASS else "band")
                key = (kind, r_eff)
                if key not in null_cache:
                    if kind == "affine":
                        null_cache[key] = affine_null_novel(
                            gt_tr, zp["x"], zp["t"], zp["u_sample"],
                            args.crop, r_eff, trials=3 * args.null_trials)
                    elif kind == "phi":
                        null_cache[key] = band_null_novel(
                            gt_tr, r_eff, trials=args.null_trials,
                            channels=(2,))
                    else:
                        null_cache[key] = band_null_novel(
                            gt_tr, r_eff, trials=args.null_trials)
                mu, sd = null_cache[key]
                parts = []
                for j, gn in enumerate(gt_n):
                    c, _ = agg([data[pde][s][m].get(f"cont_{j}", np.nan)
                                for s in seeds])
                    nv, _ = agg([data[pde][s][m].get(f"nov_{j}", np.nan)
                                 for s in seeds])
                    z = (nv - mu[j]) / (sd[j] + 1e-9)
                    parts.append(f"{gn} {c:.2f}/{nv:.2f}({z:+.1f})")
                say(f"    {m:<9} r_eff {r_eff} [{kind:<6}]: " + "  ".join(parts))
            fr = [f"{gn} {data[pde][seeds[0]][METHODS[0]].get(f'novfrac_{j}', np.nan):.2f}"
                  for j, gn in enumerate(gt_n)] if METHODS[0] in data[pde][seeds[0]] else []
            if fr:
                say("    novel fraction of each GT generator (how much of it "
                    "is NOT predictable from the others): " + ", ".join(fr))
        if args.nonaffine:
            zq = np.load(next(p for p in paths if os.path.basename(p)
                              .startswith(f"generators_{pde}")), allow_pickle=True)
            gq = gt_triples(zq, args.crop)
            Nb, sn, Qa = nonaffine_basis(gq, zq["x"], zq["t"], zq["u_sample"],
                                         args.crop)
            if Nb.shape[1] == 0:
                say("  non-affine subspace: empty (the algebra lies in the "
                    "affine class; no affine-vs-general contrast here)")
            else:
                say(f"  non-affine subspace of the GT algebra: dim {Nb.shape[1]}, "
                    f"sines to the affine class {np.round(sn, 2)} -- "
                    f"containment in each method's effective span "
                    f"(z vs band null at the same rank; affine-class "
                    f"methods are 0 by construction):")
                cache = {}
                for m in METHODS:
                    if m not in data[pde][seeds[0]]:
                        continue
                    r_eff = max(int(data[pde][seeds[0]][m].get("eff_rank", 1)), 1)
                    if r_eff not in cache:
                        cache[r_eff] = band_null_targets(gq, Nb, r_eff,
                                                         trials=args.null_trials,
                                                         Qa=Qa, sines=sn)
                    mu, sd = cache[r_eff]
                    v, sp = agg([data[pde][s][m].get("nonaff", np.nan)
                                 for s in seeds])
                    zmax = (1.0 - mu) / (sd + 1e-9)
                    per = "  ".join(
                        f"s{s}={data[pde][s][m].get('nonaff', np.nan):.3f}"
                        f"(z{(data[pde][s][m].get('nonaff', np.nan) - mu) / (sd + 1e-9):+.1f})"
                        for s in seeds)
                    say(f"    {m:<9} r_eff {r_eff}: {v:.3f} +/- {sp:.3f}   "
                        f"null {mu:.3f} +/- {sd:.3f}   z {(v - mu) / (sd + 1e-9):+.1f}"
                        f"   (perfect recovery would give z {zmax:+.1f})")
                    say(f"              per seed: {per}")
        say(f"{'null band':<10}{'-':>8}{'-':>4}"
              f"{nb[0][0]:>9.3f}±{nb[0][1]:<4.2f}{'-':>14}"
              f"{nb[1][0]:>9.3f}±{nb[1][1]:<4.2f}")
        say(f"{'null shuf':<10}{'-':>8}{'-':>4}"
              f"{ns[0][0]:>9.3f}±{ns[0][1]:<4.2f}{'-':>14}"
              f"{ns[1][0]:>9.3f}±{ns[1][1]:<4.2f}")
        say(f"\n  z against the nulls (positive = better than the floor):")
        for m in METHODS:
            if m not in data[pde][seeds[0]]:
                continue
            dm, _ = agg([data[pde][s][m]["d_G"] for s in seeds])
            cm, _ = agg([data[pde][s][m]["cos"] for s in seeds])
            say(f"    {m:<10} d_G: band {(nb[0][0] - dm) / nb[0][1]:+6.2f}"
                  f"  shuf {(ns[0][0] - dm) / ns[0][1]:+6.2f}   |   "
                  f"cos: band {(cm - nb[1][0]) / nb[1][1]:+6.2f}"
                  f"  shuf {(cm - ns[1][0]) / ns[1][1]:+6.2f}")
        say()

    if args.csv or True:
        rows = []
        for pde in sorted(data):
            for seed in sorted(data[pde]):
                for m, rec in data[pde][seed].items():
                    rows.append(dict(pde=pde, seed=seed, method=m, **rec))
        path = args.csv or os.path.join(
            args.dir, f"table1{ap_tag}_crop{args.crop}_{args.gt_rank}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(
                {k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        say(f"per-seed rows -> {path}")

    if args.latex:
        say("\n% LaTeX body, median +/- spread over seeds")
        for pde in ("heat", "burgers", "kdv"):
            if pde not in data:
                continue
            seeds = sorted(data[pde])
            say(f"\\multicolumn{{6}}{{l}}{{\\textit{{{pde}}}}} \\\\")
            for m in METHODS:
                if m not in data[pde][seeds[0]]:
                    continue
                g = lambda k: [data[pde][s][m].get(k, float("nan"))
                               for s in seeds]
                dm, dr = agg(g("d_G"))
                am, _ = agg(g("angle"))
                cm, cr = agg(g("cos"))
                em, _ = agg(g("escaped"))
                esc = "--" if em != em else f"{em:.2f}"
                say(f"{m} & {'\\checkmark' if m in EQ_FREE else '-'} & "
                    f"{data[pde][seeds[0]][m]['k']} & "
                    f"{dm:.2f}$\\pm${dr:.2f} & {am:.2f} & "
                    f"{cm:.2f}$\\pm${cr:.2f} & {esc} \\\\")


    say.save(args.out or os.path.join(
        args.dir, f"table1{ap_tag}_crop{args.crop}_{args.gt_rank}.txt"))


if __name__ == "__main__":
    main()
