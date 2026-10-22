"""Reproduction of Fig 3c and Fig 3d of Kreider, Harlim, Huang (arXiv 2511.09779) with pde/method_kreider.py and the
authors' settings (Fig_3c.m, Fig_3d.m): uniform random samples of N^2 points, N in 56, 80, 113, 160, affine generator
class, error |sin angle(truth, c)|.

  3c  u = sin(t + x) on [-1/2, 1/2]^2, first jet space, K = 20, ell = 3, J = 2, truth (t - x)(d_t - d_x),
      c = V(:, end-1)
  3d  u = exp(-x^2 / 4t) / sqrt(4 pi t) on [1, 2]^2, second jet space, K = 40, ell = 4, J = 2,
      truth 2t d_t + x d_x - u d_u, c = V(:, end)

Per run also: sin of the angle between the truth and the span of the m smallest right singular vectors (m = 1..4) and
the singular values of A. Oracle: the spectrum of A from exact jets and normals (n = 160^2), its null-space dimension.
Slopes: least squares of log(mean error) on log(N^2); theory 1 - ell/2 (d = 2) and the fitted slope of the authors'
theory curve log(N^2)^(ell/2) (N^2)^(1 - ell/2) over the same N.

Run:  python kreider_reproduce.py run --fig 3c|3d [--trials 100] [--out DIR]
      python kreider_reproduce.py summary [--out DIR]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import kreider_check as kc                                 # noqa: E402
import method_kreider as mk                                # noqa: E402

OUT = os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "kreider_dev", "validation")
NN = [56, 80, 113, 160]
SEED = 20251113
FIGS = {"3c": dict(index=0, domain=(-0.5, 0.5), K=20, ell=3, J=2, p=1, truth=kc.TRUTH_3C, col=-2,
                   u=kc.U_TRANSPORT, label="transport u = sin(t + x), first jet space"),
        "3d": dict(index=1, domain=(1.0, 2.0), K=40, ell=4, J=2, p=2, truth=kc.TRUTH_3D, col=-1,
                   u=kc.U_HEAT, label="heat kernel, second jet space")}
M_SPAN = (1, 2, 3, 4)


def oracle(fig, n_side=160):
    """Spectrum of A from exact jets and normals of n_side^2 uniform samples."""
    c = FIGS[fig]
    rng = np.random.default_rng([SEED, c["index"], 10 ** 6])
    P = rng.uniform(*c["domain"], size=(n_side ** 2, 2))
    g = mk.PolynomialGenerators(c["p"], 1)
    Z = kc.exact_jets(c["u"], c["p"], P[:, 0], P[:, 1])
    _, S = kc.exact_frames(c["u"], c["p"], P[:, 0], P[:, 1])
    sv, V = mk.null_space_svd(g.gram(Z, S))
    truth = g.from_authors_affine(c["truth"])
    return dict(sv=sv.tolist(), null_dim=int((sv < 1e-12 * sv[0]).sum()),
                dist=[mk.sin_subspace(truth, V[:, -m:]) for m in M_SPAN],
                err_col=mk.sin_subspace(truth, V[:, c["col"]]))


def run(fig, trials, out):
    c = FIGS[fig]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = kc.Log(os.path.join(out, f"kreider_reproduce_{fig}_{stamp}.log"))
    g = mk.PolynomialGenerators(c["p"], 1)
    truth = g.from_authors_affine(c["truth"])
    fu = sp.lambdify((kc.T_SYM, kc.X_SYM), c["u"], "numpy")
    cfg = {k: v for k, v in c.items() if k != "u"}
    cfg.update(u=str(c["u"]), NN=NN, trials=trials, seed=f"default_rng([{SEED}, {c['index']}, trial, N])",
               labels=g.labels, numpy=np.__version__)
    log(f"kreider_reproduce {fig} {stamp}: {c['label']}; K = {c['K']}, ell = {c['ell']}, J = {c['J']}, "
        f"p = {c['p']}, {trials} trials")
    t_or = time.time()
    orc = oracle(fig)
    log(f"oracle (exact jets and normals, 160^2): null dim {orc['null_dim']}, smallest sv/sv_max "
        f"{np.array2string(np.array(orc['sv'][-6:]) / orc['sv'][0], precision=2)}, "
        f"dist(truth, span of m smallest) {np.array2string(np.array(orc['dist']), precision=3)} "
        f"({time.time() - t_or:.1f} s)")
    path = os.path.join(out, f"kreider_reproduce_{fig}.json")
    rec = dict(config=cfg, oracle=orc, runs=[])
    t_all = time.time()
    for trial in range(trials):
        line = []
        for N in NN:
            rng = np.random.default_rng([SEED, c["index"], trial, N])
            X = c["domain"][0] + (c["domain"][1] - c["domain"][0]) * rng.random((N * N, 2))
            t0 = time.time()
            r = mk.discover(X[:, 0], X[:, 1], fu(X[:, 0], X[:, 1]), g, c["K"], c["ell"], c["J"])
            dt = time.time() - t0
            err = mk.sin_subspace(truth, r["V"][:, c["col"]])
            dist = [mk.sin_subspace(truth, r["V"][:, -m:]) for m in M_SPAN]
            rec["runs"].append(dict(trial=trial, N=N, err=err, dist=dist, sv=r["sv"].tolist(), seconds=dt))
            line.append(f"N={N} err {err:.3e} d1 {dist[0]:.2e} d2 {dist[1]:.2e} ({dt:.1f} s)")
        log(f"trial {trial:3d}: " + "; ".join(line))
        with open(path, "w") as f:
            json.dump(rec, f)
    log(f"done: {time.time() - t_all:.0f} s")


def slope(N, y):
    return float(np.polyfit(np.log(np.asarray(N, float) ** 2), np.log(y), 1)[0])


def summarize(out):
    res, md = {}, []
    md.append("# Kreider, Harlim, Huang (arXiv 2511.09779): Fig 3c and 3d reproduced with pde/method_kreider.py\n")
    chk = os.path.join(out, "kreider_check.json")
    if os.path.exists(chk):
        with open(chk) as f:
            ch = json.load(f)["results"]
        md.append("Checks (experiments/kreider_check.py): " + ", ".join(
            f"{k} {'pass' if v['passed'] else 'FAIL'}" for k, v in ch.items()) + ".\n")
    for fig, c in FIGS.items():
        path = os.path.join(out, f"kreider_reproduce_{fig}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rec = json.load(f)
        runs = rec["runs"]
        trials = sorted({r["trial"] for r in runs})
        full = [t for t in trials if sum(r["trial"] == t for r in runs) == len(NN)]
        rows = []
        for N in NN:
            rs = [r for r in runs if r["N"] == N and r["trial"] in full]
            err = np.array([r["err"] for r in rs])
            dist = np.array([r["dist"] for r in rs])
            sv = np.array([r["sv"] for r in rs])
            rows.append(dict(N=N, n=N * N, mean_err=float(err.mean()), std_err=float(err.std(ddof=1)),
                             median_err=float(np.median(err)), mean_dist=dist.mean(0).tolist(),
                             gmean_sv_smallest=np.exp(np.log(sv[:, -5:]).mean(0)).tolist(),
                             gmean_sv_max=float(np.exp(np.log(sv[:, 0]).mean())),
                             seconds=float(np.mean([r["seconds"] for r in rs]))))
        ell = c["ell"]
        theory = [np.log(N ** 2.0) ** (ell / 2) * (N ** 2.0) ** (1 - ell / 2) for N in NN]
        s = dict(trials=len(full), rows=rows, slope_mean_err=slope(NN, [r["mean_err"] for r in rows]),
                 slope_median_err=slope(NN, [r["median_err"] for r in rows]),
                 slope_dist=[slope(NN, [r["mean_dist"][i] for r in rows]) for i in range(len(M_SPAN))],
                 theory_power=1 - ell / 2, theory_fitted=slope(NN, theory), oracle=rec["oracle"])
        res[fig] = s
        md.append(f"## Fig {fig}: {c['label']}\n")
        md.append(f"K = {c['K']}, ell = {c['ell']}, J = {c['J']}, {s['trials']} trials, c = V(:, end"
                  f"{'' if c['col'] == -1 else str(c['col'] + 1)}); oracle (exact jets and normals) null-space "
                  f"dimension {rec['oracle']['null_dim']}.\n")
        md.append("| N^2 | mean err | median err | dist m=1 | m=2 | m=3 | m=4 | 5 smallest sv(A), geometric mean | "
                  "s/run |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['n']} | {r['mean_err']:.2e} | {r['median_err']:.2e} | "
                      + " | ".join(f"{v:.2e}" for v in r["mean_dist"])
                      + f" | {', '.join(f'{v:.1e}' for v in r['gmean_sv_smallest'])} | {r['seconds']:.1f} |")
        md.append(f"\nSlope of log mean err on log N^2: {s['slope_mean_err']:.2f} "
                  f"(median {s['slope_median_err']:.2f}); theory 1 - ell/2 = {s['theory_power']:.2f}, "
                  f"with the log factor fitted over these N {s['theory_fitted']:.2f}. "
                  f"Slopes of the span distances m = 1..4: " + ", ".join(f"{v:.2f}" for v in s["slope_dist"]) + ".\n")
    with open(os.path.join(out, "kreider_reproduce.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out, "kreider_validation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--fig", choices=list(FIGS))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "run":
        run(args.fig, args.trials, args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
