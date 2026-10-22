"""Experiment 57 -- fractional Burgers: the exponent of a nonlocal dissipation read from the symmetry, without the
equation (README_EXPERIMENTS.md, pre-registration of 08.10).

u_t + u u_x = -nu Lambda^alpha u on the torus (fburgers.py), alpha in {1, 1.25, 1.5, 1.75, 2} (fb100 ... fb200):
two Gaussian humps of either sign, 128 x 128 over T = 1, the band [0.2, 0.8] L scored; algebra d_x, d_t, Galilean
t d_x + d_u, scaling x d_x + alpha t d_t + (1 - alpha) u d_u (alpha = 2: viscous Burgers, plus the projective). For
alpha < 2 the dissipation has no jet expression: a library of local terms cannot represent it, and the time-scaling
exponent alpha -- the anomalous-diffusion exponent -- is what a practitioner wants from such data. Gaussian noise of
sigma times each trajectory's standard deviation (corrupt.py), sigma in CONDS; seeds 42-51, evaluation data 1000 + s:
  weak, pointwise      FINS through the exact solver (exp49 with exp51's frozen recipe; exp39 as exp52's pointwise);
  wsindy               reference weak SINDy then Lie (exp56_wsindy.py: the library d_x^k (u^j), j <= 4, k <= 6, and a
                       constant; hard and soft readouts), evaluation data;
  wsindy-train         the same on the 1024 training trajectories of the learned operators (seeds 42-44, clean);
  oracle               the library plus -Lambda^beta (u^j) columns, beta in ORACLE: the identified equation and the
                       exponent at which its terms balance (no Lie stage exists for a nonlocal term);
  weak-learned,        FINS through FNO steppers trained on 1024 clean 'mixed' trajectories (exp39's recipe; notebook 6,
  pointwise-learned    Colab), seeds 42-44, clean.
Scores: exp51's (finite algebra, non-trivial generators -- the Galilean boost and the scaling --, own counts) and the
time-scaling exponent a-hat of each span (exponent_readout.py).

Run:  python exp57_fburgers.py run [--pdes fb150,...] [--conds clean,n1e-2] [--seeds ...] [--parts ...]
      python exp57_fburgers.py summary
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp51_noise_weak as e51                             # noqa: E402
import exp56_tables as e56t                                # noqa: E402
import fburgers as fb                                      # noqa: E402

CONDS = {"clean": 0.0, "n1e-2": 1e-2}
PARTS = "weak,pointwise,wsindy,oracle,wsindy-train,weak-learned,pointwise-learned"
ORACLE = "0.5,0.75,1,1.25,1.5,1.75"
WEAK = ["--K", "16", "--stride", "1", "--wt", "2", "--wp", "2", "--wx", "auto", "--wx-factor", f"{e51.WX_FACTOR:g}"]
TORUS = ["--ic", "periodic", "--n", "10", "--K", "16", "--every", "16", "--thin", "1", "--interior", "--ut", "fd"]


def jobs(pdes, conds, seeds, parts, runs):
    out = []
    for c in conds:
        d = os.path.join(runs, c)
        nz = ["--noise", f"{CONDS[c]:g}"]
        for pde in pdes:
            for seed in seeds:
                ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
                p = ["--pde", pde]
                if "weak" in parts:
                    f = os.path.join(d, f"weak_{pde}_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", *p, "--op", "exact", *ds, *nz, *WEAK, "--save", f]))
                if "pointwise" in parts:
                    f = os.path.join(d, f"pointwise_{pde}_s{seed}.npz")
                    out.append((f, ["exp39_localised.py", *p, "--op", "exact", *ds, *TORUS, *nz, "--save", f]))
                if "wsindy" in parts:
                    pre = os.path.join(d, f"wsindy_{pde}_s{seed}")
                    out.append((pre + "_soft.npz", ["exp56_wsindy.py", *p, *ds, "--data", "eval", *nz, "--out", pre]))
                if "oracle" in parts:
                    pre = os.path.join(d, f"oracle_{pde}_s{seed}")
                    out.append((pre + "_oracle.json", ["exp56_wsindy.py", *p, *ds, "--data", "eval", *nz, "--oracle",
                                                       ORACLE, "--out", pre]))
                if seed not in e51.FIT_SEEDS or c != "clean":
                    continue
                if "wsindy-train" in parts:
                    pre = os.path.join(d, f"wsindy-train_{pde}_s{seed}")
                    out.append((pre + "_soft.npz", ["exp56_wsindy.py", *p, *ds, "--data", "train", *nz, "--out", pre]))
                if "weak-learned" in parts:
                    f = os.path.join(d, f"weak-learned_{pde}_s{seed}.npz")
                    out.append((f, ["exp49_weak.py", *p, "--op", "learned", *ds, "--tag", "", *nz, *WEAK, "--save", f]))
                if "pointwise-learned" in parts:
                    f = os.path.join(d, f"pointwise-learned_{pde}_s{seed}.npz")
                    out.append((f, ["exp39_localised.py", *p, "--op", "stepper", "--train-ic", "mixed", *ds, *TORUS,
                                    *nz, "--save", f]))
    return out


def oracle_table(args, conds):
    """The oracle's identified equations: how often the true exponent alone is selected, the balance exponent."""
    res, lines = {}, ["## Oracle (library plus -Lambda^beta columns): identified exponents beta, balance exponent a\n",
                      "| equation | condition | seeds | beta-hat = alpha, alone | balance a (per seed) | equations |",
                      "|---|---|---|---|---|---|"]
    for c in conds:
        for pde in args.pdes.split(","):
            fs = sorted(glob.glob(os.path.join(args.runs, c, f"oracle_{pde}_s*_oracle.json")))
            if not fs:
                continue
            js = [json.load(open(f)) for f in fs]
            al = fb.alpha(pde)
            mult = [sorted({t[2] for t in j["terms"] if t[2] is not None}) for j in js]
            ok = [m == ([al] if al < 2 else []) for m in mult]
            res.setdefault(c, {})[pde] = {str(j["seed"]): dict(balance=j["balance"], names=j["names"], mult=m)
                                          for j, m in zip(js, mult)}
            bal = " ".join("none" if j["balance"] is None else f"{j['balance']:.3g}" for j in js)
            eqs = sorted({j["names"] for j in js})
            lines.append(f"| {pde} | {c} | {len(js)} | {sum(ok)}/{len(js)} | {bal} | {len(eqs)} distinct: "
                         + "; ".join(e[:90] for e in eqs[:2]) + " |")
    return res, lines


def summary(args, conds):
    args.exp48 = False
    e51.summary(args, conds)
    ores, olines = oracle_table(args, conds)
    sets = [(pde, conds, ".", fb.alpha(pde)) for pde in args.pdes.split(",")]
    xa = argparse.Namespace(runs=args.runs, out_exponent=args.out)
    xres = e56t.readouts(xa, sets, write=False)
    xlines = e56t.exponent_lines(xres, sets)
    with open(os.path.join(args.out, "exponent.json"), "w") as fh:
        json.dump(xres, fh, indent=1)
    with open(os.path.join(args.out, "oracle.json"), "w") as fh:
        json.dump(ores, fh, indent=1)
    with open(os.path.join(args.out, "summary.md"), "a") as fh:
        fh.write("\n" + "\n".join(xlines) + "\n\n" + "\n".join(olines) + "\n")
    print("\n".join(xlines + [""] + olines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--pdes", default=",".join(fb.EQUATIONS))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seeds", default=e51.SEEDS)
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "fb_runs"))
    ap.add_argument("--out", default=os.path.join("results", "fburgers"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(e51.HERE, p) for p in (args.runs, args.out))
    conds = {c: CONDS[c] for c in args.conds.split(",")}
    if args.what == "run":
        e51.run(args, jobs(args.pdes.split(","), list(conds), [int(x) for x in args.seeds.split(",")],
                           args.parts.split(","), args.runs))
    else:
        summary(args, conds)


if __name__ == "__main__":
    main()
