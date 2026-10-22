"""Experiment 48 -- noise and temporal thinning (README_EXPERIMENTS.md, pre-registration of 04.10).

Every method sees the same corrupted data (corrupt.py): Gaussian noise of sigma times each trajectory's standard
deviation, or every s-th snapshot (window K / s steps: the same physical horizon). Heat (smooth data, L = 1) and KdV
(periodic solitons), seeds 42-44, evaluation data 1000 + s:
  FINS through the exact operator (exp40 heat-kernel composed s times; exp39 exact solver on the original grid);
       u_t by fourth-order differences of the corrupted snapshots (exp39 --ut fd);
  LieNLSD (exp41): exact right-hand side, determining system on the training trajectories (LieNLSD_RHS) or on the
       evaluation trajectories FINS uses (@eval; central-difference and spectral jets); F fitted on the corrupted
       training data (10 Nx Nt points of the original grid), system on either.
FINS's windows lie on snapshots 2 .. Nt - 3 (--interior: central five-point u_t everywhere, as LieNLSD's jets use
interior times). Generators are scored on the clean evaluation trajectories (exp42, --t-margin 8 // s). Condition
"protocol": FINS exactly as the protocol's exact rows (windows from snapshot 0; KdV u_t from the aligned solver
stencil), on clean data -- the reference the other conditions change.

Run:  python exp48_noise_thinning.py run [--conds clean,n1e-3] [--pdes heat,kdv] [--seeds 42,43,44]
      python exp48_noise_thinning.py summary
Workers may run side by side: each claims a job by creating <output>.claim. After a crash, with no worker running,
pass --reset-claims once to remove stale claims. Rows: LieNLSD_RHS@eval = exact right-hand side, system on the
evaluation trajectories, central differences; _spec@eval spectral x / fourth-order t; LieNLSD_RHS and LieNLSD_F =
system on the training trajectories (as published; F fitted); LieNLSD_F@eval = the same fit, evaluation system.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = {"clean": (0.0, 1), "n1e-4": (1e-4, 1), "n3e-4": (3e-4, 1), "n1e-3": (1e-3, 1), "n1e-2": (1e-2, 1),
         "t2": (0.0, 2), "t4": (0.0, 4), "t8": (0.0, 8), "protocol": (0.0, 1)}
NX = {"heat": 64, "kdv": 128}
NT0 = 128
METHODS = "LieNLSD_RHS,LieNLSD_RHS@eval,LieNLSD_RHS_spec@eval,LieNLSD_F,LieNLSD_F@eval"
FITTED = ("LieNLSD_F", "LieNLSD_F@eval")      # queued after every exact job (a fit costs 2-10 min on one core)


def jobs(conds, pdes, seeds, runs, methods=METHODS):
    out, fits = [], []
    for c in conds:
        sigma, s = CONDS[c]
        d = os.path.join(runs, c)
        corr = ["--noise", f"{sigma:g}", "--thin", str(s)]
        win = ["--K", str(16 // s), "--every", str(16 // s)]
        for seed in seeds:
            ds = ["--seed", str(seed), "--data-seed", str(1000 + seed)]
            for pde in pdes:
                f = os.path.join(d, f"exact_{pde}_s{seed}.npz")
                inner = [] if c == "protocol" else ["--interior"]
                if pde == "heat":
                    cmd = ["exp40_window.py", "--pde", "heat", "--op", "heat-kernel", "--smooth", "--only-L", "1",
                           "--n", "10", *ds, *win, *corr, *inner, "--save", f]
                else:
                    cmd = ["exp39_localised.py", "--pde", "kdv", "--op", "exact", "--ic", "periodic", "--n", "10",
                           *ds, *win, "--ut", "stencil" if c == "protocol" else "fd", *corr, *inner, "--save", f]
                out.append((f, cmd, None))
                if c == "protocol":
                    continue
                f = os.path.join(d, f"baselines_{pde}_s{seed}.npz")
                for part, queue in (([m for m in methods.split(",") if m not in FITTED], out),
                                    ([m for m in methods.split(",") if m in FITTED], fits)):
                    if part:
                        cmd = ["exp41_baselines.py", "--pde", pde, "--seed", str(seed), "--methods", ",".join(part),
                               "--lienlsd-fit-points", str(10 * NX[pde] * NT0),
                               "--lienlsd-val-points", str(2 * NX[pde] * NT0), *corr, "--out", f]
                        queue.append((f, cmd, part))
    return out + fits


def done(f, methods):
    if not os.path.exists(f) or os.path.exists(f + ".claim"):     # absent, or being written by a worker
        return False
    if methods is None:
        return True
    try:
        have = {str(m) for m in np.load(f, allow_pickle=True)["methods"]}
    except Exception:
        return False
    return set(methods) <= have


def run(args):
    conds = args.conds.split(",")
    todo = jobs(conds, args.pdes.split(","), [int(x) for x in args.seeds.split(",")], args.runs, args.methods)
    if args.reset_claims:
        for p in glob.glob(os.path.join(args.runs, "*", "*.claim")):
            os.remove(p)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads))
    failed = set()
    while True:
        left = [j for j in todo if j[0] + str(j[2]) not in failed and not done(j[0], j[2])]
        if not left:
            break
        ran = False
        for f, cmd, methods in left:
            os.makedirs(os.path.join(os.path.dirname(f), "logs"), exist_ok=True)
            claim = f + ".claim"
            if done(f, methods):
                continue
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue                          # another worker holds this file: come back to it
            log = os.path.join(os.path.dirname(f), "logs", os.path.basename(f).replace(".npz", ".log"))
            t0 = time.time()
            try:
                with open(log, "a") as fh:
                    fh.write("$ " + " ".join(cmd) + "\n")
                    fh.flush()
                    rc = subprocess.call([sys.executable, "-u", *cmd], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                         env=env)
            finally:
                os.remove(claim)
            ok = rc == 0 and done(f, methods)
            if not ok:
                failed.add(f + str(methods))
            ran = True
            print(f"{'ok' if ok else 'FAILED'} {time.time() - t0:6.0f} s  {f}  {','.join(methods or ['FINS'])}",
                  flush=True)
            break                                 # rescan: earlier jobs may have been released meanwhile
        if not ran:
            time.sleep(20)


def summary(args):
    rows = {}
    for c, (sigma, s) in CONDS.items():
        d = os.path.join(args.runs, c)
        if not glob.glob(os.path.join(d, "*.npz")):
            continue
        js = os.path.join(d, "scores.json")
        newest = max(os.path.getmtime(p) for p in glob.glob(os.path.join(d, "*.npz")))
        if not (args.reuse and os.path.exists(js) and os.path.getmtime(js) > newest):
            with open(os.path.join(d, "tables.md"), "w") as fh:
                subprocess.check_call([sys.executable, "-u", "exp42_tables.py", "--dir", d, "--pdes", args.pdes,
                                       "--seeds", args.seeds, "--t-margin", str(max(1, 8 // s)), "--json", js],
                                      cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        for pde, tab in json.load(open(js)).items():
            for lab, r in tab.items():
                if lab.startswith("_"):
                    continue
                for cv, v in r.items():
                    rows.setdefault(pde, {}).setdefault(f"{lab} [{cv}]", {})[c] = v["missing"]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    conds = [c for c in CONDS if any(c in r for t in rows.values() for r in t.values())]
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        for pde, tab in rows.items():
            fh.write(f"## {pde}: missing dimensions, m = k, mean +/- sd over seeds (per seed in the JSON)\n\n")
            fh.write("| row | " + " | ".join(conds) + " |\n|---" * 1 + "|---" * len(conds) + "|\n")
            for lab, r in tab.items():
                cells = [f"{np.mean(r[c]):.2f} +/- {np.std(r[c]):.2f}" if c in r else "" for c in conds]
                fh.write(f"| {lab} | " + " | ".join(cells) + " |\n")
            fh.write("\n")
    print(open(os.path.join(args.out, "summary.md")).read())
    # copy the small files (tables, scores, logs) next to the summary
    for c in conds:
        src = os.path.join(args.runs, c)
        dst = os.path.join(args.out, c)
        os.makedirs(os.path.join(dst, "logs"), exist_ok=True)
        for p in glob.glob(os.path.join(src, "*.md")) + glob.glob(os.path.join(src, "*.json")):
            subprocess.check_call(["cp", p, dst])
        for p in glob.glob(os.path.join(src, "logs", "*.log")):
            subprocess.check_call(["cp", p, os.path.join(dst, "logs")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--pdes", default="heat,kdv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "noise_thin_runs"),
                    help="npz files (large: LieNLSD's fields)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("FINS_RESULTS_DIR", "results"), "noise_thin"),
                    help="summary, tables, scores and logs")
    ap.add_argument("--methods", default=METHODS, help="exp41 methods")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reset-claims", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="summary: keep a scores.json newer than its npz files")
    args = ap.parse_args()
    # relative folders are the experiments folder's, like every path the scripts receive
    args.runs, args.out = (p if os.path.isabs(p) else os.path.join(HERE, p) for p in (args.runs, args.out))
    (run if args.what == "run" else summary)(args)


if __name__ == "__main__":
    main()
